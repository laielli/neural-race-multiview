"""
Run all key experiments with multiple seeds for statistical rigor.

This script runs the core experiments needed for the paper with proper
statistical controls:
1. Teacher homogeneity (Finding 1)
2. Hierarchical KD speedup (Finding 2)
3. Balance-speed trade-off (Finding 3)

Results are saved with full per-seed data for confidence intervals
and significance testing.

Usage:
    python experiments/run_statistical_experiments.py           # Full run
    python experiments/run_statistical_experiments.py --quick   # Quick test
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.statistics import (
    format_result_with_stats,
    compare_methods,
    generate_results_table,
    paired_ttest,
    cohens_d
)


# =============================================================================
# Model Definition
# =============================================================================

class SimpleLinearNet(nn.Module):
    """2-layer linear network for race dynamics experiments."""

    def __init__(self, d_input, hidden, d_output, init_scale=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_input, hidden, bias=False)
        self.W2 = nn.Linear(hidden, d_output, bias=False)

        nn.init.orthogonal_(self.W1.weight, gain=init_scale)
        nn.init.orthogonal_(self.W2.weight, gain=init_scale)

    def forward(self, x):
        return self.W2(self.W1(x))


# =============================================================================
# Data Generation
# =============================================================================

class AsymmetricMultiViewDataset:
    """Multi-view dataset with asymmetric signal strengths."""

    def __init__(self, M=5, K=10, d_view=50, n_samples=5000, signal_decay=0.5, seed=42):
        self.M = M
        self.K = K
        self.d_view = d_view
        self.n_samples = n_samples

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.signal_strengths = [signal_decay ** m for m in range(M)]
        self._generate_data()

    def _generate_data(self):
        total_dim = self.M * self.d_view

        # Generate class prototypes
        self.prototypes = {}
        for k in range(self.K):
            proto = {}
            for m in range(self.M):
                p = np.random.randn(self.d_view)
                proto[m] = p / np.linalg.norm(p)
            self.prototypes[k] = proto

        # Generate samples
        self.X = np.zeros((self.n_samples, total_dim))
        self.Y = np.zeros(self.n_samples, dtype=np.int64)

        samples_per_class = self.n_samples // self.K
        idx = 0
        for k in range(self.K):
            for _ in range(samples_per_class):
                if idx >= self.n_samples:
                    break
                x = np.zeros(total_dim)
                for m in range(self.M):
                    signal = self.prototypes[k][m] * self.signal_strengths[m]
                    noise = np.random.randn(self.d_view) * 0.1
                    x[m * self.d_view:(m + 1) * self.d_view] = signal + noise
                self.X[idx] = x
                self.Y[idx] = k
                idx += 1

        # Shuffle
        perm = np.random.permutation(self.n_samples)
        self.X = self.X[perm]
        self.Y = self.Y[perm]

    def get_tensors(self):
        return torch.tensor(self.X, dtype=torch.float32), torch.tensor(self.Y, dtype=torch.long)


class HierarchicalDataset:
    """Dataset with hierarchical class structure (superclass/subclass)."""

    def __init__(self, M=5, K_super=10, K_sub=3, d_view=50, n_samples=5000,
                 signal_decay=0.5, subclass_similarity=0.5, seed=42):
        self.M = M
        self.K_super = K_super
        self.K_sub = K_sub
        self.K_fine = K_super * K_sub
        self.d_view = d_view
        self.n_samples = n_samples
        self.subclass_similarity = subclass_similarity

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.signal_strengths = [signal_decay ** m for m in range(M)]
        self._generate_hierarchical_prototypes()
        self._generate_samples()

    def _generate_hierarchical_prototypes(self):
        self.superclass_prototypes = {}
        self.fine_prototypes = {}

        for k_super in range(self.K_super):
            super_proto = {}
            for m in range(self.M):
                p = np.random.randn(self.d_view)
                super_proto[m] = p / np.linalg.norm(p)
            self.superclass_prototypes[k_super] = super_proto

            for k_sub in range(self.K_sub):
                k_fine = k_super * self.K_sub + k_sub
                fine_proto = {}
                for m in range(self.M):
                    random_part = np.random.randn(self.d_view)
                    random_part = random_part / np.linalg.norm(random_part)
                    combined = (self.subclass_similarity * super_proto[m] +
                               (1 - self.subclass_similarity) * random_part)
                    fine_proto[m] = combined / np.linalg.norm(combined)
                self.fine_prototypes[k_fine] = fine_proto

    def _generate_samples(self):
        total_dim = self.M * self.d_view
        self.X = np.zeros((self.n_samples, total_dim))
        self.Y_fine = np.zeros(self.n_samples, dtype=np.int64)
        self.Y_coarse = np.zeros(self.n_samples, dtype=np.int64)

        samples_per_class = self.n_samples // self.K_fine
        idx = 0
        for k_fine in range(self.K_fine):
            k_super = k_fine // self.K_sub
            for _ in range(samples_per_class):
                if idx >= self.n_samples:
                    break
                x = np.zeros(total_dim)
                for m in range(self.M):
                    signal = self.fine_prototypes[k_fine][m] * self.signal_strengths[m]
                    noise = np.random.randn(self.d_view) * 0.1
                    x[m * self.d_view:(m + 1) * self.d_view] = signal + noise
                self.X[idx] = x
                self.Y_fine[idx] = k_fine
                self.Y_coarse[idx] = k_super
                idx += 1

        perm = np.random.permutation(self.n_samples)
        self.X = self.X[perm]
        self.Y_fine = self.Y_fine[perm]
        self.Y_coarse = self.Y_coarse[perm]

    def get_tensors(self, fine=False):
        X = torch.tensor(self.X, dtype=torch.float32)
        Y = torch.tensor(self.Y_fine if fine else self.Y_coarse, dtype=torch.long)
        return X, Y


# =============================================================================
# Metrics
# =============================================================================

def measure_view_contributions(model, M, d_view):
    """Measure how much each view contributes to network predictions."""
    with torch.no_grad():
        W1 = model.W1.weight.cpu().numpy()
        contributions = []
        for m in range(M):
            W1_view = W1[:, m * d_view:(m + 1) * d_view]
            contributions.append(np.linalg.norm(W1_view, 'fro'))
        total = sum(contributions)
        return [c / total for c in contributions]


def compute_effective_rank(contributions):
    """Compute effective rank (higher = more balanced view usage)."""
    p = np.array(contributions)
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    return np.exp(entropy)


# =============================================================================
# Training Functions
# =============================================================================

def train_model(model, X, Y, epochs, lr, loss_fn='mse', soft_targets=None,
                temperature=3.0, alpha=0.9, log_interval=None, verbose=False):
    """Train a model and return training history."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    num_classes = int(Y.max().item()) + 1 if soft_targets is None else soft_targets.shape[1]
    Y_onehot = F.one_hot(Y, num_classes=num_classes).float()

    history = {'epochs': [], 'accuracy': [], 'loss': []}

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, leave=False)

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()

        output = model(X)

        if soft_targets is not None:
            # KD loss
            hard_loss = F.mse_loss(output, Y_onehot)
            student_soft = F.log_softmax(output / temperature, dim=-1)
            soft_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
            soft_loss = soft_loss * (temperature ** 2)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
        elif loss_fn == 'mse':
            loss = F.mse_loss(output, Y_onehot)
        else:
            loss = F.cross_entropy(output, Y)

        loss.backward()
        optimizer.step()

        if log_interval and (epoch % log_interval == 0 or epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                preds = model(X).argmax(dim=-1)
                acc = (preds == Y).float().mean().item()
            history['epochs'].append(epoch)
            history['accuracy'].append(acc)
            history['loss'].append(loss.item())
            model.train()

    # Final metrics
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        final_acc = (preds == Y).float().mean().item()

    return model, final_acc, history


def find_epoch_to_threshold(history, threshold):
    """Find first epoch where accuracy exceeds threshold."""
    for epoch, acc in zip(history['epochs'], history['accuracy']):
        if acc >= threshold:
            return epoch
    return None


# =============================================================================
# Experiment 1: Teacher Homogeneity
# =============================================================================

def run_teacher_homogeneity_experiment(
    M=5, K=10, d_view=50, hidden=64, n_samples=5000,
    epochs=1000, lr=0.1, init_scale=0.1, n_teachers=10,
    dataset_seed=42, verbose=True
) -> Dict[str, Any]:
    """
    Finding 1: All teachers converge to identical representations.

    Train multiple teachers with different seeds and measure their
    view contributions to verify they all converge to the same solution.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Experiment 1: Teacher Homogeneity")
        print("=" * 60)

    # Create fixed dataset
    dataset = AsymmetricMultiViewDataset(
        M=M, K=K, d_view=d_view, n_samples=n_samples, seed=dataset_seed
    )
    X, Y = dataset.get_tensors()

    results = {
        'view0_contributions': [],
        'effective_ranks': [],
        'accuracies': [],
        'all_contributions': []
    }

    for seed in range(n_teachers):
        torch.manual_seed(seed)
        model = SimpleLinearNet(M * d_view, hidden, K, init_scale)
        model, acc, _ = train_model(model, X, Y, epochs, lr, verbose=False)

        contrib = measure_view_contributions(model, M, d_view)
        eff_rank = compute_effective_rank(contrib)

        results['view0_contributions'].append(contrib[0])
        results['effective_ranks'].append(eff_rank)
        results['accuracies'].append(acc)
        results['all_contributions'].append(contrib)

        if verbose:
            print(f"Teacher {seed}: acc={acc:.3f}, V0={contrib[0]:.3f}, eff_rank={eff_rank:.2f}")

    # Compute statistics
    v0_stats = format_result_with_stats(
        np.array(results['view0_contributions']), "View 0 Contribution"
    )
    rank_stats = format_result_with_stats(
        np.array(results['effective_ranks']), "Effective Rank"
    )

    results['statistics'] = {
        'view0': v0_stats,
        'effective_rank': rank_stats
    }

    if verbose:
        print(f"\nView 0 Contribution: {v0_stats['formatted_mean_ci']}")
        print(f"Effective Rank: {rank_stats['formatted_mean_ci']}")
        print(f"Std of V0: {v0_stats['std']:.4f} (should be very small)")

    return results


# =============================================================================
# Experiment 2: Hierarchical KD Speedup
# =============================================================================

def run_hierarchical_speedup_experiment(
    M=5, K_super=10, K_sub=3, d_view=50, hidden=128, n_samples=10000,
    teacher_epochs=2000, student_epochs=1000, lr=0.1, init_scale=0.1,
    n_teachers=5, n_seeds=10, temperature=3.0, alpha=0.7,
    dataset_seed=42, verbose=True
) -> Dict[str, Any]:
    """
    Finding 2: Hierarchical structure enables KD speedup.

    Compare learning speed of hard labels vs KD with collapsed soft labels
    on a hierarchical classification task.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Experiment 2: Hierarchical KD Speedup")
        print("=" * 60)

    # Create dataset
    dataset = HierarchicalDataset(
        M=M, K_super=K_super, K_sub=K_sub, d_view=d_view,
        n_samples=n_samples, seed=dataset_seed
    )
    X, Y_coarse = dataset.get_tensors(fine=False)
    X_fine, Y_fine = dataset.get_tensors(fine=True)

    if verbose:
        print(f"Dataset: {K_super} superclasses, {K_sub} subclasses each ({dataset.K_fine} total)")

    # Train teacher ensemble on fine-grained task
    if verbose:
        print(f"\nTraining {n_teachers} teachers on {dataset.K_fine}-class problem...")

    teachers = []
    for i in range(n_teachers):
        torch.manual_seed(i)
        model = SimpleLinearNet(M * d_view, hidden, dataset.K_fine, init_scale)
        model, acc, _ = train_model(model, X_fine, Y_fine, teacher_epochs, lr,
                                    loss_fn='ce', verbose=False)
        teachers.append(model)
        if verbose:
            print(f"  Teacher {i}: fine_acc={acc:.3f}")

    # Get soft targets (collapsed to superclass level)
    with torch.no_grad():
        teacher_logits = []
        for teacher in teachers:
            teacher.eval()
            teacher_logits.append(teacher(X))
        avg_logits = torch.stack(teacher_logits).mean(dim=0)
        probs = F.softmax(avg_logits / temperature, dim=-1)

        # Collapse to superclass
        soft_targets = torch.zeros(X.shape[0], K_super)
        for k_super in range(K_super):
            start = k_super * K_sub
            end = start + K_sub
            soft_targets[:, k_super] = probs[:, start:end].sum(dim=-1)

    # Train students with different methods
    results = {
        'hard_labels': {
            'final_accuracies': [],
            'effective_ranks': [],
            'epochs_to_90': [],
            'epochs_to_80': []
        },
        'kd': {
            'final_accuracies': [],
            'effective_ranks': [],
            'epochs_to_90': [],
            'epochs_to_80': []
        }
    }

    log_interval = max(1, student_epochs // 100)

    for seed in range(n_seeds):
        if verbose:
            print(f"\n--- Seed {seed} ---")

        # Hard labels student
        torch.manual_seed(100 + seed)
        model_hard = SimpleLinearNet(M * d_view, hidden, K_super, init_scale)
        model_hard, acc_hard, hist_hard = train_model(
            model_hard, X, Y_coarse, student_epochs, lr,
            log_interval=log_interval, verbose=False
        )
        contrib_hard = measure_view_contributions(model_hard, M, d_view)

        results['hard_labels']['final_accuracies'].append(acc_hard)
        results['hard_labels']['effective_ranks'].append(compute_effective_rank(contrib_hard))
        results['hard_labels']['epochs_to_90'].append(find_epoch_to_threshold(hist_hard, 0.9))
        results['hard_labels']['epochs_to_80'].append(find_epoch_to_threshold(hist_hard, 0.8))

        # KD student
        torch.manual_seed(100 + seed)
        model_kd = SimpleLinearNet(M * d_view, hidden, K_super, init_scale)
        model_kd, acc_kd, hist_kd = train_model(
            model_kd, X, Y_coarse, student_epochs, lr,
            soft_targets=soft_targets, temperature=temperature, alpha=alpha,
            log_interval=log_interval, verbose=False
        )
        contrib_kd = measure_view_contributions(model_kd, M, d_view)

        results['kd']['final_accuracies'].append(acc_kd)
        results['kd']['effective_ranks'].append(compute_effective_rank(contrib_kd))
        results['kd']['epochs_to_90'].append(find_epoch_to_threshold(hist_kd, 0.9))
        results['kd']['epochs_to_80'].append(find_epoch_to_threshold(hist_kd, 0.8))

        if verbose:
            hard_90 = results['hard_labels']['epochs_to_90'][-1]
            kd_90 = results['kd']['epochs_to_90'][-1]
            print(f"  Hard: acc={acc_hard:.3f}, epoch_to_90={hard_90}")
            print(f"  KD:   acc={acc_kd:.3f}, epoch_to_90={kd_90}")

    # Compute statistics
    def compute_speedup_stats(hard_epochs, kd_epochs):
        """Compute speedup for seeds where both reached threshold."""
        speedups = []
        for h, k in zip(hard_epochs, kd_epochs):
            if h is not None and k is not None and k > 0:
                speedups.append(h / k)
        return speedups if speedups else [1.0]

    speedups_90 = compute_speedup_stats(
        results['hard_labels']['epochs_to_90'],
        results['kd']['epochs_to_90']
    )
    speedups_80 = compute_speedup_stats(
        results['hard_labels']['epochs_to_80'],
        results['kd']['epochs_to_80']
    )

    results['statistics'] = {
        'hard_accuracy': format_result_with_stats(
            np.array(results['hard_labels']['final_accuracies']), "Hard Labels Accuracy"
        ),
        'kd_accuracy': format_result_with_stats(
            np.array(results['kd']['final_accuracies']), "KD Accuracy"
        ),
        'speedup_90': format_result_with_stats(np.array(speedups_90), "Speedup at 90%"),
        'speedup_80': format_result_with_stats(np.array(speedups_80), "Speedup at 80%"),
        'accuracy_comparison': compare_methods(
            np.array(results['kd']['final_accuracies']),
            np.array(results['hard_labels']['final_accuracies']),
            "KD", "Hard Labels", paired=True
        )
    }

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Hard Labels Accuracy: {results['statistics']['hard_accuracy']['formatted_mean_ci']}")
        print(f"KD Accuracy: {results['statistics']['kd_accuracy']['formatted_mean_ci']}")
        print(f"Speedup at 90%: {results['statistics']['speedup_90']['formatted_mean_ci']}x")
        print(f"Speedup at 80%: {results['statistics']['speedup_80']['formatted_mean_ci']}x")
        print(f"\nComparison: {results['statistics']['accuracy_comparison']['formatted_comparison']}")

    return results


# =============================================================================
# Experiment 3: Balance-Speed Trade-off
# =============================================================================

def run_balance_speed_experiment(
    M=5, K=10, d_view=50, hidden=64, n_samples=5000,
    epochs=1000, lr=0.1, init_scale=0.1,
    n_teachers=5, n_seeds=10, temperature=3.0, alpha=0.9,
    dataset_seed=42, verbose=True
) -> Dict[str, Any]:
    """
    Finding 3: Balance-speed trade-off.

    Compare three methods:
    1. Hard labels baseline
    2. KD with uniform teacher weighting
    3. KD with inverse-strength teacher weighting (promotes balance)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Experiment 3: Balance-Speed Trade-off")
        print("=" * 60)

    # Create dataset
    dataset = AsymmetricMultiViewDataset(
        M=M, K=K, d_view=d_view, n_samples=n_samples, seed=dataset_seed
    )
    X, Y = dataset.get_tensors()
    Y_onehot = F.one_hot(Y, num_classes=K).float()

    # Train diverse teachers (each sees only one view)
    if verbose:
        print(f"\nTraining {M} single-view teachers...")

    single_view_teachers = []
    for m in range(M):
        # Create masked input (only view m visible)
        X_masked = X.clone()
        for other_m in range(M):
            if other_m != m:
                X_masked[:, other_m * d_view:(other_m + 1) * d_view] = 0

        torch.manual_seed(m)
        model = SimpleLinearNet(M * d_view, hidden, K, init_scale)
        model, acc, _ = train_model(model, X_masked, Y, epochs, lr, verbose=False)
        single_view_teachers.append(model)

        contrib = measure_view_contributions(model, M, d_view)
        if verbose:
            print(f"  Teacher {m} (sees V{m}): acc={acc:.3f}, V{m} contrib={contrib[m]:.3f}")

    # Generate soft targets with different weightings
    with torch.no_grad():
        teacher_logits = []
        for teacher in single_view_teachers:
            teacher.eval()
            teacher_logits.append(teacher(X))

        # Uniform weighting
        uniform_logits = torch.stack(teacher_logits).mean(dim=0)
        soft_targets_uniform = F.softmax(uniform_logits / temperature, dim=-1)

        # Inverse weighting (give more weight to weaker-view teachers)
        inverse_weights = [1.0 / s for s in dataset.signal_strengths]
        inverse_weights = [w / sum(inverse_weights) for w in inverse_weights]
        inverse_logits = sum(w * l for w, l in zip(inverse_weights, teacher_logits))
        soft_targets_inverse = F.softmax(inverse_logits / temperature, dim=-1)

    # Run experiments
    results = {
        'hard_labels': {
            'final_accuracies': [],
            'effective_ranks': [],
            'epochs_to_90': []
        },
        'kd_uniform': {
            'final_accuracies': [],
            'effective_ranks': [],
            'epochs_to_90': []
        },
        'kd_inverse': {
            'final_accuracies': [],
            'effective_ranks': [],
            'epochs_to_90': []
        }
    }

    log_interval = max(1, epochs // 100)

    for seed in range(n_seeds):
        if verbose and seed < 3:
            print(f"\n--- Seed {seed} ---")

        # Hard labels
        torch.manual_seed(100 + seed)
        model = SimpleLinearNet(M * d_view, hidden, K, init_scale)
        model, acc, hist = train_model(model, X, Y, epochs, lr,
                                       log_interval=log_interval, verbose=False)
        contrib = measure_view_contributions(model, M, d_view)
        results['hard_labels']['final_accuracies'].append(acc)
        results['hard_labels']['effective_ranks'].append(compute_effective_rank(contrib))
        results['hard_labels']['epochs_to_90'].append(find_epoch_to_threshold(hist, 0.9))

        # KD uniform
        torch.manual_seed(100 + seed)
        model = SimpleLinearNet(M * d_view, hidden, K, init_scale)
        model, acc, hist = train_model(model, X, Y, epochs, lr,
                                       soft_targets=soft_targets_uniform,
                                       temperature=temperature, alpha=alpha,
                                       log_interval=log_interval, verbose=False)
        contrib = measure_view_contributions(model, M, d_view)
        results['kd_uniform']['final_accuracies'].append(acc)
        results['kd_uniform']['effective_ranks'].append(compute_effective_rank(contrib))
        results['kd_uniform']['epochs_to_90'].append(find_epoch_to_threshold(hist, 0.9))

        # KD inverse
        torch.manual_seed(100 + seed)
        model = SimpleLinearNet(M * d_view, hidden, K, init_scale)
        model, acc, hist = train_model(model, X, Y, epochs, lr,
                                       soft_targets=soft_targets_inverse,
                                       temperature=temperature, alpha=alpha,
                                       log_interval=log_interval, verbose=False)
        contrib = measure_view_contributions(model, M, d_view)
        results['kd_inverse']['final_accuracies'].append(acc)
        results['kd_inverse']['effective_ranks'].append(compute_effective_rank(contrib))
        results['kd_inverse']['epochs_to_90'].append(find_epoch_to_threshold(hist, 0.9))

        if verbose and seed < 3:
            print(f"  Hard:    acc={results['hard_labels']['final_accuracies'][-1]:.3f}, "
                  f"rank={results['hard_labels']['effective_ranks'][-1]:.2f}")
            print(f"  Uniform: acc={results['kd_uniform']['final_accuracies'][-1]:.3f}, "
                  f"rank={results['kd_uniform']['effective_ranks'][-1]:.2f}")
            print(f"  Inverse: acc={results['kd_inverse']['final_accuracies'][-1]:.3f}, "
                  f"rank={results['kd_inverse']['effective_ranks'][-1]:.2f}")

    # Compute statistics
    results['statistics'] = {
        'hard_labels': {
            'accuracy': format_result_with_stats(
                np.array(results['hard_labels']['final_accuracies']), "Accuracy"
            ),
            'effective_rank': format_result_with_stats(
                np.array(results['hard_labels']['effective_ranks']), "Effective Rank"
            )
        },
        'kd_uniform': {
            'accuracy': format_result_with_stats(
                np.array(results['kd_uniform']['final_accuracies']), "Accuracy"
            ),
            'effective_rank': format_result_with_stats(
                np.array(results['kd_uniform']['effective_ranks']), "Effective Rank"
            )
        },
        'kd_inverse': {
            'accuracy': format_result_with_stats(
                np.array(results['kd_inverse']['final_accuracies']), "Accuracy"
            ),
            'effective_rank': format_result_with_stats(
                np.array(results['kd_inverse']['effective_ranks']), "Effective Rank"
            )
        },
        'rank_comparison_uniform_vs_hard': compare_methods(
            np.array(results['kd_uniform']['effective_ranks']),
            np.array(results['hard_labels']['effective_ranks']),
            "KD Uniform", "Hard Labels", paired=True
        ),
        'rank_comparison_inverse_vs_hard': compare_methods(
            np.array(results['kd_inverse']['effective_ranks']),
            np.array(results['hard_labels']['effective_ranks']),
            "KD Inverse", "Hard Labels", paired=True
        )
    }

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")

        for method in ['hard_labels', 'kd_uniform', 'kd_inverse']:
            stats = results['statistics'][method]
            name = method.replace('_', ' ').title()
            print(f"\n{name}:")
            print(f"  Accuracy: {stats['accuracy']['formatted_mean_ci']}")
            print(f"  Effective Rank: {stats['effective_rank']['formatted_mean_ci']}")

        print(f"\nUniform vs Hard (rank): {results['statistics']['rank_comparison_uniform_vs_hard']['formatted_comparison']}")
        print(f"Inverse vs Hard (rank): {results['statistics']['rank_comparison_inverse_vs_hard']['formatted_comparison']}")

    return results


# =============================================================================
# Main Runner
# =============================================================================

def run_all_experiments(quick=False, output_dir='../log/results/statistical'):
    """Run all experiments and save results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Experiment parameters
    if quick:
        print("Running in QUICK mode (reduced epochs/seeds)")
        params = {
            'epochs': 500,
            'teacher_epochs': 500,
            'student_epochs': 500,
            'n_teachers': 3,
            'n_seeds': 5
        }
    else:
        params = {
            'epochs': 1000,
            'teacher_epochs': 2000,
            'student_epochs': 1000,
            'n_teachers': 5,
            'n_seeds': 10
        }

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'quick_mode': quick
    }

    # Experiment 1: Teacher Homogeneity
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TEACHER HOMOGENEITY")
    print("=" * 70)

    exp1_results = run_teacher_homogeneity_experiment(
        epochs=params['epochs'],
        n_teachers=params['n_teachers'] * 2,  # More teachers for better stats
        verbose=True
    )
    all_results['exp1_teacher_homogeneity'] = exp1_results

    # Experiment 2: Hierarchical Speedup
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: HIERARCHICAL KD SPEEDUP")
    print("=" * 70)

    exp2_results = run_hierarchical_speedup_experiment(
        teacher_epochs=params['teacher_epochs'],
        student_epochs=params['student_epochs'],
        n_teachers=params['n_teachers'],
        n_seeds=params['n_seeds'],
        verbose=True
    )
    all_results['exp2_hierarchical_speedup'] = exp2_results

    # Experiment 3: Balance-Speed Trade-off
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: BALANCE-SPEED TRADE-OFF")
    print("=" * 70)

    exp3_results = run_balance_speed_experiment(
        epochs=params['epochs'],
        n_teachers=params['n_teachers'],
        n_seeds=params['n_seeds'],
        verbose=True
    )
    all_results['exp3_balance_speed'] = exp3_results

    # Save results
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj

    results_file = output_path / 'statistical_results.json'
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {results_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    print("\nFinding 1: Teacher Homogeneity")
    print(f"  View 0 Contribution: {exp1_results['statistics']['view0']['formatted_mean_ci']}")
    print(f"  Standard deviation: {exp1_results['statistics']['view0']['std']:.4f}")

    print("\nFinding 2: Hierarchical KD Speedup")
    print(f"  Hard Labels: {exp2_results['statistics']['hard_accuracy']['formatted_mean_ci']}")
    print(f"  KD: {exp2_results['statistics']['kd_accuracy']['formatted_mean_ci']}")
    print(f"  Speedup at 90%: {exp2_results['statistics']['speedup_90']['formatted_mean_ci']}x")

    print("\nFinding 3: Balance-Speed Trade-off")
    for method in ['hard_labels', 'kd_uniform', 'kd_inverse']:
        stats = exp3_results['statistics'][method]
        name = method.replace('_', ' ').title()
        print(f"  {name}: Rank={stats['effective_rank']['formatted_mean_ci']}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run statistical experiments')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default='../log/results/statistical',
                       help='Output directory')
    args = parser.parse_args()

    run_all_experiments(quick=args.quick, output_dir=args.output)


if __name__ == '__main__':
    main()
