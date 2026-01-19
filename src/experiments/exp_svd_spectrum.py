"""
Experiment: SVD Spectrum Analysis - Knowledge Breadth Diagnostic

Theory: Effective rank = exp(entropy of normalized singular values).
Higher effective rank = more balanced "knowledge" across modes.

Prediction: SVD spectrum of KD models is "flatter" (more uniform) than hard labels.
This provides a diagnostic tool for measuring "knowledge breadth."

Setup:
1. Train model with hard labels
2. Train model with KD
3. Compare full SVD spectrum of weight matrices
4. Compute spectral entropy and effective rank
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_asymmetric import AsymmetricMultiViewDataset


class SimpleLinearNet(nn.Module):
    """2-layer linear network for race dynamics experiments."""

    def __init__(self, d_input, hidden, d_output, init_scale=0.1):
        super().__init__()
        self.d_input = d_input
        self.hidden = hidden
        self.d_output = d_output

        self.W1 = nn.Linear(d_input, hidden, bias=False)
        self.W2 = nn.Linear(hidden, d_output, bias=False)

        nn.init.orthogonal_(self.W1.weight, gain=init_scale)
        nn.init.orthogonal_(self.W2.weight, gain=init_scale)

    def forward(self, x):
        h = self.W1(x)
        return self.W2(h)

    def get_svd_spectrum(self):
        """Get singular values of both weight matrices."""
        with torch.no_grad():
            W1 = self.W1.weight.cpu()
            W2 = self.W2.weight.cpu()

            _, s1, _ = torch.linalg.svd(W1)
            _, s2, _ = torch.linalg.svd(W2)

            # Combined spectrum (product for effective weight)
            W_eff = W2 @ W1  # (d_output, d_input)
            _, s_eff, _ = torch.linalg.svd(W_eff)

            return {
                'W1': s1.numpy(),
                'W2': s2.numpy(),
                'W_eff': s_eff.numpy()
            }


def compute_spectral_metrics(singular_values):
    """Compute metrics from singular value spectrum."""
    s = np.array(singular_values)
    s = s[s > 1e-10]  # Remove near-zero

    # Normalize to probability distribution
    s_norm = s / s.sum()

    # Effective rank = exp(entropy)
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    effective_rank = np.exp(entropy)

    # Nuclear norm (sum of singular values)
    nuclear_norm = s.sum()

    # Frobenius norm (sqrt of sum of squared singular values)
    frobenius_norm = np.sqrt((s ** 2).sum())

    # Top-k concentration
    top1_frac = s[0] / s.sum() if len(s) > 0 else 0
    top3_frac = s[:3].sum() / s.sum() if len(s) >= 3 else 1

    # Condition number
    condition_number = s[0] / s[-1] if s[-1] > 0 else float('inf')

    return {
        'effective_rank': effective_rank,
        'spectral_entropy': entropy,
        'nuclear_norm': nuclear_norm,
        'frobenius_norm': frobenius_norm,
        'top1_fraction': top1_frac,
        'top3_fraction': top3_frac,
        'condition_number': condition_number,
        'num_sv': len(s)
    }


def measure_view_contributions(model, dataset):
    """Measure how much each view contributes."""
    with torch.no_grad():
        W1 = model.W1.weight.cpu().numpy()

        contributions = []
        for m in range(dataset.M):
            slot_start = m * dataset.d_view
            slot_end = (m + 1) * dataset.d_view
            W1_view = W1[:, slot_start:slot_end]
            contrib = np.linalg.norm(W1_view, 'fro')
            contributions.append(contrib)

        total = sum(contributions)
        return [c / total for c in contributions]


def compute_effective_rank(contributions):
    """Compute effective rank of contribution distribution."""
    p = np.array(contributions)
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    return np.exp(entropy)


def train_hard(dataset, hidden, epochs, lr, init_scale, seed, verbose=True):
    """Train with hard labels."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleLinearNet(
        d_input=dataset.M * dataset.d_view,
        hidden=hidden,
        d_output=dataset.K,
        init_scale=init_scale
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    X, Y = dataset.get_tensors()
    Y_onehot = F.one_hot(Y, num_classes=dataset.K).float()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Hard Training", leave=False)

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = F.mse_loss(output, Y_onehot)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        acc = (preds == Y).float().mean().item()

    return model, acc


def train_kd(dataset, teachers, hidden, epochs, lr, init_scale,
             seed, temperature, alpha, verbose=True):
    """Train with knowledge distillation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    student = SimpleLinearNet(
        d_input=dataset.M * dataset.d_view,
        hidden=hidden,
        d_output=dataset.K,
        init_scale=init_scale
    )

    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0)

    X, Y = dataset.get_tensors()
    Y_onehot = F.one_hot(Y, num_classes=dataset.K).float()

    with torch.no_grad():
        teacher_logits = []
        for teacher in teachers:
            teacher.eval()
            teacher_logits.append(teacher(X))
        avg_logits = torch.stack(teacher_logits).mean(dim=0)
        soft_targets = F.softmax(avg_logits / temperature, dim=-1)

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="KD Training", leave=False)

    for epoch in iterator:
        student.train()
        optimizer.zero_grad()

        student_logits = student(X)

        hard_loss = F.mse_loss(student_logits, Y_onehot)
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
        soft_loss = soft_loss * (temperature ** 2)

        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        loss.backward()
        optimizer.step()

    student.eval()
    with torch.no_grad():
        preds = student(X).argmax(dim=-1)
        acc = (preds == Y).float().mean().item()

    return student, acc


def run_experiment(
    M: int = 5,
    K: int = 10,
    d_view: int = 50,
    hidden: int = 64,
    n_samples: int = 5000,
    epochs: int = 1000,
    lr: float = 0.1,
    init_scale: float = 0.1,
    n_teachers: int = 5,
    temperature: float = 3.0,
    alpha: float = 0.7,
    n_seeds: int = 5,
    output_dir: str = 'results',
    verbose: bool = True
):
    """Run SVD spectrum analysis experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("SVD Spectrum Analysis Experiment")
    print("=" * 70)
    print(f"Seeds: {n_seeds}")
    print("=" * 70)

    # Create dataset
    torch.manual_seed(42)
    dataset = AsymmetricMultiViewDataset(
        M=M, K=K, d_view=d_view,
        n_samples=n_samples, seed=42
    )

    print(f"\nDataset: M={M} views, K={K} classes")
    print(f"Signal strengths: {dataset.signal_strengths}")

    # =========================================================================
    # Train teachers
    # =========================================================================
    print(f"\n" + "=" * 50)
    print(f"Training {n_teachers} Teachers")
    print("=" * 50)

    teachers = []
    for i in range(n_teachers):
        teacher, acc = train_hard(
            dataset, hidden, epochs, lr, init_scale,
            seed=i, verbose=verbose
        )
        teachers.append(teacher)
        print(f"Teacher {i}: acc={acc:.3f}")

    # =========================================================================
    # Train hard label models
    # =========================================================================
    print(f"\n" + "=" * 50)
    print("Training Hard Label Models")
    print("=" * 50)

    hard_models = []
    hard_metrics = []

    for seed in range(n_seeds):
        model, acc = train_hard(
            dataset, hidden, epochs, lr, init_scale,
            seed=100 + seed, verbose=verbose
        )
        hard_models.append(model)

        svd_spec = model.get_svd_spectrum()
        view_contrib = measure_view_contributions(model, dataset)

        metrics = {
            'seed': seed,
            'accuracy': acc,
            'view_effective_rank': compute_effective_rank(view_contrib),
            'view_contributions': view_contrib,
            'W1_metrics': compute_spectral_metrics(svd_spec['W1']),
            'W2_metrics': compute_spectral_metrics(svd_spec['W2']),
            'W_eff_metrics': compute_spectral_metrics(svd_spec['W_eff']),
            'W1_spectrum': svd_spec['W1'].tolist(),
            'W_eff_spectrum': svd_spec['W_eff'].tolist()
        }
        hard_metrics.append(metrics)
        print(f"Hard {seed}: acc={acc:.3f}, "
              f"W_eff eff_rank={metrics['W_eff_metrics']['effective_rank']:.3f}")

    # =========================================================================
    # Train KD models
    # =========================================================================
    print(f"\n" + "=" * 50)
    print("Training KD Models")
    print("=" * 50)

    kd_models = []
    kd_metrics = []

    for seed in range(n_seeds):
        model, acc = train_kd(
            dataset, teachers, hidden, epochs, lr, init_scale,
            seed=100 + seed, temperature=temperature, alpha=alpha,
            verbose=verbose
        )
        kd_models.append(model)

        svd_spec = model.get_svd_spectrum()
        view_contrib = measure_view_contributions(model, dataset)

        metrics = {
            'seed': seed,
            'accuracy': acc,
            'view_effective_rank': compute_effective_rank(view_contrib),
            'view_contributions': view_contrib,
            'W1_metrics': compute_spectral_metrics(svd_spec['W1']),
            'W2_metrics': compute_spectral_metrics(svd_spec['W2']),
            'W_eff_metrics': compute_spectral_metrics(svd_spec['W_eff']),
            'W1_spectrum': svd_spec['W1'].tolist(),
            'W_eff_spectrum': svd_spec['W_eff'].tolist()
        }
        kd_metrics.append(metrics)
        print(f"KD {seed}: acc={acc:.3f}, "
              f"W_eff eff_rank={metrics['W_eff_metrics']['effective_rank']:.3f}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    # Average metrics
    def avg_metric(metrics_list, key1, key2=None):
        if key2 is None:
            return np.mean([m[key1] for m in metrics_list])
        return np.mean([m[key1][key2] for m in metrics_list])

    print(f"\n{'Metric':<30} | {'Hard Labels':<15} | {'KD':<15}")
    print("-" * 65)

    # View-level metrics
    hard_view_rank = avg_metric(hard_metrics, 'view_effective_rank')
    kd_view_rank = avg_metric(kd_metrics, 'view_effective_rank')
    print(f"{'View Effective Rank':<30} | {hard_view_rank:.3f}{'':<10} | {kd_view_rank:.3f}")

    # W_eff metrics
    hard_eff_rank = avg_metric(hard_metrics, 'W_eff_metrics', 'effective_rank')
    kd_eff_rank = avg_metric(kd_metrics, 'W_eff_metrics', 'effective_rank')
    print(f"{'W_eff Spectral Eff Rank':<30} | {hard_eff_rank:.3f}{'':<10} | {kd_eff_rank:.3f}")

    hard_entropy = avg_metric(hard_metrics, 'W_eff_metrics', 'spectral_entropy')
    kd_entropy = avg_metric(kd_metrics, 'W_eff_metrics', 'spectral_entropy')
    print(f"{'W_eff Spectral Entropy':<30} | {hard_entropy:.3f}{'':<10} | {kd_entropy:.3f}")

    hard_top3 = avg_metric(hard_metrics, 'W_eff_metrics', 'top3_fraction')
    kd_top3 = avg_metric(kd_metrics, 'W_eff_metrics', 'top3_fraction')
    print(f"{'W_eff Top-3 Fraction':<30} | {hard_top3:.3f}{'':<10} | {kd_top3:.3f}")

    hard_acc = avg_metric(hard_metrics, 'accuracy')
    kd_acc = avg_metric(kd_metrics, 'accuracy')
    print(f"{'Accuracy':<30} | {hard_acc:.3f}{'':<10} | {kd_acc:.3f}")

    # Compute differences
    entropy_diff = kd_entropy - hard_entropy
    rank_diff = kd_eff_rank - hard_eff_rank
    top3_diff = kd_top3 - hard_top3

    print(f"\nDifferences (KD - Hard):")
    print(f"  Spectral Entropy: {entropy_diff:+.3f}")
    print(f"  Effective Rank: {rank_diff:+.3f}")
    print(f"  Top-3 Fraction: {top3_diff:+.3f}")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    # Criterion 1: KD has higher spectral entropy
    if kd_entropy > hard_entropy:
        print(f"✓ KD has higher spectral entropy: {kd_entropy:.3f} > {hard_entropy:.3f}")
    else:
        print(f"✗ KD doesn't have higher entropy: {kd_entropy:.3f} <= {hard_entropy:.3f}")
        success = False

    # Criterion 2: KD has lower top-3 concentration
    if kd_top3 < hard_top3:
        print(f"✓ KD has lower top-3 concentration: {kd_top3:.3f} < {hard_top3:.3f}")
    else:
        print(f"✗ KD doesn't have lower top-3: {kd_top3:.3f} >= {hard_top3:.3f}")

    # Criterion 3: Both achieve good accuracy
    if hard_acc > 0.9 and kd_acc > 0.9:
        print(f"✓ Both achieve high accuracy: hard={hard_acc:.3f}, kd={kd_acc:.3f}")
    else:
        print(f"~ Accuracy issue: hard={hard_acc:.3f}, kd={kd_acc:.3f}")

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - KD has flatter SVD spectrum!")
    else:
        print("EXPERIMENT INCONCLUSIVE - SVD difference unclear")
    print("=" * 50)

    # =========================================================================
    # Plots
    # =========================================================================
    plot_svd_spectra(hard_metrics, kd_metrics,
                     output_path / 'figures' / 'svd_spectrum_comparison.png')
    plot_spectrum_distributions(hard_metrics, kd_metrics,
                               output_path / 'figures' / 'svd_spectrum_distributions.png')

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'n_teachers': n_teachers,
            'temperature': temperature, 'alpha': alpha, 'n_seeds': n_seeds
        },
        'hard_labels': {
            'avg_view_effective_rank': float(hard_view_rank),
            'avg_spectral_effective_rank': float(hard_eff_rank),
            'avg_spectral_entropy': float(hard_entropy),
            'avg_top3_fraction': float(hard_top3),
            'avg_accuracy': float(hard_acc),
        },
        'kd': {
            'avg_view_effective_rank': float(kd_view_rank),
            'avg_spectral_effective_rank': float(kd_eff_rank),
            'avg_spectral_entropy': float(kd_entropy),
            'avg_top3_fraction': float(kd_top3),
            'avg_accuracy': float(kd_acc),
        },
        'analysis': {
            'entropy_difference': float(entropy_diff),
            'rank_difference': float(rank_diff),
            'top3_difference': float(top3_diff)
        },
        'success': success
    }

    with open(output_path / 'svd_spectrum_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'svd_spectrum_results.json'}")

    return save_results, success


def plot_svd_spectra(hard_metrics, kd_metrics, output_path):
    """Plot SVD spectra comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average spectra
    hard_spectra = [m['W_eff_spectrum'] for m in hard_metrics]
    kd_spectra = [m['W_eff_spectrum'] for m in kd_metrics]

    # Pad to same length
    max_len = max(max(len(s) for s in hard_spectra), max(len(s) for s in kd_spectra))
    hard_padded = [s + [0]*(max_len - len(s)) for s in hard_spectra]
    kd_padded = [s + [0]*(max_len - len(s)) for s in kd_spectra]

    hard_avg = np.mean(hard_padded, axis=0)
    kd_avg = np.mean(kd_padded, axis=0)

    # Log scale plot
    ax = axes[0]
    x = np.arange(1, len(hard_avg) + 1)
    ax.semilogy(x, hard_avg, 'b-', linewidth=2, marker='o', markersize=4, label='Hard Labels')
    ax.semilogy(x, kd_avg, 'r-', linewidth=2, marker='s', markersize=4, label='KD')
    ax.set_xlabel('Singular Value Index', fontsize=12)
    ax.set_ylabel('Singular Value (log scale)', fontsize=12)
    ax.set_title('SVD Spectrum (W_eff)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Normalized (cumulative) plot
    ax = axes[1]
    hard_norm = hard_avg / hard_avg.sum()
    kd_norm = kd_avg / kd_avg.sum()
    hard_cum = np.cumsum(hard_norm)
    kd_cum = np.cumsum(kd_norm)

    ax.plot(x, hard_cum, 'b-', linewidth=2, marker='o', markersize=4, label='Hard Labels')
    ax.plot(x, kd_cum, 'r-', linewidth=2, marker='s', markersize=4, label='KD')
    ax.set_xlabel('Number of Singular Values', fontsize=12)
    ax.set_ylabel('Cumulative Fraction', fontsize=12)
    ax.set_title('Cumulative SVD Energy', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_spectrum_distributions(hard_metrics, kd_metrics, output_path):
    """Plot distribution of spectral metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Effective rank
    ax = axes[0]
    hard_ranks = [m['W_eff_metrics']['effective_rank'] for m in hard_metrics]
    kd_ranks = [m['W_eff_metrics']['effective_rank'] for m in kd_metrics]

    bp = ax.boxplot([hard_ranks, kd_ranks], positions=[1, 2], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('orange')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Hard Labels', 'KD'])
    ax.set_ylabel('Effective Rank')
    ax.set_title('Spectral Effective Rank')
    ax.grid(True, alpha=0.3, axis='y')

    # Entropy
    ax = axes[1]
    hard_ent = [m['W_eff_metrics']['spectral_entropy'] for m in hard_metrics]
    kd_ent = [m['W_eff_metrics']['spectral_entropy'] for m in kd_metrics]

    bp = ax.boxplot([hard_ent, kd_ent], positions=[1, 2], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('orange')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Hard Labels', 'KD'])
    ax.set_ylabel('Spectral Entropy')
    ax.set_title('Spectral Entropy (higher = flatter)')
    ax.grid(True, alpha=0.3, axis='y')

    # Top-3 fraction
    ax = axes[2]
    hard_top3 = [m['W_eff_metrics']['top3_fraction'] for m in hard_metrics]
    kd_top3 = [m['W_eff_metrics']['top3_fraction'] for m in kd_metrics]

    bp = ax.boxplot([hard_top3, kd_top3], positions=[1, 2], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('orange')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Hard Labels', 'KD'])
    ax.set_ylabel('Top-3 Fraction')
    ax.set_title('Top-3 Concentration (lower = broader)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SVD Spectrum Analysis Experiment')
    parser.add_argument('--M', type=int, default=5, help='Number of views')
    parser.add_argument('--K', type=int, default=10, help='Number of classes')
    parser.add_argument('--d-view', type=int, default=50, help='Dimension per view')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer width')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.1, help='Init scale')
    parser.add_argument('--n-teachers', type=int, default=5, help='Number of teachers')
    parser.add_argument('--temperature', type=float, default=3.0, help='KD temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Soft label weight')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 500
        args.seeds = 3
        args.n_teachers = 3
        print("Quick mode: epochs=500, seeds=3, n_teachers=3")

    run_experiment(
        M=args.M,
        K=args.K,
        d_view=args.d_view,
        hidden=args.hidden,
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        init_scale=args.init_scale,
        n_teachers=args.n_teachers,
        temperature=args.temperature,
        alpha=args.alpha,
        n_seeds=args.seeds,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
