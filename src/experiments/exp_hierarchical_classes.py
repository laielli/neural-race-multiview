"""
Experiment: Hierarchical Class Structure - When Do Soft Labels Accelerate?

Hypothesis: Soft labels provide additional learning signal when there's latent
subclass structure that hard labels cannot encode.

Setup:
- K_super = 10 superclasses (what student predicts)
- K_sub = 5 subclasses per superclass (50 total, what teacher sees)
- Teacher trained on 50-class problem
- Student trained on 10-class problem

Key insight: When a "husky" sample is shown:
- Hard 10-class label: [1,0,0,...] (just "dog")
- Soft label from teacher: encodes that huskies look somewhat like wolves
- This inter-class similarity is information hard labels CANNOT provide

Prediction: Soft labels should help the student learn better representations
because they encode subclass structure.
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


class HierarchicalMultiViewDataset:
    """
    Multi-view dataset with hierarchical class structure.

    - M views, each with different signal strength
    - K_super superclasses (coarse labels)
    - K_sub subclasses per superclass (fine labels)
    - Total classes = K_super * K_sub

    Subclasses within a superclass share some view structure but differ in others,
    creating inter-class similarity that soft labels can capture.
    """

    def __init__(
        self,
        M: int = 5,
        K_super: int = 10,
        K_sub: int = 5,
        d_view: int = 50,
        n_samples: int = 5000,
        signal_decay: float = 0.5,
        subclass_similarity: float = 0.7,  # How similar subclasses are within superclass
        seed: int = 42
    ):
        self.M = M
        self.K_super = K_super
        self.K_sub = K_sub
        self.K_fine = K_super * K_sub  # Total fine classes
        self.d_view = d_view
        self.n_samples = n_samples
        self.subclass_similarity = subclass_similarity

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Signal strengths per view (decaying)
        self.signal_strengths = [signal_decay ** m for m in range(M)]

        # Generate class prototypes with hierarchical structure
        self._generate_hierarchical_prototypes()

        # Generate samples
        self._generate_samples()

    def _generate_hierarchical_prototypes(self):
        """
        Generate prototypes with hierarchical structure.

        For each superclass, we create:
        1. A superclass prototype (shared across subclasses)
        2. Subclass-specific variations

        Subclasses within a superclass share (subclass_similarity)% of their prototype.
        """
        # Superclass prototypes: K_super x M x d_view
        self.superclass_prototypes = {}

        # Fine class prototypes: K_fine x M x d_view
        self.fine_prototypes = {}

        for k_super in range(self.K_super):
            # Random superclass prototype for each view
            super_proto = {}
            for m in range(self.M):
                super_proto[m] = np.random.randn(self.d_view)
                super_proto[m] = super_proto[m] / np.linalg.norm(super_proto[m])

            self.superclass_prototypes[k_super] = super_proto

            # Generate subclass prototypes as variations of superclass
            for k_sub in range(self.K_sub):
                k_fine = k_super * self.K_sub + k_sub
                fine_proto = {}

                for m in range(self.M):
                    # Subclass = similarity * superclass + (1-similarity) * random
                    random_part = np.random.randn(self.d_view)
                    random_part = random_part / np.linalg.norm(random_part)

                    combined = (self.subclass_similarity * super_proto[m] +
                               (1 - self.subclass_similarity) * random_part)
                    combined = combined / np.linalg.norm(combined)

                    fine_proto[m] = combined

                self.fine_prototypes[k_fine] = fine_proto

    def _generate_samples(self):
        """Generate samples with hierarchical labels."""
        total_dim = self.M * self.d_view

        self.X = np.zeros((self.n_samples, total_dim))
        self.Y_fine = np.zeros(self.n_samples, dtype=np.int64)  # 50-class labels
        self.Y_coarse = np.zeros(self.n_samples, dtype=np.int64)  # 10-class labels

        samples_per_class = self.n_samples // self.K_fine

        idx = 0
        for k_fine in range(self.K_fine):
            k_super = k_fine // self.K_sub

            for _ in range(samples_per_class):
                if idx >= self.n_samples:
                    break

                # Build input from all views
                x = np.zeros(total_dim)
                for m in range(self.M):
                    view_signal = self.fine_prototypes[k_fine][m] * self.signal_strengths[m]
                    noise = np.random.randn(self.d_view) * 0.1
                    x[m * self.d_view:(m + 1) * self.d_view] = view_signal + noise

                self.X[idx] = x
                self.Y_fine[idx] = k_fine
                self.Y_coarse[idx] = k_super
                idx += 1

        # Shuffle
        perm = np.random.permutation(self.n_samples)
        self.X = self.X[perm]
        self.Y_fine = self.Y_fine[perm]
        self.Y_coarse = self.Y_coarse[perm]

    def get_tensors(self, fine=False):
        """Return X and Y as tensors."""
        X = torch.tensor(self.X, dtype=torch.float32)
        if fine:
            Y = torch.tensor(self.Y_fine, dtype=torch.long)
        else:
            Y = torch.tensor(self.Y_coarse, dtype=torch.long)
        return X, Y

    def get_superclass_from_fine(self, y_fine):
        """Convert fine labels to coarse labels."""
        return y_fine // self.K_sub


class SimpleLinearNet(nn.Module):
    """2-layer linear network."""

    def __init__(self, d_input, hidden, d_output, init_scale=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_input, hidden, bias=False)
        self.W2 = nn.Linear(hidden, d_output, bias=False)

        nn.init.orthogonal_(self.W1.weight, gain=init_scale)
        nn.init.orthogonal_(self.W2.weight, gain=init_scale)

    def forward(self, x):
        return self.W2(self.W1(x))


def measure_view_contributions(model, M, d_view):
    """Measure per-view weight norms."""
    with torch.no_grad():
        W1 = model.W1.weight.cpu().numpy()
        contributions = []
        for m in range(M):
            W1_view = W1[:, m * d_view:(m + 1) * d_view]
            contributions.append(np.linalg.norm(W1_view, 'fro'))
        total = sum(contributions)
        return [c / total for c in contributions]


def compute_effective_rank(contributions):
    """Effective rank from contribution distribution."""
    p = np.array(contributions)
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    return np.exp(entropy)


def train_teacher(dataset, hidden, epochs, lr, init_scale, seed, verbose=True,
                  use_cross_entropy=True):
    """
    Train teacher on fine-grained (K_fine-class) problem.

    Args:
        use_cross_entropy: If True, use cross-entropy loss (produces proper logits).
                          If False, use MSE loss (produces near-zero logits).
    """
    torch.manual_seed(seed)

    model = SimpleLinearNet(
        d_input=dataset.M * dataset.d_view,
        hidden=hidden,
        d_output=dataset.K_fine,
        init_scale=init_scale
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    X, Y_fine = dataset.get_tensors(fine=True)

    if not use_cross_entropy:
        Y_onehot = F.one_hot(Y_fine, num_classes=dataset.K_fine).float()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"Teacher {seed}", leave=False)

    for _ in iterator:
        optimizer.zero_grad()
        logits = model(X)
        if use_cross_entropy:
            loss = F.cross_entropy(logits, Y_fine)
        else:
            loss = F.mse_loss(logits, Y_onehot)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # Fine accuracy
        fine_preds = model(X).argmax(dim=-1)
        fine_acc = (fine_preds == Y_fine).float().mean().item()

        # Coarse accuracy (superclass)
        coarse_preds = fine_preds // dataset.K_sub
        Y_coarse = Y_fine // dataset.K_sub
        coarse_acc = (coarse_preds == Y_coarse).float().mean().item()

    return model, fine_acc, coarse_acc


def get_soft_labels_collapsed(teacher, X, dataset, temperature):
    """
    Get soft labels from teacher, collapsed to superclass level.

    Teacher outputs 50-class probabilities.
    We sum probabilities within each superclass to get 10-class soft labels.
    """
    with torch.no_grad():
        logits = teacher(X)  # [N, 50]
        probs = F.softmax(logits / temperature, dim=-1)  # [N, 50]

        # Collapse to superclass: sum probs for subclasses within each superclass
        collapsed = torch.zeros(X.shape[0], dataset.K_super)
        for k_super in range(dataset.K_super):
            start = k_super * dataset.K_sub
            end = start + dataset.K_sub
            collapsed[:, k_super] = probs[:, start:end].sum(dim=-1)

        return collapsed  # [N, 10]


def train_student_hard(dataset, hidden, epochs, lr, init_scale, seed,
                       log_interval=10, verbose=True):
    """Train student on coarse (10-class) problem with hard labels."""
    torch.manual_seed(seed)

    model = SimpleLinearNet(
        d_input=dataset.M * dataset.d_view,
        hidden=hidden,
        d_output=dataset.K_super,  # 10 classes
        init_scale=init_scale
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    X, Y_coarse = dataset.get_tensors(fine=False)
    Y_onehot = F.one_hot(Y_coarse, num_classes=dataset.K_super).float()

    history = {'epochs': [], 'accuracy': [], 'loss': [], 'view_contributions': []}

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Hard labels", leave=False)

    for epoch in iterator:
        optimizer.zero_grad()
        output = model(X)
        loss = F.mse_loss(output, Y_onehot)
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = model(X).argmax(dim=-1)
                acc = (preds == Y_coarse).float().mean().item()

            contrib = measure_view_contributions(model, dataset.M, dataset.d_view)
            history['epochs'].append(epoch)
            history['accuracy'].append(acc)
            history['loss'].append(loss.item())
            history['view_contributions'].append(contrib)
            model.train()

    return model, history


def train_student_kd(dataset, teachers, hidden, epochs, lr, init_scale, seed,
                     temperature, alpha, log_interval=10, verbose=True):
    """Train student with KD using collapsed soft labels from teacher ensemble."""
    torch.manual_seed(seed)

    model = SimpleLinearNet(
        d_input=dataset.M * dataset.d_view,
        hidden=hidden,
        d_output=dataset.K_super,  # 10 classes
        init_scale=init_scale
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    X, Y_coarse = dataset.get_tensors(fine=False)
    Y_onehot = F.one_hot(Y_coarse, num_classes=dataset.K_super).float()

    # Get soft targets from teacher ensemble (collapsed to 10 classes)
    soft_targets_list = []
    for teacher in teachers:
        soft_targets_list.append(get_soft_labels_collapsed(teacher, X, dataset, temperature))
    soft_targets = torch.stack(soft_targets_list).mean(dim=0)  # [N, 10]

    history = {'epochs': [], 'accuracy': [], 'loss': [], 'view_contributions': []}

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="KD", leave=False)

    for epoch in iterator:
        optimizer.zero_grad()
        output = model(X)

        # Hard label loss
        hard_loss = F.mse_loss(output, Y_onehot)

        # Soft label loss (KL divergence)
        student_soft = F.log_softmax(output / temperature, dim=-1)
        soft_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
        soft_loss = soft_loss * (temperature ** 2)

        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = model(X).argmax(dim=-1)
                acc = (preds == Y_coarse).float().mean().item()

            contrib = measure_view_contributions(model, dataset.M, dataset.d_view)
            history['epochs'].append(epoch)
            history['accuracy'].append(acc)
            history['loss'].append(loss.item())
            history['view_contributions'].append(contrib)
            model.train()

    return model, history


def analyze_soft_label_information(dataset, teachers, temperature, verbose=False):
    """
    Analyze how much inter-class information soft labels contain.

    For each sample, measure:
    - Entropy of soft labels (higher = more spread)
    - True class probability (higher = more confident)
    - Off-diagonal mass (probability on non-true classes)
    """
    X, Y_coarse = dataset.get_tensors(fine=False)
    _, Y_fine = dataset.get_tensors(fine=True)

    # Get soft labels from each teacher
    soft_targets_list = []
    for teacher in teachers:
        soft_targets_list.append(get_soft_labels_collapsed(teacher, X, dataset, temperature))
    soft_targets = torch.stack(soft_targets_list).mean(dim=0)  # [N, 10]

    # Also get raw (non-collapsed) soft labels to analyze fine-grained predictions
    raw_soft_list = []
    raw_logits_list = []
    for teacher in teachers:
        with torch.no_grad():
            logits = teacher(X)
            raw_logits_list.append(logits)
            raw_soft = F.softmax(logits / temperature, dim=-1)
            raw_soft_list.append(raw_soft)
    raw_soft = torch.stack(raw_soft_list).mean(dim=0)  # [N, K_fine]
    raw_logits = torch.stack(raw_logits_list).mean(dim=0)  # [N, K_fine]

    if verbose:
        # Debug: Check logit magnitudes
        print(f"\n  Debug: Logit statistics:")
        print(f"    Logit range: [{raw_logits.min().item():.3f}, {raw_logits.max().item():.3f}]")
        print(f"    Logit std: {raw_logits.std().item():.3f}")
        print(f"    After temp division: [{(raw_logits/temperature).min().item():.3f}, {(raw_logits/temperature).max().item():.3f}]")

        # Check raw probs (with τ=1) vs tempered probs
        raw_probs_no_temp = F.softmax(raw_logits, dim=-1)
        print(f"\n  Debug: Sample 0 (true fine={Y_fine[0].item()}, true coarse={Y_coarse[0].item()}):")
        print(f"    Raw logits (first 10): {raw_logits[0, :10].numpy().round(3)}")
        print(f"    Probs τ=1 (first 10): {raw_probs_no_temp[0, :10].numpy().round(3)}")
        print(f"    Probs τ={temperature} (first 10): {raw_soft[0, :10].numpy().round(3)}")

    # Entropy of collapsed soft labels
    entropy = -torch.sum(soft_targets * torch.log(soft_targets + 1e-10), dim=-1)
    avg_entropy = entropy.mean().item()

    # True class probability (collapsed)
    true_class_prob = soft_targets[torch.arange(len(Y_coarse)), Y_coarse]
    avg_true_prob = true_class_prob.mean().item()

    # Off-diagonal mass
    off_diag_mass = 1 - true_class_prob
    avg_off_diag = off_diag_mass.mean().item()

    # Fine-grained analysis: how much does teacher confuse subclasses?
    # For each sample, what fraction goes to OTHER subclasses of the SAME superclass?
    same_super_other_sub = []
    for i in range(len(Y_fine)):
        k_fine_true = Y_fine[i].item()
        k_super_true = k_fine_true // dataset.K_sub

        # Indices of other subclasses in same superclass
        start = k_super_true * dataset.K_sub
        end = start + dataset.K_sub

        # Probability on other subclasses of same superclass
        mask = torch.ones(dataset.K_fine, dtype=torch.bool)
        mask[k_fine_true] = False  # Exclude true fine class
        mask[:start] = False  # Exclude other superclasses
        mask[end:] = False

        other_sub_prob = raw_soft[i, mask].sum().item()
        same_super_other_sub.append(other_sub_prob)

    avg_same_super_confusion = np.mean(same_super_other_sub)

    if verbose:
        print(f"\n  Debug: Sample soft labels (first 3 samples):")
        for i in range(min(3, len(X))):
            print(f"    Sample {i}: true_coarse={Y_coarse[i].item()}, "
                  f"soft_probs={soft_targets[i].numpy().round(3)}")

    return {
        'avg_entropy': avg_entropy,
        'max_entropy': np.log(dataset.K_super),
        'avg_true_class_prob': avg_true_prob,
        'avg_off_diagonal_mass': avg_off_diag,
        'avg_same_superclass_confusion': avg_same_super_confusion,
    }


def run_experiment(
    M: int = 5,
    K_super: int = 10,
    K_sub: int = 3,
    d_view: int = 50,
    hidden: int = 128,
    n_samples: int = 10000,
    epochs: int = 1000,
    teacher_epochs: int = 2000,
    lr: float = 0.1,
    init_scale: float = 0.1,
    n_teachers: int = 5,
    temperature: float = 3.0,
    alpha: float = 0.7,
    subclass_similarity: float = 0.5,
    log_interval: int = 10,
    n_seeds: int = 5,
    output_dir: str = 'results',
    verbose: bool = True
):
    """Run hierarchical class experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("Hierarchical Class Experiment")
    print("=" * 70)
    print(f"Superclasses: {K_super}, Subclasses per super: {K_sub}")
    print(f"Total fine classes: {K_super * K_sub}")
    print(f"Subclass similarity: {subclass_similarity}")
    print(f"Teacher epochs: {teacher_epochs}, Student epochs: {epochs}")
    print(f"Seeds: {n_seeds}, Temperature: {temperature}, Alpha: {alpha}")
    print("=" * 70)

    # Create dataset
    torch.manual_seed(42)
    dataset = HierarchicalMultiViewDataset(
        M=M, K_super=K_super, K_sub=K_sub, d_view=d_view,
        n_samples=n_samples, subclass_similarity=subclass_similarity, seed=42
    )

    print(f"\nDataset: M={M} views, {K_super} superclasses, {K_sub} subclasses each")
    print(f"Signal strengths: {[f'{s:.3f}' for s in dataset.signal_strengths]}")

    # =========================================================================
    # Phase 1: Train teacher ensemble on fine-grained problem
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 1: Training Teacher Ensemble (50-class)")
    print("=" * 50)

    teachers = []
    teacher_metrics = []

    for i in range(n_teachers):
        teacher, fine_acc, coarse_acc = train_teacher(
            dataset, hidden, teacher_epochs, lr, init_scale, seed=i, verbose=verbose
        )
        teachers.append(teacher)

        contrib = measure_view_contributions(teacher, M, d_view)
        eff_rank = compute_effective_rank(contrib)

        teacher_metrics.append({
            'seed': i,
            'fine_accuracy': fine_acc,
            'coarse_accuracy': coarse_acc,
            'effective_rank': eff_rank,
            'view0_contribution': contrib[0]
        })

        print(f"Teacher {i}: fine_acc={fine_acc:.3f}, coarse_acc={coarse_acc:.3f}, "
              f"eff_rank={eff_rank:.3f}")

    # Analyze soft label information content
    soft_label_info = analyze_soft_label_information(dataset, teachers, temperature, verbose=True)
    print(f"\nSoft label analysis (collapsed to {K_super} superclasses):")
    print(f"  Avg entropy: {soft_label_info['avg_entropy']:.3f} / {soft_label_info['max_entropy']:.3f} (max)")
    print(f"  Avg true superclass prob: {soft_label_info['avg_true_class_prob']:.3f}")
    print(f"  Avg off-diagonal mass: {soft_label_info['avg_off_diagonal_mass']:.3f}")
    print(f"  Avg same-superclass confusion: {soft_label_info['avg_same_superclass_confusion']:.3f}")

    # =========================================================================
    # Phase 2: Train students with hard labels vs KD
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 2: Training Students (10-class)")
    print("=" * 50)

    hard_results = []
    kd_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")

        # Hard labels
        model_hard, history_hard = train_student_hard(
            dataset, hidden, epochs, lr, init_scale, seed=100+seed,
            log_interval=log_interval, verbose=verbose
        )

        contrib_hard = measure_view_contributions(model_hard, M, d_view)
        hard_results.append({
            'seed': seed,
            'final_accuracy': history_hard['accuracy'][-1],
            'final_loss': history_hard['loss'][-1],
            'effective_rank': compute_effective_rank(contrib_hard),
            'view0_contribution': contrib_hard[0],
            'history': history_hard
        })

        # KD
        model_kd, history_kd = train_student_kd(
            dataset, teachers, hidden, epochs, lr, init_scale, seed=100+seed,
            temperature=temperature, alpha=alpha, log_interval=log_interval,
            verbose=verbose
        )

        contrib_kd = measure_view_contributions(model_kd, M, d_view)
        kd_results.append({
            'seed': seed,
            'final_accuracy': history_kd['accuracy'][-1],
            'final_loss': history_kd['loss'][-1],
            'effective_rank': compute_effective_rank(contrib_kd),
            'view0_contribution': contrib_kd[0],
            'history': history_kd
        })

        print(f"  Hard: acc={history_hard['accuracy'][-1]:.3f}, "
              f"eff_rank={compute_effective_rank(contrib_hard):.3f}")
        print(f"  KD:   acc={history_kd['accuracy'][-1]:.3f}, "
              f"eff_rank={compute_effective_rank(contrib_kd):.3f}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    avg_hard_acc = np.mean([r['final_accuracy'] for r in hard_results])
    avg_kd_acc = np.mean([r['final_accuracy'] for r in kd_results])
    avg_hard_rank = np.mean([r['effective_rank'] for r in hard_results])
    avg_kd_rank = np.mean([r['effective_rank'] for r in kd_results])
    avg_hard_v0 = np.mean([r['view0_contribution'] for r in hard_results])
    avg_kd_v0 = np.mean([r['view0_contribution'] for r in kd_results])

    print(f"\n{'Metric':<25} | {'Hard Labels':<15} | {'KD':<15} | {'Diff':<10}")
    print("-" * 70)
    print(f"{'Final Accuracy':<25} | {avg_hard_acc:<15.4f} | {avg_kd_acc:<15.4f} | {avg_kd_acc - avg_hard_acc:+.4f}")
    print(f"{'Effective Rank':<25} | {avg_hard_rank:<15.3f} | {avg_kd_rank:<15.3f} | {avg_kd_rank - avg_hard_rank:+.3f}")
    print(f"{'View 0 Contribution':<25} | {avg_hard_v0:<15.3f} | {avg_kd_v0:<15.3f} | {avg_kd_v0 - avg_hard_v0:+.3f}")

    # Compare learning curves
    hard_acc_curve = np.mean([r['history']['accuracy'] for r in hard_results], axis=0)
    kd_acc_curve = np.mean([r['history']['accuracy'] for r in kd_results], axis=0)
    epochs_list = hard_results[0]['history']['epochs']

    # Find epoch where each reaches 90% accuracy
    hard_90_epoch = None
    kd_90_epoch = None
    for i, (h, k) in enumerate(zip(hard_acc_curve, kd_acc_curve)):
        if hard_90_epoch is None and h >= 0.9:
            hard_90_epoch = epochs_list[i]
        if kd_90_epoch is None and k >= 0.9:
            kd_90_epoch = epochs_list[i]

    print(f"\nLearning Speed:")
    print(f"  Hard labels reach 90% at epoch: {hard_90_epoch}")
    print(f"  KD reaches 90% at epoch: {kd_90_epoch}")
    if hard_90_epoch and kd_90_epoch:
        speedup = hard_90_epoch / kd_90_epoch
        print(f"  Speedup factor: {speedup:.2f}x")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    # Criterion 1: KD has higher or equal accuracy
    if avg_kd_acc >= avg_hard_acc - 0.01:
        print(f"✓ KD accuracy >= Hard accuracy: {avg_kd_acc:.4f} >= {avg_hard_acc:.4f}")
    else:
        print(f"✗ KD accuracy < Hard accuracy: {avg_kd_acc:.4f} < {avg_hard_acc:.4f}")
        success = False

    # Criterion 2: KD learns faster (reaches 90% sooner)
    if kd_90_epoch and hard_90_epoch and kd_90_epoch <= hard_90_epoch:
        print(f"✓ KD learns faster: {kd_90_epoch} <= {hard_90_epoch} epochs to 90%")
    elif kd_90_epoch is None and hard_90_epoch is None:
        print("~ Neither reached 90% accuracy")
    else:
        print(f"✗ KD not faster: {kd_90_epoch} vs {hard_90_epoch} epochs")
        success = False

    # Criterion 3: Soft labels contain meaningful information
    if soft_label_info['avg_off_diagonal_mass'] > 0.1:
        print(f"✓ Soft labels contain inter-class info: {soft_label_info['avg_off_diagonal_mass']:.3f} off-diagonal mass")
    else:
        print(f"✗ Soft labels too peaked: {soft_label_info['avg_off_diagonal_mass']:.3f} off-diagonal mass")

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - Soft labels provide learning signal!")
    else:
        print("EXPERIMENT INCONCLUSIVE - Need different conditions")
    print("=" * 50)

    # =========================================================================
    # Plots
    # =========================================================================
    plot_learning_curves(epochs_list, hard_acc_curve, kd_acc_curve,
                        output_path / 'figures' / 'hierarchical_learning_curves.png')

    plot_view_evolution(hard_results, kd_results, M,
                       output_path / 'figures' / 'hierarchical_view_evolution.png')

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K_super': K_super, 'K_sub': K_sub, 'd_view': d_view,
            'hidden': hidden, 'n_samples': n_samples, 'epochs': epochs,
            'teacher_epochs': teacher_epochs,
            'lr': lr, 'init_scale': init_scale, 'n_teachers': n_teachers,
            'temperature': temperature, 'alpha': alpha,
            'subclass_similarity': subclass_similarity, 'n_seeds': n_seeds
        },
        'dataset': {
            'signal_strengths': dataset.signal_strengths,
            'K_fine': dataset.K_fine
        },
        'teachers': {
            'avg_fine_accuracy': np.mean([t['fine_accuracy'] for t in teacher_metrics]),
            'avg_coarse_accuracy': np.mean([t['coarse_accuracy'] for t in teacher_metrics]),
            'avg_effective_rank': np.mean([t['effective_rank'] for t in teacher_metrics])
        },
        'soft_label_info': soft_label_info,
        'hard_labels': {
            'avg_accuracy': float(avg_hard_acc),
            'std_accuracy': float(np.std([r['final_accuracy'] for r in hard_results])),
            'avg_effective_rank': float(avg_hard_rank),
            'avg_view0_contribution': float(avg_hard_v0),
            'epoch_to_90': hard_90_epoch
        },
        'kd': {
            'avg_accuracy': float(avg_kd_acc),
            'std_accuracy': float(np.std([r['final_accuracy'] for r in kd_results])),
            'avg_effective_rank': float(avg_kd_rank),
            'avg_view0_contribution': float(avg_kd_v0),
            'epoch_to_90': kd_90_epoch
        },
        'comparison': {
            'accuracy_diff': float(avg_kd_acc - avg_hard_acc),
            'rank_diff': float(avg_kd_rank - avg_hard_rank),
            'speedup': float(hard_90_epoch / kd_90_epoch) if (hard_90_epoch and kd_90_epoch) else None
        },
        'success': success
    }

    with open(output_path / 'hierarchical_classes_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'hierarchical_classes_results.json'}")

    return save_results, success


def plot_learning_curves(epochs, hard_curve, kd_curve, output_path):
    """Plot accuracy learning curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, hard_curve, 'b-', linewidth=2, label='Hard Labels')
    ax.plot(epochs, kd_curve, 'r-', linewidth=2, label='KD (Soft Labels)')

    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curves: Hard Labels vs KD\n(Hierarchical Class Structure)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_view_evolution(hard_results, kd_results, M, output_path):
    """Plot view contribution evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.8, M))

    # Hard labels
    ax = axes[0]
    epochs = hard_results[0]['history']['epochs']
    hard_contrib = np.mean([r['history']['view_contributions'] for r in hard_results], axis=0)

    for m in range(M):
        ax.plot(epochs, hard_contrib[:, m], color=colors[m], linewidth=2, label=f'View {m}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative Contribution', fontsize=12)
    ax.set_title('Hard Labels: View Contributions', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # KD
    ax = axes[1]
    kd_contrib = np.mean([r['history']['view_contributions'] for r in kd_results], axis=0)

    for m in range(M):
        ax.plot(epochs, kd_contrib[:, m], color=colors[m], linewidth=2, label=f'View {m}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative Contribution', fontsize=12)
    ax.set_title('KD: View Contributions', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Hierarchical Class Experiment')
    parser.add_argument('--M', type=int, default=5, help='Number of views')
    parser.add_argument('--K-super', type=int, default=10, help='Number of superclasses')
    parser.add_argument('--K-sub', type=int, default=3, help='Subclasses per superclass')
    parser.add_argument('--d-view', type=int, default=50, help='Dimension per view')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden layer width')
    parser.add_argument('--samples', type=int, default=10000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--teacher-epochs', type=int, default=2000, help='Teacher training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.1, help='Init scale')
    parser.add_argument('--n-teachers', type=int, default=5, help='Number of teachers')
    parser.add_argument('--temperature', type=float, default=3.0, help='KD temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Soft label weight')
    parser.add_argument('--subclass-similarity', type=float, default=0.5,
                       help='How similar subclasses are within superclass (lower = more distinct)')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 500
        args.teacher_epochs = 1000
        args.seeds = 3
        args.n_teachers = 3
        args.log_interval = 25
        print("Quick mode: epochs=500, teacher_epochs=1000, seeds=3")

    run_experiment(
        M=args.M,
        K_super=args.K_super,
        K_sub=args.K_sub,
        d_view=args.d_view,
        hidden=args.hidden,
        n_samples=args.samples,
        epochs=args.epochs,
        teacher_epochs=args.teacher_epochs,
        lr=args.lr,
        init_scale=args.init_scale,
        n_teachers=args.n_teachers,
        temperature=args.temperature,
        alpha=args.alpha,
        subclass_similarity=args.subclass_similarity,
        log_interval=args.log_interval,
        n_seeds=args.seeds,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
