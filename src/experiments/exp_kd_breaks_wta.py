"""
Experiment: KD Breaks Winner-Take-All with Asymmetric Signals

Core hypothesis test for the neural-race-multiview paper:
- Hard labels: Student learns mainly dominant view (WTA)
- KD from diverse teachers: Student learns from ALL views (breaks WTA)

Setup:
1. Asymmetric view signals (View 0 strongest, View M-1 weakest)
2. Train N teachers (different seeds) → each has WTA but possibly different biases
3. Train hard-label student → should show strong WTA on View 0
4. Train KD student from teacher ensemble → should have more balanced view usage

Success criteria:
- Hard-label dominance (View 0 contribution) > 0.5
- KD dominance (View 0 contribution) < Hard-label dominance
- KD shows more balanced view contributions (higher effective rank)
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

    def get_logits(self, x):
        """Alias for forward, used in KD."""
        return self.forward(x)


def measure_view_contributions(model, dataset):
    """Measure how much each view contributes to the network's predictions."""
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
    """Compute effective rank of contribution distribution (higher = more balanced)."""
    p = np.array(contributions)
    p = p / p.sum()  # Normalize to probability
    # Effective rank = exp(entropy)
    entropy = -np.sum(p * np.log(p + 1e-10))
    return np.exp(entropy)


def train_teacher(dataset, hidden, epochs, lr, init_scale, seed, verbose=True):
    """Train a single teacher with hard labels."""
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
        iterator = tqdm(iterator, desc=f"Teacher {seed}", leave=False)

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = F.mse_loss(output, Y_onehot)
        loss.backward()
        optimizer.step()

    # Final accuracy
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        acc = (preds == Y).float().mean().item()

    return model, acc


def train_hard_labels(dataset, hidden, epochs, lr, init_scale, seed, verbose=True):
    """Train student with hard labels only."""
    return train_teacher(dataset, hidden, epochs, lr, init_scale, seed, verbose)


def train_kd(dataset, teachers, hidden, epochs, lr, init_scale, seed,
             temperature=3.0, alpha=0.9, verbose=True):
    """Train student with knowledge distillation from teacher ensemble."""
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

    # Pre-compute teacher soft targets
    with torch.no_grad():
        teacher_logits = []
        for teacher in teachers:
            teacher.eval()
            teacher_logits.append(teacher(X))
        # Average soft targets from all teachers
        avg_logits = torch.stack(teacher_logits).mean(dim=0)
        soft_targets = F.softmax(avg_logits / temperature, dim=-1)

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"KD Student {seed}", leave=False)

    for epoch in iterator:
        student.train()
        optimizer.zero_grad()

        student_logits = student(X)

        # Hard label loss (MSE)
        hard_loss = F.mse_loss(student_logits, Y_onehot)

        # Soft label loss (KL divergence)
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
        soft_loss = soft_loss * (temperature ** 2)  # Scale by T^2

        # Combined loss
        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        loss.backward()
        optimizer.step()

    # Final accuracy
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
    alpha: float = 0.9,
    n_students: int = 5,
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Run KD vs hard-label comparison experiment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("KD Breaks WTA Experiment")
    print("=" * 70)

    # Create dataset (same for all)
    torch.manual_seed(42)
    dataset = AsymmetricMultiViewDataset(
        M=M, K=K, d_view=d_view,
        n_samples=n_samples, seed=42
    )

    print(f"M={M} views, K={K} classes")
    print(f"Signal strengths: {dataset.signal_strengths}")

    corr_strengths = dataset.get_view_correlation_strengths()
    predicted_winner = np.argmax(corr_strengths)
    print(f"\nCorrelation strengths (σ_1 per view):")
    for m, s in enumerate(corr_strengths):
        marker = " <- PREDICTED WINNER" if m == predicted_winner else ""
        print(f"  View {m}: {s:.4f}{marker}")
    print("=" * 70)

    # =========================================================================
    # Phase 1: Train teacher ensemble
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 1: Training Teacher Ensemble")
    print("=" * 50)

    teachers = []
    teacher_results = []

    for i in range(n_teachers):
        teacher, acc = train_teacher(
            dataset, hidden, epochs, lr, init_scale,
            seed=i, verbose=verbose
        )
        teachers.append(teacher)

        view_contrib = measure_view_contributions(teacher, dataset)
        winner = np.argmax(view_contrib)

        teacher_results.append({
            'seed': i,
            'accuracy': acc,
            'view_contributions': view_contrib,
            'winner_view': int(winner)
        })

        if verbose:
            print(f"Teacher {i}: acc={acc:.3f}, winner=View {winner}, "
                  f"contributions={[f'{c:.3f}' for c in view_contrib]}")

    # Ensemble coverage
    ensemble_contrib = np.mean([t['view_contributions'] for t in teacher_results], axis=0)
    print(f"\nEnsemble avg contributions: {[f'{c:.3f}' for c in ensemble_contrib]}")

    # =========================================================================
    # Phase 2: Train hard-label students
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 2: Training Hard-Label Students")
    print("=" * 50)

    hard_results = []

    for i in range(n_students):
        student, acc = train_hard_labels(
            dataset, hidden, epochs, lr, init_scale,
            seed=100 + i, verbose=verbose
        )

        view_contrib = measure_view_contributions(student, dataset)
        winner = np.argmax(view_contrib)
        eff_rank = compute_effective_rank(view_contrib)

        hard_results.append({
            'seed': 100 + i,
            'accuracy': acc,
            'view_contributions': view_contrib,
            'winner_view': int(winner),
            'view0_contribution': view_contrib[0],
            'effective_rank': eff_rank
        })

        if verbose:
            print(f"Hard {i}: acc={acc:.3f}, winner=View {winner}, "
                  f"View0={view_contrib[0]:.3f}, eff_rank={eff_rank:.2f}")

    # =========================================================================
    # Phase 3: Train KD students
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 3: Training KD Students")
    print("=" * 50)

    kd_results = []

    for i in range(n_students):
        student, acc = train_kd(
            dataset, teachers, hidden, epochs, lr, init_scale,
            seed=100 + i, temperature=temperature, alpha=alpha,
            verbose=verbose
        )

        view_contrib = measure_view_contributions(student, dataset)
        winner = np.argmax(view_contrib)
        eff_rank = compute_effective_rank(view_contrib)

        kd_results.append({
            'seed': 100 + i,
            'accuracy': acc,
            'view_contributions': view_contrib,
            'winner_view': int(winner),
            'view0_contribution': view_contrib[0],
            'effective_rank': eff_rank
        })

        if verbose:
            print(f"KD {i}: acc={acc:.3f}, winner=View {winner}, "
                  f"View0={view_contrib[0]:.3f}, eff_rank={eff_rank:.2f}")

    # =========================================================================
    # Results comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    hard_view0 = np.mean([r['view0_contribution'] for r in hard_results])
    hard_view0_std = np.std([r['view0_contribution'] for r in hard_results])
    hard_eff_rank = np.mean([r['effective_rank'] for r in hard_results])
    hard_acc = np.mean([r['accuracy'] for r in hard_results])

    kd_view0 = np.mean([r['view0_contribution'] for r in kd_results])
    kd_view0_std = np.std([r['view0_contribution'] for r in kd_results])
    kd_eff_rank = np.mean([r['effective_rank'] for r in kd_results])
    kd_acc = np.mean([r['accuracy'] for r in kd_results])

    print(f"\n{'Metric':<30} | {'Hard Labels':<20} | {'KD':<20}")
    print("-" * 75)
    print(f"{'View 0 Contribution':<30} | {hard_view0:.3f} ± {hard_view0_std:.3f}{'':<8} | {kd_view0:.3f} ± {kd_view0_std:.3f}")
    print(f"{'Effective Rank':<30} | {hard_eff_rank:.3f}{'':<16} | {kd_eff_rank:.3f}")
    print(f"{'Accuracy':<30} | {hard_acc:.3f}{'':<16} | {kd_acc:.3f}")

    # Average view contributions
    hard_avg_contrib = np.mean([r['view_contributions'] for r in hard_results], axis=0)
    kd_avg_contrib = np.mean([r['view_contributions'] for r in kd_results], axis=0)

    print(f"\nAverage View Contributions:")
    print(f"  Hard: {[f'{c:.3f}' for c in hard_avg_contrib]}")
    print(f"  KD:   {[f'{c:.3f}' for c in kd_avg_contrib]}")

    # =========================================================================
    # Success criteria
    # =========================================================================
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    # Criterion 1: Hard labels show WTA (View 0 > 0.35)
    if hard_view0 > 0.35:
        print(f"✓ Hard labels show WTA: View 0 = {hard_view0:.3f} > 0.35")
    else:
        print(f"✗ Hard labels don't show WTA: View 0 = {hard_view0:.3f} <= 0.35")
        success = False

    # Criterion 2: KD reduces View 0 dominance
    if kd_view0 < hard_view0:
        reduction = (hard_view0 - kd_view0) / hard_view0 * 100
        print(f"✓ KD reduces View 0 dominance: {hard_view0:.3f} → {kd_view0:.3f} ({reduction:.0f}% reduction)")
    else:
        print(f"✗ KD doesn't reduce View 0 dominance: {hard_view0:.3f} → {kd_view0:.3f}")
        success = False

    # Criterion 3: KD has higher effective rank (more balanced)
    if kd_eff_rank > hard_eff_rank:
        print(f"✓ KD more balanced: eff_rank {kd_eff_rank:.3f} > {hard_eff_rank:.3f}")
    else:
        print(f"✗ KD not more balanced: eff_rank {kd_eff_rank:.3f} <= {hard_eff_rank:.3f}")
        success = False

    # Criterion 4: Both achieve good accuracy
    if hard_acc > 0.9 and kd_acc > 0.9:
        print(f"✓ Both achieve high accuracy: hard={hard_acc:.3f}, kd={kd_acc:.3f}")
    else:
        print(f"✗ Accuracy issue: hard={hard_acc:.3f}, kd={kd_acc:.3f}")
        success = False

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - KD breaks WTA!")
    else:
        print("EXPERIMENT FAILED - KD did not break WTA as expected")
    print("=" * 50)

    # =========================================================================
    # Plot results
    # =========================================================================
    plot_view_comparison(hard_results, kd_results, dataset,
                        output_path / 'figures' / 'kd_breaks_wta_comparison.png')
    plot_contribution_distributions(hard_results, kd_results,
                                   output_path / 'figures' / 'kd_breaks_wta_distributions.png')

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'n_teachers': n_teachers,
            'temperature': temperature, 'alpha': alpha, 'n_students': n_students
        },
        'dataset': {
            'signal_strengths': dataset.signal_strengths,
            'correlation_strengths': corr_strengths,
            'predicted_winner': int(predicted_winner)
        },
        'teachers': teacher_results,
        'hard_labels': {
            'avg_view0_contribution': float(hard_view0),
            'std_view0_contribution': float(hard_view0_std),
            'avg_effective_rank': float(hard_eff_rank),
            'avg_accuracy': float(hard_acc),
            'avg_view_contributions': hard_avg_contrib.tolist(),
            'per_student': hard_results
        },
        'kd': {
            'avg_view0_contribution': float(kd_view0),
            'std_view0_contribution': float(kd_view0_std),
            'avg_effective_rank': float(kd_eff_rank),
            'avg_accuracy': float(kd_acc),
            'avg_view_contributions': kd_avg_contrib.tolist(),
            'per_student': kd_results
        },
        'success': success
    }

    with open(output_path / 'kd_breaks_wta_results.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)

    print(f"\nResults saved to: {output_path / 'kd_breaks_wta_results.json'}")

    return save_results, success


def plot_view_comparison(hard_results, kd_results, dataset, output_path):
    """Plot view contributions comparison: Hard vs KD."""
    fig, ax = plt.subplots(figsize=(12, 6))

    M = dataset.M
    x = np.arange(M)
    width = 0.35

    # Average contributions
    hard_avg = np.mean([r['view_contributions'] for r in hard_results], axis=0)
    kd_avg = np.mean([r['view_contributions'] for r in kd_results], axis=0)

    # Standard deviations
    hard_std = np.std([r['view_contributions'] for r in hard_results], axis=0)
    kd_std = np.std([r['view_contributions'] for r in kd_results], axis=0)

    bars1 = ax.bar(x - width/2, hard_avg, width, yerr=hard_std,
                   label='Hard Labels', color='steelblue', capsize=3)
    bars2 = ax.bar(x + width/2, kd_avg, width, yerr=kd_std,
                   label='KD', color='orange', capsize=3)

    # Theory line (normalized correlation strengths)
    corr = np.array(dataset.get_view_correlation_strengths())
    corr_norm = corr / corr.sum()
    ax.plot(x, corr_norm, 'k--', marker='o', markersize=8, linewidth=2,
            label='Theory (σ₁ normalized)')

    ax.axhline(y=1/M, color='gray', linestyle=':', linewidth=1, label=f'Equal (1/{M})')

    ax.set_xlabel('View', fontsize=12)
    ax.set_ylabel('Contribution (normalized)', fontsize=12)
    ax.set_title('View Contributions: Hard Labels vs KD', fontsize=14)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_contribution_distributions(hard_results, kd_results, output_path):
    """Plot distribution of View 0 contributions and effective rank."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # View 0 contributions
    ax = axes[0]
    hard_v0 = [r['view0_contribution'] for r in hard_results]
    kd_v0 = [r['view0_contribution'] for r in kd_results]

    positions = [1, 2]
    bp = ax.boxplot([hard_v0, kd_v0], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('orange')

    ax.set_xticks(positions)
    ax.set_xticklabels(['Hard Labels', 'KD'])
    ax.set_ylabel('View 0 Contribution')
    ax.set_title('View 0 Dominance (lower = less WTA)')
    ax.grid(True, alpha=0.3, axis='y')

    # Effective rank
    ax = axes[1]
    hard_er = [r['effective_rank'] for r in hard_results]
    kd_er = [r['effective_rank'] for r in kd_results]

    bp = ax.boxplot([hard_er, kd_er], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('orange')

    ax.set_xticks(positions)
    ax.set_xticklabels(['Hard Labels', 'KD'])
    ax.set_ylabel('Effective Rank')
    ax.set_title('View Balance (higher = more balanced)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='KD Breaks WTA Experiment')
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
    parser.add_argument('--alpha', type=float, default=0.9, help='Soft label weight')
    parser.add_argument('--n-students', type=int, default=5, help='Number of students')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 500
        args.n_teachers = 3
        args.n_students = 3
        print("Quick mode: epochs=500, n_teachers=3, n_students=3")

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
        n_students=args.n_students,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
