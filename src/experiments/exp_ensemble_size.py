"""
Experiment: Ensemble Size - Minimum Teacher Diversity for Coverage Transfer

Theory: KD from single teacher ≈ hard labels (teacher learned one view).
Multi-teacher ensemble provides gradient for multiple views.

Prediction: Effective rank of student grows with N teachers, saturates at ensemble coverage.

Setup:
1. Train N ∈ {1, 2, 3, 5, 10} teachers
2. For each N, train KD student from that ensemble
3. Measure student effective rank vs N teachers
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
    """Compute effective rank of contribution distribution."""
    p = np.array(contributions)
    p = p / p.sum()
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

    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        acc = (preds == Y).float().mean().item()

    return model, acc


def train_kd(dataset, teachers, hidden, epochs, lr, init_scale,
             seed, temperature, alpha, verbose=True):
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
        avg_logits = torch.stack(teacher_logits).mean(dim=0)
        soft_targets = F.softmax(avg_logits / temperature, dim=-1)

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"KD (N={len(teachers)})", leave=False)

    for epoch in iterator:
        student.train()
        optimizer.zero_grad()

        student_logits = student(X)

        # Hard label loss (MSE)
        hard_loss = F.mse_loss(student_logits, Y_onehot)

        # Soft label loss (KL divergence)
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
        soft_loss = soft_loss * (temperature ** 2)

        # Combined loss
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
    ensemble_sizes: list = None,
    temperature: float = 3.0,
    alpha: float = 0.7,
    n_student_seeds: int = 3,
    output_dir: str = 'results',
    verbose: bool = True
):
    """Run ensemble size experiment."""
    if ensemble_sizes is None:
        ensemble_sizes = [1, 2, 3, 5, 10]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("Ensemble Size Experiment")
    print("=" * 70)
    print(f"Ensemble sizes: {ensemble_sizes}")
    print(f"Student seeds: {n_student_seeds}")
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
    # Train all teachers upfront (max ensemble size)
    # =========================================================================
    max_teachers = max(ensemble_sizes)
    print(f"\n" + "=" * 50)
    print(f"Training {max_teachers} teachers")
    print("=" * 50)

    all_teachers = []
    teacher_metrics = []

    for i in range(max_teachers):
        teacher, acc = train_teacher(
            dataset, hidden, epochs, lr, init_scale,
            seed=i, verbose=verbose
        )
        all_teachers.append(teacher)
        view_contrib = measure_view_contributions(teacher, dataset)
        eff_rank = compute_effective_rank(view_contrib)
        teacher_metrics.append({
            'seed': i,
            'accuracy': acc,
            'effective_rank': eff_rank,
            'view0_contribution': view_contrib[0]
        })
        print(f"Teacher {i}: acc={acc:.3f}, eff_rank={eff_rank:.3f}, v0={view_contrib[0]:.3f}")

    # =========================================================================
    # Hard label baseline (single model, no ensemble)
    # =========================================================================
    print(f"\n" + "=" * 50)
    print("Hard Label Baseline")
    print("=" * 50)

    hard_results = []
    for seed in range(n_student_seeds):
        model, acc = train_teacher(
            dataset, hidden, epochs, lr, init_scale,
            seed=100 + seed, verbose=verbose
        )
        view_contrib = measure_view_contributions(model, dataset)
        eff_rank = compute_effective_rank(view_contrib)
        hard_results.append({
            'seed': seed,
            'accuracy': acc,
            'effective_rank': eff_rank,
            'view0_contribution': view_contrib[0]
        })

    hard_avg_eff_rank = np.mean([r['effective_rank'] for r in hard_results])
    hard_std_eff_rank = np.std([r['effective_rank'] for r in hard_results])
    print(f"Hard labels: eff_rank = {hard_avg_eff_rank:.3f} ± {hard_std_eff_rank:.3f}")

    # =========================================================================
    # Ensemble size sweep
    # =========================================================================
    print(f"\n" + "=" * 50)
    print("Ensemble Size Sweep")
    print("=" * 50)

    ensemble_results = {}

    for n_teachers in ensemble_sizes:
        print(f"\n--- N = {n_teachers} teachers ---")

        # Use first n_teachers from the pool
        teachers = all_teachers[:n_teachers]

        # Compute ensemble metrics
        ensemble_contribs = [measure_view_contributions(t, dataset) for t in teachers]
        ensemble_avg_contrib = np.mean(ensemble_contribs, axis=0)
        ensemble_eff_rank = compute_effective_rank(ensemble_avg_contrib)

        print(f"  Ensemble avg contributions: {[f'{c:.3f}' for c in ensemble_avg_contrib]}")
        print(f"  Ensemble eff_rank: {ensemble_eff_rank:.3f}")

        # Train students with this ensemble
        student_results = []

        for seed in range(n_student_seeds):
            student, acc = train_kd(
                dataset, teachers, hidden, epochs, lr, init_scale,
                seed=100 + seed, temperature=temperature, alpha=alpha,
                verbose=verbose
            )

            view_contrib = measure_view_contributions(student, dataset)
            eff_rank = compute_effective_rank(view_contrib)

            student_results.append({
                'seed': seed,
                'accuracy': acc,
                'effective_rank': eff_rank,
                'view0_contribution': view_contrib[0],
                'view_contributions': view_contrib
            })

        avg_eff_rank = np.mean([r['effective_rank'] for r in student_results])
        std_eff_rank = np.std([r['effective_rank'] for r in student_results])
        avg_acc = np.mean([r['accuracy'] for r in student_results])

        ensemble_results[n_teachers] = {
            'ensemble_effective_rank': ensemble_eff_rank,
            'ensemble_avg_contributions': ensemble_avg_contrib.tolist(),
            'student_avg_effective_rank': avg_eff_rank,
            'student_std_effective_rank': std_eff_rank,
            'student_avg_accuracy': avg_acc,
            'per_student': student_results
        }

        print(f"  Student eff_rank: {avg_eff_rank:.3f} ± {std_eff_rank:.3f}")
        print(f"  Student accuracy: {avg_acc:.3f}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'N Teachers':<12} | {'Ensemble Rank':<15} | {'Student Rank':<20} | {'Accuracy':<10}")
    print("-" * 65)
    print(f"{'Hard (0)':<12} | {'--':<15} | {hard_avg_eff_rank:.3f} ± {hard_std_eff_rank:.3f}{'':<7} | {'1.000':<10}")

    for n_teachers in ensemble_sizes:
        r = ensemble_results[n_teachers]
        print(f"{n_teachers:<12} | {r['ensemble_effective_rank']:.3f}{'':<10} | "
              f"{r['student_avg_effective_rank']:.3f} ± {r['student_std_effective_rank']:.3f}{'':<5} | "
              f"{r['student_avg_accuracy']:.3f}")

    # Compute correlation between ensemble size and student effective rank
    student_ranks = [ensemble_results[n]['student_avg_effective_rank'] for n in ensemble_sizes]
    from scipy import stats
    correlation, p_value = stats.spearmanr(ensemble_sizes, student_ranks)

    print(f"\nCorrelation Analysis:")
    print(f"  Spearman ρ(N, student_eff_rank): {correlation:.3f} (p={p_value:.4f})")

    # Check improvement from N=1 to max
    rank_n1 = ensemble_results[1]['student_avg_effective_rank']
    rank_max = ensemble_results[max(ensemble_sizes)]['student_avg_effective_rank']
    improvement = rank_max - rank_n1

    print(f"\nImprovement from N=1 to N={max(ensemble_sizes)}: {improvement:.3f}")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    # Criterion 1: Positive correlation with ensemble size
    if correlation > 0.5:
        print(f"✓ Positive correlation with ensemble size: ρ = {correlation:.3f}")
    else:
        print(f"✗ Weak/no correlation with ensemble size: ρ = {correlation:.3f}")
        success = False

    # Criterion 2: N=1 worse than N>1
    if rank_n1 < rank_max:
        print(f"✓ Single teacher worse than ensemble: {rank_n1:.3f} < {rank_max:.3f}")
    else:
        print(f"✗ Single teacher not worse: {rank_n1:.3f} >= {rank_max:.3f}")
        success = False

    # Criterion 3: KD beats hard labels at some N
    kd_beats_hard = any(ensemble_results[n]['student_avg_effective_rank'] > hard_avg_eff_rank
                       for n in ensemble_sizes)
    if kd_beats_hard:
        print(f"✓ KD beats hard labels at some ensemble size")
    else:
        print(f"✗ KD never beats hard labels")

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - Ensemble size matters for KD!")
    else:
        print("EXPERIMENT INCONCLUSIVE - Ensemble size effect unclear")
    print("=" * 50)

    # =========================================================================
    # Plots
    # =========================================================================
    plot_ensemble_size(ensemble_sizes, ensemble_results, hard_avg_eff_rank,
                       output_path / 'figures' / 'ensemble_size.png')

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'ensemble_sizes': ensemble_sizes,
            'temperature': temperature, 'alpha': alpha,
            'n_student_seeds': n_student_seeds
        },
        'dataset': {
            'signal_strengths': dataset.signal_strengths,
        },
        'teachers': teacher_metrics,
        'hard_labels': {
            'avg_effective_rank': float(hard_avg_eff_rank),
            'std_effective_rank': float(hard_std_eff_rank),
            'per_seed': hard_results
        },
        'ensemble_sweep': {
            n: {
                'ensemble_effective_rank': float(ensemble_results[n]['ensemble_effective_rank']),
                'student_avg_effective_rank': float(ensemble_results[n]['student_avg_effective_rank']),
                'student_std_effective_rank': float(ensemble_results[n]['student_std_effective_rank']),
                'student_avg_accuracy': float(ensemble_results[n]['student_avg_accuracy']),
            }
            for n in ensemble_sizes
        },
        'analysis': {
            'spearman_correlation': float(correlation),
            'spearman_p_value': float(p_value),
            'improvement_n1_to_max': float(improvement)
        },
        'success': success
    }

    with open(output_path / 'ensemble_size_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'ensemble_size_results.json'}")

    return save_results, success


def plot_ensemble_size(ensemble_sizes, ensemble_results, hard_baseline, output_path):
    """Plot student effective rank vs ensemble size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Student effective rank
    student_ranks = [ensemble_results[n]['student_avg_effective_rank'] for n in ensemble_sizes]
    student_stds = [ensemble_results[n]['student_std_effective_rank'] for n in ensemble_sizes]

    ax.errorbar(ensemble_sizes, student_ranks, yerr=student_stds, marker='o',
                markersize=10, linewidth=2, capsize=5, color='steelblue',
                label='Student (KD)')

    # Ensemble effective rank
    ensemble_ranks = [ensemble_results[n]['ensemble_effective_rank'] for n in ensemble_sizes]
    ax.plot(ensemble_sizes, ensemble_ranks, marker='s', markersize=8, linewidth=2,
            linestyle='--', color='green', label='Teacher Ensemble')

    # Hard label baseline
    ax.axhline(y=hard_baseline, color='red', linestyle=':', linewidth=2,
               label='Hard Labels Baseline')

    # Max possible (uniform)
    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=1,
               label='Max (uniform, M=5)')

    ax.set_xlabel('Number of Teachers (N)', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Student Effective Rank vs Ensemble Size\n'
                 '(More teachers → More balanced knowledge)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ensemble_sizes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Ensemble Size Experiment')
    parser.add_argument('--M', type=int, default=5, help='Number of views')
    parser.add_argument('--K', type=int, default=10, help='Number of classes')
    parser.add_argument('--d-view', type=int, default=50, help='Dimension per view')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer width')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.1, help='Init scale')
    parser.add_argument('--temperature', type=float, default=3.0, help='KD temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Soft label weight')
    parser.add_argument('--seeds', type=int, default=3, help='Student seeds per ensemble')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    ensemble_sizes = [1, 2, 3, 5, 10]

    if args.quick:
        args.epochs = 500
        args.seeds = 2
        ensemble_sizes = [1, 2, 5]
        print("Quick mode: epochs=500, seeds=2, ensemble_sizes=[1, 2, 5]")

    run_experiment(
        M=args.M,
        K=args.K,
        d_view=args.d_view,
        hidden=args.hidden,
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        init_scale=args.init_scale,
        ensemble_sizes=ensemble_sizes,
        temperature=args.temperature,
        alpha=args.alpha,
        n_student_seeds=args.seeds,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
