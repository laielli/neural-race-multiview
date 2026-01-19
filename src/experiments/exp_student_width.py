"""
Experiment: Student Width - Capacity Requirements for Knowledge Inheritance

Theory: Narrower student has fewer pathways to distribute learning across.
Must trade off pathway breadth vs pathway depth.

Prediction: Narrow students can't inherit multi-view knowledge (capacity bottleneck).
There exists a minimum width below which KD cannot transfer balanced knowledge.

Setup:
1. Train fixed (wide) teacher ensemble
2. Train students of varying widths: {32, 64, 128, 256, 512}
3. Measure effective rank vs student width
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


def train_kd(dataset, teachers, student_hidden, epochs, lr, init_scale,
             seed, temperature, alpha, verbose=True):
    """Train student with knowledge distillation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    student = SimpleLinearNet(
        d_input=dataset.M * dataset.d_view,
        hidden=student_hidden,  # Variable width!
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
        iterator = tqdm(iterator, desc=f"KD (width={student_hidden})", leave=False)

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
    teacher_hidden: int = 256,  # Wide teachers
    n_samples: int = 5000,
    epochs: int = 1000,
    lr: float = 0.1,
    init_scale: float = 0.1,
    n_teachers: int = 5,
    student_widths: list = None,
    temperature: float = 3.0,
    alpha: float = 0.7,
    n_seeds: int = 3,
    output_dir: str = 'results',
    verbose: bool = True
):
    """Run student width experiment."""
    if student_widths is None:
        student_widths = [16, 32, 64, 128, 256, 512]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("Student Width Experiment")
    print("=" * 70)
    print(f"Teacher width: {teacher_hidden}")
    print(f"Student widths: {student_widths}")
    print(f"Seeds per width: {n_seeds}")
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
    # Train wide teacher ensemble
    # =========================================================================
    print(f"\n" + "=" * 50)
    print(f"Training {n_teachers} Wide Teachers (width={teacher_hidden})")
    print("=" * 50)

    teachers = []
    teacher_metrics = []

    for i in range(n_teachers):
        teacher, acc = train_teacher(
            dataset, teacher_hidden, epochs, lr, init_scale,
            seed=i, verbose=verbose
        )
        teachers.append(teacher)
        view_contrib = measure_view_contributions(teacher, dataset)
        eff_rank = compute_effective_rank(view_contrib)
        teacher_metrics.append({
            'seed': i,
            'accuracy': acc,
            'effective_rank': eff_rank,
            'view0_contribution': view_contrib[0]
        })
        print(f"Teacher {i}: acc={acc:.3f}, eff_rank={eff_rank:.3f}")

    teacher_avg_eff_rank = np.mean([t['effective_rank'] for t in teacher_metrics])
    print(f"\nTeacher ensemble avg eff_rank: {teacher_avg_eff_rank:.3f}")

    # =========================================================================
    # Student width sweep
    # =========================================================================
    print(f"\n" + "=" * 50)
    print("Student Width Sweep")
    print("=" * 50)

    width_results = {}

    for width in student_widths:
        print(f"\n--- Student Width = {width} ---")

        seed_results = []

        for seed in range(n_seeds):
            student, acc = train_kd(
                dataset, teachers, width, epochs, lr, init_scale,
                seed=100 + seed, temperature=temperature, alpha=alpha,
                verbose=verbose
            )

            view_contrib = measure_view_contributions(student, dataset)
            eff_rank = compute_effective_rank(view_contrib)

            seed_results.append({
                'seed': seed,
                'accuracy': acc,
                'effective_rank': eff_rank,
                'view0_contribution': view_contrib[0],
                'view_contributions': view_contrib
            })

        avg_eff_rank = np.mean([r['effective_rank'] for r in seed_results])
        std_eff_rank = np.std([r['effective_rank'] for r in seed_results])
        avg_acc = np.mean([r['accuracy'] for r in seed_results])
        avg_v0 = np.mean([r['view0_contribution'] for r in seed_results])

        width_results[width] = {
            'avg_effective_rank': avg_eff_rank,
            'std_effective_rank': std_eff_rank,
            'avg_accuracy': avg_acc,
            'avg_view0_contribution': avg_v0,
            'per_seed': seed_results
        }

        print(f"  eff_rank: {avg_eff_rank:.3f} ± {std_eff_rank:.3f}")
        print(f"  accuracy: {avg_acc:.3f}")
        print(f"  view0: {avg_v0:.3f}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Width':<10} | {'Eff Rank':<20} | {'View 0':<10} | {'Accuracy':<10}")
    print("-" * 55)

    for width in student_widths:
        r = width_results[width]
        print(f"{width:<10} | {r['avg_effective_rank']:.3f} ± {r['std_effective_rank']:.3f}{'':<7} | "
              f"{r['avg_view0_contribution']:.3f}{'':<5} | {r['avg_accuracy']:.3f}")

    # Compute correlation
    eff_ranks = [width_results[w]['avg_effective_rank'] for w in student_widths]
    from scipy import stats
    correlation, p_value = stats.spearmanr(student_widths, eff_ranks)

    print(f"\nCorrelation Analysis:")
    print(f"  Spearman ρ(width, eff_rank): {correlation:.3f} (p={p_value:.4f})")

    # Compare narrowest to widest
    narrow = width_results[min(student_widths)]
    wide = width_results[max(student_widths)]
    rank_diff = wide['avg_effective_rank'] - narrow['avg_effective_rank']
    acc_diff = wide['avg_accuracy'] - narrow['avg_accuracy']

    print(f"\nNarrow vs Wide:")
    print(f"  Eff rank: {narrow['avg_effective_rank']:.3f} → {wide['avg_effective_rank']:.3f} ({rank_diff:+.3f})")
    print(f"  Accuracy: {narrow['avg_accuracy']:.3f} → {wide['avg_accuracy']:.3f} ({acc_diff:+.3f})")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    # Criterion 1: Positive correlation with width
    if correlation > 0.5:
        print(f"✓ Positive correlation with width: ρ = {correlation:.3f}")
    else:
        print(f"~ Weak/no correlation: ρ = {correlation:.3f}")

    # Criterion 2: Narrow students have lower effective rank
    if narrow['avg_effective_rank'] < wide['avg_effective_rank']:
        print(f"✓ Narrow worse than wide: {narrow['avg_effective_rank']:.3f} < {wide['avg_effective_rank']:.3f}")
    else:
        print(f"✗ Narrow not worse: {narrow['avg_effective_rank']:.3f} >= {wide['avg_effective_rank']:.3f}")
        success = False

    # Criterion 3: All widths achieve reasonable accuracy
    min_acc = min(width_results[w]['avg_accuracy'] for w in student_widths)
    if min_acc > 0.5:
        print(f"✓ All widths functional: min accuracy = {min_acc:.3f}")
    else:
        print(f"~ Some widths fail: min accuracy = {min_acc:.3f}")

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - Student width affects knowledge transfer!")
    else:
        print("EXPERIMENT INCONCLUSIVE - Width effect unclear")
    print("=" * 50)

    # =========================================================================
    # Plots
    # =========================================================================
    plot_width_sweep(student_widths, width_results, teacher_avg_eff_rank,
                     output_path / 'figures' / 'student_width.png')
    plot_width_accuracy_tradeoff(student_widths, width_results,
                                  output_path / 'figures' / 'student_width_accuracy.png')

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view,
            'teacher_hidden': teacher_hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'n_teachers': n_teachers,
            'student_widths': student_widths,
            'temperature': temperature, 'alpha': alpha, 'n_seeds': n_seeds
        },
        'dataset': {
            'signal_strengths': dataset.signal_strengths,
        },
        'teachers': {
            'avg_effective_rank': float(teacher_avg_eff_rank),
            'per_teacher': teacher_metrics
        },
        'width_sweep': {
            w: {
                'avg_effective_rank': float(width_results[w]['avg_effective_rank']),
                'std_effective_rank': float(width_results[w]['std_effective_rank']),
                'avg_accuracy': float(width_results[w]['avg_accuracy']),
                'avg_view0_contribution': float(width_results[w]['avg_view0_contribution']),
            }
            for w in student_widths
        },
        'analysis': {
            'spearman_correlation': float(correlation),
            'spearman_p_value': float(p_value),
            'rank_difference_narrow_to_wide': float(rank_diff),
            'acc_difference_narrow_to_wide': float(acc_diff)
        },
        'success': success
    }

    with open(output_path / 'student_width_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'student_width_results.json'}")

    return save_results, success


def plot_width_sweep(widths, width_results, teacher_baseline, output_path):
    """Plot effective rank vs student width."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Effective rank vs width
    ax = axes[0]
    eff_ranks = [width_results[w]['avg_effective_rank'] for w in widths]
    eff_stds = [width_results[w]['std_effective_rank'] for w in widths]

    ax.errorbar(widths, eff_ranks, yerr=eff_stds, marker='o', markersize=10,
                linewidth=2, capsize=5, color='steelblue')

    ax.axhline(y=teacher_baseline, color='green', linestyle='--', linewidth=2,
               label='Teacher Ensemble')
    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=1,
               label='Max (uniform)')

    ax.set_xlabel('Student Hidden Width', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Knowledge Breadth vs Student Capacity', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Accuracy vs width
    ax = axes[1]
    accuracies = [width_results[w]['avg_accuracy'] for w in widths]

    ax.plot(widths, accuracies, marker='s', markersize=10, linewidth=2, color='orange')

    ax.set_xlabel('Student Hidden Width', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Task Performance vs Student Capacity', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_width_accuracy_tradeoff(widths, width_results, output_path):
    """Plot accuracy vs effective rank colored by width."""
    fig, ax = plt.subplots(figsize=(10, 6))

    eff_ranks = [width_results[w]['avg_effective_rank'] for w in widths]
    accuracies = [width_results[w]['avg_accuracy'] for w in widths]

    colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))

    for i, width in enumerate(widths):
        ax.scatter(eff_ranks[i], accuracies[i], c=[colors[i]], s=200,
                   label=f'width={width}', edgecolors='black', linewidth=1)

    ax.set_xlabel('Effective Rank (Knowledge Breadth)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Capacity-Knowledge Tradeoff', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Student Width Experiment')
    parser.add_argument('--M', type=int, default=5, help='Number of views')
    parser.add_argument('--K', type=int, default=10, help='Number of classes')
    parser.add_argument('--d-view', type=int, default=50, help='Dimension per view')
    parser.add_argument('--teacher-hidden', type=int, default=256, help='Teacher width')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.1, help='Init scale')
    parser.add_argument('--n-teachers', type=int, default=5, help='Number of teachers')
    parser.add_argument('--temperature', type=float, default=3.0, help='KD temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Soft label weight')
    parser.add_argument('--seeds', type=int, default=3, help='Seeds per width')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    student_widths = [16, 32, 64, 128, 256, 512]

    if args.quick:
        args.epochs = 500
        args.seeds = 2
        args.n_teachers = 3
        student_widths = [32, 64, 128]
        print("Quick mode: epochs=500, seeds=2, widths=[32, 64, 128]")

    run_experiment(
        M=args.M,
        K=args.K,
        d_view=args.d_view,
        teacher_hidden=args.teacher_hidden,
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        init_scale=args.init_scale,
        n_teachers=args.n_teachers,
        student_widths=student_widths,
        temperature=args.temperature,
        alpha=args.alpha,
        n_seeds=args.seeds,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
