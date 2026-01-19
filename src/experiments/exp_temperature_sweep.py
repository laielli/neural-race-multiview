"""
Experiment: Temperature Sweep - How τ Affects KD Pathway Balance

Theory: Soft labels distribute gradient proportional to teacher's confidence.
Higher τ → more uniform softmax distribution → more balanced gradient → more balanced pathways.

Prediction: Effective rank increases monotonically with τ (up to saturation).

Setup:
1. Train fixed teacher ensemble
2. Sweep τ ∈ {1.0, 2.0, 3.0, 5.0, 10.0, 20.0}
3. For each τ, train KD student and measure effective rank
4. Plot effective rank vs τ
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
from scipy import stats
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
    """Compute effective rank of contribution distribution (higher = more balanced)."""
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


def train_kd_with_temperature(dataset, teachers, hidden, epochs, lr, init_scale,
                               seed, temperature, alpha=0.7, verbose=True):
    """Train student with knowledge distillation at specific temperature."""
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

    # Pre-compute teacher soft targets at this temperature
    with torch.no_grad():
        teacher_logits = []
        for teacher in teachers:
            teacher.eval()
            teacher_logits.append(teacher(X))
        avg_logits = torch.stack(teacher_logits).mean(dim=0)
        soft_targets = F.softmax(avg_logits / temperature, dim=-1)

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"KD τ={temperature}", leave=False)

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
    n_teachers: int = 5,
    temperatures: list = None,
    alpha: float = 0.7,
    n_seeds: int = 5,
    output_dir: str = 'results',
    verbose: bool = True
):
    """Run temperature sweep experiment."""
    if temperatures is None:
        temperatures = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("Temperature Sweep Experiment")
    print("=" * 70)
    print(f"Temperatures: {temperatures}")
    print(f"Seeds: {n_seeds}, Alpha: {alpha}")
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
    # Phase 1: Train teacher ensemble (fixed for all temperatures)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 1: Training Teacher Ensemble")
    print("=" * 50)

    teachers = []
    for i in range(n_teachers):
        teacher, acc = train_teacher(
            dataset, hidden, epochs, lr, init_scale,
            seed=i, verbose=verbose
        )
        teachers.append(teacher)
        view_contrib = measure_view_contributions(teacher, dataset)
        if verbose:
            print(f"Teacher {i}: acc={acc:.3f}, contributions={[f'{c:.3f}' for c in view_contrib]}")

    # =========================================================================
    # Phase 2: Hard-label baseline
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 2: Hard-Label Baseline")
    print("=" * 50)

    hard_results = []
    for seed in range(n_seeds):
        model, acc = train_teacher(
            dataset, hidden, epochs, lr, init_scale,
            seed=100 + seed, verbose=verbose
        )
        view_contrib = measure_view_contributions(model, dataset)
        eff_rank = compute_effective_rank(view_contrib)
        hard_results.append({
            'seed': seed,
            'accuracy': acc,
            'view_contributions': view_contrib,
            'effective_rank': eff_rank,
            'view0_contribution': view_contrib[0]
        })

    hard_avg_eff_rank = np.mean([r['effective_rank'] for r in hard_results])
    hard_std_eff_rank = np.std([r['effective_rank'] for r in hard_results])
    hard_avg_v0 = np.mean([r['view0_contribution'] for r in hard_results])

    print(f"\nHard Labels Baseline:")
    print(f"  Effective Rank: {hard_avg_eff_rank:.3f} ± {hard_std_eff_rank:.3f}")
    print(f"  View 0 Contribution: {hard_avg_v0:.3f}")

    # =========================================================================
    # Phase 3: Temperature sweep
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 3: Temperature Sweep")
    print("=" * 50)

    temp_results = {}

    for tau in temperatures:
        print(f"\n--- Temperature τ = {tau} ---")
        tau_results = []

        for seed in range(n_seeds):
            student, acc = train_kd_with_temperature(
                dataset, teachers, hidden, epochs, lr, init_scale,
                seed=100 + seed, temperature=tau, alpha=alpha,
                verbose=verbose
            )

            view_contrib = measure_view_contributions(student, dataset)
            eff_rank = compute_effective_rank(view_contrib)

            tau_results.append({
                'seed': seed,
                'accuracy': acc,
                'view_contributions': view_contrib,
                'effective_rank': eff_rank,
                'view0_contribution': view_contrib[0]
            })

        avg_eff_rank = np.mean([r['effective_rank'] for r in tau_results])
        std_eff_rank = np.std([r['effective_rank'] for r in tau_results])
        avg_v0 = np.mean([r['view0_contribution'] for r in tau_results])
        avg_acc = np.mean([r['accuracy'] for r in tau_results])

        temp_results[tau] = {
            'per_seed': tau_results,
            'avg_effective_rank': avg_eff_rank,
            'std_effective_rank': std_eff_rank,
            'avg_view0_contribution': avg_v0,
            'avg_accuracy': avg_acc
        }

        print(f"  τ={tau}: eff_rank={avg_eff_rank:.3f}±{std_eff_rank:.3f}, "
              f"v0={avg_v0:.3f}, acc={avg_acc:.3f}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'τ':<8} | {'Eff Rank':<15} | {'View 0':<10} | {'Accuracy':<10}")
    print("-" * 50)
    print(f"{'Hard':<8} | {hard_avg_eff_rank:.3f} ± {hard_std_eff_rank:.3f}{'':<5} | {hard_avg_v0:.3f}{'':<5} | {'--':<10}")

    taus = sorted(temp_results.keys())
    eff_ranks = []
    for tau in taus:
        r = temp_results[tau]
        eff_ranks.append(r['avg_effective_rank'])
        print(f"{tau:<8} | {r['avg_effective_rank']:.3f} ± {r['std_effective_rank']:.3f}{'':<5} | "
              f"{r['avg_view0_contribution']:.3f}{'':<5} | {r['avg_accuracy']:.3f}")

    # Correlation analysis
    correlation, p_value = stats.pearsonr(taus, eff_ranks)
    monotonic_corr, mono_p = stats.spearmanr(taus, eff_ranks)

    print(f"\nCorrelation Analysis:")
    print(f"  Pearson r(τ, eff_rank): {correlation:.3f} (p={p_value:.4f})")
    print(f"  Spearman ρ(τ, eff_rank): {monotonic_corr:.3f} (p={mono_p:.4f})")

    # Check if monotonically increasing
    is_monotonic = all(eff_ranks[i] <= eff_ranks[i+1] for i in range(len(eff_ranks)-1))
    print(f"  Strictly monotonic increasing: {'Yes' if is_monotonic else 'No'}")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    # Criterion 1: Positive correlation
    if correlation > 0.8:
        print(f"✓ Strong positive correlation: r = {correlation:.3f} > 0.8")
    elif correlation > 0.5:
        print(f"~ Moderate positive correlation: r = {correlation:.3f}")
    else:
        print(f"✗ Weak/no positive correlation: r = {correlation:.3f}")
        success = False

    # Criterion 2: KD at any τ beats hard labels
    all_beat_hard = all(temp_results[tau]['avg_effective_rank'] > hard_avg_eff_rank
                        for tau in temperatures)
    if all_beat_hard:
        print(f"✓ All KD temperatures beat hard labels baseline")
    else:
        print(f"✗ Some KD temperatures don't beat hard labels")

    # Criterion 3: High τ significantly higher than low τ
    low_tau_rank = temp_results[temperatures[0]]['avg_effective_rank']
    high_tau_rank = temp_results[temperatures[-1]]['avg_effective_rank']
    improvement = (high_tau_rank - low_tau_rank) / low_tau_rank * 100
    if improvement > 5:
        print(f"✓ High τ improvement over low τ: {improvement:.1f}%")
    else:
        print(f"~ Marginal improvement from low to high τ: {improvement:.1f}%")

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - Temperature affects pathway balance!")
    else:
        print("EXPERIMENT INCONCLUSIVE - Weak temperature effect")
    print("=" * 50)

    # =========================================================================
    # Plots
    # =========================================================================
    plot_temperature_sweep(temperatures, temp_results, hard_avg_eff_rank,
                          output_path / 'figures' / 'temperature_sweep.png')
    plot_view_contributions_by_temp(temperatures, temp_results, dataset,
                                    output_path / 'figures' / 'temperature_view_contributions.png')

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'n_teachers': n_teachers,
            'temperatures': temperatures, 'alpha': alpha, 'n_seeds': n_seeds
        },
        'dataset': {
            'signal_strengths': dataset.signal_strengths,
        },
        'hard_labels': {
            'avg_effective_rank': float(hard_avg_eff_rank),
            'std_effective_rank': float(hard_std_eff_rank),
            'avg_view0_contribution': float(hard_avg_v0),
            'per_seed': hard_results
        },
        'temperature_sweep': {
            tau: {
                'avg_effective_rank': float(temp_results[tau]['avg_effective_rank']),
                'std_effective_rank': float(temp_results[tau]['std_effective_rank']),
                'avg_view0_contribution': float(temp_results[tau]['avg_view0_contribution']),
                'avg_accuracy': float(temp_results[tau]['avg_accuracy']),
            }
            for tau in temperatures
        },
        'analysis': {
            'pearson_correlation': float(correlation),
            'pearson_p_value': float(p_value),
            'spearman_correlation': float(monotonic_corr),
            'spearman_p_value': float(mono_p),
            'is_monotonic': is_monotonic
        },
        'success': success
    }

    with open(output_path / 'temperature_sweep_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'temperature_sweep_results.json'}")

    return save_results, success


def plot_temperature_sweep(temperatures, temp_results, hard_baseline, output_path):
    """Plot effective rank vs temperature."""
    fig, ax = plt.subplots(figsize=(10, 6))

    taus = sorted(temperatures)
    eff_ranks = [temp_results[tau]['avg_effective_rank'] for tau in taus]
    eff_stds = [temp_results[tau]['std_effective_rank'] for tau in taus]

    ax.errorbar(taus, eff_ranks, yerr=eff_stds, marker='o', markersize=10,
                linewidth=2, capsize=5, color='steelblue', label='KD')

    ax.axhline(y=hard_baseline, color='red', linestyle='--', linewidth=2,
               label='Hard Labels Baseline')

    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=1,
               label='Max (uniform, M=5)')

    ax.set_xlabel('Temperature (τ)', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Temperature Effect on Pathway Balance\n'
                 '(Higher = More Balanced View Usage)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(taus, eff_ranks, 1)
    p = np.poly1d(z)
    ax.plot(taus, p(taus), 'b--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_view_contributions_by_temp(temperatures, temp_results, dataset, output_path):
    """Plot view contributions for each temperature."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    taus = sorted(temperatures)
    M = dataset.M

    for i, tau in enumerate(taus):
        if i >= len(axes):
            break
        ax = axes[i]

        # Get all view contributions for this temperature
        all_contrib = [r['view_contributions'] for r in temp_results[tau]['per_seed']]
        avg_contrib = np.mean(all_contrib, axis=0)
        std_contrib = np.std(all_contrib, axis=0)

        x = np.arange(M)
        ax.bar(x, avg_contrib, yerr=std_contrib, capsize=3, color='steelblue', alpha=0.7)

        ax.axhline(y=1/M, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('View')
        ax.set_ylabel('Contribution')
        ax.set_title(f'τ = {tau}')
        ax.set_xticks(x)
        ax.set_ylim(0, 0.5)

    # Hide unused axes
    for i in range(len(taus), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('View Contributions by Temperature', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Temperature Sweep Experiment')
    parser.add_argument('--M', type=int, default=5, help='Number of views')
    parser.add_argument('--K', type=int, default=10, help='Number of classes')
    parser.add_argument('--d-view', type=int, default=50, help='Dimension per view')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer width')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.1, help='Init scale')
    parser.add_argument('--n-teachers', type=int, default=5, help='Number of teachers')
    parser.add_argument('--alpha', type=float, default=0.7, help='Soft label weight')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    temperatures = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    if args.quick:
        args.epochs = 500
        args.seeds = 3
        args.n_teachers = 3
        temperatures = [1.0, 3.0, 10.0]
        print("Quick mode: epochs=500, seeds=3, temperatures=[1, 3, 10]")

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
        temperatures=temperatures,
        alpha=args.alpha,
        n_seeds=args.seeds,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
