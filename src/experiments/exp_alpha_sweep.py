"""
Experiment: Alpha Sweep - Phase Transition in KD Effectiveness

Theory: The gradient has two components:
- External (KD): α · teacher_signal
- Self-reinforcing (hard): (1-α) · s_m · competition_term

Below critical α*, the hard label term dominates → KD fails to break WTA.

Prediction: Phase transition - dominance drops sharply around some critical α*.

Setup:
1. Train fixed teacher ensemble
2. Sweep α ∈ {0.0, 0.1, 0.2, ..., 1.0}
3. For each α, train KD student and measure dominance/effective rank
4. Identify phase transition point
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
from scipy.ndimage import gaussian_filter1d
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


def train_kd_with_alpha(dataset, teachers, hidden, epochs, lr, init_scale,
                        seed, temperature, alpha, verbose=True):
    """Train student with knowledge distillation at specific alpha."""
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
        iterator = tqdm(iterator, desc=f"KD α={alpha:.2f}", leave=False)

    for epoch in iterator:
        student.train()
        optimizer.zero_grad()

        student_logits = student(X)

        if alpha == 0.0:
            # Pure hard labels
            loss = F.mse_loss(student_logits, Y_onehot)
        elif alpha == 1.0:
            # Pure soft labels
            student_soft = F.log_softmax(student_logits / temperature, dim=-1)
            loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
            loss = loss * (temperature ** 2)
        else:
            # Combined loss
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


def find_phase_transition(alphas, dominances):
    """Find the critical alpha where phase transition occurs."""
    # Use derivative to find steepest descent
    smoothed = gaussian_filter1d(dominances, sigma=1)
    derivative = np.diff(smoothed) / np.diff(alphas)

    # Phase transition is where derivative is most negative (dominance drops fastest)
    transition_idx = np.argmin(derivative)
    critical_alpha = (alphas[transition_idx] + alphas[transition_idx + 1]) / 2

    return critical_alpha, derivative


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
    alphas: list = None,
    n_seeds: int = 5,
    output_dir: str = 'results',
    verbose: bool = True
):
    """Run alpha sweep experiment."""
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("Alpha Sweep Experiment")
    print("=" * 70)
    print(f"Alphas: {alphas}")
    print(f"Seeds: {n_seeds}, Temperature: {temperature}")
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
    # Phase 1: Train teacher ensemble
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

    # Teacher ensemble statistics
    teacher_contribs = [measure_view_contributions(t, dataset) for t in teachers]
    ensemble_v0 = np.mean([c[0] for c in teacher_contribs])
    ensemble_eff_rank = np.mean([compute_effective_rank(c) for c in teacher_contribs])
    print(f"\nTeacher ensemble: View 0 = {ensemble_v0:.3f}, eff_rank = {ensemble_eff_rank:.3f}")

    # =========================================================================
    # Phase 2: Alpha sweep
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 2: Alpha Sweep")
    print("=" * 50)

    alpha_results = {}

    for alpha in alphas:
        print(f"\n--- Alpha = {alpha:.2f} ---")
        alpha_seed_results = []

        for seed in range(n_seeds):
            student, acc = train_kd_with_alpha(
                dataset, teachers, hidden, epochs, lr, init_scale,
                seed=100 + seed, temperature=temperature, alpha=alpha,
                verbose=verbose
            )

            view_contrib = measure_view_contributions(student, dataset)
            eff_rank = compute_effective_rank(view_contrib)

            alpha_seed_results.append({
                'seed': seed,
                'accuracy': acc,
                'view_contributions': view_contrib,
                'effective_rank': eff_rank,
                'view0_contribution': view_contrib[0],
                'dominance': view_contrib[0]  # View 0 contribution as dominance
            })

        avg_dominance = np.mean([r['dominance'] for r in alpha_seed_results])
        std_dominance = np.std([r['dominance'] for r in alpha_seed_results])
        avg_eff_rank = np.mean([r['effective_rank'] for r in alpha_seed_results])
        avg_acc = np.mean([r['accuracy'] for r in alpha_seed_results])

        alpha_results[alpha] = {
            'per_seed': alpha_seed_results,
            'avg_dominance': avg_dominance,
            'std_dominance': std_dominance,
            'avg_effective_rank': avg_eff_rank,
            'avg_accuracy': avg_acc
        }

        print(f"  α={alpha:.2f}: dominance={avg_dominance:.3f}±{std_dominance:.3f}, "
              f"eff_rank={avg_eff_rank:.3f}, acc={avg_acc:.3f}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'α':<8} | {'Dominance':<20} | {'Eff Rank':<12} | {'Accuracy':<10}")
    print("-" * 55)

    dominances = []
    eff_ranks = []
    accuracies = []

    for alpha in alphas:
        r = alpha_results[alpha]
        dominances.append(r['avg_dominance'])
        eff_ranks.append(r['avg_effective_rank'])
        accuracies.append(r['avg_accuracy'])
        print(f"{alpha:<8} | {r['avg_dominance']:.3f} ± {r['std_dominance']:.3f}{'':<8} | "
              f"{r['avg_effective_rank']:.3f}{'':<7} | {r['avg_accuracy']:.3f}")

    # Find phase transition
    critical_alpha, derivative = find_phase_transition(np.array(alphas), np.array(dominances))
    print(f"\nPhase Transition Analysis:")
    print(f"  Critical α* (steepest decline): {critical_alpha:.2f}")

    # Measure transition sharpness
    dom_low = alpha_results[alphas[0]]['avg_dominance']
    dom_high = alpha_results[alphas[-1]]['avg_dominance']
    transition_magnitude = dom_low - dom_high
    print(f"  Transition magnitude: {transition_magnitude:.3f} (α=0 to α=1)")

    # Check for clear transition
    mid_point = len(alphas) // 2
    dom_first_half = np.mean(dominances[:mid_point])
    dom_second_half = np.mean(dominances[mid_point:])
    transition_ratio = dom_first_half / (dom_second_half + 1e-10)
    print(f"  First half vs second half ratio: {transition_ratio:.2f}")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    # Criterion 1: Clear transition (low α has higher dominance than high α)
    if transition_magnitude > 0.05:
        print(f"✓ Clear transition: dominance drops by {transition_magnitude:.3f}")
    else:
        print(f"✗ No clear transition: dominance only drops by {transition_magnitude:.3f}")
        success = False

    # Criterion 2: α=0 (hard labels) has higher dominance
    if dominances[0] > dominances[-1]:
        print(f"✓ α=0 has higher dominance than α=1: {dominances[0]:.3f} > {dominances[-1]:.3f}")
    else:
        print(f"✗ α=0 does not have higher dominance: {dominances[0]:.3f} vs {dominances[-1]:.3f}")
        success = False

    # Criterion 3: Accuracy maintained across α
    acc_range = max(accuracies) - min(accuracies)
    if acc_range < 0.2:
        print(f"✓ Accuracy stable across α: range = {acc_range:.3f}")
    else:
        print(f"~ Accuracy varies with α: range = {acc_range:.3f}")

    print("\n" + "=" * 50)
    if success:
        print(f"EXPERIMENT PASSED - Phase transition found at α* ≈ {critical_alpha:.2f}")
    else:
        print("EXPERIMENT INCONCLUSIVE - No clear phase transition")
    print("=" * 50)

    # =========================================================================
    # Plots
    # =========================================================================
    plot_alpha_sweep(alphas, alpha_results, critical_alpha,
                     output_path / 'figures' / 'alpha_sweep.png')
    plot_alpha_accuracy_tradeoff(alphas, alpha_results,
                                  output_path / 'figures' / 'alpha_accuracy_tradeoff.png')

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'n_teachers': n_teachers,
            'temperature': temperature, 'alphas': alphas, 'n_seeds': n_seeds
        },
        'dataset': {
            'signal_strengths': dataset.signal_strengths,
        },
        'teacher_ensemble': {
            'avg_view0_contribution': float(ensemble_v0),
            'avg_effective_rank': float(ensemble_eff_rank)
        },
        'alpha_sweep': {
            alpha: {
                'avg_dominance': float(alpha_results[alpha]['avg_dominance']),
                'std_dominance': float(alpha_results[alpha]['std_dominance']),
                'avg_effective_rank': float(alpha_results[alpha]['avg_effective_rank']),
                'avg_accuracy': float(alpha_results[alpha]['avg_accuracy']),
            }
            for alpha in alphas
        },
        'analysis': {
            'critical_alpha': float(critical_alpha),
            'transition_magnitude': float(transition_magnitude),
            'first_half_avg_dominance': float(dom_first_half),
            'second_half_avg_dominance': float(dom_second_half)
        },
        'success': success
    }

    with open(output_path / 'alpha_sweep_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'alpha_sweep_results.json'}")

    return save_results, success


def plot_alpha_sweep(alphas, alpha_results, critical_alpha, output_path):
    """Plot dominance and effective rank vs alpha."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dominance vs alpha
    ax = axes[0]
    dominances = [alpha_results[a]['avg_dominance'] for a in alphas]
    dom_stds = [alpha_results[a]['std_dominance'] for a in alphas]

    ax.errorbar(alphas, dominances, yerr=dom_stds, marker='o', markersize=10,
                linewidth=2, capsize=5, color='steelblue')

    ax.axvline(x=critical_alpha, color='red', linestyle='--', linewidth=2,
               label=f'Critical α* ≈ {critical_alpha:.2f}')

    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('View 0 Dominance', fontsize=12)
    ax.set_title('Dominance vs Soft Label Weight\n'
                 '(Lower = Less WTA)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Effective rank vs alpha
    ax = axes[1]
    eff_ranks = [alpha_results[a]['avg_effective_rank'] for a in alphas]

    ax.plot(alphas, eff_ranks, marker='s', markersize=10,
            linewidth=2, color='green')

    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=1,
               label='Max (uniform, M=5)')

    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Effective Rank vs Soft Label Weight\n'
                 '(Higher = More Balanced)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_alpha_accuracy_tradeoff(alphas, alpha_results, output_path):
    """Plot accuracy vs dominance tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dominances = [alpha_results[a]['avg_dominance'] for a in alphas]
    accuracies = [alpha_results[a]['avg_accuracy'] for a in alphas]

    # Color by alpha
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    for i, alpha in enumerate(alphas):
        ax.scatter(dominances[i], accuracies[i], c=[colors[i]], s=150,
                   label=f'α={alpha:.1f}', edgecolors='black', linewidth=1)

    ax.set_xlabel('View 0 Dominance (WTA strength)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs WTA Tradeoff Across Alpha Values', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Alpha Sweep Experiment')
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
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if args.quick:
        args.epochs = 500
        args.seeds = 3
        args.n_teachers = 3
        alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
        print("Quick mode: epochs=500, seeds=3, alphas=[0, 0.3, 0.5, 0.7, 1.0]")

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
        alphas=alphas,
        n_seeds=args.seeds,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
