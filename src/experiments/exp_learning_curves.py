"""
Experiment: Learning Curves - KD Accelerates Weak Pathways

Theory: Under hard labels, pathway strength evolves as:
  ds_m/dt = σ_1(Σ_m) · s_m  (rich get richer)

Under KD, the gradient has an additional external term:
  ds_m/dt = α_m(teacher) + γ · σ_1(Σ_m) · s_m  (constant boost + rich get richer)

The external term matters more when s_m is small, so weak pathways grow faster under KD.

Prediction: Per-pathway strength curves show weak pathways accelerated under KD.

Setup:
1. Train model with hard labels, track per-view weight norm every N epochs
2. Train model with KD, track per-view weight norm every N epochs
3. Compare growth rates for weak views (views 2, 3, 4)
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


def measure_view_contributions(model, M, d_view):
    """Measure how much each view contributes to the network's predictions."""
    with torch.no_grad():
        W1 = model.W1.weight.cpu().numpy()

        contributions = []
        for m in range(M):
            slot_start = m * d_view
            slot_end = (m + 1) * d_view
            W1_view = W1[:, slot_start:slot_end]
            contrib = np.linalg.norm(W1_view, 'fro')
            contributions.append(contrib)

        return contributions


def train_with_tracking(model, dataset, epochs, lr, log_interval, is_kd=False,
                       teachers=None, temperature=3.0, alpha=0.7, verbose=True):
    """Train model and track per-view weight norms over time."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    X, Y = dataset.get_tensors()
    Y_onehot = F.one_hot(Y, num_classes=dataset.K).float()

    # Pre-compute soft targets if KD
    soft_targets = None
    if is_kd and teachers is not None:
        with torch.no_grad():
            teacher_logits = []
            for teacher in teachers:
                teacher.eval()
                teacher_logits.append(teacher(X))
            avg_logits = torch.stack(teacher_logits).mean(dim=0)
            soft_targets = F.softmax(avg_logits / temperature, dim=-1)

    # Track history
    history = {
        'epochs': [],
        'loss': [],
        'accuracy': [],
        'view_contributions': []  # List of [contrib_0, contrib_1, ..., contrib_M-1]
    }

    iterator = range(epochs)
    if verbose:
        mode = "KD" if is_kd else "Hard"
        iterator = tqdm(iterator, desc=f"{mode} Training", leave=False)

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()

        output = model(X)

        if is_kd and soft_targets is not None:
            # Combined loss
            hard_loss = F.mse_loss(output, Y_onehot)
            student_soft = F.log_softmax(output / temperature, dim=-1)
            soft_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
            soft_loss = soft_loss * (temperature ** 2)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
        else:
            loss = F.mse_loss(output, Y_onehot)

        loss.backward()
        optimizer.step()

        # Log metrics at intervals
        if epoch % log_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = model(X).argmax(dim=-1)
                acc = (preds == Y).float().mean().item()

            contributions = measure_view_contributions(model, dataset.M, dataset.d_view)

            history['epochs'].append(epoch)
            history['loss'].append(loss.item())
            history['accuracy'].append(acc)
            history['view_contributions'].append(contributions)

    return history


def compute_growth_rates(history):
    """Compute growth rates for each view from the learning curves."""
    epochs = np.array(history['epochs'])
    contributions = np.array(history['view_contributions'])

    M = contributions.shape[1]
    growth_rates = []

    for m in range(M):
        # Linear regression on log contributions for exponential fit
        log_contrib = np.log(contributions[:, m] + 1e-10)
        # Use early phase for growth rate
        early_idx = len(epochs) // 3
        if early_idx < 2:
            early_idx = len(epochs)

        slope, _ = np.polyfit(epochs[:early_idx], log_contrib[:early_idx], 1)
        growth_rates.append(slope)

    return growth_rates


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
    log_interval: int = 10,
    n_seeds: int = 3,
    output_dir: str = 'results',
    verbose: bool = True
):
    """Run learning curves comparison experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("Learning Curves Experiment")
    print("=" * 70)
    print(f"Tracking per-view weight norms over {epochs} epochs")
    print(f"Seeds: {n_seeds}, Log interval: {log_interval}")
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
        torch.manual_seed(i)
        teacher = SimpleLinearNet(
            d_input=M * d_view,
            hidden=hidden,
            d_output=K,
            init_scale=init_scale
        )
        history = train_with_tracking(
            teacher, dataset, epochs, lr, log_interval,
            is_kd=False, verbose=verbose
        )
        teachers.append(teacher)
        final_contrib = history['view_contributions'][-1]
        print(f"Teacher {i}: contributions={[f'{c:.3f}' for c in final_contrib]}")

    # =========================================================================
    # Phase 2: Train with hard labels (multiple seeds)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 2: Hard Label Training with Tracking")
    print("=" * 50)

    hard_histories = []

    for seed in range(n_seeds):
        torch.manual_seed(100 + seed)
        model = SimpleLinearNet(
            d_input=M * d_view,
            hidden=hidden,
            d_output=K,
            init_scale=init_scale
        )
        history = train_with_tracking(
            model, dataset, epochs, lr, log_interval,
            is_kd=False, verbose=verbose
        )
        hard_histories.append(history)
        final_contrib = history['view_contributions'][-1]
        print(f"Hard {seed}: final contributions={[f'{c:.3f}' for c in final_contrib]}")

    # =========================================================================
    # Phase 3: Train with KD (multiple seeds)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 3: KD Training with Tracking")
    print("=" * 50)

    kd_histories = []

    for seed in range(n_seeds):
        torch.manual_seed(100 + seed)
        model = SimpleLinearNet(
            d_input=M * d_view,
            hidden=hidden,
            d_output=K,
            init_scale=init_scale
        )
        history = train_with_tracking(
            model, dataset, epochs, lr, log_interval,
            is_kd=True, teachers=teachers, temperature=temperature,
            alpha=alpha, verbose=verbose
        )
        kd_histories.append(history)
        final_contrib = history['view_contributions'][-1]
        print(f"KD {seed}: final contributions={[f'{c:.3f}' for c in final_contrib]}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    # Compute average learning curves
    epochs_list = hard_histories[0]['epochs']

    hard_avg_curves = np.mean([h['view_contributions'] for h in hard_histories], axis=0)
    kd_avg_curves = np.mean([h['view_contributions'] for h in kd_histories], axis=0)

    # Compute growth rates for each view
    hard_growth_rates = []
    kd_growth_rates = []

    for seed in range(n_seeds):
        hard_growth_rates.append(compute_growth_rates(hard_histories[seed]))
        kd_growth_rates.append(compute_growth_rates(kd_histories[seed]))

    hard_avg_growth = np.mean(hard_growth_rates, axis=0)
    kd_avg_growth = np.mean(kd_growth_rates, axis=0)

    print("\nGrowth Rates by View:")
    print(f"{'View':<8} | {'Hard Rate':<12} | {'KD Rate':<12} | {'KD/Hard Ratio':<15}")
    print("-" * 50)

    for m in range(M):
        ratio = kd_avg_growth[m] / (hard_avg_growth[m] + 1e-10)
        print(f"{m:<8} | {hard_avg_growth[m]:.6f}{'':<4} | {kd_avg_growth[m]:.6f}{'':<4} | {ratio:.2f}")

    # Check if weak views (m > 0) are accelerated under KD
    weak_view_acceleration = []
    for m in range(1, M):
        ratio = kd_avg_growth[m] / (hard_avg_growth[m] + 1e-10)
        weak_view_acceleration.append(ratio)

    avg_weak_acceleration = np.mean(weak_view_acceleration)

    print(f"\nWeak View Acceleration (avg ratio for views 1-{M-1}): {avg_weak_acceleration:.2f}")

    # Compare final contributions
    hard_final = hard_avg_curves[-1]
    kd_final = kd_avg_curves[-1]

    print("\nFinal View Contributions:")
    print(f"  Hard: {[f'{c:.3f}' for c in hard_final]}")
    print(f"  KD:   {[f'{c:.3f}' for c in kd_final]}")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    # Criterion 1: Weak views have higher growth rate under KD
    if avg_weak_acceleration > 1.0:
        print(f"✓ Weak views accelerated under KD: avg ratio = {avg_weak_acceleration:.2f} > 1.0")
    else:
        print(f"✗ Weak views not accelerated under KD: avg ratio = {avg_weak_acceleration:.2f}")
        success = False

    # Criterion 2: KD has more balanced final contributions
    hard_dominance = hard_final[0] / np.sum(hard_final)
    kd_dominance = kd_final[0] / np.sum(kd_final)

    if kd_dominance < hard_dominance:
        print(f"✓ KD more balanced: View 0 dominance {kd_dominance:.3f} < {hard_dominance:.3f}")
    else:
        print(f"✗ KD not more balanced: View 0 dominance {kd_dominance:.3f} >= {hard_dominance:.3f}")
        success = False

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - KD accelerates weak pathways!")
    else:
        print("EXPERIMENT INCONCLUSIVE - Acceleration effect unclear")
    print("=" * 50)

    # =========================================================================
    # Plots
    # =========================================================================
    plot_learning_curves(epochs_list, hard_avg_curves, kd_avg_curves, M,
                        output_path / 'figures' / 'learning_curves.png')
    plot_normalized_curves(epochs_list, hard_avg_curves, kd_avg_curves, M,
                          output_path / 'figures' / 'learning_curves_normalized.png')

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'n_teachers': n_teachers,
            'temperature': temperature, 'alpha': alpha,
            'log_interval': log_interval, 'n_seeds': n_seeds
        },
        'dataset': {
            'signal_strengths': dataset.signal_strengths,
        },
        'hard_labels': {
            'epochs': epochs_list,
            'avg_view_curves': hard_avg_curves.tolist(),
            'avg_growth_rates': hard_avg_growth.tolist(),
            'final_contributions': hard_final.tolist()
        },
        'kd': {
            'epochs': epochs_list,
            'avg_view_curves': kd_avg_curves.tolist(),
            'avg_growth_rates': kd_avg_growth.tolist(),
            'final_contributions': kd_final.tolist()
        },
        'analysis': {
            'weak_view_acceleration': float(avg_weak_acceleration),
            'hard_view0_dominance': float(hard_dominance),
            'kd_view0_dominance': float(kd_dominance)
        },
        'success': success
    }

    with open(output_path / 'learning_curves_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'learning_curves_results.json'}")

    return save_results, success


def plot_learning_curves(epochs, hard_curves, kd_curves, M, output_path):
    """Plot per-view weight norm curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.8, M))

    # Hard labels
    ax = axes[0]
    for m in range(M):
        ax.plot(epochs, hard_curves[:, m], color=colors[m], linewidth=2,
                label=f'View {m}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Weight Norm (Frobenius)', fontsize=12)
    ax.set_title('Hard Labels: Per-View Weight Norms', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # KD
    ax = axes[1]
    for m in range(M):
        ax.plot(epochs, kd_curves[:, m], color=colors[m], linewidth=2,
                label=f'View {m}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Weight Norm (Frobenius)', fontsize=12)
    ax.set_title('KD: Per-View Weight Norms', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_normalized_curves(epochs, hard_curves, kd_curves, M, output_path):
    """Plot normalized per-view curves (relative to total)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Normalize
    hard_totals = hard_curves.sum(axis=1, keepdims=True)
    hard_norm = hard_curves / (hard_totals + 1e-10)

    kd_totals = kd_curves.sum(axis=1, keepdims=True)
    kd_norm = kd_curves / (kd_totals + 1e-10)

    colors = plt.cm.viridis(np.linspace(0, 0.8, M))

    # Hard labels
    ax = axes[0]
    ax.stackplot(epochs, hard_norm.T, colors=colors, alpha=0.7,
                 labels=[f'View {m}' for m in range(M)])
    ax.axhline(y=1/M, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative Contribution', fontsize=12)
    ax.set_title('Hard Labels: View Contribution Over Time', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='center right', fontsize=9)

    # KD
    ax = axes[1]
    ax.stackplot(epochs, kd_norm.T, colors=colors, alpha=0.7,
                 labels=[f'View {m}' for m in range(M)])
    ax.axhline(y=1/M, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative Contribution', fontsize=12)
    ax.set_title('KD: View Contribution Over Time', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='center right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Learning Curves Experiment')
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
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 500
        args.seeds = 2
        args.n_teachers = 3
        args.log_interval = 25
        print("Quick mode: epochs=500, seeds=2, log_interval=25")

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
        log_interval=args.log_interval,
        n_seeds=args.seeds,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
