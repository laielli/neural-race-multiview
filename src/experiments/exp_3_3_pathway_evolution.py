"""
Experiment 3.3: Pathway Evolution Under KD

Compares pathway evolution: hard labels (winner-take-all) vs KD (multiple survive).

Expected:
- Hard labels: one pathway dominates
- KD: multiple pathways survive
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MultiViewDataset
from src.model import MultiViewNet, init_weights
from src.metrics import count_surviving_pathways
from src.train import train_hard_labels, train_kd


def experiment_3_3_pathway_evolution_comparison(
    seed: int = 0,
    y_target: int = 0,
    num_teachers: int = 5,
    K: int = 10,
    M: int = 3,
    d_view: int = 50,
    hidden: int = 200,
    epochs: int = 200,
    temperature: float = 4.0,
    lr: float = 0.001,  # Smaller LR for gradient flow approximation
    batch_size: int = 128,
    n_samples: int = 10000,
    log_interval: int = 5,
    loss_type: str = 'mse',  # MSE to match Saxe et al. theory
    gradient_flow: bool = True,  # No momentum for gradient flow
    output_dir: str = 'results/figures',
    verbose: bool = True
):
    """
    Compare pathway evolution: hard labels vs KD.

    Args:
        seed: Random seed
        y_target: Class to visualize
        num_teachers: Number of teachers for KD
        K: Number of classes
        M: Views per class
        d_view: Dimensions per view slot
        hidden: Hidden layer width
        epochs: Training epochs
        temperature: KD temperature
        lr: Learning rate
        batch_size: Batch size
        n_samples: Training samples
        log_interval: How often to log pathway strengths
        output_dir: Directory for saving figures
        verbose: Print progress

    Returns:
        dict: Experiment results with both histories
    """
    if verbose:
        print("=" * 60)
        print("Experiment 3.3: Pathway Evolution Comparison")
        print(f"Config: seed={seed}, target_class={y_target}")
        print(f"K={K}, M={M}, {num_teachers} teachers")
        print(f"Loss: {loss_type.upper()}, Gradient flow: {gradient_flow}")
        print("=" * 60)

    dataset = MultiViewDataset(
        K=K, M=M, d_view=d_view,
        n_samples=n_samples, seed=seed
    )

    # Train teacher ensemble first
    if verbose:
        print("\nTraining teacher ensemble...")

    teachers = []
    for i in range(num_teachers):
        torch.manual_seed(seed * 100 + i)
        teacher = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
        teacher.apply(init_weights)

        train_hard_labels(teacher, dataset, epochs=100, lr=lr,
                          batch_size=batch_size, verbose=False,
                          loss_type=loss_type, gradient_flow=gradient_flow)
        teachers.append(teacher)

    # Train with hard labels (with pathway tracking)
    if verbose:
        print("\nTraining student with hard labels...")

    torch.manual_seed(seed + 5000)
    model_hard = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
    model_hard.apply(init_weights)

    history_hard = train_hard_labels(
        model_hard, dataset,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        log_interval=log_interval,
        track_pathways=True,
        track_classes=[y_target],
        verbose=verbose,
        loss_type=loss_type,
        gradient_flow=gradient_flow
    )

    # Train with KD (same init, with pathway tracking)
    if verbose:
        print("\nTraining student with KD...")

    torch.manual_seed(seed + 5000)  # Same init!
    model_kd = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
    model_kd.apply(init_weights)

    history_kd = train_kd(
        model_kd, teachers, dataset,
        epochs=epochs,
        temperature=temperature,
        lr=lr,
        batch_size=batch_size,
        log_interval=log_interval,
        track_pathways=True,
        track_classes=[y_target],
        verbose=verbose,
        gradient_flow=gradient_flow
    )

    # Count surviving pathways
    surviving_hard = count_surviving_pathways(model_hard, dataset)
    surviving_kd = count_surviving_pathways(model_kd, dataset)

    if verbose:
        print(f"\nSurviving pathways (class {y_target}):")
        print(f"  Hard labels: {surviving_hard[y_target]}")
        print(f"  KD: {surviving_kd[y_target]}")

    # Create side-by-side visualization
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, M))

    # Hard labels plot
    ax = axes[0]
    epochs_logged = history_hard['epochs_logged']
    for m in range(M):
        strengths = history_hard['pathway_strengths'][y_target][m]
        ax.plot(epochs_logged, strengths, label=f'View {m}',
                linewidth=2.5, color=colors[m])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pathway Strength', fontsize=12)
    ax.set_title('Hard Labels (Winner-Take-All)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # KD plot
    ax = axes[1]
    epochs_logged = history_kd['epochs_logged']
    for m in range(M):
        strengths = history_kd['pathway_strengths'][y_target][m]
        ax.plot(epochs_logged, strengths, label=f'View {m}',
                linewidth=2.5, color=colors[m])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pathway Strength', fontsize=12)
    ax.set_title('Knowledge Distillation (Multiple Views)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_path / 'pathway_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nSaved figure: {fig_path}")

    # Final pathway strengths
    final_hard = [history_hard['pathway_strengths'][y_target][m][-1] for m in range(M)]
    final_kd = [history_kd['pathway_strengths'][y_target][m][-1] for m in range(M)]

    if verbose:
        print(f"\nFinal pathway strengths for class {y_target}:")
        print("Hard labels:", [f"{s:.3f}" for s in final_hard])
        print("KD:         ", [f"{s:.3f}" for s in final_kd])

        # Check winner-take-all vs multi-view
        hard_ratio = max(final_hard) / (sum(final_hard) + 1e-10)
        kd_ratio = max(final_kd) / (sum(final_kd) + 1e-10)

        print(f"\nDominance ratio (winner/total):")
        print(f"  Hard labels: {hard_ratio:.3f}")
        print(f"  KD: {kd_ratio:.3f}")

        if hard_ratio > 0.7 and kd_ratio < 0.6:
            print("✓ PASSED: Hard labels show winner-take-all, KD preserves multiple views")
        else:
            print("✗ Results inconclusive")

    return {
        'experiment': 'exp_3_3_pathway_evolution',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': seed,
            'y_target': y_target,
            'num_teachers': num_teachers,
            'K': K,
            'M': M,
            'd_view': d_view,
            'hidden': hidden,
            'epochs': epochs,
            'temperature': temperature,
            'lr': lr,
            'log_interval': log_interval,
            'loss_type': loss_type,
            'gradient_flow': gradient_flow
        },
        'metrics': {
            'hard_label': {
                'final_strengths': {str(m): float(s) for m, s in enumerate(final_hard)},
                'surviving_pathways': int(surviving_hard[y_target]),
                'dominance_ratio': float(max(final_hard) / (sum(final_hard) + 1e-10))
            },
            'kd': {
                'final_strengths': {str(m): float(s) for m, s in enumerate(final_kd)},
                'surviving_pathways': int(surviving_kd[y_target]),
                'dominance_ratio': float(max(final_kd) / (sum(final_kd) + 1e-10))
            },
            'passed': bool(max(final_hard) / (sum(final_hard) + 1e-10) > 0.7 and
                       max(final_kd) / (sum(final_kd) + 1e-10) < 0.6)
        },
        'history_hard': {
            'epochs_logged': history_hard['epochs_logged'],
            'pathway_strengths': {
                str(m): history_hard['pathway_strengths'][y_target][m]
                for m in range(M)
            },
            'accuracy': history_hard['accuracy']
        },
        'history_kd': {
            'epochs_logged': history_kd['epochs_logged'],
            'pathway_strengths': {
                str(m): history_kd['pathway_strengths'][y_target][m]
                for m in range(M)
            },
            'accuracy': history_kd['accuracy']
        },
        'figure_path': str(fig_path)
    }


def run_multiple_classes(
    seed: int = 0,
    num_classes: int = 3,
    num_teachers: int = 5,
    epochs: int = 200,
    lr: float = 0.001,
    loss_type: str = 'mse',
    gradient_flow: bool = True,
    output_dir: str = 'results/figures',
    verbose: bool = True
):
    """
    Run pathway comparison for multiple classes.

    Provides more robust evidence of winner-take-all vs multi-view preservation.
    """
    if verbose:
        print("=" * 60)
        print("Pathway Evolution: Multiple Classes")
        print("=" * 60)

    all_results = []

    for y in range(num_classes):
        if verbose:
            print(f"\n--- Class {y} ---")

        result = experiment_3_3_pathway_evolution_comparison(
            seed=seed,
            y_target=y,
            num_teachers=num_teachers,
            epochs=epochs,
            lr=lr,
            loss_type=loss_type,
            gradient_flow=gradient_flow,
            output_dir=output_dir,
            verbose=False
        )
        all_results.append(result)

        hard_surv = result['metrics']['hard_label']['surviving_pathways']
        kd_surv = result['metrics']['kd']['surviving_pathways']
        if verbose:
            print(f"Surviving pathways - Hard: {hard_surv}, KD: {kd_surv}")

    # Summary
    hard_surviving = [r['metrics']['hard_label']['surviving_pathways'] for r in all_results]
    kd_surviving = [r['metrics']['kd']['surviving_pathways'] for r in all_results]

    if verbose:
        print("\n" + "=" * 60)
        print("Summary across classes:")
        print(f"Hard labels - mean surviving: {np.mean(hard_surviving):.2f}")
        print(f"KD - mean surviving: {np.mean(kd_surviving):.2f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Experiment 3.3: Pathway Evolution')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--class', dest='y_target', type=int, default=0, help='Target class')
    parser.add_argument('--teachers', type=int, default=5, help='Number of teachers')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (small for gradient flow)')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD temperature')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss type')
    parser.add_argument('--no-gradient-flow', action='store_true', help='Disable gradient flow (use momentum)')
    parser.add_argument('--multi-class', type=int, default=0, help='Run for multiple classes')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 50
        args.teachers = 3

    output_dir = Path(args.output) / 'figures'
    gradient_flow = not args.no_gradient_flow

    if args.multi_class > 0:
        results = run_multiple_classes(
            seed=args.seed,
            num_classes=args.multi_class,
            num_teachers=args.teachers,
            epochs=args.epochs,
            lr=args.lr,
            loss_type=args.loss,
            gradient_flow=gradient_flow,
            output_dir=str(output_dir)
        )
    else:
        results = experiment_3_3_pathway_evolution_comparison(
            seed=args.seed,
            y_target=args.y_target,
            num_teachers=args.teachers,
            epochs=args.epochs,
            lr=args.lr,
            temperature=args.temperature,
            loss_type=args.loss,
            gradient_flow=gradient_flow,
            output_dir=str(output_dir)
        )

        # Save results
        results_dir = Path(args.output)
        results_dir.mkdir(parents=True, exist_ok=True)

        output_file = results_dir / 'results_exp_3_3.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
