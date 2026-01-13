"""
Experiment 2.3: Race Dynamics Visualization

Visualizes pathway strengths over training to show winner-take-all dynamics.

Expected: One pathway grows to dominate, others decay.
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
from src.train import train_hard_labels


def experiment_2_3_race_visualization(
    seed: int = 0,
    y_target: int = 0,
    K: int = 10,
    M: int = 3,
    d_view: int = 50,
    hidden: int = 200,
    epochs: int = 200,
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
    Visualize pathway strengths over training to show winner-take-all dynamics.

    Args:
        seed: Random seed
        y_target: Class to visualize
        K: Number of classes
        M: Views per class
        d_view: Dimensions per view slot
        hidden: Hidden layer width
        epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        n_samples: Training samples
        log_interval: How often to log pathway strengths
        output_dir: Directory for saving figures
        verbose: Print progress

    Returns:
        dict: Experiment results including history
    """
    if verbose:
        print("=" * 60)
        print("Experiment 2.3: Race Dynamics Visualization")
        print(f"Config: K={K}, M={M}, seed={seed}, target_class={y_target}")
        print(f"Loss: {loss_type.upper()}, Gradient flow: {gradient_flow}")
        print("=" * 60)

    # Create dataset and model
    dataset = MultiViewDataset(
        K=K, M=M, d_view=d_view,
        n_samples=n_samples, seed=seed
    )

    torch.manual_seed(seed + 5000)
    model = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
    model.apply(init_weights)

    # Train with pathway tracking (MSE + gradient flow to match theory)
    history = train_hard_labels(
        model, dataset,
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

    # Create visualization
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, M))
    epochs_logged = history['epochs_logged']

    for m in range(M):
        strengths = history['pathway_strengths'][y_target][m]
        ax.plot(epochs_logged, strengths, label=f'View {m}',
                linewidth=2.5, color=colors[m])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pathway Strength', fontsize=12)
    ax.set_title(f'Race Dynamics (Class {y_target}, {loss_type.upper()} Loss)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation about winner-take-all
    final_strengths = [history['pathway_strengths'][y_target][m][-1] for m in range(M)]
    winner = np.argmax(final_strengths)
    ax.annotate(f'Winner: View {winner}',
                xy=(epochs_logged[-1], final_strengths[winner]),
                xytext=(epochs_logged[-1] * 0.7, max(final_strengths) * 0.9),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))

    fig_path = output_path / 'race_dynamics_hard.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nSaved figure: {fig_path}")
        print(f"\nFinal pathway strengths for class {y_target}:")
        for m in range(M):
            s = final_strengths[m]
            marker = "← WINNER" if m == winner else ""
            print(f"  View {m}: {s:.4f} {marker}")

    return {
        'experiment': 'exp_2_3_race_dynamics',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': seed,
            'y_target': y_target,
            'K': K,
            'M': M,
            'd_view': d_view,
            'hidden': hidden,
            'epochs': epochs,
            'lr': lr,
            'log_interval': log_interval,
            'loss_type': loss_type,
            'gradient_flow': gradient_flow
        },
        'metrics': {
            'winning_view': int(winner),
            'final_strengths': {str(m): float(s) for m, s in enumerate(final_strengths)},
            'winner_ratio': float(final_strengths[winner] / (sum(final_strengths) + 1e-10))
        },
        'history': {
            'epochs_logged': epochs_logged,
            'pathway_strengths': {
                str(m): history['pathway_strengths'][y_target][m]
                for m in range(M)
            },
            'accuracy': history['accuracy'],
            'loss': history['loss']
        },
        'figure_path': str(fig_path)
    }


def run_multiple_seeds(
    num_seeds: int = 5,
    y_target: int = 0,
    epochs: int = 200,
    lr: float = 0.001,
    loss_type: str = 'mse',
    gradient_flow: bool = True,
    output_dir: str = 'results/figures',
    verbose: bool = True
):
    """
    Run race dynamics for multiple seeds and create combined visualization.

    Shows that different seeds lead to different winners.
    """
    if verbose:
        print("=" * 60)
        print("Race Dynamics: Multiple Seeds Comparison")
        print("=" * 60)

    all_results = []
    winners = []

    for seed in range(num_seeds):
        if verbose:
            print(f"\n--- Seed {seed} ---")

        result = experiment_2_3_race_visualization(
            seed=seed,
            y_target=y_target,
            epochs=epochs,
            lr=lr,
            loss_type=loss_type,
            gradient_flow=gradient_flow,
            output_dir=output_dir,
            verbose=False
        )
        all_results.append(result)
        winners.append(result['metrics']['winning_view'])

        if verbose:
            print(f"Winner: View {result['metrics']['winning_view']}")

    # Create combined figure
    output_path = Path(output_dir)
    fig, axes = plt.subplots(1, num_seeds, figsize=(4 * num_seeds, 4), sharey=True)

    if num_seeds == 1:
        axes = [axes]

    M = all_results[0]['config']['M']
    colors = plt.cm.tab10(np.linspace(0, 1, M))

    for idx, (ax, result) in enumerate(zip(axes, all_results)):
        epochs_logged = result['history']['epochs_logged']
        for m in range(M):
            strengths = result['history']['pathway_strengths'][str(m)]
            ax.plot(epochs_logged, strengths, label=f'View {m}',
                    linewidth=2, color=colors[m])

        ax.set_xlabel('Epoch')
        if idx == 0:
            ax.set_ylabel('Pathway Strength')
        ax.set_title(f'Seed {idx} (Winner: {result["metrics"]["winning_view"]})')
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    plt.tight_layout()

    fig_path = output_path / 'race_dynamics_multi_seed.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nSaved combined figure: {fig_path}")
        print(f"\nWinners across seeds: {winners}")
        print(f"Unique winners: {len(set(winners))} / {M} possible views")

    return all_results, winners


def main():
    parser = argparse.ArgumentParser(description='Experiment 2.3: Race Dynamics')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--class', dest='y_target', type=int, default=0, help='Target class')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (small for gradient flow)')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss type')
    parser.add_argument('--no-gradient-flow', action='store_true', help='Disable gradient flow (use momentum)')
    parser.add_argument('--multi', type=int, default=0, help='Run multiple seeds')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 50

    output_dir = Path(args.output) / 'figures'
    gradient_flow = not args.no_gradient_flow

    if args.multi > 0:
        results, winners = run_multiple_seeds(
            num_seeds=args.multi,
            y_target=args.y_target,
            epochs=args.epochs,
            lr=args.lr,
            loss_type=args.loss,
            gradient_flow=gradient_flow,
            output_dir=str(output_dir)
        )
    else:
        results = experiment_2_3_race_visualization(
            seed=args.seed,
            y_target=args.y_target,
            epochs=args.epochs,
            lr=args.lr,
            loss_type=args.loss,
            gradient_flow=gradient_flow,
            output_dir=str(output_dir)
        )

        # Save results
        results_dir = Path(args.output)
        results_dir.mkdir(parents=True, exist_ok=True)

        output_file = results_dir / 'results_exp_2_3.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
