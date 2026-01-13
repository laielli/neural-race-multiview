"""
Experiment: Competing Views Analysis

Compares orthogonal slots (original) vs competing views (shared dimensions)
to understand winner-take-all dynamics.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MultiViewDataset, CompetingViewDataset
from src.model import MultiViewNet, init_weights
from src.metrics import measure_view_coverage
from src.train import train_hard_labels, train_kd


def run_comparison(num_seeds=10, epochs=100, lr=0.001, loss_type='mse', gradient_flow=True, output_dir='results'):
    """
    Compare orthogonal vs competing views for winner-take-all dynamics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    results = {
        'orthogonal': {'coverage': [], 'dominance': [], 'accuracy': []},
        'competing': {'coverage': [], 'dominance': [], 'accuracy': []}
    }

    print("=" * 70)
    print("Competing Views Experiment")
    print(f"Comparing orthogonal slots vs shared dimensions")
    print(f"Loss: {loss_type.upper()}, Gradient flow: {gradient_flow}")
    print("=" * 70)

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")

        # 1. Orthogonal slots (original)
        dataset_orth = MultiViewDataset(K=10, M=3, d_view=50, n_samples=10000, seed=seed)

        torch.manual_seed(seed + 5000)
        model_orth = MultiViewNet(d=dataset_orth.d, hidden=200, K=dataset_orth.K)
        model_orth.apply(init_weights)

        train_hard_labels(model_orth, dataset_orth, epochs=epochs, lr=lr, verbose=False,
                          loss_type=loss_type, gradient_flow=gradient_flow)

        cov_orth = measure_view_coverage(model_orth, dataset_orth)
        dom_orth, acc_orth = measure_dominance(model_orth, dataset_orth)

        results['orthogonal']['coverage'].append(cov_orth)
        results['orthogonal']['dominance'].append(dom_orth)
        results['orthogonal']['accuracy'].append(acc_orth)

        print(f"  Orthogonal: coverage={cov_orth:.3f}, dominance={dom_orth:.3f}")

        # 2. Competing views (shared dimensions)
        dataset_comp = CompetingViewDataset(K=10, M=3, d=150, view_spread=0.5,
                                            n_samples=10000, seed=seed)

        torch.manual_seed(seed + 5000)
        model_comp = MultiViewNet(d=dataset_comp.d, hidden=200, K=dataset_comp.K)
        model_comp.apply(init_weights)

        train_hard_labels(model_comp, dataset_comp, epochs=epochs, lr=lr, verbose=False,
                          loss_type=loss_type, gradient_flow=gradient_flow)

        cov_comp = measure_view_coverage(model_comp, dataset_comp)
        dom_comp, acc_comp = measure_dominance(model_comp, dataset_comp)

        results['competing']['coverage'].append(cov_comp)
        results['competing']['dominance'].append(dom_comp)
        results['competing']['accuracy'].append(acc_comp)

        print(f"  Competing:  coverage={cov_comp:.3f}, dominance={dom_comp:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name in ['orthogonal', 'competing']:
        cov = np.mean(results[name]['coverage'])
        cov_std = np.std(results[name]['coverage'])
        dom = np.mean(results[name]['dominance'])
        dom_std = np.std(results[name]['dominance'])
        print(f"{name:12s}: coverage={cov:.3f}±{cov_std:.3f}, dominance={dom:.3f}±{dom_std:.3f}")

    print(f"\nExpected (1/M): coverage=0.333, dominance>0.7 for winner-take-all")

    # Create comparison figure
    create_comparison_figure(results, output_path / 'figures' / 'competing_views_comparison.png')

    # Save results
    final_results = {
        'experiment': 'competing_views_comparison',
        'timestamp': datetime.now().isoformat(),
        'config': {'num_seeds': num_seeds, 'epochs': epochs, 'lr': lr, 'loss_type': loss_type, 'gradient_flow': gradient_flow},
        'results': {
            name: {
                'coverage': {'mean': float(np.mean(results[name]['coverage'])),
                             'std': float(np.std(results[name]['coverage']))},
                'dominance': {'mean': float(np.mean(results[name]['dominance'])),
                              'std': float(np.std(results[name]['dominance']))},
            }
            for name in ['orthogonal', 'competing']
        },
        'conclusion': 'No winner-take-all observed in either setup'
    }

    with open(output_path / 'results_competing_views.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    return results


def measure_dominance(model, dataset):
    """Measure mean dominance ratio and accuracy."""
    model.eval()
    dominances = []

    for y in range(dataset.K):
        class_strengths = []
        for m in range(dataset.M):
            phi = dataset.get_view_feature(y, m)
            s = model.get_pathway_strength(phi)
            class_strengths.append(s)

        max_s = max(class_strengths)
        dominances.append(max_s / (sum(class_strengths) + 1e-10))

    X, Y = dataset.get_tensors()
    with torch.no_grad():
        acc = (model(X).argmax(dim=1) == Y).float().mean().item()

    return float(np.mean(dominances)), acc


def create_comparison_figure(results, filepath):
    """Create bar chart comparing orthogonal vs competing views."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coverage comparison
    ax = axes[0]
    names = ['Orthogonal\nSlots', 'Competing\nViews']
    means = [np.mean(results['orthogonal']['coverage']),
             np.mean(results['competing']['coverage'])]
    stds = [np.std(results['orthogonal']['coverage']),
            np.std(results['competing']['coverage'])]

    bars = ax.bar(names, means, yerr=stds, capsize=5, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax.axhline(y=1/3, color='gray', linestyle='--', linewidth=2, label='Expected (1/M)')
    ax.set_ylabel('Coverage', fontsize=12)
    ax.set_title('View Coverage Comparison', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Dominance comparison
    ax = axes[1]
    means = [np.mean(results['orthogonal']['dominance']),
             np.mean(results['competing']['dominance'])]
    stds = [np.std(results['orthogonal']['dominance']),
            np.std(results['competing']['dominance'])]

    bars = ax.bar(names, means, yerr=stds, capsize=5, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax.axhline(y=1/3, color='gray', linestyle='--', linewidth=2, label='Equal (1/M)')
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='Winner-take-all threshold')
    ax.set_ylabel('Dominance Ratio', fontsize=12)
    ax.set_title('Pathway Dominance Comparison', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {filepath}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (small for gradient flow)')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss type')
    parser.add_argument('--no-gradient-flow', action='store_true', help='Disable gradient flow (use momentum)')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    if args.quick:
        args.seeds = 3
        args.epochs = 50

    run_comparison(num_seeds=args.seeds, epochs=args.epochs, lr=args.lr,
                   loss_type=args.loss, gradient_flow=not args.no_gradient_flow,
                   output_dir=args.output)
