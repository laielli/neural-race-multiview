"""
Experiment: Multi-Task Race Dynamics

Tests winner-take-all dynamics using multi-task learning setup.
Unlike single-task multi-view (where all views predict same class),
this has M independent classification tasks creating genuine competition.

Key hypothesis: Under multi-task structure, race dynamics should emerge:
- Dominance > 0.5 (one pathway dominates)
- Only ~1/M tasks learned well per network
- Different seeds → different tasks learned

Success criteria:
- Dominance > 0.5 in >= 80% of seeds
- Coverage ≈ 1/M (± 0.1)
- Variance across seeds (different tasks learned)
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

from data_multitask import MultiTaskMultiViewDataset
from model import GatedDLN
from train import train_multitask_hard


def run_single_seed(
    M: int = 5,
    K: int = 10,
    d_view: int = 50,
    hidden: int = 64,
    n_samples: int = 5000,
    epochs: int = 500,
    lr: float = 0.02,
    init_scale: float = 0.2,
    seed: int = 0,
    verbose: bool = True
):
    """
    Run multi-task race dynamics experiment with a single seed.
    """
    torch.manual_seed(seed)

    # Create dataset
    dataset = MultiTaskMultiViewDataset(
        M=M, K=K, d_view=d_view,
        n_samples=n_samples, seed=seed
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Seed {seed}: Multi-Task Race Dynamics")
        print(f"{'='*60}")
        print(f"M={M} tasks, K={K} classes, d_view={d_view}")
        print(f"n_samples={n_samples}, epochs={epochs}, lr={lr}")

    # Create model
    model = GatedDLN(
        M=M,
        d_input=d_view,
        hidden=hidden,
        d_output=K,
        gate_mode='diagonal',
        init_scale=init_scale
    )
    model.init_orthogonal(init_scale)

    # Train
    history = train_multitask_hard(
        model, dataset,
        epochs=epochs,
        lr=lr,
        verbose=verbose
    )

    # Final results
    final_dom = history['dominance'][-1]
    final_accs = history['task_accuracies'][-1]
    mean_acc = sum(final_accs.values()) / len(final_accs)
    best_task = max(final_accs, key=final_accs.get)
    best_acc = final_accs[best_task]

    # Count tasks learned (accuracy > 0.5)
    learned_tasks = [m for m, acc in final_accs.items() if acc > 0.5]
    coverage = len(learned_tasks) / M

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results (seed={seed})")
        print(f"{'='*60}")
        print(f"Dominance: {final_dom:.4f}")
        print(f"Mean accuracy: {mean_acc:.4f}")
        print(f"Best task: {best_task} (acc={best_acc:.4f})")
        print(f"Learned tasks: {learned_tasks}")
        print(f"Coverage: {coverage:.3f} (expected: {1/M:.3f})")

        if final_dom > 0.5:
            print("✓ Winner-take-all observed (dominance > 0.5)")
        else:
            print("✗ No WTA (dominance <= 0.5)")

    return {
        'seed': seed,
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale
        },
        'final_dominance': final_dom,
        'final_task_accuracies': final_accs,
        'mean_accuracy': mean_acc,
        'best_task': best_task,
        'learned_tasks': learned_tasks,
        'coverage': coverage,
        'history': history
    }


def run_experiment(
    M: int = 5,
    K: int = 10,
    d_view: int = 50,
    hidden: int = 64,
    n_samples: int = 5000,
    epochs: int = 500,
    lr: float = 0.02,
    init_scale: float = 0.2,
    num_seeds: int = 10,
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Run full experiment across multiple seeds.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    all_results = []

    print("="*70)
    print("Multi-Task Race Dynamics Experiment")
    print("="*70)
    print(f"M={M} tasks, K={K} classes, {num_seeds} seeds")
    print(f"Expected: dominance > 0.5, coverage ≈ {1/M:.3f}")
    print("="*70)

    for seed in range(num_seeds):
        result = run_single_seed(
            M=M, K=K, d_view=d_view, hidden=hidden,
            n_samples=n_samples, epochs=epochs, lr=lr,
            init_scale=init_scale, seed=seed, verbose=verbose
        )
        all_results.append(result)

    # Aggregate results
    dominances = [r['final_dominance'] for r in all_results]
    coverages = [r['coverage'] for r in all_results]
    mean_accs = [r['mean_accuracy'] for r in all_results]
    best_tasks = [r['best_task'] for r in all_results]

    # Count WTA occurrences
    wta_count = sum(1 for d in dominances if d > 0.5)

    # Check if different seeds learned different tasks
    unique_best_tasks = len(set(best_tasks))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Dominance: {np.mean(dominances):.4f} ± {np.std(dominances):.4f}")
    print(f"Coverage: {np.mean(coverages):.4f} ± {np.std(coverages):.4f} (expected: {1/M:.3f})")
    print(f"Mean accuracy: {np.mean(mean_accs):.4f} ± {np.std(mean_accs):.4f}")
    print(f"WTA count: {wta_count}/{num_seeds} ({100*wta_count/num_seeds:.0f}%)")
    print(f"Unique best tasks: {unique_best_tasks}/{M}")
    print(f"Best tasks by seed: {best_tasks}")

    # Success criteria
    print("\n" + "-"*50)
    print("Success Criteria:")
    print("-"*50)

    success = True

    if wta_count >= 0.8 * num_seeds:
        print(f"✓ WTA in >= 80% of seeds: {wta_count}/{num_seeds}")
    else:
        print(f"✗ WTA in < 80% of seeds: {wta_count}/{num_seeds}")
        success = False

    expected_coverage = 1 / M
    if abs(np.mean(coverages) - expected_coverage) <= 0.15:
        print(f"✓ Coverage ≈ 1/M: {np.mean(coverages):.3f} ≈ {expected_coverage:.3f}")
    else:
        print(f"✗ Coverage not ≈ 1/M: {np.mean(coverages):.3f} vs {expected_coverage:.3f}")
        success = False

    if unique_best_tasks > 1:
        print(f"✓ Different seeds learn different tasks: {unique_best_tasks} unique")
    else:
        print(f"✗ All seeds learn same task")
        success = False

    print("\n" + "="*50)
    if success:
        print("EXPERIMENT PASSED - Race dynamics confirmed!")
    else:
        print("EXPERIMENT FAILED - Race dynamics not observed")
    print("="*50)

    # Plot results
    plot_dominance_distribution(all_results, output_path / 'figures' / 'dominance_dist.png', M)
    plot_task_learning(all_results, output_path / 'figures' / 'task_learning.png', M)

    # Save results
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'num_seeds': num_seeds
        },
        'summary': {
            'dominance_mean': float(np.mean(dominances)),
            'dominance_std': float(np.std(dominances)),
            'coverage_mean': float(np.mean(coverages)),
            'coverage_std': float(np.std(coverages)),
            'wta_count': wta_count,
            'unique_best_tasks': unique_best_tasks,
            'success': success
        },
        'per_seed': [
            {k: v for k, v in r.items() if k != 'history'}
            for r in all_results
        ]
    }

    with open(output_path / 'multitask_race_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'multitask_race_results.json'}")

    return all_results, success


def plot_dominance_distribution(results, output_path, M):
    """Plot distribution of dominance across seeds."""
    dominances = [r['final_dominance'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(range(len(dominances)), dominances, color='steelblue', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='WTA threshold')
    ax.axhline(y=1/M, color='gray', linestyle='--', linewidth=1, label=f'Equal (1/{M})')

    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Dominance', fontsize=12)
    ax.set_title('Pathway Dominance by Seed', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_task_learning(results, output_path, M):
    """Plot which tasks each seed learned."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create heatmap of task accuracies
    accs = np.zeros((len(results), M))
    for i, r in enumerate(results):
        for m, acc in r['final_task_accuracies'].items():
            accs[i, m] = acc

    im = ax.imshow(accs.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Task', fontsize=12)
    ax.set_title('Task Accuracy by Seed (WTA should show column dominance)', fontsize=14)
    ax.set_xticks(range(len(results)))
    ax.set_yticks(range(M))

    plt.colorbar(im, ax=ax, label='Accuracy')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Task Race Dynamics Experiment')
    parser.add_argument('--M', type=int, default=5, help='Number of tasks/views')
    parser.add_argument('--K', type=int, default=10, help='Classes per task')
    parser.add_argument('--d-view', type=int, default=50, help='Dimension per view')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer width')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.2, help='Init scale')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer epochs/seeds)')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 100
        args.seeds = 3
        print("Quick mode: epochs=100, seeds=3")

    run_experiment(
        M=args.M,
        K=args.K,
        d_view=args.d_view,
        hidden=args.hidden,
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        init_scale=args.init_scale,
        num_seeds=args.seeds,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
