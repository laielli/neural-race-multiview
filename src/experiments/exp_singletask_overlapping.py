"""
Experiment: Single-Task Multi-View with Overlapping Gates

Tests winner-take-all dynamics when all views predict the SAME task
using overlapping gates (not diagonal).

Key insight from Saxe replication:
- Race dynamics require multiple pathways competing for SAME target
- Diagonal gates create independent pathways (no competition)
- Overlapping gates create competition for shared representation

This experiment uses:
- Single-task: All views predict same class label
- Overlapping gates: k_plus_minus_mod pattern
- MSE loss: Matches Saxe theory

Expected results:
- Dominance > 0.5 (one view wins the race)
- SVD dynamics match theory (sigmoidal growth, different rates)
- Different seeds → different views win
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import MultiViewDataset
from model import GatedDLN


def train_singletask_overlapping(
    model,
    dataset,
    epochs: int = 500,
    lr: float = 0.02,
    log_interval: int = 10,
    verbose: bool = True
):
    """
    Train GatedDLN on single-task multi-view data with overlapping gates.

    All views predict the SAME classification task.
    Overlapping gates create competition between pathways.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    history = {
        'loss': [],
        'accuracy': [],
        'epochs_logged': [],
        'dominance': [],
        'pathway_strengths': [],
        'svd': {
            'hidden': []
        }
    }

    # Get full dataset
    X, Y = dataset.get_tensors()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training Single-Task Overlapping")

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()

        # Split input into views
        x_list = [X[:, m * dataset.d_view:(m + 1) * dataset.d_view]
                  for m in range(model.M)]

        # One-hot targets (same for all views)
        y_onehot = F.one_hot(Y, num_classes=model.d_output).float()

        # Forward pass with gated loss
        loss = model.forward(x_list, y_target=y_onehot)

        loss.backward()
        optimizer.step()

        # Log metrics
        if epoch % log_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                history['loss'].append(loss.item())
                history['epochs_logged'].append(epoch)

                # Compute accuracy (using first gated output)
                outputs = model.forward(x_list)
                if outputs:
                    # Average over all active pathways
                    avg_output = torch.stack(outputs).mean(dim=0)
                    preds = avg_output.argmax(dim=-1)
                    acc = (preds == Y).float().mean().item()
                    history['accuracy'].append(acc)
                else:
                    history['accuracy'].append(0.0)

                # Pathway strengths and dominance
                strengths = model.get_all_pathway_strengths()
                history['pathway_strengths'].append(strengths)
                history['dominance'].append(model.compute_dominance())

                # Hidden layer SVs
                svs = model.get_singular_values()
                history['svd']['hidden'].append(svs['hidden'].tolist())

            if verbose:
                iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{history['accuracy'][-1]:.3f}",
                    'dom': f"{history['dominance'][-1]:.3f}"
                })

    return history


def run_experiment(
    M: int = 5,
    K: int = 10,
    d_view: int = 50,
    hidden: int = 64,
    n_samples: int = 5000,
    epochs: int = 500,
    lr: float = 0.02,
    init_scale: float = 0.2,
    gate_k: int = 3,
    num_seeds: int = 10,
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Run single-task multi-view experiment with overlapping gates.

    Args:
        gate_k: Number of neighbors for k_plus_minus_mod gating
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    all_results = []

    print("=" * 70)
    print("Single-Task Multi-View with Overlapping Gates")
    print("=" * 70)
    print(f"M={M} views, K={K} classes")
    print(f"Gate mode: k_plus_minus_mod with k={gate_k}")
    print(f"Expected: dominance > 0.5 (WTA), SVD race dynamics")
    print("=" * 70)

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create single-task multi-view dataset
        dataset = MultiViewDataset(
            M=M, K=K, d_view=d_view,
            n_samples=n_samples, seed=seed
        )

        # Create model with overlapping gates
        model = GatedDLN(
            M=M,
            d_input=d_view,
            hidden=hidden,
            d_output=K,
            gate_mode='k_plus_minus_mod',
            k_neighbors=gate_k,
            init_scale=init_scale
        )
        model.init_orthogonal(init_scale)

        if verbose:
            print(f"\nSeed {seed}: Gate pattern")
            print(model.gate)

        # Train
        history = train_singletask_overlapping(
            model, dataset,
            epochs=epochs, lr=lr,
            verbose=verbose
        )

        # Final metrics
        final_dom = history['dominance'][-1]
        final_acc = history['accuracy'][-1]
        final_strengths = history['pathway_strengths'][-1]

        # Which pathway won?
        winner = np.argmax(final_strengths)

        result = {
            'seed': seed,
            'final_dominance': final_dom,
            'final_accuracy': final_acc,
            'final_pathway_strengths': final_strengths,
            'winner_pathway': int(winner),
            'history': history
        }
        all_results.append(result)

        if verbose:
            print(f"\nSeed {seed} Results:")
            print(f"  Dominance: {final_dom:.4f}")
            print(f"  Winner: pathway {winner}")
            print(f"  Accuracy: {final_acc:.4f}")

    # Aggregate results
    dominances = [r['final_dominance'] for r in all_results]
    winners = [r['winner_pathway'] for r in all_results]
    accuracies = [r['final_accuracy'] for r in all_results]

    wta_count = sum(1 for d in dominances if d > 0.5)
    unique_winners = len(set(winners))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dominance: {np.mean(dominances):.4f} +/- {np.std(dominances):.4f}")
    print(f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    print(f"WTA count (dom > 0.5): {wta_count}/{num_seeds} ({100*wta_count/num_seeds:.0f}%)")
    print(f"Unique winners: {unique_winners}/{M}")
    print(f"Winners by seed: {winners}")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    if wta_count >= 0.8 * num_seeds:
        print(f"✓ WTA in >= 80% of seeds: {wta_count}/{num_seeds}")
    else:
        print(f"✗ WTA in < 80% of seeds: {wta_count}/{num_seeds}")
        success = False

    if unique_winners > 1:
        print(f"✓ Different seeds produce different winners: {unique_winners} unique")
    else:
        print(f"✗ All seeds produce same winner")
        success = False

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - WTA dynamics confirmed!")
    else:
        print("EXPERIMENT FAILED - WTA dynamics not observed")
    print("=" * 50)

    # Plot results
    plot_dominance_evolution(all_results, output_path / 'figures' / 'singletask_overlapping_dominance.png')
    plot_svd_evolution(all_results[0], output_path / 'figures' / 'singletask_overlapping_svd.png')

    # Save results
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'gate_k': gate_k, 'num_seeds': num_seeds
        },
        'summary': {
            'dominance_mean': float(np.mean(dominances)),
            'dominance_std': float(np.std(dominances)),
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'wta_count': wta_count,
            'unique_winners': unique_winners,
            'success': success
        },
        'per_seed': [
            {k: v for k, v in r.items() if k != 'history'}
            for r in all_results
        ]
    }

    with open(output_path / 'singletask_overlapping_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'singletask_overlapping_results.json'}")

    return all_results, success


def plot_dominance_evolution(results, output_path):
    """Plot dominance evolution for all seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        epochs = r['history']['epochs_logged']
        dominances = r['history']['dominance']
        ax.plot(epochs, dominances, alpha=0.7, label=f"Seed {r['seed']}")

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='WTA threshold')
    ax.axhline(y=1/len(results[0]['final_pathway_strengths']), color='gray',
               linestyle='--', linewidth=1, label='Equal (1/M)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dominance', fontsize=12)
    ax.set_title('Pathway Dominance Evolution (Single-Task, Overlapping Gates)', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='center right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_svd_evolution(result, output_path):
    """Plot hidden layer SVD evolution for a single seed."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = result['history']['epochs_logged']
    svd_history = result['history']['svd']['hidden']

    # Get number of SVs to plot (top 10)
    n_svs = min(10, len(svd_history[0]))

    for i in range(n_svs):
        svs = [epoch_svs[i] for epoch_svs in svd_history]
        ax.plot(epochs, svs, label=f'SV {i+1}', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title(f'Hidden Layer SVD Evolution (Seed {result["seed"]})', fontsize=14)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Single-Task Overlapping Gates Experiment')
    parser.add_argument('--M', type=int, default=5, help='Number of views')
    parser.add_argument('--K', type=int, default=10, help='Number of classes')
    parser.add_argument('--d-view', type=int, default=50, help='Dimension per view')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer width')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.2, help='Init scale')
    parser.add_argument('--gate-k', type=int, default=3, help='Gate k_neighbors parameter')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test')
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
        gate_k=args.gate_k,
        num_seeds=args.seeds,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
