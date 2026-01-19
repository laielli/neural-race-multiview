"""
Experiment: Race Dynamics with Asymmetric View Signals

Tests whether neural race dynamics emerge when views have different
predictive power (signal-to-noise ratio).

Theory prediction:
- View with highest σ_1(Σ_yx) should learn first
- SVD of hidden layer should show sigmoidal growth at different rates
- Final representation should be dominated by high-signal views

This experiment uses:
- AsymmetricMultiViewDataset with hierarchical signal strengths
- Simple 2-layer linear network (matches Saxe theory)
- MSE loss (required for theory match)
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
    """
    2-layer linear network for race dynamics experiments.

    Architecture: x -> W1 -> h -> W2 -> y
    No nonlinearities - matches Saxe deep linear network theory.
    """

    def __init__(self, d_input, hidden, d_output, init_scale=0.1):
        super().__init__()
        self.d_input = d_input
        self.hidden = hidden
        self.d_output = d_output

        self.W1 = nn.Linear(d_input, hidden, bias=False)
        self.W2 = nn.Linear(hidden, d_output, bias=False)

        # Small orthogonal init
        nn.init.orthogonal_(self.W1.weight, gain=init_scale)
        nn.init.orthogonal_(self.W2.weight, gain=init_scale)

    def forward(self, x):
        h = self.W1(x)
        return self.W2(h)

    def get_hidden_svd(self):
        """Get singular values of W1 (input -> hidden mapping)."""
        with torch.no_grad():
            _, s, _ = torch.linalg.svd(self.W1.weight)
            return s.cpu().numpy()

    def get_output_svd(self):
        """Get singular values of W2 (hidden -> output mapping)."""
        with torch.no_grad():
            _, s, _ = torch.linalg.svd(self.W2.weight)
            return s.cpu().numpy()

    def get_end_to_end_svd(self):
        """Get singular values of W2 @ W1 (full network)."""
        with torch.no_grad():
            W = self.W2.weight @ self.W1.weight
            _, s, _ = torch.linalg.svd(W)
            return s.cpu().numpy()


def train_linear_mse(model, dataset, epochs=500, lr=0.01, log_interval=10, verbose=True):
    """
    Train linear network with MSE loss (matches Saxe theory).
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    history = {
        'loss': [],
        'accuracy': [],
        'epochs_logged': [],
        'svd_hidden': [],
        'svd_output': [],
        'svd_end_to_end': []
    }

    X, Y = dataset.get_tensors()
    Y_onehot = F.one_hot(Y, num_classes=model.d_output).float()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training Linear MSE")

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()

        output = model(X)
        loss = F.mse_loss(output, Y_onehot)

        loss.backward()
        optimizer.step()

        # Log metrics
        if epoch % log_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                history['loss'].append(loss.item())
                history['epochs_logged'].append(epoch)

                # Accuracy
                preds = output.argmax(dim=-1)
                acc = (preds == Y).float().mean().item()
                history['accuracy'].append(acc)

                # SVD metrics
                history['svd_hidden'].append(model.get_hidden_svd().tolist())
                history['svd_output'].append(model.get_output_svd().tolist())
                history['svd_end_to_end'].append(model.get_end_to_end_svd().tolist())

            if verbose:
                iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{acc:.3f}"
                })

    return history


def measure_view_contributions(model, dataset):
    """
    Measure how much each view contributes to the network's predictions.

    Uses the projection of W1 onto each view's feature space.
    """
    with torch.no_grad():
        W1 = model.W1.weight.numpy()  # (hidden, d_input)

        contributions = []
        for m in range(dataset.M):
            # Extract weights for view m
            slot_start = m * dataset.d_view
            slot_end = (m + 1) * dataset.d_view
            W1_view = W1[:, slot_start:slot_end]

            # Contribution = Frobenius norm of this slice
            contrib = np.linalg.norm(W1_view, 'fro')
            contributions.append(contrib)

        # Normalize
        total = sum(contributions)
        return [c / total for c in contributions]


def run_experiment(
    M: int = 5,
    K: int = 10,
    d_view: int = 50,
    hidden: int = 64,
    n_samples: int = 5000,
    epochs: int = 500,
    lr: float = 0.01,
    init_scale: float = 0.1,
    num_seeds: int = 5,
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Run asymmetric race dynamics experiment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    all_results = []

    print("=" * 70)
    print("Asymmetric Race Dynamics Experiment")
    print("=" * 70)

    # Create dataset (same for all seeds - only model init varies)
    torch.manual_seed(42)
    dataset = AsymmetricMultiViewDataset(
        M=M, K=K, d_view=d_view,
        n_samples=n_samples, seed=42
    )

    print(f"M={M} views, K={K} classes")
    print(f"Signal strengths: {dataset.signal_strengths}")

    # Get correlation strengths (theory prediction)
    corr_strengths = dataset.get_view_correlation_strengths()
    predicted_winner = np.argmax(corr_strengths)
    print(f"\nCorrelation strengths (σ_1 per view):")
    for m, s in enumerate(corr_strengths):
        marker = " <- WINNER" if m == predicted_winner else ""
        print(f"  View {m}: {s:.4f}{marker}")
    print("=" * 70)

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model
        model = SimpleLinearNet(
            d_input=M * d_view,
            hidden=hidden,
            d_output=K,
            init_scale=init_scale
        )

        if verbose:
            print(f"\nSeed {seed}: Training...")

        # Train
        history = train_linear_mse(
            model, dataset,
            epochs=epochs, lr=lr,
            verbose=verbose
        )

        # Measure final view contributions
        view_contributions = measure_view_contributions(model, dataset)
        winner = np.argmax(view_contributions)

        result = {
            'seed': seed,
            'final_loss': history['loss'][-1],
            'final_accuracy': history['accuracy'][-1],
            'view_contributions': view_contributions,
            'winner_view': int(winner),
            'predicted_winner': int(predicted_winner),
            'correct_prediction': winner == predicted_winner,
            'history': history
        }
        all_results.append(result)

        if verbose:
            print(f"\nSeed {seed} Results:")
            print(f"  Accuracy: {history['accuracy'][-1]:.4f}")
            print(f"  View contributions: {[f'{c:.3f}' for c in view_contributions]}")
            print(f"  Winner: View {winner} {'✓' if winner == predicted_winner else '✗'}")

    # Aggregate results
    correct_predictions = sum(r['correct_prediction'] for r in all_results)
    winners = [r['winner_view'] for r in all_results]
    accuracies = [r['final_accuracy'] for r in all_results]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Predicted winner (from σ_1): View {predicted_winner}")
    print(f"Correct predictions: {correct_predictions}/{num_seeds}")
    print(f"Winners by seed: {winners}")
    print(f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    if correct_predictions >= 0.8 * num_seeds:
        print(f"✓ Theory prediction correct in >= 80% of seeds: {correct_predictions}/{num_seeds}")
    else:
        print(f"✗ Theory prediction correct in < 80% of seeds: {correct_predictions}/{num_seeds}")
        success = False

    # Check that dominant view has highest contribution
    avg_contributions = np.mean([r['view_contributions'] for r in all_results], axis=0)
    if avg_contributions[predicted_winner] == max(avg_contributions):
        print(f"✓ Predicted view has highest average contribution: {avg_contributions[predicted_winner]:.3f}")
    else:
        print(f"✗ Predicted view not highest: {avg_contributions[predicted_winner]:.3f} vs {max(avg_contributions):.3f}")
        success = False

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - Race dynamics match theory!")
    else:
        print("EXPERIMENT FAILED - Race dynamics not as predicted")
    print("=" * 50)

    # Plot results
    plot_svd_evolution(all_results[0], output_path / 'figures' / 'asymmetric_svd_evolution.png')
    plot_view_contributions(all_results, dataset, output_path / 'figures' / 'asymmetric_view_contributions.png')
    plot_loss_accuracy(all_results, output_path / 'figures' / 'asymmetric_training.png')

    # Save results
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'num_seeds': num_seeds
        },
        'dataset': {
            'signal_strengths': dataset.signal_strengths,
            'correlation_strengths': corr_strengths,
            'predicted_winner': int(predicted_winner)
        },
        'summary': {
            'correct_predictions': correct_predictions,
            'avg_accuracy': float(np.mean(accuracies)),
            'avg_view_contributions': avg_contributions.tolist(),
            'success': success
        },
        'per_seed': [
            {k: (int(v) if isinstance(v, (np.integer, np.int64)) else
                 float(v) if isinstance(v, (np.floating, np.float64)) else
                 [float(x) for x in v] if isinstance(v, (list, np.ndarray)) and len(v) > 0 and isinstance(v[0], (np.floating, np.float64)) else v)
             for k, v in r.items() if k != 'history'}
            for r in all_results
        ]
    }

    with open(output_path / 'asymmetric_race_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'asymmetric_race_results.json'}")

    return all_results, success


def plot_svd_evolution(result, output_path):
    """Plot SVD evolution during training."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = result['history']['epochs_logged']

    # Hidden layer SVD
    ax = axes[0]
    svd_hidden = np.array(result['history']['svd_hidden'])
    for i in range(min(10, svd_hidden.shape[1])):
        ax.plot(epochs, svd_hidden[:, i], label=f'SV {i+1}', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Singular Value')
    ax.set_title('Hidden Layer (W1) SVD')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Output layer SVD
    ax = axes[1]
    svd_output = np.array(result['history']['svd_output'])
    for i in range(min(10, svd_output.shape[1])):
        ax.plot(epochs, svd_output[:, i], label=f'SV {i+1}', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Singular Value')
    ax.set_title('Output Layer (W2) SVD')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # End-to-end SVD
    ax = axes[2]
    svd_e2e = np.array(result['history']['svd_end_to_end'])
    for i in range(min(10, svd_e2e.shape[1])):
        ax.plot(epochs, svd_e2e[:, i], label=f'SV {i+1}', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Singular Value')
    ax.set_title('End-to-End (W2@W1) SVD')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_view_contributions(results, dataset, output_path):
    """Plot view contributions vs correlation strengths."""
    fig, ax = plt.subplots(figsize=(10, 6))

    M = dataset.M
    x = np.arange(M)
    width = 0.35

    # Correlation strengths (normalized)
    corr_strengths = np.array(dataset.get_view_correlation_strengths())
    corr_strengths = corr_strengths / corr_strengths.sum()

    # Average view contributions
    avg_contributions = np.mean([r['view_contributions'] for r in results], axis=0)

    ax.bar(x - width/2, corr_strengths, width, label='Theory (σ_1 normalized)', color='steelblue')
    ax.bar(x + width/2, avg_contributions, width, label='Learned (W1 contrib)', color='orange')

    ax.set_xlabel('View', fontsize=12)
    ax.set_ylabel('Contribution (normalized)', fontsize=12)
    ax.set_title('View Contributions: Theory vs Learned', fontsize=14)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_loss_accuracy(results, output_path):
    """Plot loss and accuracy during training."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for r in results:
        epochs = r['history']['epochs_logged']
        axes[0].plot(epochs, r['history']['loss'], alpha=0.7, label=f"Seed {r['seed']}")
        axes[1].plot(epochs, r['history']['accuracy'], alpha=0.7, label=f"Seed {r['seed']}")

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Asymmetric Race Dynamics Experiment')
    parser.add_argument('--M', type=int, default=5, help='Number of views')
    parser.add_argument('--K', type=int, default=10, help='Number of classes')
    parser.add_argument('--d-view', type=int, default=50, help='Dimension per view')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer width')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.1, help='Init scale')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 200
        args.seeds = 3
        print("Quick mode: epochs=200, seeds=3")

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
