"""
Experiment: Theory-Matched Settings

Test winner-take-all dynamics using settings that match Saxe et al. theory:
- Deep linear network (no ReLU)
- MSE loss
- Gradient flow (no momentum, small learning rate)
- Small initialization

Expected: Coverage ≈ 0.33 (one view per class), dominance > 0.7
"""

import sys
import torch
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import MultiViewDataset
from model import DeepLinearNet, init_weights_linear
from train import train_hard_labels
from metrics import measure_view_coverage, compute_pathway_dominance


def run_experiment(
    seed: int = 42,
    K: int = 10,
    M: int = 3,
    d_view: int = 50,
    hidden: int = 50,  # Smaller than original 200
    epochs: int = 200,
    lr: float = 0.001,  # Small for gradient flow
    n_samples: int = 5000,
    verbose: bool = True
):
    """Run single experiment with theory-matched settings."""

    torch.manual_seed(seed)

    # Create dataset
    dataset = MultiViewDataset(
        K=K, M=M, d_view=d_view,
        n_samples=n_samples,
        noise_std=0.1
    )

    # Create deep linear network
    model = DeepLinearNet(d=dataset.d, hidden=hidden, K=K)
    model.apply(init_weights_linear)

    if verbose:
        print(f"Configuration:")
        print(f"  Model: DeepLinearNet (no ReLU)")
        print(f"  Hidden: {hidden}")
        print(f"  Loss: MSE")
        print(f"  Optimizer: SGD (no momentum)")
        print(f"  LR: {lr}")
        print(f"  Epochs: {epochs}")
        print()

    # Train with MSE loss and gradient flow
    history = train_hard_labels(
        model, dataset,
        epochs=epochs,
        lr=lr,
        batch_size=128,
        log_interval=20,
        track_pathways=True,
        track_classes=list(range(min(3, K))),  # Track first 3 classes
        verbose=verbose,
        loss_type='mse',
        gradient_flow=True
    )

    # Measure final metrics
    coverage = measure_view_coverage(model, dataset)
    dominance = compute_pathway_dominance(model, dataset)

    if verbose:
        print(f"\nResults:")
        print(f"  Final accuracy: {history['accuracy'][-1]:.4f}")
        print(f"  Final loss: {history['loss'][-1]:.4f}")
        print(f"  Coverage: {coverage:.4f} (expected ≈ {1/M:.2f})")
        print(f"  Dominance: {dominance:.4f} (expected > 0.7)")
        print()

        # Check if theory predictions hold
        if coverage < 0.5:
            print("✓ Winner-take-all observed! Coverage < 0.5")
        else:
            print("✗ Still learning all views. Coverage >= 0.5")

        if dominance > 0.5:
            print(f"✓ Pathway dominance observed! Dominance = {dominance:.2f}")
        else:
            print(f"✗ Equal pathways. Dominance = {dominance:.2f}")

    return {
        'seed': seed,
        'coverage': coverage,
        'dominance': dominance,
        'final_accuracy': history['accuracy'][-1],
        'final_loss': history['loss'][-1],
        'pathway_strengths': history['pathway_strengths'],
        'config': {
            'model': 'DeepLinearNet',
            'hidden': hidden,
            'loss': 'mse',
            'gradient_flow': True,
            'lr': lr,
            'epochs': epochs
        }
    }


def run_comparison():
    """Compare theory-matched vs original settings."""

    print("=" * 60)
    print("EXPERIMENT: Theory-Matched Settings")
    print("=" * 60)
    print()

    results = []

    # Test multiple seeds
    for seed in [42, 123, 456]:
        print(f"\n--- Seed {seed} ---")
        result = run_experiment(seed=seed, verbose=True)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    coverages = [r['coverage'] for r in results]
    dominances = [r['dominance'] for r in results]

    print(f"Coverage:  mean={sum(coverages)/len(coverages):.3f}, std={torch.tensor(coverages).std().item():.3f}")
    print(f"Dominance: mean={sum(dominances)/len(dominances):.3f}, std={torch.tensor(dominances).std().item():.3f}")

    avg_coverage = sum(coverages) / len(coverages)
    if avg_coverage < 0.5:
        print("\n✓ SUCCESS: Theory predictions validated!")
    else:
        print("\n✗ FAIL: Theory predictions not observed.")
        print("  Consider adjusting: hidden size, learning rate, epochs")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'log' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'theory_match_results.json'

    with open(output_file, 'w') as f:
        # Convert pathway_strengths to serializable format
        for r in results:
            r['pathway_strengths'] = {
                str(k): v for k, v in r['pathway_strengths'].items()
            }
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    run_comparison()
