"""
Experiment 2.1: Single-View Convergence

Validates Theorem 2 prediction that hard label training leads to C(f) ≈ 1/M.

Expected result: mean coverage ≈ 0.33 for M=3
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MultiViewDataset, verify_dataset
from src.model import MultiViewNet, init_weights
from src.metrics import measure_view_coverage, measure_view_coverage_detailed
from src.train import train_hard_labels


def experiment_2_1_single_view_convergence(
    num_seeds: int = 30,
    K: int = 10,
    M: int = 3,
    d_view: int = 50,
    hidden: int = 200,
    epochs: int = 100,
    lr: float = 0.001,  # Smaller LR for gradient flow approximation
    batch_size: int = 128,
    n_samples: int = 10000,
    loss_type: str = 'mse',  # MSE to match Saxe et al. theory
    gradient_flow: bool = True,  # No momentum for gradient flow
    verbose: bool = True
):
    """
    Validate that hard label training leads to C(f) ≈ 1/M.

    Args:
        num_seeds: Number of random seeds to test
        K: Number of classes
        M: Views per class
        d_view: Dimensions per view slot
        hidden: Hidden layer width
        epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        n_samples: Training samples
        verbose: Print progress

    Returns:
        dict: Experiment results
    """
    results = []
    detailed_results = []

    if verbose:
        print("=" * 60)
        print("Experiment 2.1: Single-View Convergence")
        print(f"Config: K={K}, M={M}, d_view={d_view}, epochs={epochs}")
        print(f"Loss: {loss_type.upper()}, Gradient flow: {gradient_flow}")
        print(f"Expected coverage: {1/M:.3f}")
        print("=" * 60)

    for seed in range(num_seeds):
        # Create fresh dataset and model
        dataset = MultiViewDataset(
            K=K, M=M, d_view=d_view,
            n_samples=n_samples, seed=seed
        )

        # Initialize model with different seed
        torch.manual_seed(seed + 5000)
        model = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
        model.apply(init_weights)

        # Train with MSE loss + gradient flow to match Saxe et al. theory
        history = train_hard_labels(
            model, dataset,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            verbose=False,
            loss_type=loss_type,
            gradient_flow=gradient_flow
        )

        # Measure coverage
        coverage = measure_view_coverage(model, dataset)
        coverage_detail = measure_view_coverage_detailed(model, dataset)

        results.append(coverage)
        detailed_results.append({
            'seed': seed,
            'coverage': coverage,
            'final_accuracy': history['accuracy'][-1],
            'per_class_views': {str(k): v for k, v in coverage_detail['per_class'].items()}
        })

        if verbose:
            print(f"Seed {seed:2d}: coverage = {coverage:.3f}, accuracy = {history['accuracy'][-1]:.3f}")

    # Summary statistics
    mean_cov = np.mean(results)
    std_cov = np.std(results)
    expected = 1 / M

    if verbose:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Mean coverage: {mean_cov:.3f} ± {std_cov:.3f}")
        print(f"Expected (1/M): {expected:.3f}")
        print(f"Difference: {abs(mean_cov - expected):.3f}")

        # Check if result matches expectation
        if abs(mean_cov - expected) < 0.05:
            print("✓ PASSED: Coverage matches 1/M prediction")
        else:
            print("✗ FAILED: Coverage differs from 1/M prediction")

    return {
        'experiment': 'exp_2_1_single_view_convergence',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'K': K,
            'M': M,
            'd_view': d_view,
            'hidden': hidden,
            'num_seeds': num_seeds,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'n_samples': n_samples,
            'loss_type': loss_type,
            'gradient_flow': gradient_flow
        },
        'metrics': {
            'mean_coverage': float(mean_cov),
            'std_coverage': float(std_cov),
            'expected': float(expected),
            'difference': float(abs(mean_cov - expected)),
            'passed': bool(abs(mean_cov - expected) < 0.05)
        },
        'raw_data': detailed_results
    }


def main():
    parser = argparse.ArgumentParser(description='Experiment 2.1: Single-View Convergence')
    parser.add_argument('--seeds', type=int, default=30, help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (small for gradient flow)')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss type')
    parser.add_argument('--no-gradient-flow', action='store_true', help='Disable gradient flow (use momentum)')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer seeds')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    args = parser.parse_args()

    if args.quick:
        args.seeds = 3
        args.epochs = 50

    results = experiment_2_1_single_view_convergence(
        num_seeds=args.seeds,
        epochs=args.epochs,
        lr=args.lr,
        loss_type=args.loss,
        gradient_flow=not args.no_gradient_flow
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'results_exp_2_1.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
