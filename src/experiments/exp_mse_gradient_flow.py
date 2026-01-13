"""
Experiment: MSE + Gradient Flow for Emergent Competition

Based on Saxe et al. 2022 analysis, winner-take-all should emerge from:
1. MSE loss (not cross-entropy)
2. Gradient flow (no momentum, small learning rate)
3. Deep linear networks (exact match to theory)

This experiment tests whether these theory-matched conditions produce
emergent winner-take-all without explicit competition loss.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import MultiViewDataset
from model import DeepLinearNet, MultiViewNet, init_weights_linear, init_weights
from train import train_hard_labels
from metrics import measure_view_coverage, compute_pathway_dominance


def run_single_experiment(
    model_type='linear',
    loss_type='mse',
    gradient_flow=True,
    lr=0.001,
    epochs=500,
    hidden=50,
    seed=42,
    verbose=False
):
    """Run single training experiment."""
    torch.manual_seed(seed)

    dataset = MultiViewDataset(
        K=10, M=3, d_view=50,
        n_samples=5000,
        noise_std=0.1
    )

    if model_type == 'linear':
        model = DeepLinearNet(d=dataset.d, hidden=hidden, K=dataset.K)
        model.apply(init_weights_linear)
    else:
        model = MultiViewNet(d=dataset.d, K=dataset.K, hidden=hidden)
        model.apply(init_weights)

    history = train_hard_labels(
        model, dataset,
        epochs=epochs,
        lr=lr,
        batch_size=128,
        log_interval=50,
        track_pathways=True,
        track_classes=[0, 1, 2],
        verbose=verbose,
        loss_type=loss_type,
        gradient_flow=gradient_flow
    )

    coverage = measure_view_coverage(model, dataset, threshold=0.1)
    dominance = compute_pathway_dominance(model, dataset)

    return {
        'coverage': coverage,
        'dominance': dominance,
        'accuracy': history['accuracy'][-1],
        'loss': history['loss'][-1],
        'pathway_strengths': history['pathway_strengths']
    }


def test_learning_rates():
    """Test different learning rates for gradient flow approximation."""
    print("=" * 70)
    print("TEST 1: Learning Rate Sweep (MSE + Gradient Flow)")
    print("=" * 70)
    print("\nTheory: Smaller LR = better gradient flow approximation")
    print("        Winner-take-all should emerge at sufficiently small LR\n")

    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    print(f"{'LR':<12} | {'Dominance':<10} {'Coverage':<10} {'Accuracy':<10} | {'WTA?':<5}")
    print("-" * 70)

    for lr in learning_rates:
        # Scale epochs inversely with LR to give similar total gradient
        epochs = int(500 * (0.001 / lr)) if lr > 0.0001 else 5000
        epochs = min(epochs, 5000)  # Cap at 5000

        result = run_single_experiment(
            model_type='linear',
            loss_type='mse',
            gradient_flow=True,
            lr=lr,
            epochs=epochs,
            verbose=False
        )

        wta = "YES" if result['dominance'] > 0.5 else "NO"
        print(f"{lr:<12} | {result['dominance']:<10.3f} {result['coverage']:<10.3f} {result['accuracy']:<10.3f} | {wta:<5}")


def test_mse_vs_ce():
    """Compare MSE vs cross-entropy under gradient flow."""
    print("\n" + "=" * 70)
    print("TEST 2: MSE vs Cross-Entropy (both with gradient flow)")
    print("=" * 70)
    print("\nTheory: MSE has natural saturation, CE doesn't")
    print("        Only MSE should produce winner-take-all\n")

    configs = [
        ('MSE + GradFlow', 'mse', True),
        ('CE + GradFlow', 'ce', True),
        ('MSE + SGD', 'mse', False),
        ('CE + SGD', 'ce', False),
    ]

    print(f"{'Config':<20} | {'Dominance':<10} {'Coverage':<10} {'Accuracy':<10} | {'WTA?':<5}")
    print("-" * 70)

    for name, loss, gf in configs:
        result = run_single_experiment(
            model_type='linear',
            loss_type=loss,
            gradient_flow=gf,
            lr=0.001,
            epochs=1000,
            verbose=False
        )

        wta = "YES" if result['dominance'] > 0.5 else "NO"
        print(f"{name:<20} | {result['dominance']:<10.3f} {result['coverage']:<10.3f} {result['accuracy']:<10.3f} | {wta:<5}")


def test_linear_vs_relu():
    """Compare deep linear vs ReLU networks."""
    print("\n" + "=" * 70)
    print("TEST 3: Linear vs ReLU Networks (MSE + gradient flow)")
    print("=" * 70)
    print("\nTheory: Deep linear networks match theory exactly")
    print("        ReLU networks have changing gating patterns\n")

    configs = [
        ('DeepLinear', 'linear'),
        ('ReLU MLP', 'relu'),
    ]

    print(f"{'Model':<20} | {'Dominance':<10} {'Coverage':<10} {'Accuracy':<10} | {'WTA?':<5}")
    print("-" * 70)

    for name, model_type in configs:
        result = run_single_experiment(
            model_type=model_type,
            loss_type='mse',
            gradient_flow=True,
            lr=0.001,
            epochs=1000,
            verbose=False
        )

        wta = "YES" if result['dominance'] > 0.5 else "NO"
        print(f"{name:<20} | {result['dominance']:<10.3f} {result['coverage']:<10.3f} {result['accuracy']:<10.3f} | {wta:<5}")


def test_extended_training():
    """Test very long training to see if WTA emerges eventually."""
    print("\n" + "=" * 70)
    print("TEST 4: Extended Training (MSE + small LR + long epochs)")
    print("=" * 70)
    print("\nTheory: WTA may require many iterations to emerge")
    print("        Testing up to 10000 epochs\n")

    epochs_list = [100, 500, 1000, 2000, 5000, 10000]

    print(f"{'Epochs':<12} | {'Dominance':<10} {'Coverage':<10} {'Accuracy':<10} | {'WTA?':<5}")
    print("-" * 70)

    for epochs in epochs_list:
        result = run_single_experiment(
            model_type='linear',
            loss_type='mse',
            gradient_flow=True,
            lr=0.0001,
            epochs=epochs,
            verbose=False
        )

        wta = "YES" if result['dominance'] > 0.5 else "NO"
        print(f"{epochs:<12} | {result['dominance']:<10.3f} {result['coverage']:<10.3f} {result['accuracy']:<10.3f} | {wta:<5}")


def test_seed_variance():
    """Test if WTA emerges for some seeds but not others."""
    print("\n" + "=" * 70)
    print("TEST 5: Seed Variance (MSE + gradient flow)")
    print("=" * 70)
    print("\nTheory: Different seeds should produce different winners")
    print("        If WTA exists, seeds should show high variance\n")

    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 9999]
    dominances = []

    print(f"{'Seed':<12} | {'Dominance':<10} {'Coverage':<10} {'Accuracy':<10} | {'WTA?':<5}")
    print("-" * 70)

    for seed in seeds:
        result = run_single_experiment(
            model_type='linear',
            loss_type='mse',
            gradient_flow=True,
            lr=0.001,
            epochs=1000,
            seed=seed,
            verbose=False
        )

        dominances.append(result['dominance'])
        wta = "YES" if result['dominance'] > 0.5 else "NO"
        print(f"{seed:<12} | {result['dominance']:<10.3f} {result['coverage']:<10.3f} {result['accuracy']:<10.3f} | {wta:<5}")

    print("-" * 70)
    print(f"{'Mean':<12} | {np.mean(dominances):<10.3f}")
    print(f"{'Std':<12} | {np.std(dominances):<10.3f}")
    print(f"{'Min':<12} | {np.min(dominances):<10.3f}")
    print(f"{'Max':<12} | {np.max(dominances):<10.3f}")


def test_small_init():
    """Test very small initialization (as in Saxe theory)."""
    print("\n" + "=" * 70)
    print("TEST 6: Initialization Scale (MSE + gradient flow)")
    print("=" * 70)
    print("\nTheory: Small initialization required for race dynamics")
    print("        Testing different init scales\n")

    import torch.nn as nn

    init_scales = [0.1, 0.01, 0.001, 0.0001]

    print(f"{'Init Scale':<12} | {'Dominance':<10} {'Coverage':<10} {'Accuracy':<10} | {'WTA?':<5}")
    print("-" * 70)

    for scale in init_scales:
        torch.manual_seed(42)

        dataset = MultiViewDataset(K=10, M=3, d_view=50, n_samples=5000, noise_std=0.1)
        model = DeepLinearNet(d=dataset.d, hidden=50, K=dataset.K)

        # Custom initialization with specific scale
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=scale)

        history = train_hard_labels(
            model, dataset,
            epochs=2000,
            lr=0.001,
            batch_size=128,
            log_interval=200,
            verbose=False,
            loss_type='mse',
            gradient_flow=True
        )

        coverage = measure_view_coverage(model, dataset, threshold=0.1)
        dominance = compute_pathway_dominance(model, dataset)

        wta = "YES" if dominance > 0.5 else "NO"
        print(f"{scale:<12} | {dominance:<10.3f} {coverage:<10.3f} {history['accuracy'][-1]:<10.3f} | {wta:<5}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("EXPERIMENT: MSE + Gradient Flow for Emergent Competition")
    print("=" * 70)
    print("\nBased on Saxe et al. 2022, testing if theory-matched conditions")
    print("produce winner-take-all without explicit competition loss.\n")

    test_mse_vs_ce()
    test_learning_rates()
    test_linear_vs_relu()
    test_small_init()
    test_seed_variance()
    test_extended_training()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
