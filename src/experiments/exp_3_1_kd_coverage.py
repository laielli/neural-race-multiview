"""
Experiment 3.1: KD Coverage Transfer

Compares view coverage: hard labels vs KD from teacher ensemble.

Expected: C(student_KD) ≈ C(ensemble) >> C(student_hard)
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
from src.metrics import measure_view_coverage, measure_ensemble_coverage
from src.train import train_hard_labels, train_kd


def experiment_3_1_kd_coverage(
    num_teachers: int = 5,
    num_trials: int = 10,
    K: int = 10,
    M: int = 3,
    d_view: int = 50,
    hidden: int = 200,
    epochs: int = 100,
    temperature: float = 4.0,
    lr: float = 0.001,  # Smaller LR for gradient flow approximation
    batch_size: int = 128,
    n_samples: int = 10000,
    loss_type: str = 'mse',  # MSE to match Saxe et al. theory
    gradient_flow: bool = True,  # No momentum for gradient flow
    output_dir: str = 'results/figures',
    verbose: bool = True
):
    """
    Compare coverage: hard labels vs KD from ensemble.

    Args:
        num_teachers: Number of teachers in ensemble
        num_trials: Number of trials to run
        K: Number of classes
        M: Views per class
        d_view: Dimensions per view slot
        hidden: Hidden layer width
        epochs: Training epochs
        temperature: KD temperature
        lr: Learning rate
        batch_size: Batch size
        n_samples: Training samples
        output_dir: Directory for saving figures
        verbose: Print progress

    Returns:
        dict: Experiment results
    """
    results = {
        'hard_label': [],
        'kd': [],
        'ensemble': [],
        'individual_teachers': []
    }

    if verbose:
        print("=" * 60)
        print("Experiment 3.1: KD Coverage Transfer")
        print(f"Config: {num_teachers} teachers, {num_trials} trials")
        print(f"K={K}, M={M}, temperature={temperature}")
        print(f"Loss: {loss_type.upper()}, Gradient flow: {gradient_flow}")
        print("=" * 60)

    for trial in range(num_trials):
        if verbose:
            print(f"\n=== Trial {trial + 1}/{num_trials} ===")

        # Shared dataset for fair comparison
        dataset = MultiViewDataset(
            K=K, M=M, d_view=d_view,
            n_samples=n_samples, seed=trial
        )

        # Train teacher ensemble
        teachers = []
        teacher_coverages = []

        for i in range(num_teachers):
            torch.manual_seed(trial * 100 + i)
            teacher = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
            teacher.apply(init_weights)

            train_hard_labels(teacher, dataset, epochs=epochs, lr=lr,
                              batch_size=batch_size, verbose=False,
                              loss_type=loss_type, gradient_flow=gradient_flow)
            teachers.append(teacher)

            t_cov = measure_view_coverage(teacher, dataset)
            teacher_coverages.append(t_cov)

            if verbose:
                print(f"  Teacher {i}: coverage = {t_cov:.3f}")

        results['individual_teachers'].append(teacher_coverages)

        # Measure ensemble coverage
        ensemble_cov = measure_ensemble_coverage(teachers, dataset)
        results['ensemble'].append(ensemble_cov)
        if verbose:
            print(f"  Ensemble coverage: {ensemble_cov:.3f}")

        # Train student with hard labels (baseline)
        torch.manual_seed(trial * 1000)
        student_hard = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
        student_hard.apply(init_weights)

        train_hard_labels(student_hard, dataset, epochs=epochs, lr=lr,
                          batch_size=batch_size, verbose=False,
                          loss_type=loss_type, gradient_flow=gradient_flow)

        hard_cov = measure_view_coverage(student_hard, dataset)
        results['hard_label'].append(hard_cov)
        if verbose:
            print(f"  Student (hard) coverage: {hard_cov:.3f}")

        # Train student with KD (same initialization as hard label student)
        torch.manual_seed(trial * 1000)  # Same init!
        student_kd = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
        student_kd.apply(init_weights)

        train_kd(student_kd, teachers, dataset, epochs=epochs,
                 temperature=temperature, lr=lr, batch_size=batch_size,
                 verbose=False, gradient_flow=gradient_flow)

        kd_cov = measure_view_coverage(student_kd, dataset)
        results['kd'].append(kd_cov)
        if verbose:
            print(f"  Student (KD) coverage: {kd_cov:.3f}")

    # Summary statistics
    mean_hard = np.mean(results['hard_label'])
    std_hard = np.std(results['hard_label'])
    mean_kd = np.mean(results['kd'])
    std_kd = np.std(results['kd'])
    mean_ensemble = np.mean(results['ensemble'])
    std_ensemble = np.std(results['ensemble'])

    if verbose:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Hard label:  {mean_hard:.3f} ± {std_hard:.3f}")
        print(f"KD:          {mean_kd:.3f} ± {std_kd:.3f}")
        print(f"Ensemble:    {mean_ensemble:.3f} ± {std_ensemble:.3f}")
        print(f"\nExpected (1/M): {1/M:.3f}")

        # Check if KD matches ensemble
        kd_vs_ensemble = abs(mean_kd - mean_ensemble)
        if kd_vs_ensemble < 0.1:
            print("✓ PASSED: KD student coverage ≈ ensemble coverage")
        else:
            print(f"✗ FAILED: KD student differs from ensemble by {kd_vs_ensemble:.3f}")

        # Check if KD > hard
        if mean_kd > mean_hard + 0.1:
            print("✓ PASSED: KD coverage > hard label coverage")
        else:
            print("✗ FAILED: KD coverage not significantly higher than hard label")

    # Create visualization
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Hard Labels', 'KD', 'Ensemble']
    means = [mean_hard, mean_kd, mean_ensemble]
    stds = [std_hard, std_kd, std_ensemble]
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    bars = ax.bar(categories, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

    # Add individual trial points
    for i, (cat, data) in enumerate(zip(categories, [results['hard_label'], results['kd'], results['ensemble']])):
        x_jitter = np.random.normal(i, 0.05, len(data))
        ax.scatter(x_jitter, data, color='black', alpha=0.3, s=30, zorder=3)

    # Add expected line
    ax.axhline(y=1/M, color='gray', linestyle='--', linewidth=2, label=f'Expected (1/M = {1/M:.2f})')

    ax.set_ylabel('View Coverage', fontsize=12)
    ax.set_title('KD Coverage Transfer: Hard Labels vs KD vs Ensemble', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig_path = output_path / 'kd_coverage_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nSaved figure: {fig_path}")

    return {
        'experiment': 'exp_3_1_kd_coverage',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_teachers': num_teachers,
            'num_trials': num_trials,
            'K': K,
            'M': M,
            'd_view': d_view,
            'hidden': hidden,
            'epochs': epochs,
            'temperature': temperature,
            'lr': lr,
            'batch_size': batch_size,
            'n_samples': n_samples,
            'loss_type': loss_type,
            'gradient_flow': gradient_flow
        },
        'metrics': {
            'hard_label': {'mean': float(mean_hard), 'std': float(std_hard)},
            'kd': {'mean': float(mean_kd), 'std': float(std_kd)},
            'ensemble': {'mean': float(mean_ensemble), 'std': float(std_ensemble)},
            'kd_vs_ensemble_diff': float(abs(mean_kd - mean_ensemble)),
            'kd_vs_hard_diff': float(mean_kd - mean_hard),
            'passed_kd_matches_ensemble': bool(abs(mean_kd - mean_ensemble) < 0.1),
            'passed_kd_better_than_hard': bool(mean_kd > mean_hard + 0.1)
        },
        'raw_data': {
            'hard_label': [float(x) for x in results['hard_label']],
            'kd': [float(x) for x in results['kd']],
            'ensemble': [float(x) for x in results['ensemble']],
            'individual_teachers': [[float(x) for x in trial] for trial in results['individual_teachers']]
        },
        'figure_path': str(fig_path)
    }


def main():
    parser = argparse.ArgumentParser(description='Experiment 3.1: KD Coverage Transfer')
    parser.add_argument('--teachers', type=int, default=5, help='Number of teachers')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (small for gradient flow)')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD temperature')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss type')
    parser.add_argument('--no-gradient-flow', action='store_true', help='Disable gradient flow (use momentum)')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    args = parser.parse_args()

    if args.quick:
        args.teachers = 3
        args.trials = 3
        args.epochs = 50

    results = experiment_3_1_kd_coverage(
        num_teachers=args.teachers,
        num_trials=args.trials,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        loss_type=args.loss,
        gradient_flow=not args.no_gradient_flow,
        output_dir=str(Path(args.output) / 'figures')
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'results_exp_3_1.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
