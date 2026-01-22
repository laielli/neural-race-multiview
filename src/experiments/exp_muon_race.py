"""
Experiment: Does Muon Break the Neural Race?

Tests whether gradient orthogonalization (Muon) disrupts
winner-take-all dynamics in GatedDLN architecture.

This is Experiment 1 from IDEA-014: Newton-Schulz Orthogonalization
as a Probe for Neural Race Dynamics in KD.

Hypothesis: Muon removes singular value information from gradients,
disrupting the SVD-based race mechanism.

Predictions:
- SGD: Dominance → ~0.5-0.6 (winner-take-all)
- Adam: Similar to SGD
- Muon: Dominance → ~0.33 (1/M, balanced pathways)

Usage:
    cd papers/neural-race-multiview/src
    python -m experiments.exp_muon_race
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GatedDLN
from train import train_gated_dln_with_optimizer


# Experiment config
CONFIG = {
    'M': 3,           # Number of pathways
    'd_input': 20,    # Input dimension per pathway
    'hidden': 32,     # Hidden layer width
    'd_output': 10,   # Output dimension
    'n_samples': 1000,
    'epochs': 500,
    'lr_sgd': 0.02,
    'lr_adam': 0.001,
    'lr_muon': 0.02,
    'lr_muon_exact': 0.02,  # Exact SVD orthogonalization
    'n_seeds': 5,
    'log_interval': 10,
    'noise_std': 0.1,
}


def create_structured_regression_data(M, d_input, d_output, n_samples, seed, noise_std=0.1):
    """
    Create regression data with controlled singular value structure.

    Different pathways have different input-output correlation strengths,
    which should produce race dynamics under standard training.

    Args:
        M: Number of pathways (views)
        d_input: Input dimension per pathway
        d_output: Output dimension
        n_samples: Number of samples
        seed: Random seed
        noise_std: Noise level

    Returns:
        X: Input tensor (n_samples, M * d_input)
        Y: Target tensor (n_samples, d_output)
        strengths: True pathway strengths [1.0, 0.67, 0.33] for M=3
    """
    torch.manual_seed(seed)

    d_total = M * d_input
    X = torch.randn(n_samples, d_total)

    # Create ground truth weights with different pathway strengths
    # Pathway m has strength (M - m) / M
    W_true = torch.zeros(d_output, d_total)
    strengths = []

    for m in range(M):
        strength = (M - m) / M  # [1.0, 0.67, 0.33] for M=3
        strengths.append(strength)

        # Random projection for this pathway
        W_m = torch.randn(d_output, d_input) * strength / np.sqrt(d_input)
        W_true[:, m*d_input:(m+1)*d_input] = W_m

    Y = X @ W_true.T + noise_std * torch.randn(n_samples, d_output)

    return X, Y, strengths


def run_single_experiment(optimizer_type, seed, config):
    """Run single training run and return metrics."""
    torch.manual_seed(seed)

    # Create data with structured pathway strengths
    X, Y, true_strengths = create_structured_regression_data(
        M=config['M'],
        d_input=config['d_input'],
        d_output=config['d_output'],
        n_samples=config['n_samples'],
        seed=seed,
        noise_std=config['noise_std']
    )

    # Create model - diagonal gate means each pathway is separate
    model = GatedDLN(
        M=config['M'],
        d_input=config['d_input'],
        hidden=config['hidden'],
        d_output=config['d_output'],
        gate_mode='diagonal',
        init_scale=0.1
    )
    model.init_orthogonal(scale=0.1)

    # Select learning rate
    lr_key = f'lr_{optimizer_type}'
    lr = config[lr_key]

    # Train
    history = train_gated_dln_with_optimizer(
        model, X, Y,
        optimizer_type=optimizer_type,
        epochs=config['epochs'],
        lr=lr,
        momentum=0.0,  # Gradient flow for theory match
        log_interval=config['log_interval'],
        track_svd=True,
        verbose=False
    )

    return {
        'final_dominance': history['dominance'][-1],
        'final_loss': history['loss'][-1],
        'dominance_trajectory': history['dominance'],
        'pathway_strengths': history['pathway_strengths'][-1],
        'true_strengths': true_strengths,
    }


def main():
    print("=" * 60)
    print("EXPERIMENT: Does Muon Break the Neural Race?")
    print("=" * 60)
    print(f"\nConfig: M={CONFIG['M']} pathways, {CONFIG['epochs']} epochs, {CONFIG['n_seeds']} seeds")
    print(f"Prediction: SGD/Adam → high dominance, Muon → low dominance (≈{1/CONFIG['M']:.2f})")

    results = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'optimizers': {}
    }

    all_dominances = {}

    for opt in ['sgd', 'adam', 'muon', 'muon_exact']:
        print(f"\n{'='*50}")
        print(f"Running {opt.upper()} experiments")
        print('='*50)

        opt_results = []
        for seed in range(CONFIG['n_seeds']):
            print(f"  Seed {seed}...", end=' ', flush=True)
            try:
                res = run_single_experiment(opt, seed, CONFIG)
                opt_results.append(res)
                print(f"dominance={res['final_dominance']:.3f}, loss={res['final_loss']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        if not opt_results:
            print(f"  WARNING: All runs failed for {opt}")
            continue

        # Aggregate
        dominances = [r['final_dominance'] for r in opt_results]
        all_dominances[opt] = dominances

        results['optimizers'][opt] = {
            'runs': opt_results,
            'mean_dominance': float(np.mean(dominances)),
            'std_dominance': float(np.std(dominances)),
            'min_dominance': float(np.min(dominances)),
            'max_dominance': float(np.max(dominances)),
        }
        print(f"\n{opt.upper()} mean dominance: {np.mean(dominances):.3f} ± {np.std(dominances):.3f}")

    # Statistical tests
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    if 'sgd' in all_dominances and 'muon' in all_dominances:
        t_stat, p_value = stats.ttest_ind(all_dominances['sgd'], all_dominances['muon'])
        print(f"\nSGD vs Muon t-test: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("  ✓ Difference is statistically significant (p < 0.05)")
        else:
            print("  ✗ Difference is NOT statistically significant")
        results['sgd_vs_muon_pvalue'] = float(p_value)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nExpected: SGD/Adam dominance > 0.4, Muon dominance ≈ {1/CONFIG['M']:.2f}")
    print()

    baseline_dominance = 1 / CONFIG['M']
    success = True

    for opt in ['sgd', 'adam', 'muon', 'muon_exact']:
        if opt not in results['optimizers']:
            continue
        r = results['optimizers'][opt]
        status = ""
        if opt in ['sgd', 'adam']:
            if r['mean_dominance'] > 0.4:
                status = "✓ Shows race"
            else:
                status = "✗ No clear race"
                success = False
        else:  # muon or muon_exact
            if r['mean_dominance'] < 0.4 and abs(r['mean_dominance'] - baseline_dominance) < 0.1:
                status = "✓ Disrupts race"
            elif r['mean_dominance'] < 0.38:
                status = "~ Partially disrupts race"
            else:
                status = "✗ Did not disrupt race as expected"
                success = False

        print(f"{opt.upper():6s}: {r['mean_dominance']:.3f} ± {r['std_dominance']:.3f}  {status}")

    print()
    if success:
        print("🎉 HYPOTHESIS SUPPORTED: Muon disrupts neural race dynamics!")
    else:
        print("⚠️  HYPOTHESIS NOT FULLY SUPPORTED - investigate results")

    # Save results
    out_dir = Path(__file__).parent.parent.parent / 'log' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'exp_muon_race_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == '__main__':
    main()
