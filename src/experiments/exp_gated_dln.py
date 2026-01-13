"""
Experiment: GatedDLN Race Dynamics

Tests winner-take-all dynamics using the Saxe et al. GatedDLN architecture
which has explicit pathway separation.

Key differences from standard MLP experiments:
- M separate encoders and decoders (one per pathway)
- Shared hidden layer
- Binary gate controlling connectivity
- Track singular values to observe race dynamics

Expected: Winner-take-all in SVD space - different singular value modes
grow at different rates, with dominant modes winning.
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

from data import MultiViewDataset
from model import GatedDLN, GatedMultiViewNet
from train import train_gated_dln, train_gated_multiview


def test_basic_gated_dln(
    M: int = 3,
    d_input: int = 4,
    hidden: int = 64,
    d_output: int = 7,
    epochs: int = 500,
    lr: float = 0.02,
    init_scale: float = 0.2,
    gate_mode: str = 'diagonal',
    seed: int = 42,
    verbose: bool = True
):
    """
    Test basic GatedDLN with structured regression task (matching gated-dln notebook).

    Uses the same setup as Saxe et al.:
    - X = identity matrix (orthogonal inputs)
    - Y = structured output matrix
    """
    torch.manual_seed(seed)

    # Create structured input-output mapping (like gated-dln notebook)
    X = torch.eye(d_input)

    # Structured output: some features shared, some unique
    Y = torch.zeros(d_input, d_output)
    Y[:, 0] = 6.0  # All inputs map to feature 0
    Y[0:2, 1] = 4.0  # Inputs 0,1 map to feature 1
    Y[2:4, 2] = 4.0  # Inputs 2,3 map to feature 2
    for i in range(d_input):
        Y[i, 3 + i] = 3.0  # Each input has unique feature

    if verbose:
        print("=" * 60)
        print("Test: Basic GatedDLN (Saxe et al. style)")
        print("=" * 60)
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"Gate mode: {gate_mode}")
        print(f"Init scale: {init_scale}")
        print()

    # Create model
    model = GatedDLN(
        M=M,
        d_input=d_input,
        hidden=hidden,
        d_output=d_output,
        gate_mode=gate_mode,
        init_scale=init_scale
    )
    model.init_orthogonal(init_scale)

    if verbose:
        print(f"Gate matrix:\n{model.gate}")
        print()

    # Train
    history = train_gated_dln(
        model, X, Y,
        epochs=epochs,
        lr=lr,
        log_interval=epochs // 20,
        track_svd=True,
        verbose=verbose
    )

    # Results
    final_dom = history['dominance'][-1]
    final_loss = history['loss'][-1]

    if verbose:
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Final loss: {final_loss:.6f}")
        print(f"Final dominance: {final_dom:.4f}")
        print(f"Pathway strengths: {[f'{s:.3f}' for s in history['pathway_strengths'][-1]]}")

        # Check for WTA
        if final_dom > 0.5:
            print("Winner-take-all observed (dominance > 0.5)")
        else:
            print(f"Equal pathways (dominance = {final_dom:.3f} ~ 1/{M} = {1/M:.3f})")

    return {
        'config': {
            'M': M, 'd_input': d_input, 'hidden': hidden, 'd_output': d_output,
            'epochs': epochs, 'lr': lr, 'init_scale': init_scale,
            'gate_mode': gate_mode, 'seed': seed
        },
        'final_loss': final_loss,
        'final_dominance': final_dom,
        'pathway_strengths': history['pathway_strengths'][-1],
        'history': history
    }


def test_gated_multiview(
    M: int = 3,
    K: int = 10,
    d_view: int = 50,
    hidden: int = 64,
    n_samples: int = 5000,
    epochs: int = 500,
    lr: float = 0.01,
    init_scale: float = 0.2,
    gate_mode: str = 'diagonal',
    seed: int = 42,
    verbose: bool = True
):
    """
    Test GatedMultiViewNet on multi-view classification dataset.
    """
    torch.manual_seed(seed)

    # Create dataset
    dataset = MultiViewDataset(
        K=K, M=M, d_view=d_view,
        n_samples=n_samples,
        seed=seed
    )

    if verbose:
        print("=" * 60)
        print("Test: GatedMultiViewNet on Multi-View Dataset")
        print("=" * 60)
        print(f"Dataset: K={K} classes, M={M} views, d_view={d_view}")
        print(f"Gate mode: {gate_mode}")
        print(f"Init scale: {init_scale}")
        print()

    # Create model
    model = GatedMultiViewNet.from_dataset(
        dataset,
        hidden=hidden,
        gate_mode=gate_mode,
        init_scale=init_scale
    )
    model.init_orthogonal(init_scale)

    if verbose:
        print(f"Gate matrix:\n{model.gate}")
        print()

    # Train
    history = train_gated_multiview(
        model, dataset,
        epochs=epochs,
        lr=lr,
        batch_size=None,  # Full batch for theory match
        log_interval=epochs // 20,
        track_svd=True,
        verbose=verbose
    )

    # Results
    final_dom = history['dominance'][-1]
    final_acc = history['accuracy'][-1]
    final_cov = history['view_coverage'][-1]

    if verbose:
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Final accuracy: {final_acc:.4f}")
        print(f"Final view coverage: {final_cov:.4f}")
        print(f"Final dominance: {final_dom:.4f}")
        print(f"Pathway strengths: {[f'{s:.3f}' for s in history['pathway_strengths'][-1]]}")

        # Check for WTA
        if final_dom > 0.5:
            print("Winner-take-all observed (dominance > 0.5)")
        elif final_cov < 0.5:
            print("Partial view learning (coverage < 0.5)")
        else:
            print(f"All views learned (coverage = {final_cov:.3f})")

    return {
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'gate_mode': gate_mode, 'seed': seed
        },
        'final_accuracy': final_acc,
        'final_coverage': final_cov,
        'final_dominance': final_dom,
        'pathway_strengths': history['pathway_strengths'][-1],
        'history': history
    }


def compare_gate_modes(
    epochs: int = 500,
    lr: float = 0.01,
    seed: int = 42,
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Compare different gating modes to see how they affect race dynamics.
    """
    gate_modes = ['diagonal', 'full', 'k_neighbors']
    results = {}

    if verbose:
        print("=" * 60)
        print("Comparing Gate Modes")
        print("=" * 60)

    for mode in gate_modes:
        if verbose:
            print(f"\n--- Gate mode: {mode} ---")

        k_neighbors = 2 if mode == 'k_neighbors' else 1
        result = test_basic_gated_dln(
            gate_mode=mode,
            epochs=epochs,
            lr=lr,
            seed=seed,
            verbose=False
        )
        results[mode] = result

        if verbose:
            print(f"  Dominance: {result['final_dominance']:.4f}")
            print(f"  Strengths: {[f'{s:.3f}' for s in result['pathway_strengths']]}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'Mode':<15} | {'Dominance':<10} | {'WTA?'}")
        print("-" * 40)
        for mode, result in results.items():
            dom = result['final_dominance']
            wta = "YES" if dom > 0.5 else "NO"
            print(f"{mode:<15} | {dom:<10.4f} | {wta}")

    return results


def plot_svd_dynamics(history, output_path=None, title="SVD Dynamics"):
    """
    Plot singular value evolution during training.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    epochs = history['epochs_logged']

    # Encoder SVs
    ax = axes[0]
    enc_svs = np.array(history['svd']['encoders'])  # (time, M, num_svs)
    for m in range(enc_svs.shape[1]):
        for sv_idx in range(min(3, enc_svs.shape[2])):
            ax.plot(epochs, enc_svs[:, m, sv_idx],
                   label=f'Enc {m}, SV {sv_idx}' if sv_idx == 0 else None,
                   alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Singular Value')
    ax.set_title('Encoder SVs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hidden SVs
    ax = axes[1]
    hid_svs = np.array(history['svd']['hidden'])  # (time, num_svs)
    for sv_idx in range(min(5, hid_svs.shape[1])):
        ax.plot(epochs, hid_svs[:, sv_idx], label=f'SV {sv_idx}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Singular Value')
    ax.set_title('Hidden Layer SVs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Decoder SVs
    ax = axes[2]
    dec_svs = np.array(history['svd']['decoders'])  # (time, M, num_svs)
    for m in range(dec_svs.shape[1]):
        for sv_idx in range(min(3, dec_svs.shape[2])):
            ax.plot(epochs, dec_svs[:, m, sv_idx],
                   label=f'Dec {m}, SV {sv_idx}' if sv_idx == 0 else None,
                   alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Singular Value')
    ax.set_title('Decoder SVs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()
    return fig


def plot_race_dynamics(history, output_path=None, title="Pathway Race Dynamics"):
    """
    Plot pathway strength evolution (race dynamics).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = history['epochs_logged']

    # Pathway strengths over time
    ax = axes[0]
    strengths = np.array(history['pathway_strengths'])
    M = strengths.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, M))

    for m in range(M):
        ax.plot(epochs, strengths[:, m], label=f'Pathway {m}',
               linewidth=2, color=colors[m])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pathway Strength')
    ax.set_title('Pathway Strengths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dominance over time
    ax = axes[1]
    ax.plot(epochs, history['dominance'], linewidth=2, color='blue')
    ax.axhline(y=1/M, color='gray', linestyle='--', label=f'Equal (1/{M})')
    ax.axhline(y=0.5, color='red', linestyle='--', label='WTA threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dominance Ratio')
    ax.set_title('Pathway Dominance')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()
    return fig


def run_full_experiment(
    epochs: int = 500,
    lr: float = 0.02,
    init_scale: float = 0.2,
    num_seeds: int = 5,
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Run full experiment comparing GatedDLN vs standard MLP.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    all_results = {
        'basic_gated_dln': [],
        'gated_multiview': []
    }

    if verbose:
        print("=" * 70)
        print("Full GatedDLN Experiment")
        print("=" * 70)

    for seed in range(num_seeds):
        if verbose:
            print(f"\n{'='*20} Seed {seed} {'='*20}")

        # Test basic GatedDLN (Saxe style)
        result1 = test_basic_gated_dln(
            epochs=epochs, lr=lr, init_scale=init_scale,
            seed=seed, verbose=verbose
        )
        all_results['basic_gated_dln'].append(result1)

        # Plot for first seed
        if seed == 0:
            plot_svd_dynamics(
                result1['history'],
                output_path / 'figures' / 'gated_dln_svd.png',
                "GatedDLN SVD Dynamics"
            )
            plot_race_dynamics(
                result1['history'],
                output_path / 'figures' / 'gated_dln_race.png',
                "GatedDLN Race Dynamics"
            )

        # Test on multi-view dataset
        result2 = test_gated_multiview(
            epochs=epochs, lr=lr, init_scale=init_scale,
            seed=seed, verbose=verbose
        )
        all_results['gated_multiview'].append(result2)

        if seed == 0:
            plot_svd_dynamics(
                result2['history'],
                output_path / 'figures' / 'gated_multiview_svd.png',
                "GatedMultiView SVD Dynamics"
            )
            plot_race_dynamics(
                result2['history'],
                output_path / 'figures' / 'gated_multiview_race.png',
                "GatedMultiView Race Dynamics"
            )

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for exp_name, results in all_results.items():
            doms = [r['final_dominance'] for r in results]
            print(f"\n{exp_name}:")
            print(f"  Dominance: {np.mean(doms):.4f} +/- {np.std(doms):.4f}")
            print(f"  WTA count: {sum(1 for d in doms if d > 0.5)}/{len(doms)}")

    # Save results
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': epochs, 'lr': lr, 'init_scale': init_scale,
            'num_seeds': num_seeds
        },
        'summary': {
            exp_name: {
                'dominance_mean': float(np.mean([r['final_dominance'] for r in results])),
                'dominance_std': float(np.std([r['final_dominance'] for r in results])),
                'wta_count': sum(1 for r in results if r['final_dominance'] > 0.5)
            }
            for exp_name, results in all_results.items()
        }
    }

    with open(output_path / 'gated_dln_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {output_path / 'gated_dln_results.json'}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='GatedDLN Race Dynamics Experiment')
    parser.add_argument('--test', type=str, default='basic',
                       choices=['basic', 'multiview', 'compare', 'full'],
                       help='Which test to run')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.2, help='Init scale')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--gate-mode', type=str, default='diagonal',
                       choices=['diagonal', 'full', 'k_neighbors'],
                       help='Gate mode')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--output', type=str, default='results', help='Output dir')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 100
        args.seeds = 2

    if args.test == 'basic':
        result = test_basic_gated_dln(
            epochs=args.epochs, lr=args.lr, init_scale=args.init_scale,
            gate_mode=args.gate_mode
        )
    elif args.test == 'multiview':
        result = test_gated_multiview(
            epochs=args.epochs, lr=args.lr, init_scale=args.init_scale,
            gate_mode=args.gate_mode
        )
    elif args.test == 'compare':
        result = compare_gate_modes(
            epochs=args.epochs, lr=args.lr
        )
    elif args.test == 'full':
        result = run_full_experiment(
            epochs=args.epochs, lr=args.lr, init_scale=args.init_scale,
            num_seeds=args.seeds, output_dir=args.output
        )


if __name__ == '__main__':
    main()
