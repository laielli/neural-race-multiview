"""
Experiment: KD with GatedDLN Architecture

Tests Theorem 3: Does KD from teacher ensemble preserve multiple pathways
in GatedDLN architecture where race dynamics occur?

Expected:
- Hard labels: Winner-take-all (one pathway dominates)
- KD: Multiple pathways preserved (gradients distributed)

This is the critical test - GatedDLN has race dynamics (confirmed),
now we test if KD can break winner-take-all as theory predicts.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import MultiViewDataset
from src.model import GatedMultiViewNet
from src.train import train_gated_multiview, train_gated_multiview_kd


def experiment_gated_kd(
    seed: int = 42,
    num_teachers: int = 5,
    K: int = 10,
    M: int = 3,
    d_view: int = 50,
    hidden: int = 64,
    epochs: int = 500,
    temperature: float = 4.0,
    lr: float = 0.02,
    log_interval: int = 10,
    gate_mode: str = 'full',  # 'full' creates competition, 'diagonal' isolates pathways
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Compare hard label vs KD training with GatedDLN architecture.

    Args:
        seed: Random seed
        num_teachers: Number of teachers for KD
        K: Number of classes
        M: Views per class (= number of pathways)
        d_view: Dimensions per view
        hidden: Hidden layer width
        epochs: Training epochs
        temperature: KD temperature
        lr: Learning rate
        log_interval: Logging frequency
        output_dir: Output directory
        verbose: Print progress

    Returns:
        dict: Experiment results
    """
    if verbose:
        print("=" * 60)
        print("Experiment: KD with GatedDLN Architecture")
        print("=" * 60)
        print(f"Config: seed={seed}, K={K}, M={M}, hidden={hidden}")
        print(f"Teachers: {num_teachers}, Temperature: {temperature}")
        print(f"Gate mode: {gate_mode}")
        print("=" * 60)

    # Create dataset
    torch.manual_seed(seed)
    dataset = MultiViewDataset(
        K=K, M=M, d_view=d_view,
        n_samples=10000, seed=seed
    )

    # Train teacher ensemble with different seeds
    if verbose:
        print("\n--- Training Teacher Ensemble ---")

    teachers = []
    teacher_dominances = []

    for i in range(num_teachers):
        if verbose:
            print(f"\nTeacher {i+1}/{num_teachers}")

        torch.manual_seed(seed * 100 + i)
        teacher = GatedMultiViewNet(
            M=M, d_view=d_view, K=K, hidden=hidden,
            gate_mode=gate_mode, init_scale=0.2
        )
        teacher.init_orthogonal()

        history = train_gated_multiview(
            teacher, dataset,
            epochs=epochs,
            lr=lr,
            log_interval=log_interval,
            track_svd=False,
            verbose=verbose
        )

        teachers.append(teacher)
        teacher_dominances.append(history['dominance'][-1])

        if verbose:
            print(f"  Final dominance: {history['dominance'][-1]:.3f}")
            print(f"  Final accuracy: {history['accuracy'][-1]:.3f}")

    # Average teacher dominance
    avg_teacher_dom = np.mean(teacher_dominances)
    if verbose:
        print(f"\nTeacher ensemble avg dominance: {avg_teacher_dom:.3f}")

    # Train student with HARD LABELS (same seed for fair comparison)
    if verbose:
        print("\n--- Training Student (Hard Labels) ---")

    torch.manual_seed(seed + 5000)
    student_hard = GatedMultiViewNet(
        M=M, d_view=d_view, K=K, hidden=hidden,
        gate_mode=gate_mode, init_scale=0.2
    )
    student_hard.init_orthogonal()

    history_hard = train_gated_multiview(
        student_hard, dataset,
        epochs=epochs,
        lr=lr,
        log_interval=log_interval,
        track_svd=True,
        verbose=verbose
    )

    # Train student with KD (same init!)
    if verbose:
        print("\n--- Training Student (KD) ---")

    torch.manual_seed(seed + 5000)  # Same init!
    student_kd = GatedMultiViewNet(
        M=M, d_view=d_view, K=K, hidden=hidden,
        gate_mode=gate_mode, init_scale=0.2
    )
    student_kd.init_orthogonal()

    history_kd = train_gated_multiview_kd(
        student_kd, teachers, dataset,
        epochs=epochs,
        temperature=temperature,
        lr=lr,
        log_interval=log_interval,
        track_svd=True,
        verbose=verbose
    )

    # Compare results
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)

        print(f"\nFinal Dominance (1/M = {1/M:.3f} is equal, 1.0 is WTA):")
        print(f"  Hard labels: {history_hard['dominance'][-1]:.3f}")
        print(f"  KD:          {history_kd['dominance'][-1]:.3f}")

        print(f"\nFinal Pathway Strengths:")
        print(f"  Hard labels: {[f'{s:.3f}' for s in history_hard['pathway_strengths'][-1]]}")
        print(f"  KD:          {[f'{s:.3f}' for s in history_kd['pathway_strengths'][-1]]}")

        print(f"\nFinal Accuracy:")
        print(f"  Hard labels: {history_hard['accuracy'][-1]:.3f}")
        print(f"  KD:          {history_kd['accuracy'][-1]:.3f}")

        print(f"\nView Coverage:")
        print(f"  Hard labels: {history_hard['view_coverage'][-1]:.3f}")
        print(f"  KD:          {history_kd['view_coverage'][-1]:.3f}")

    # Create visualization
    output_path = Path(output_dir) / 'figures'
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Dominance over training
    ax = axes[0, 0]
    ax.plot(history_hard['epochs_logged'], history_hard['dominance'],
            'b-', linewidth=2, label='Hard Labels')
    ax.plot(history_kd['epochs_logged'], history_kd['dominance'],
            'r-', linewidth=2, label='KD')
    ax.axhline(y=1/M, color='gray', linestyle='--', label=f'Equal (1/M={1/M:.2f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dominance')
    ax.set_title('Pathway Dominance During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Pathway strengths - Hard Labels
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, M))
    for m in range(M):
        strengths = [ps[m] for ps in history_hard['pathway_strengths']]
        ax.plot(history_hard['epochs_logged'], strengths,
                linewidth=2, color=colors[m], label=f'Pathway {m}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pathway Strength')
    ax.set_title('Hard Labels: Pathway Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Pathway strengths - KD
    ax = axes[1, 0]
    for m in range(M):
        strengths = [ps[m] for ps in history_kd['pathway_strengths']]
        ax.plot(history_kd['epochs_logged'], strengths,
                linewidth=2, color=colors[m], label=f'Pathway {m}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pathway Strength')
    ax.set_title('KD: Pathway Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Final comparison bar chart
    ax = axes[1, 1]
    x = np.arange(M)
    width = 0.35
    hard_strengths = history_hard['pathway_strengths'][-1]
    kd_strengths = history_kd['pathway_strengths'][-1]
    ax.bar(x - width/2, hard_strengths, width, label='Hard Labels', color='blue', alpha=0.7)
    ax.bar(x + width/2, kd_strengths, width, label='KD', color='red', alpha=0.7)
    ax.set_xlabel('Pathway')
    ax.set_ylabel('Final Strength')
    ax.set_title('Final Pathway Strengths Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{m}' for m in range(M)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = output_path / 'gated_kd_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nSaved figure: {fig_path}")

    # Determine if KD preserved pathways better
    hard_dom = history_hard['dominance'][-1]
    kd_dom = history_kd['dominance'][-1]
    kd_preserves = kd_dom < hard_dom

    if verbose:
        print("\n" + "=" * 60)
        print("THEOREM 3 TEST")
        print("=" * 60)
        if kd_preserves:
            print(f"PASSED: KD has lower dominance ({kd_dom:.3f}) than hard labels ({hard_dom:.3f})")
            print("KD preserves pathway diversity as predicted!")
        else:
            print(f"FAILED: KD dominance ({kd_dom:.3f}) >= hard labels ({hard_dom:.3f})")
            print("KD did NOT preserve pathway diversity better than hard labels")

    # Compile results
    results = {
        'experiment': 'exp_gated_kd',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': seed,
            'num_teachers': num_teachers,
            'K': K,
            'M': M,
            'd_view': d_view,
            'hidden': hidden,
            'epochs': epochs,
            'temperature': temperature,
            'lr': lr,
            'gate_mode': gate_mode
        },
        'teachers': {
            'dominances': [float(d) for d in teacher_dominances],
            'avg_dominance': float(avg_teacher_dom)
        },
        'hard_labels': {
            'final_dominance': float(hard_dom),
            'final_accuracy': float(history_hard['accuracy'][-1]),
            'final_coverage': float(history_hard['view_coverage'][-1]),
            'final_pathway_strengths': [float(s) for s in hard_strengths]
        },
        'kd': {
            'final_dominance': float(kd_dom),
            'final_accuracy': float(history_kd['accuracy'][-1]),
            'final_coverage': float(history_kd['view_coverage'][-1]),
            'final_pathway_strengths': [float(s) for s in kd_strengths]
        },
        'theorem_3_test': {
            'passed': bool(kd_preserves),
            'hard_dominance': float(hard_dom),
            'kd_dominance': float(kd_dom),
            'dominance_reduction': float(hard_dom - kd_dom)
        },
        'figure_path': str(fig_path)
    }

    return results, history_hard, history_kd


def run_temperature_sweep(
    seed: int = 42,
    temperatures: list = None,
    epochs: int = 500,
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Sweep over KD temperatures to find optimal pathway preservation.

    Higher temperature = softer targets = more gradient distribution
    """
    if temperatures is None:
        temperatures = [1.0, 2.0, 4.0, 8.0, 16.0]

    if verbose:
        print("=" * 60)
        print("Temperature Sweep for KD + GatedDLN")
        print("=" * 60)

    results = []

    for temp in temperatures:
        if verbose:
            print(f"\n--- Temperature = {temp} ---")

        result, _, _ = experiment_gated_kd(
            seed=seed,
            temperature=temp,
            epochs=epochs,
            output_dir=output_dir,
            verbose=False
        )

        results.append({
            'temperature': temp,
            'hard_dominance': result['hard_labels']['final_dominance'],
            'kd_dominance': result['kd']['final_dominance'],
            'kd_preserves': result['theorem_3_test']['passed']
        })

        if verbose:
            print(f"  Hard dom: {result['hard_labels']['final_dominance']:.3f}")
            print(f"  KD dom:   {result['kd']['final_dominance']:.3f}")
            print(f"  Preserves: {result['theorem_3_test']['passed']}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Temperature Sweep Summary")
        print("=" * 60)
        print(f"{'Temp':<8} {'Hard Dom':<12} {'KD Dom':<12} {'Preserves':<10}")
        print("-" * 42)
        for r in results:
            print(f"{r['temperature']:<8.1f} {r['hard_dominance']:<12.3f} {r['kd_dominance']:<12.3f} {str(r['kd_preserves']):<10}")

    return results


def main():
    parser = argparse.ArgumentParser(description='KD with GatedDLN Architecture')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--teachers', type=int, default=5, help='Number of teachers')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD temperature')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer width')
    parser.add_argument('--M', type=int, default=3, help='Number of views/pathways')
    parser.add_argument('--K', type=int, default=10, help='Number of classes')
    parser.add_argument('--gate', type=str, default='full', choices=['full', 'diagonal', 'k_neighbors'],
                        help='Gate mode (full=competition, diagonal=isolated)')
    parser.add_argument('--sweep', action='store_true', help='Run temperature sweep')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 100
        args.teachers = 3

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        results = run_temperature_sweep(
            seed=args.seed,
            epochs=args.epochs,
            output_dir=str(output_dir)
        )
        # Save sweep results
        sweep_file = output_dir / 'results_gated_kd_sweep.json'
        with open(sweep_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSweep results saved to: {sweep_file}")
    else:
        results, _, _ = experiment_gated_kd(
            seed=args.seed,
            num_teachers=args.teachers,
            K=args.K,
            M=args.M,
            hidden=args.hidden,
            epochs=args.epochs,
            temperature=args.temperature,
            lr=args.lr,
            gate_mode=args.gate,
            output_dir=str(output_dir)
        )

        # Save results
        results_file = output_dir / 'results_gated_kd.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
