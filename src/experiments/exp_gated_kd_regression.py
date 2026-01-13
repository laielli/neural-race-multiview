"""
Experiment: KD with GatedDLN on Structured Regression

Uses the original Saxe-style setup where race dynamics are confirmed:
- Structured input-output regression task
- Input-output correlations with different singular values
- Race dynamics emerge in SVD space

Tests if KD can preserve multiple modes (pathways) when teachers
have learned different correlation structures.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import GatedDLN
from src.train import train_gated_dln, train_gated_kd


def create_structured_regression_data(
    N: int = 1000,
    M: int = 4,
    d_input: int = 20,
    d_output: int = 10,
    singular_values: list = None,
    seed: int = 42
):
    """
    Create structured regression data with known input-output correlations.

    The input-output correlation matrix has specific singular values,
    which control which modes grow fastest during training.

    Args:
        N: Number of samples
        M: Number of pathways/modes
        d_input: Input dimension per pathway
        d_output: Output dimension
        singular_values: List of singular values (strength of each mode)
        seed: Random seed

    Returns:
        X: Input tensor (N, M * d_input)
        Y: Output tensor (N, d_output)
        true_svs: The true singular values
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if singular_values is None:
        # Default: decreasing singular values to create a race
        singular_values = [10.0, 5.0, 2.0, 1.0][:M]

    # Ensure we have M singular values
    while len(singular_values) < M:
        singular_values.append(0.5)

    # Create structured correlation matrix for each pathway
    # X_m -> Y with strength singular_values[m]

    X_list = []
    Y_accum = torch.zeros(N, d_output)

    for m in range(M):
        # Random orthogonal input projection
        U_in = torch.randn(d_input, d_input)
        U_in, _ = torch.linalg.qr(U_in)

        # Random orthogonal output projection
        U_out = torch.randn(d_output, d_output)
        U_out, _ = torch.linalg.qr(U_out)

        # Generate input for this pathway
        X_m = torch.randn(N, d_input) @ U_in.T

        # Contribution to output (scaled by singular value)
        # Only use first few dimensions to create rank-limited correlation
        rank = min(d_input, d_output, 5)
        W_m = U_out[:, :rank] @ torch.diag(torch.ones(rank) * singular_values[m]) @ U_in[:rank, :]
        Y_m = X_m @ W_m.T

        X_list.append(X_m)
        Y_accum = Y_accum + Y_m

    # Concatenate inputs
    X = torch.cat(X_list, dim=1)

    # Add some noise to outputs
    Y = Y_accum + 0.1 * torch.randn(N, d_output)

    return X, Y, singular_values


def experiment_gated_kd_regression(
    seed: int = 42,
    num_teachers: int = 5,
    M: int = 4,
    d_input: int = 20,
    d_output: int = 10,
    hidden: int = 32,
    epochs: int = 500,
    temperature: float = 4.0,
    lr: float = 0.02,
    log_interval: int = 10,
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Compare hard label vs KD training on structured regression with GatedDLN.

    Args:
        seed: Random seed
        num_teachers: Number of teachers for KD
        M: Number of pathways
        d_input: Input dimension per pathway
        d_output: Output dimension
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
        print("Experiment: KD with GatedDLN (Structured Regression)")
        print("=" * 60)
        print(f"Config: seed={seed}, M={M}, hidden={hidden}")
        print(f"Input dim: {d_input * M}, Output dim: {d_output}")
        print(f"Teachers: {num_teachers}, Temperature: {temperature}")
        print("=" * 60)

    # Create structured data with different correlation strengths
    singular_values = [10.0, 5.0, 2.0, 1.0][:M]
    X, Y, true_svs = create_structured_regression_data(
        N=2000, M=M, d_input=d_input, d_output=d_output,
        singular_values=singular_values, seed=seed
    )

    if verbose:
        print(f"\nTarget singular values: {true_svs}")
        print("These control which modes should grow fastest")

    # Train teacher ensemble with different seeds
    if verbose:
        print("\n--- Training Teacher Ensemble ---")

    teachers = []
    teacher_results = []

    for i in range(num_teachers):
        if verbose:
            print(f"\nTeacher {i+1}/{num_teachers}")

        torch.manual_seed(seed * 100 + i)
        teacher = GatedDLN(
            M=M, d_input=d_input, hidden=hidden, d_output=d_output,
            gate_mode='diagonal', init_scale=0.2
        )
        teacher.init_orthogonal()

        history = train_gated_dln(
            teacher, X, Y,
            epochs=epochs,
            lr=lr,
            log_interval=log_interval,
            track_svd=True,
            verbose=verbose
        )

        teachers.append(teacher)
        teacher_results.append({
            'final_loss': history['loss'][-1],
            'final_dominance': history['dominance'][-1],
            'final_pathway_strengths': history['pathway_strengths'][-1]
        })

        if verbose:
            print(f"  Final loss: {history['loss'][-1]:.4f}")
            print(f"  Final dominance: {history['dominance'][-1]:.3f}")
            print(f"  Pathway strengths: {[f'{s:.2f}' for s in history['pathway_strengths'][-1]]}")

    # Train student with HARD LABELS
    if verbose:
        print("\n--- Training Student (Hard Labels = MSE to Y) ---")

    torch.manual_seed(seed + 5000)
    student_hard = GatedDLN(
        M=M, d_input=d_input, hidden=hidden, d_output=d_output,
        gate_mode='diagonal', init_scale=0.2
    )
    student_hard.init_orthogonal()

    history_hard = train_gated_dln(
        student_hard, X, Y,
        epochs=epochs,
        lr=lr,
        log_interval=log_interval,
        track_svd=True,
        verbose=verbose
    )

    # Train student with KD (same init!)
    if verbose:
        print("\n--- Training Student (KD from Teachers) ---")

    torch.manual_seed(seed + 5000)  # Same init!
    student_kd = GatedDLN(
        M=M, d_input=d_input, hidden=hidden, d_output=d_output,
        gate_mode='diagonal', init_scale=0.2
    )
    student_kd.init_orthogonal()

    history_kd = train_gated_kd(
        student_kd, teachers, X, Y,
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
        print(f"  Hard labels: {[f'{s:.2f}' for s in history_hard['pathway_strengths'][-1]]}")
        print(f"  KD:          {[f'{s:.2f}' for s in history_kd['pathway_strengths'][-1]]}")

        print(f"\nFinal Loss:")
        print(f"  Hard labels: {history_hard['loss'][-1]:.4f}")
        print(f"  KD:          {history_kd['loss'][-1]:.4f}")

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
                linewidth=2, color=colors[m], label=f'P{m} (SV={true_svs[m]:.1f})')
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
                linewidth=2, color=colors[m], label=f'P{m} (SV={true_svs[m]:.1f})')
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
    ax.set_xticklabels([f'P{m}\n(SV={true_svs[m]:.1f})' for m in range(M)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = output_path / 'gated_kd_regression.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nSaved figure: {fig_path}")

    # Analyze pathway strength distribution
    hard_strengths = np.array(history_hard['pathway_strengths'][-1])
    kd_strengths = np.array(history_kd['pathway_strengths'][-1])

    # Compute variance (higher = more unequal = more WTA)
    hard_var = np.var(hard_strengths / hard_strengths.sum())
    kd_var = np.var(kd_strengths / kd_strengths.sum())

    # Determine if KD preserved more pathways
    hard_dom = history_hard['dominance'][-1]
    kd_dom = history_kd['dominance'][-1]
    kd_preserves = kd_dom < hard_dom

    if verbose:
        print("\n" + "=" * 60)
        print("THEOREM 3 TEST")
        print("=" * 60)
        print(f"Pathway strength variance:")
        print(f"  Hard labels: {hard_var:.4f}")
        print(f"  KD:          {kd_var:.4f}")
        print()
        if kd_preserves:
            print(f"PASSED: KD has lower dominance ({kd_dom:.3f}) than hard labels ({hard_dom:.3f})")
            print("KD preserves pathway diversity as predicted!")
        else:
            print(f"FAILED: KD dominance ({kd_dom:.3f}) >= hard labels ({hard_dom:.3f})")
            print("KD did NOT preserve pathway diversity better than hard labels")

    # Compile results
    results = {
        'experiment': 'exp_gated_kd_regression',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': seed,
            'num_teachers': num_teachers,
            'M': M,
            'd_input': d_input,
            'd_output': d_output,
            'hidden': hidden,
            'epochs': epochs,
            'temperature': temperature,
            'lr': lr,
            'target_singular_values': true_svs
        },
        'teachers': {
            'results': teacher_results,
            'avg_dominance': float(np.mean([r['final_dominance'] for r in teacher_results]))
        },
        'hard_labels': {
            'final_dominance': float(hard_dom),
            'final_loss': float(history_hard['loss'][-1]),
            'final_pathway_strengths': [float(s) for s in hard_strengths],
            'strength_variance': float(hard_var)
        },
        'kd': {
            'final_dominance': float(kd_dom),
            'final_loss': float(history_kd['loss'][-1]),
            'final_pathway_strengths': [float(s) for s in kd_strengths],
            'strength_variance': float(kd_var)
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


def main():
    parser = argparse.ArgumentParser(description='KD with GatedDLN (Structured Regression)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--teachers', type=int, default=5, help='Number of teachers')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD temperature')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=32, help='Hidden layer width')
    parser.add_argument('--M', type=int, default=4, help='Number of pathways')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 200
        args.teachers = 3

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, _, _ = experiment_gated_kd_regression(
        seed=args.seed,
        num_teachers=args.teachers,
        M=args.M,
        hidden=args.hidden,
        epochs=args.epochs,
        temperature=args.temperature,
        lr=args.lr,
        output_dir=str(output_dir)
    )

    # Save results
    results_file = output_dir / 'results_gated_kd_regression.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
