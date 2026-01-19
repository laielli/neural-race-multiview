"""
Experiment: Knowledge Distillation Breaks WTA in Multi-Task

Tests whether KD from an ensemble of single-task teachers
enables a student to learn multiple tasks (breaking WTA).

Key hypothesis: Soft labels from teacher ensemble should:
1. Provide external signal for all tasks (not just the one initialization favors)
2. Prevent winner-take-all dynamics
3. Transfer multi-task coverage from ensemble to single student

Success criteria:
- Student coverage > 0.5 × ensemble coverage
- Student dominance < 0.5 (WTA broken)
- SVD effective rank higher for KD student than hard-label student
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

from data_multitask import MultiTaskMultiViewDataset, compute_coverage
from model import GatedDLN
from train import (
    train_multitask_hard,
    train_multitask_kd,
    train_multitask_teachers,
    compute_ensemble_coverage
)


def run_experiment(
    M: int = 5,
    K: int = 10,
    d_view: int = 50,
    hidden: int = 64,
    n_samples: int = 5000,
    epochs: int = 500,
    lr: float = 0.02,
    init_scale: float = 0.2,
    n_teachers: int = 5,
    temperature: float = 3.0,
    alpha: float = 0.9,
    student_seed: int = 99,
    output_dir: str = 'results',
    verbose: bool = True
):
    """
    Run full KD experiment: train teachers, distill to student, compare.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    print("="*70)
    print("Multi-Task KD Experiment")
    print("="*70)
    print(f"M={M} tasks, K={K} classes")
    print(f"n_teachers={n_teachers}, temperature={temperature}, alpha={alpha}")
    print(f"Expected: student coverage ≈ ensemble coverage, dominance < 0.5")
    print("="*70)

    # Create dataset (same for all)
    torch.manual_seed(42)
    dataset = MultiTaskMultiViewDataset(
        M=M, K=K, d_view=d_view,
        n_samples=n_samples, seed=42
    )

    # =========================================================================
    # Phase 1: Train teacher ensemble
    # =========================================================================
    print("\n" + "="*50)
    print("Phase 1: Training Teacher Ensemble")
    print("="*50)

    teachers, teacher_histories = train_multitask_teachers(
        dataset,
        num_teachers=n_teachers,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        init_scale=init_scale,
        verbose=verbose
    )

    # Compute ensemble coverage
    ensemble_coverage = compute_ensemble_coverage(teachers, dataset, threshold=0.5)
    expected_ensemble_coverage = 1 - (1 - 1/M) ** n_teachers

    print(f"\n{'='*50}")
    print("Teacher Ensemble Summary")
    print(f"{'='*50}")
    print(f"Ensemble coverage: {ensemble_coverage:.3f}")
    print(f"Expected (theoretical): {expected_ensemble_coverage:.3f}")

    teacher_dominances = [h['dominance'][-1] for h in teacher_histories]
    print(f"Teacher dominances: {[f'{d:.3f}' for d in teacher_dominances]}")

    # Which tasks did each teacher learn best?
    teacher_best_tasks = []
    for h in teacher_histories:
        final_accs = h['task_accuracies'][-1]
        best_task = max(final_accs, key=final_accs.get)
        teacher_best_tasks.append(best_task)
    print(f"Teacher best tasks: {teacher_best_tasks}")
    print(f"Unique tasks covered: {len(set(teacher_best_tasks))}")

    # =========================================================================
    # Phase 2: Train KD student
    # =========================================================================
    print("\n" + "="*50)
    print("Phase 2: Training KD Student")
    print("="*50)

    torch.manual_seed(student_seed)
    kd_student = GatedDLN(
        M=M, d_input=d_view, hidden=hidden, d_output=K,
        gate_mode='diagonal', init_scale=init_scale
    )
    kd_student.init_orthogonal(init_scale)

    kd_history = train_multitask_kd(
        kd_student, teachers, dataset,
        epochs=epochs, lr=lr,
        temperature=temperature, alpha=alpha,
        verbose=verbose
    )

    # =========================================================================
    # Phase 3: Train hard-label baseline (same seed as KD student)
    # =========================================================================
    print("\n" + "="*50)
    print("Phase 3: Training Hard-Label Baseline")
    print("="*50)

    torch.manual_seed(student_seed)
    hard_student = GatedDLN(
        M=M, d_input=d_view, hidden=hidden, d_output=K,
        gate_mode='diagonal', init_scale=init_scale
    )
    hard_student.init_orthogonal(init_scale)

    hard_history = train_multitask_hard(
        hard_student, dataset,
        epochs=epochs, lr=lr,
        verbose=verbose
    )

    # =========================================================================
    # Phase 4: Compare results
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    # Final metrics
    kd_dom = kd_history['dominance'][-1]
    hard_dom = hard_history['dominance'][-1]

    kd_accs = kd_history['task_accuracies'][-1]
    hard_accs = hard_history['task_accuracies'][-1]

    kd_mean_acc = sum(kd_accs.values()) / len(kd_accs)
    hard_mean_acc = sum(hard_accs.values()) / len(hard_accs)

    kd_learned = sum(1 for acc in kd_accs.values() if acc > 0.5)
    hard_learned = sum(1 for acc in hard_accs.values() if acc > 0.5)

    kd_coverage = kd_learned / M
    hard_coverage = hard_learned / M

    print(f"\n{'Metric':<25} | {'Hard Labels':<15} | {'KD':<15}")
    print("-" * 60)
    print(f"{'Dominance':<25} | {hard_dom:<15.4f} | {kd_dom:<15.4f}")
    print(f"{'Mean Accuracy':<25} | {hard_mean_acc:<15.4f} | {kd_mean_acc:<15.4f}")
    print(f"{'Tasks Learned (>0.5)':<25} | {hard_learned:<15} | {kd_learned:<15}")
    print(f"{'Coverage':<25} | {hard_coverage:<15.3f} | {kd_coverage:<15.3f}")

    print(f"\nEnsemble coverage: {ensemble_coverage:.3f}")
    print(f"Coverage transfer (KD/Ensemble): {kd_coverage/max(ensemble_coverage, 0.01):.2f}")

    # SVD metrics comparison
    kd_svd = kd_history['svd_metrics'][-1]
    hard_svd = hard_history['svd_metrics'][-1]

    kd_eff_ranks = [kd_svd[f'enc_{m}_eff_rank'] for m in range(M)]
    hard_eff_ranks = [hard_svd[f'enc_{m}_eff_rank'] for m in range(M)]

    print(f"\nSVD Effective Ranks (mean across encoders):")
    print(f"  Hard labels: {np.mean(hard_eff_ranks):.3f}")
    print(f"  KD: {np.mean(kd_eff_ranks):.3f}")

    # =========================================================================
    # Success criteria
    # =========================================================================
    print("\n" + "-"*50)
    print("Success Criteria:")
    print("-"*50)

    success = True

    # Criterion 1: KD coverage > 0.5 × ensemble coverage
    coverage_transfer = kd_coverage / max(ensemble_coverage, 0.01)
    if coverage_transfer > 0.5:
        print(f"✓ Coverage transfer > 50%: {100*coverage_transfer:.0f}%")
    else:
        print(f"✗ Coverage transfer <= 50%: {100*coverage_transfer:.0f}%")
        success = False

    # Criterion 2: KD dominance < 0.5 (WTA broken)
    if kd_dom < 0.5:
        print(f"✓ KD dominance < 0.5: {kd_dom:.4f}")
    else:
        print(f"✗ KD dominance >= 0.5: {kd_dom:.4f}")
        success = False

    # Criterion 3: KD effective rank > hard effective rank
    if np.mean(kd_eff_ranks) > np.mean(hard_eff_ranks):
        print(f"✓ KD effective rank > hard: {np.mean(kd_eff_ranks):.3f} > {np.mean(hard_eff_ranks):.3f}")
    else:
        print(f"✗ KD effective rank <= hard: {np.mean(kd_eff_ranks):.3f} <= {np.mean(hard_eff_ranks):.3f}")
        success = False

    print("\n" + "="*50)
    if success:
        print("EXPERIMENT PASSED - KD breaks WTA and transfers coverage!")
    else:
        print("EXPERIMENT FAILED - KD did not break WTA as expected")
    print("="*50)

    # =========================================================================
    # Plot results
    # =========================================================================
    plot_comparison(
        hard_history, kd_history, teacher_histories,
        output_path / 'figures' / 'multitask_kd_comparison.png',
        M
    )

    plot_task_accuracies(
        hard_accs, kd_accs, teacher_histories,
        output_path / 'figures' / 'multitask_kd_task_accs.png',
        M
    )

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'd_view': d_view, 'hidden': hidden,
            'n_samples': n_samples, 'epochs': epochs, 'lr': lr,
            'init_scale': init_scale, 'n_teachers': n_teachers,
            'temperature': temperature, 'alpha': alpha, 'student_seed': student_seed
        },
        'ensemble': {
            'coverage': ensemble_coverage,
            'expected_coverage': expected_ensemble_coverage,
            'teacher_dominances': teacher_dominances,
            'teacher_best_tasks': teacher_best_tasks
        },
        'hard_labels': {
            'dominance': hard_dom,
            'mean_accuracy': hard_mean_acc,
            'coverage': hard_coverage,
            'task_accuracies': hard_accs,
            'svd_eff_ranks': hard_eff_ranks
        },
        'kd': {
            'dominance': kd_dom,
            'mean_accuracy': kd_mean_acc,
            'coverage': kd_coverage,
            'task_accuracies': kd_accs,
            'svd_eff_ranks': kd_eff_ranks,
            'coverage_transfer': coverage_transfer
        },
        'success': success
    }

    with open(output_path / 'multitask_kd_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'multitask_kd_results.json'}")

    return save_results, success


def plot_comparison(hard_history, kd_history, teacher_histories, output_path, M):
    """Plot dominance and accuracy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dominance over training
    ax = axes[0]
    epochs = hard_history['epochs_logged']
    ax.plot(epochs, hard_history['dominance'], 'b-', linewidth=2, label='Hard Labels')
    ax.plot(epochs, kd_history['dominance'], 'g-', linewidth=2, label='KD')
    ax.axhline(y=0.5, color='red', linestyle='--', label='WTA threshold')
    ax.axhline(y=1/M, color='gray', linestyle='--', label=f'Equal (1/{M})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dominance')
    ax.set_title('Pathway Dominance During Training')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean accuracy over training
    ax = axes[1]
    hard_accs = [sum(a.values())/len(a) for a in hard_history['task_accuracies']]
    kd_accs = [sum(a.values())/len(a) for a in kd_history['task_accuracies']]
    ax.plot(epochs, hard_accs, 'b-', linewidth=2, label='Hard Labels')
    ax.plot(epochs, kd_accs, 'g-', linewidth=2, label='KD')

    # Teacher ensemble mean (horizontal line)
    teacher_final_accs = []
    for th in teacher_histories:
        final_acc = sum(th['task_accuracies'][-1].values()) / M
        teacher_final_accs.append(final_acc)
    ax.axhline(y=np.mean(teacher_final_accs), color='orange', linestyle='--',
               label=f'Teacher Mean')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Task Accuracy')
    ax.set_title('Mean Accuracy During Training')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_task_accuracies(hard_accs, kd_accs, teacher_histories, output_path, M):
    """Plot per-task accuracy comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(M)
    width = 0.25

    # Teacher ensemble (max across teachers for each task)
    teacher_max_accs = []
    for m in range(M):
        task_accs = [th['task_accuracies'][-1][m] for th in teacher_histories]
        teacher_max_accs.append(max(task_accs))

    ax.bar(x - width, [hard_accs[m] for m in range(M)], width, label='Hard Labels', color='steelblue')
    ax.bar(x, [kd_accs[m] for m in range(M)], width, label='KD Student', color='green')
    ax.bar(x + width, teacher_max_accs, width, label='Teacher (best)', color='orange', alpha=0.7)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Learned threshold')

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Task Accuracy Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Task KD Experiment')
    parser.add_argument('--M', type=int, default=5, help='Number of tasks/views')
    parser.add_argument('--K', type=int, default=10, help='Classes per task')
    parser.add_argument('--d-view', type=int, default=50, help='Dimension per view')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden layer width')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--init-scale', type=float, default=0.2, help='Init scale')
    parser.add_argument('--n-teachers', type=int, default=5, help='Number of teachers')
    parser.add_argument('--temperature', type=float, default=3.0, help='KD temperature')
    parser.add_argument('--alpha', type=float, default=0.9, help='Soft label weight')
    parser.add_argument('--student-seed', type=int, default=99, help='Student seed')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 100
        args.n_teachers = 3
        print("Quick mode: epochs=100, n_teachers=3")

    run_experiment(
        M=args.M,
        K=args.K,
        d_view=args.d_view,
        hidden=args.hidden,
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        init_scale=args.init_scale,
        n_teachers=args.n_teachers,
        temperature=args.temperature,
        alpha=args.alpha,
        student_seed=args.student_seed,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
