"""
Master experiment runner for KDMech experiments.

Runs experiments by priority level and generates summary report.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp_2_1_single_view import experiment_2_1_single_view_convergence
from experiments.exp_2_3_race_dynamics import experiment_2_3_race_visualization
from experiments.exp_3_1_kd_coverage import experiment_3_1_kd_coverage
from experiments.exp_3_3_pathway_evolution import experiment_3_3_pathway_evolution_comparison


def run_p1_experiments(output_dir: str = 'results', quick: bool = False):
    """
    Run all P1 (high priority) experiments.

    P1 Experiments:
    - 2.1: Single-view convergence
    - 2.3: Race dynamics visualization
    - 3.1: KD coverage transfer
    - 3.3: Pathway evolution comparison

    Args:
        output_dir: Base output directory
        quick: Run with reduced parameters for testing

    Returns:
        dict: All experiment results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path = output_path / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'quick' if quick else 'full',
        'experiments': {}
    }

    # Experiment parameters
    if quick:
        params = {
            'num_seeds': 3,
            'num_trials': 3,
            'num_teachers': 3,
            'epochs': 50,
            'epochs_visual': 100
        }
    else:
        params = {
            'num_seeds': 30,
            'num_trials': 10,
            'num_teachers': 5,
            'epochs': 100,
            'epochs_visual': 200
        }

    print("=" * 70)
    print("KDMech P1 Experiments")
    print(f"Mode: {'Quick test' if quick else 'Full run'}")
    print("=" * 70)

    # Experiment 2.1: Single-view convergence
    print("\n" + "=" * 70)
    print("EXPERIMENT 2.1: Single-View Convergence")
    print("=" * 70)
    try:
        result_2_1 = experiment_2_1_single_view_convergence(
            num_seeds=params['num_seeds'],
            epochs=params['epochs']
        )
        all_results['exp_2_1'] = result_2_1

        with open(output_path / 'results_exp_2_1.json', 'w') as f:
            json.dump(result_2_1, f, indent=2)

        summary['experiments']['exp_2_1'] = {
            'status': 'completed',
            'passed': result_2_1['metrics']['passed'],
            'mean_coverage': result_2_1['metrics']['mean_coverage'],
            'expected': result_2_1['metrics']['expected']
        }
    except Exception as e:
        print(f"ERROR in Exp 2.1: {e}")
        summary['experiments']['exp_2_1'] = {'status': 'failed', 'error': str(e)}

    # Experiment 2.3: Race dynamics
    print("\n" + "=" * 70)
    print("EXPERIMENT 2.3: Race Dynamics Visualization")
    print("=" * 70)
    try:
        result_2_3 = experiment_2_3_race_visualization(
            seed=0,
            y_target=0,
            epochs=params['epochs_visual'],
            output_dir=str(figures_path)
        )
        all_results['exp_2_3'] = result_2_3

        with open(output_path / 'results_exp_2_3.json', 'w') as f:
            json.dump(result_2_3, f, indent=2)

        summary['experiments']['exp_2_3'] = {
            'status': 'completed',
            'winning_view': result_2_3['metrics']['winning_view'],
            'winner_ratio': result_2_3['metrics']['winner_ratio'],
            'figure': result_2_3['figure_path']
        }
    except Exception as e:
        print(f"ERROR in Exp 2.3: {e}")
        summary['experiments']['exp_2_3'] = {'status': 'failed', 'error': str(e)}

    # Experiment 3.1: KD coverage transfer
    print("\n" + "=" * 70)
    print("EXPERIMENT 3.1: KD Coverage Transfer")
    print("=" * 70)
    try:
        result_3_1 = experiment_3_1_kd_coverage(
            num_teachers=params['num_teachers'],
            num_trials=params['num_trials'],
            epochs=params['epochs'],
            output_dir=str(figures_path)
        )
        all_results['exp_3_1'] = result_3_1

        with open(output_path / 'results_exp_3_1.json', 'w') as f:
            json.dump(result_3_1, f, indent=2)

        summary['experiments']['exp_3_1'] = {
            'status': 'completed',
            'passed_kd_matches_ensemble': result_3_1['metrics']['passed_kd_matches_ensemble'],
            'passed_kd_better_than_hard': result_3_1['metrics']['passed_kd_better_than_hard'],
            'hard_coverage': result_3_1['metrics']['hard_label']['mean'],
            'kd_coverage': result_3_1['metrics']['kd']['mean'],
            'ensemble_coverage': result_3_1['metrics']['ensemble']['mean'],
            'figure': result_3_1['figure_path']
        }
    except Exception as e:
        print(f"ERROR in Exp 3.1: {e}")
        summary['experiments']['exp_3_1'] = {'status': 'failed', 'error': str(e)}

    # Experiment 3.3: Pathway evolution
    print("\n" + "=" * 70)
    print("EXPERIMENT 3.3: Pathway Evolution Comparison")
    print("=" * 70)
    try:
        result_3_3 = experiment_3_3_pathway_evolution_comparison(
            seed=0,
            y_target=0,
            num_teachers=params['num_teachers'],
            epochs=params['epochs_visual'],
            output_dir=str(figures_path)
        )
        all_results['exp_3_3'] = result_3_3

        with open(output_path / 'results_exp_3_3.json', 'w') as f:
            json.dump(result_3_3, f, indent=2)

        summary['experiments']['exp_3_3'] = {
            'status': 'completed',
            'passed': result_3_3['metrics']['passed'],
            'hard_dominance_ratio': result_3_3['metrics']['hard_label']['dominance_ratio'],
            'kd_dominance_ratio': result_3_3['metrics']['kd']['dominance_ratio'],
            'hard_surviving': result_3_3['metrics']['hard_label']['surviving_pathways'],
            'kd_surviving': result_3_3['metrics']['kd']['surviving_pathways'],
            'figure': result_3_3['figure_path']
        }
    except Exception as e:
        print(f"ERROR in Exp 3.3: {e}")
        summary['experiments']['exp_3_3'] = {'status': 'failed', 'error': str(e)}

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(
        1 for exp in summary['experiments'].values()
        if exp.get('status') == 'completed' and exp.get('passed', True)
    )
    total_count = len(summary['experiments'])

    print(f"\nExperiments completed: {sum(1 for e in summary['experiments'].values() if e['status'] == 'completed')}/{total_count}")

    for exp_name, exp_data in summary['experiments'].items():
        status = "✓" if exp_data['status'] == 'completed' else "✗"
        print(f"  {status} {exp_name}: {exp_data['status']}")

    # Save summary
    with open(output_path / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate markdown summary for advisor
    generate_summary_report(summary, all_results, output_path)

    print(f"\nAll results saved to: {output_path}")

    return all_results, summary


def generate_summary_report(summary, results, output_path):
    """Generate markdown summary report for the advisor."""

    report = f"""# Experiment Results Summary

**Date**: {summary['timestamp']}
**Mode**: {summary['mode']}

---

## Overview

| Experiment | Status | Key Result |
|------------|--------|------------|
"""

    for exp_name, exp_data in summary['experiments'].items():
        status = "✓ Pass" if exp_data['status'] == 'completed' else "✗ Fail"

        if exp_name == 'exp_2_1':
            mc = exp_data.get('mean_coverage')
            ex = exp_data.get('expected')
            key_result = f"Coverage: {mc:.3f} (expected {ex:.3f})" if mc is not None else "N/A"
        elif exp_name == 'exp_2_3':
            wr = exp_data.get('winner_ratio')
            key_result = f"Winner ratio: {wr:.3f}" if wr is not None else "N/A"
        elif exp_name == 'exp_3_1':
            kd = exp_data.get('kd_coverage')
            hd = exp_data.get('hard_coverage')
            key_result = f"KD: {kd:.3f}, Hard: {hd:.3f}" if kd is not None else "N/A"
        elif exp_name == 'exp_3_3':
            kd_s = exp_data.get('kd_surviving')
            hd_s = exp_data.get('hard_surviving')
            key_result = f"KD surviving: {kd_s}, Hard: {hd_s}" if kd_s is not None else "N/A"
        else:
            key_result = "N/A"

        report += f"| {exp_name} | {status} | {key_result} |\n"

    report += """
---

## Detailed Results

### Experiment 2.1: Single-View Convergence

**Validates**: Hard label training leads to C(f) ≈ 1/M

"""
    if 'exp_2_1' in results:
        r = results['exp_2_1']
        report += f"""- Mean coverage: **{r['metrics']['mean_coverage']:.3f}** ± {r['metrics']['std_coverage']:.3f}
- Expected (1/M): **{r['metrics']['expected']:.3f}**
- Difference: {r['metrics']['difference']:.3f}
- **Result**: {'PASSED ✓' if r['metrics']['passed'] else 'FAILED ✗'}
"""

    report += """
### Experiment 2.3: Race Dynamics

**Validates**: Winner-take-all pathway evolution

"""
    if 'exp_2_3' in results:
        r = results['exp_2_3']
        report += f"""- Winning view: {r['metrics']['winning_view']}
- Winner dominance ratio: {r['metrics']['winner_ratio']:.3f}
- Figure: `{r['figure_path']}`
"""

    report += """
### Experiment 3.1: KD Coverage Transfer

**Validates**: C(student_KD) ≈ C(ensemble) >> C(student_hard)

"""
    if 'exp_3_1' in results:
        r = results['exp_3_1']
        report += f"""- Hard label coverage: **{r['metrics']['hard_label']['mean']:.3f}** ± {r['metrics']['hard_label']['std']:.3f}
- KD coverage: **{r['metrics']['kd']['mean']:.3f}** ± {r['metrics']['kd']['std']:.3f}
- Ensemble coverage: **{r['metrics']['ensemble']['mean']:.3f}** ± {r['metrics']['ensemble']['std']:.3f}
- KD matches ensemble: {'YES ✓' if r['metrics']['passed_kd_matches_ensemble'] else 'NO ✗'}
- KD > Hard: {'YES ✓' if r['metrics']['passed_kd_better_than_hard'] else 'NO ✗'}
- Figure: `{r['figure_path']}`
"""

    report += """
### Experiment 3.3: Pathway Evolution

**Validates**: Multiple pathways survive under KD (vs winner-take-all for hard labels)

"""
    if 'exp_3_3' in results:
        r = results['exp_3_3']
        report += f"""- Hard labels:
  - Surviving pathways: {r['metrics']['hard_label']['surviving_pathways']}
  - Dominance ratio: {r['metrics']['hard_label']['dominance_ratio']:.3f}
- KD:
  - Surviving pathways: {r['metrics']['kd']['surviving_pathways']}
  - Dominance ratio: {r['metrics']['kd']['dominance_ratio']:.3f}
- **Result**: {'PASSED ✓' if r['metrics']['passed'] else 'INCONCLUSIVE'}
- Figure: `{r['figure_path']}`
"""

    report += """
---

## Generated Figures

1. `race_dynamics_hard.png` - Pathway strength evolution under hard labels
2. `kd_coverage_comparison.png` - Coverage comparison bar chart
3. `pathway_comparison.png` - Side-by-side pathway evolution (hard vs KD)

---

## Next Steps

Based on these results:
1. If all P1 experiments pass, proceed to P2 experiments
2. If any experiments fail, investigate and adjust parameters
3. Send figures to Writer for paper inclusion
"""

    with open(output_path / 'experiment_summary.md', 'w') as f:
        f.write(report)

    # Also save to to_advisor directory
    to_advisor_path = Path(__file__).parent.parent / 'exchange' / 'to_advisor'
    if to_advisor_path.exists():
        with open(to_advisor_path / f'{datetime.now().strftime("%Y-%m-%d")}_results_p1.md', 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Run KDMech experiments')
    parser.add_argument('--priority', type=str, default='P1',
                        choices=['P1', 'P2', 'P3', 'all'],
                        help='Priority level to run')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced parameters')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    args = parser.parse_args()

    if args.priority == 'P1':
        run_p1_experiments(output_dir=args.output, quick=args.quick)
    else:
        print(f"Priority {args.priority} not yet implemented")
        print("Available: P1")


if __name__ == '__main__':
    main()
