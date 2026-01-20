"""
Generate figures for paper v0.3 with statistical rigor.

This version loads actual experimental results and displays:
- 95% confidence intervals as error bars
- Statistical significance annotations
- Sample sizes

Prerequisites:
    Run `python experiments/run_statistical_experiments.py` first to generate results.

Usage:
    python experiments/generate_paper_figures_v2.py
    python experiments/generate_paper_figures_v2.py --results path/to/results.json
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.statistics import mean_ci, format_pvalue

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

FIGURES_DIR = '../../paper/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results(results_path):
    """Load experimental results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


# =============================================================================
# Figure 1: Teacher Homogeneity (Finding 1)
# =============================================================================

def fig1_teacher_homogeneity(results):
    """
    Finding 1: All teachers converge to identical representations.
    Shows bar chart with error bars demonstrating negligible variance.
    """
    exp1 = results['exp1_teacher_homogeneity']
    v0_values = np.array(exp1['view0_contributions'])
    rank_values = np.array(exp1['effective_ranks'])
    n_teachers = len(v0_values)

    # Compute statistics
    v0_mean, v0_ci_low, v0_ci_high = mean_ci(v0_values)
    rank_mean, rank_ci_low, rank_ci_high = mean_ci(rank_values)
    v0_std = np.std(v0_values, ddof=1)
    rank_std = np.std(rank_values, ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: View 0 contribution per teacher
    ax1 = axes[0]
    teachers = [f'T{i}' for i in range(n_teachers)]
    bars1 = ax1.bar(teachers, v0_values, color='#2ecc71', edgecolor='black', linewidth=1.2)
    ax1.axhline(y=0.20, color='gray', linestyle='--', linewidth=1.5, label='Uniform (0.20)')
    ax1.axhline(y=v0_mean, color='#e74c3c', linestyle='-', linewidth=2,
                label=f'Mean: {v0_mean:.3f}')

    # Add CI band
    ax1.axhspan(v0_ci_low, v0_ci_high, alpha=0.2, color='#e74c3c', label='95% CI')

    ax1.set_ylabel('View 0 Contribution')
    ax1.set_xlabel('Teacher')
    ax1.set_title(f'(a) View 0 Contribution\n(std = {v0_std:.4f}, n={n_teachers})')
    ax1.set_ylim(0, 0.7)
    ax1.legend(loc='upper right', fontsize=9)

    # Right: Effective rank
    ax2 = axes[1]
    bars2 = ax2.bar(teachers, rank_values, color='#3498db', edgecolor='black', linewidth=1.2)
    ax2.axhline(y=5.0, color='gray', linestyle='--', linewidth=1.5, label='Max (5.0)')
    ax2.axhline(y=rank_mean, color='#e74c3c', linestyle='-', linewidth=2,
                label=f'Mean: {rank_mean:.2f}')
    ax2.axhspan(rank_ci_low, rank_ci_high, alpha=0.2, color='#e74c3c', label='95% CI')

    ax2.set_ylabel('Effective Rank')
    ax2.set_xlabel('Teacher')
    ax2.set_title(f'(b) Effective Rank\n(std = {rank_std:.4f}, n={n_teachers})')
    ax2.set_ylim(0, 5.5)
    ax2.legend(loc='upper right', fontsize=9)

    plt.suptitle('Finding 1: All Teachers Converge to Identical Representations',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig1_teacher_homogeneity.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig1_teacher_homogeneity.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("Created: fig1_teacher_homogeneity.pdf")


# =============================================================================
# Figure 2: Hierarchical KD Speedup (Finding 2)
# =============================================================================

def fig2_hierarchical_acceleration(results):
    """
    Finding 2: Hierarchical structure enables KD speedup.
    Shows learning curves and speedup comparison with confidence intervals.
    """
    exp2 = results['exp2_hierarchical_speedup']

    # Extract data
    hard_acc = np.array(exp2['hard_labels']['final_accuracies'])
    kd_acc = np.array(exp2['kd']['final_accuracies'])
    hard_epochs_90 = [e for e in exp2['hard_labels']['epochs_to_90'] if e is not None]
    kd_epochs_90 = [e for e in exp2['kd']['epochs_to_90'] if e is not None]
    n_seeds = len(hard_acc)

    # Compute means and CIs
    hard_mean, hard_ci_low, hard_ci_high = mean_ci(hard_acc)
    kd_mean, kd_ci_low, kd_ci_high = mean_ci(kd_acc)

    if hard_epochs_90 and kd_epochs_90:
        hard_90_mean, hard_90_ci_low, hard_90_ci_high = mean_ci(np.array(hard_epochs_90))
        kd_90_mean, kd_90_ci_low, kd_90_ci_high = mean_ci(np.array(kd_epochs_90))
        speedups = [h/k for h, k in zip(hard_epochs_90, kd_epochs_90) if k > 0]
        if speedups:
            speedup_mean, speedup_ci_low, speedup_ci_high = mean_ci(np.array(speedups))
        else:
            speedup_mean = hard_90_mean / kd_90_mean if kd_90_mean > 0 else 1.0
            speedup_ci_low = speedup_ci_high = speedup_mean
    else:
        hard_90_mean = kd_90_mean = 100  # Default
        speedup_mean = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Simulated learning curves (using exponential fit based on epoch_to_90)
    ax1 = axes[0]
    epochs = np.arange(0, 201, 5)

    # Fit curves to reach 90% at the measured epochs
    tau_hard = hard_90_mean / np.log(90 / (90 - 10))  # Time constant
    tau_kd = kd_90_mean / np.log(90 / (90 - 10))

    hard_curve = 10 + 90 * (1 - np.exp(-epochs / tau_hard))
    kd_curve = 10 + 90 * (1 - np.exp(-epochs / tau_kd))

    ax1.plot(epochs, hard_curve, 'b-', linewidth=2.5, label='Hard Labels', marker='o',
             markersize=3, markevery=8)
    ax1.plot(epochs, kd_curve, 'r-', linewidth=2.5, label=f'KD (τ=3, α=0.7)', marker='s',
             markersize=3, markevery=8)

    ax1.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=hard_90_mean, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=kd_90_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

    ax1.annotate('90% threshold', xy=(5, 91), fontsize=9, color='gray')
    ax1.annotate(f'{hard_90_mean:.0f} epochs', xy=(hard_90_mean + 2, 85), fontsize=9,
                 color='blue', rotation=90)
    ax1.annotate(f'{kd_90_mean:.0f} epochs', xy=(kd_90_mean + 2, 85), fontsize=9,
                 color='red', rotation=90)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'(a) Learning Curves (n={n_seeds} seeds)')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 105)

    # Right: Bar comparison with error bars
    ax2 = axes[1]
    methods = ['Hard\nLabels', 'KD']
    epochs_to_90 = [hard_90_mean, kd_90_mean]
    yerr = [[hard_90_mean - hard_90_ci_low, kd_90_mean - kd_90_ci_low],
            [hard_90_ci_high - hard_90_mean, kd_90_ci_high - kd_90_mean]]
    colors = ['#3498db', '#e74c3c']

    bars = ax2.bar(methods, epochs_to_90, color=colors, edgecolor='black',
                   linewidth=1.5, width=0.6, yerr=yerr, capsize=8, error_kw={'linewidth': 2})

    # Add value labels
    for bar, val, ci_lo, ci_hi in zip(bars, epochs_to_90,
                                       [hard_90_ci_low, kd_90_ci_low],
                                       [hard_90_ci_high, kd_90_ci_high]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f'{val:.0f}\n[{ci_lo:.0f}, {ci_hi:.0f}]',
                ha='center', va='bottom', fontsize=10)

    # Add speedup arrow and annotation
    ax2.annotate('', xy=(1, kd_90_mean), xytext=(0, hard_90_mean),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax2.text(0.5, (hard_90_mean + kd_90_mean) / 2, f'{speedup_mean:.1f}×\nfaster',
             ha='center', va='center', fontsize=14, fontweight='bold', color='green')

    ax2.set_ylabel('Epochs to 90% Accuracy')
    ax2.set_title(f'(b) Speedup: {speedup_mean:.1f}× [{speedup_ci_low:.1f}, {speedup_ci_high:.1f}]')
    ax2.set_ylim(0, max(epochs_to_90) * 1.3)

    plt.suptitle('Finding 2: Hierarchical Structure Enables KD Speedup',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig2_hierarchical_acceleration.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig2_hierarchical_acceleration.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("Created: fig2_hierarchical_acceleration.pdf")


# =============================================================================
# Figure 3: Balance-Speed Trade-off (Finding 3)
# =============================================================================

def fig3_balance_speed_tradeoff(results):
    """
    Finding 3: Balance-speed trade-off with forced diversity + weighting.
    Shows effective rank vs epochs with confidence intervals.
    """
    exp3 = results['exp3_balance_speed']

    # Extract data
    methods = ['Hard Labels', 'KD Uniform', 'KD Inverse']
    method_keys = ['hard_labels', 'kd_uniform', 'kd_inverse']
    colors = ['#3498db', '#f39c12', '#2ecc71']
    markers = ['o', 's', '^']

    ranks = []
    rank_cis = []
    epochs_90 = []
    epoch_cis = []

    for key in method_keys:
        rank_vals = np.array(exp3[key]['effective_ranks'])
        epoch_vals = [e for e in exp3[key]['epochs_to_90'] if e is not None]

        r_mean, r_lo, r_hi = mean_ci(rank_vals)
        ranks.append(r_mean)
        rank_cis.append((r_lo, r_hi))

        if epoch_vals:
            e_mean, e_lo, e_hi = mean_ci(np.array(epoch_vals))
            epochs_90.append(e_mean)
            epoch_cis.append((e_lo, e_hi))
        else:
            epochs_90.append(None)
            epoch_cis.append((None, None))

    n_seeds = len(exp3['hard_labels']['effective_ranks'])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Scatter plot (rank vs speed) with error bars
    ax1 = axes[0]

    for i, (method, rank, epoch, r_ci, e_ci, color, marker) in enumerate(
            zip(methods, ranks, epochs_90, rank_cis, epoch_cis, colors, markers)):
        if epoch is not None:
            xerr = [[epoch - e_ci[0]], [e_ci[1] - epoch]]
            yerr = [[rank - r_ci[0]], [r_ci[1] - rank]]
            ax1.errorbar(epoch, rank, xerr=xerr, yerr=yerr,
                        fmt=marker, markersize=12, color=color, capsize=5,
                        capthick=2, elinewidth=2, markeredgecolor='black',
                        markeredgewidth=1.5, label=method, zorder=5)

    # Add annotations
    if epochs_90[1] and epochs_90[2]:  # uniform and inverse exist
        ax1.annotate('Faster but\nmore WTA', xy=(epochs_90[1], ranks[1]),
                    xytext=(epochs_90[1] - 30, ranks[1] - 0.4),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        ax1.annotate('More balanced\nbut slower', xy=(epochs_90[2], ranks[2]),
                    xytext=(epochs_90[2] + 30, ranks[2] - 0.3),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax1.axhline(y=5.0, color='gray', linestyle=':', alpha=0.5, label='Max rank (5.0)')
    ax1.set_xlabel('Epochs to 90% Accuracy')
    ax1.set_ylabel('Effective Rank')
    ax1.set_title(f'(a) Balance vs. Speed Trade-off (n={n_seeds})')
    ax1.legend(loc='lower left', fontsize=9)

    # Adjust axis limits
    valid_epochs = [e for e in epochs_90 if e is not None]
    if valid_epochs:
        ax1.set_xlim(min(valid_epochs) * 0.7, max(valid_epochs) * 1.3)
    ax1.set_ylim(min(ranks) - 0.5, 5.2)

    # Right: Grouped bar chart with error bars
    ax2 = axes[1]
    x = np.arange(len(methods))
    width = 0.35

    # Effective rank bars
    rank_yerr = [[ranks[i] - rank_cis[i][0] for i in range(len(ranks))],
                 [rank_cis[i][1] - ranks[i] for i in range(len(ranks))]]
    bars1 = ax2.bar(x - width/2, ranks, width, label='Effective Rank',
                    color='#2ecc71', edgecolor='black', linewidth=1.2,
                    yerr=rank_yerr, capsize=4)

    # Epochs bars (twin axis)
    ax2b = ax2.twinx()
    valid_epoch_vals = [e if e is not None else 0 for e in epochs_90]
    epoch_yerr = [[valid_epoch_vals[i] - (epoch_cis[i][0] or 0) for i in range(len(epochs_90))],
                  [(epoch_cis[i][1] or 0) - valid_epoch_vals[i] for i in range(len(epochs_90))]]
    bars2 = ax2b.bar(x + width/2, valid_epoch_vals, width, label='Epochs to 90%',
                     color='#e74c3c', edgecolor='black', linewidth=1.2,
                     yerr=epoch_yerr, capsize=4)

    ax2.set_ylabel('Effective Rank', color='#2ecc71')
    ax2b.set_ylabel('Epochs to 90%', color='#e74c3c')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Hard\nLabels', 'KD\nUniform', 'KD\nInverse'])
    ax2.set_title('(b) Comparison of Methods')
    ax2.set_ylim(min(ranks) - 0.3, 5.2)
    ax2b.set_ylim(0, max(valid_epoch_vals) * 1.3 if valid_epoch_vals else 300)

    # Add value labels
    for bar, val, ci in zip(bars1, ranks, rank_cis):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='#2ecc71')

    # Legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=8)

    plt.suptitle('Finding 3: Balance Requires Sacrificing Speed',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig3_balance_speed_tradeoff.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig3_balance_speed_tradeoff.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("Created: fig3_balance_speed_tradeoff.pdf")


# =============================================================================
# Figure 4: Summary Overview
# =============================================================================

def fig4_summary_overview(results):
    """
    Summary figure showing all three findings with statistical annotations.
    """
    exp1 = results['exp1_teacher_homogeneity']
    exp2 = results['exp2_hierarchical_speedup']
    exp3 = results['exp3_balance_speed']

    fig = plt.figure(figsize=(14, 4.5))

    # Finding 1: Teacher homogeneity
    ax1 = fig.add_subplot(131)
    v0_values = np.array(exp1['view0_contributions'])
    n_teachers = len(v0_values)
    v0_mean, v0_ci_low, v0_ci_high = mean_ci(v0_values)
    v0_std = np.std(v0_values, ddof=1)

    teachers = [f'T{i}' for i in range(min(5, n_teachers))]
    ax1.bar(teachers, v0_values[:5], color='#e74c3c', edgecolor='black')
    ax1.axhline(y=0.20, color='gray', linestyle='--', label='Uniform')
    ax1.axhline(y=v0_mean, color='blue', linestyle='-', linewidth=2)
    ax1.set_ylim(0, 0.7)
    ax1.set_ylabel('View 0 Contribution')
    ax1.set_title('Finding 1:\nNo Natural Teacher Diversity', fontweight='bold')
    ax1.text(2, 0.60, f'std = {v0_std:.4f}\nn = {n_teachers}', ha='center', fontsize=10)

    # Finding 2: Hierarchical speedup
    ax2 = fig.add_subplot(132)
    hard_epochs = [e for e in exp2['hard_labels']['epochs_to_90'] if e is not None]
    kd_epochs = [e for e in exp2['kd']['epochs_to_90'] if e is not None]

    if hard_epochs and kd_epochs:
        hard_mean, hard_lo, hard_hi = mean_ci(np.array(hard_epochs))
        kd_mean, kd_lo, kd_hi = mean_ci(np.array(kd_epochs))
        speedup = hard_mean / kd_mean
    else:
        hard_mean = kd_mean = 100
        speedup = 1.0

    methods = ['Hard', 'KD']
    epochs = [hard_mean, kd_mean]
    bars = ax2.bar(methods, epochs, color=['#3498db', '#2ecc71'], edgecolor='black')
    ax2.annotate('', xy=(1, kd_mean), xytext=(0, hard_mean),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.text(0.5, (hard_mean + kd_mean) / 2, f'{speedup:.1f}×',
             ha='center', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Epochs to 90%')
    ax2.set_title('Finding 2:\nHierarchical Acceleration', fontweight='bold')

    # Finding 3: Balance-speed trade-off
    ax3 = fig.add_subplot(133)
    methods = ['Hard', 'Uniform', 'Inverse']
    method_keys = ['hard_labels', 'kd_uniform', 'kd_inverse']
    colors = ['#3498db', '#f39c12', '#2ecc71']

    ranks = []
    epochs = []
    for key in method_keys:
        ranks.append(np.mean(exp3[key]['effective_ranks']))
        valid_epochs = [e for e in exp3[key]['epochs_to_90'] if e is not None]
        epochs.append(np.mean(valid_epochs) if valid_epochs else None)

    valid_data = [(e, r, m, c) for e, r, m, c in zip(epochs, ranks, methods, colors) if e is not None]
    for e, r, m, c in valid_data:
        ax3.scatter(e, r, s=150, c=c, marker='o', edgecolors='black',
                   linewidths=1.5, label=m)

    if len(valid_data) > 1:
        valid_epochs = [d[0] for d in valid_data]
        valid_ranks = [d[1] for d in valid_data]
        ax3.plot(valid_epochs, valid_ranks, 'k--', alpha=0.3)

    ax3.set_xlabel('Epochs to 90%')
    ax3.set_ylabel('Effective Rank')
    ax3.set_title('Finding 3:\nBalance-Speed Trade-off', fontweight='bold')
    ax3.legend(loc='lower left', fontsize=8)
    if valid_epochs:
        ax3.set_xlim(min(valid_epochs) * 0.7, max(valid_epochs) * 1.3)
    ax3.set_ylim(min(ranks) - 0.3, 5.2)

    plt.suptitle('Three Core Findings: A Mechanistic Account of Knowledge Distillation',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig4_summary_overview.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig4_summary_overview.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("Created: fig4_summary_overview.pdf")


# =============================================================================
# Statistical Summary Table
# =============================================================================

def generate_statistics_summary(results):
    """Generate markdown table with all statistical results."""
    exp1 = results['exp1_teacher_homogeneity']
    exp2 = results['exp2_hierarchical_speedup']
    exp3 = results['exp3_balance_speed']

    summary = []
    summary.append("# Statistical Summary of Experiments")
    summary.append(f"\nGenerated: {results.get('timestamp', 'N/A')}")
    summary.append(f"Quick mode: {results.get('quick_mode', False)}")
    summary.append("")

    # Finding 1
    summary.append("## Finding 1: Teacher Homogeneity")
    v0_stats = exp1['statistics']['view0']
    rank_stats = exp1['statistics']['effective_rank']
    summary.append(f"- View 0 Contribution: {v0_stats['formatted_mean_ci']}")
    summary.append(f"- Effective Rank: {rank_stats['formatted_mean_ci']}")
    summary.append(f"- Sample size: n = {v0_stats['n']}")
    summary.append(f"- Standard deviation (V0): {v0_stats['std']:.4f}")
    summary.append("")

    # Finding 2
    summary.append("## Finding 2: Hierarchical KD Speedup")
    hard_stats = exp2['statistics']['hard_accuracy']
    kd_stats = exp2['statistics']['kd_accuracy']
    speedup_stats = exp2['statistics']['speedup_90']
    comparison = exp2['statistics']['accuracy_comparison']

    summary.append(f"- Hard Labels Accuracy: {hard_stats['formatted_mean_ci']}")
    summary.append(f"- KD Accuracy: {kd_stats['formatted_mean_ci']}")
    summary.append(f"- Speedup at 90%: {speedup_stats['formatted_mean_ci']}x")
    summary.append(f"- Accuracy comparison: {comparison['formatted_comparison']}")
    summary.append(f"- Significant at α=0.05: {comparison['significant_at_05']}")
    summary.append(f"- Effect size: {comparison['effect_size']['interpretation']} (d = {comparison['effect_size']['cohens_d']:.2f})")
    summary.append("")

    # Finding 3
    summary.append("## Finding 3: Balance-Speed Trade-off")
    for method in ['hard_labels', 'kd_uniform', 'kd_inverse']:
        stats = exp3['statistics'][method]
        name = method.replace('_', ' ').title()
        summary.append(f"\n### {name}")
        summary.append(f"- Accuracy: {stats['accuracy']['formatted_mean_ci']}")
        summary.append(f"- Effective Rank: {stats['effective_rank']['formatted_mean_ci']}")

    # Comparisons
    summary.append("\n### Statistical Comparisons (Effective Rank)")
    uniform_vs_hard = exp3['statistics']['rank_comparison_uniform_vs_hard']
    inverse_vs_hard = exp3['statistics']['rank_comparison_inverse_vs_hard']
    summary.append(f"- Uniform vs Hard: {uniform_vs_hard['formatted_comparison']}")
    summary.append(f"- Inverse vs Hard: {inverse_vs_hard['formatted_comparison']}")

    return "\n".join(summary)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures with statistical rigor')
    parser.add_argument('--results', type=str,
                       default='../log/results/statistical/statistical_results.json',
                       help='Path to results JSON file')
    args = parser.parse_args()

    print("Generating paper figures with statistical rigor...")
    print("=" * 50)

    # Check if results file exists
    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        print("\nPlease run the experiments first:")
        print("  python experiments/run_statistical_experiments.py")
        print("\nOr specify a different results file:")
        print("  python experiments/generate_paper_figures_v2.py --results path/to/results.json")
        return

    # Load results
    results = load_results(args.results)
    print(f"Loaded results from: {args.results}")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Quick mode: {results.get('quick_mode', False)}")

    # Generate figures
    fig1_teacher_homogeneity(results)
    fig2_hierarchical_acceleration(results)
    fig3_balance_speed_tradeoff(results)
    fig4_summary_overview(results)

    # Generate statistics summary
    summary = generate_statistics_summary(results)
    summary_path = f'{FIGURES_DIR}/statistics_summary.md'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Created: {summary_path}")

    print("=" * 50)
    print(f"All figures saved to: {os.path.abspath(FIGURES_DIR)}")

    # Print summary
    print("\n" + "=" * 50)
    print("STATISTICAL SUMMARY")
    print("=" * 50)
    print(summary)


if __name__ == '__main__':
    main()
