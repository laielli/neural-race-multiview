# Statistical Experiments for Paper

This directory contains tools for running experiments with proper statistical rigor.

## Quick Start

```bash
# Quick test (~3-5 min on CPU)
python experiments/run_statistical_experiments.py --quick

# Generate figures from results
python experiments/generate_paper_figures_v2.py
```

## Full Experiments (for paper)

```bash
# Full experiments (~20-30 min on CPU)
python experiments/run_statistical_experiments.py --output ../log/results/statistical

# Generate figures with 95% CI error bars
python experiments/generate_paper_figures_v2.py --results ../log/results/statistical/statistical_results.json
```

## What the Scripts Do

### `run_statistical_experiments.py`

Runs three core experiments with multiple seeds:

1. **Experiment 1: Teacher Homogeneity** (Finding 1)
   - Trains N teachers with different seeds
   - Measures view contributions and effective rank
   - Verifies all teachers converge to identical representations (low std)

2. **Experiment 2: Hierarchical KD Speedup** (Finding 2)
   - Creates hierarchical dataset (superclass/subclass structure)
   - Compares hard labels vs KD learning speed
   - Reports speedup with confidence intervals

3. **Experiment 3: Balance-Speed Trade-off** (Finding 3)
   - Trains single-view teachers for forced diversity
   - Compares hard labels, KD uniform, KD inverse weighting
   - Measures effective rank vs learning speed trade-off

### `generate_paper_figures_v2.py`

Generates publication-ready figures with:
- 95% confidence intervals (error bars)
- Statistical significance annotations
- Sample sizes
- Markdown summary of all statistics

## Output Files

```
../log/results/statistical/
├── statistical_results.json    # All raw results with statistics

../../paper/figures/
├── fig1_teacher_homogeneity.pdf
├── fig2_hierarchical_acceleration.pdf
├── fig3_balance_speed_tradeoff.pdf
├── fig4_summary_overview.pdf
└── statistics_summary.md       # Formatted statistics for paper
```

## Statistical Methods

All results include:
- **Mean with 95% CI** (t-distribution based)
- **Paired t-tests** for same-seed comparisons
- **Cohen's d** effect size interpretation
- **Standard deviation** and standard error

## Parameters

| Parameter | Quick | Full | Description |
|-----------|-------|------|-------------|
| `epochs` | 500 | 1000 | Training epochs |
| `teacher_epochs` | 500 | 2000 | Teacher training epochs |
| `n_teachers` | 3 | 5 | Number of teachers in ensemble |
| `n_seeds` | 5 | 10 | Number of random seeds |

## Utilities

The `utils/statistics.py` module provides:
- `mean_ci(data)` - Mean with t-distribution CI
- `bootstrap_ci(data)` - Bootstrap CI for any statistic
- `paired_ttest(a, b)` - Paired t-test with effect size
- `compare_methods(a, b)` - Full comparison with formatting
- `generate_results_table(results)` - Markdown table generation
