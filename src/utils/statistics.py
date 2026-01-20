"""
Statistical utilities for adding rigor to experimental results.

Provides:
- Bootstrap confidence intervals
- Paired t-tests and Wilcoxon tests
- Effect sizes (Cohen's d)
- Result formatting for papers
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
import warnings


# =============================================================================
# Confidence Intervals
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
        seed: Random seed

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    data = np.asarray(data)
    n = len(data)

    # Point estimate
    point = statistic(data)

    # Bootstrap samples
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample = data[np.random.randint(0, n, size=n)]
        boot_stats[i] = statistic(boot_sample)

    # Percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return point, ci_lower, ci_upper


def mean_ci(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute mean with confidence interval using t-distribution.

    For small samples, this is more accurate than bootstrap.

    Args:
        data: 1D array of observations
        confidence: Confidence level

    Returns:
        (mean, ci_lower, ci_upper)
    """
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)

    if n < 2:
        return mean, mean, mean

    # t critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se

    return mean, mean - margin, mean + margin


def std_error(data: np.ndarray) -> float:
    """Compute standard error of the mean."""
    return stats.sem(np.asarray(data))


# =============================================================================
# Hypothesis Tests
# =============================================================================

def paired_ttest(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Paired t-test comparing two related samples.

    Args:
        a: First sample (e.g., method A results)
        b: Second sample (e.g., method B results, same seeds)
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dict with t-statistic, p-value, effect size (Cohen's d)
    """
    a, b = np.asarray(a), np.asarray(b)

    if len(a) != len(b):
        raise ValueError("Samples must have same length for paired test")

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(a, b, alternative=alternative)

    # Cohen's d for paired samples
    diff = a - b
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': d,
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff, ddof=1),
        'n': len(a)
    }


def independent_ttest(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = 'two-sided',
    equal_var: bool = False
) -> Dict[str, float]:
    """
    Independent samples t-test (Welch's by default).

    Args:
        a: First sample
        b: Second sample
        alternative: 'two-sided', 'less', or 'greater'
        equal_var: If False, use Welch's t-test (recommended)

    Returns:
        Dict with t-statistic, p-value, effect size (Cohen's d)
    """
    a, b = np.asarray(a), np.asarray(b)

    # t-test
    t_stat, p_value = stats.ttest_ind(a, b, alternative=alternative, equal_var=equal_var)

    # Cohen's d (pooled std)
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': d,
        'mean_a': np.mean(a),
        'mean_b': np.mean(b),
        'std_a': np.std(a, ddof=1),
        'std_b': np.std(b, ddof=1),
        'n_a': n1,
        'n_b': n2
    }


def wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test (non-parametric paired test).

    Use when data is not normally distributed or sample size is small.

    Args:
        a: First sample
        b: Second sample
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dict with test statistic, p-value
    """
    a, b = np.asarray(a), np.asarray(b)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value = stats.wilcoxon(a, b, alternative=alternative)

    return {
        'statistic': stat,
        'p_value': p_value,
        'n': len(a)
    }


def mannwhitney_test(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Mann-Whitney U test (non-parametric independent test).

    Args:
        a: First sample
        b: Second sample
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dict with test statistic, p-value
    """
    a, b = np.asarray(a), np.asarray(b)
    stat, p_value = stats.mannwhitneyu(a, b, alternative=alternative)

    return {
        'statistic': stat,
        'p_value': p_value,
        'n_a': len(a),
        'n_b': len(b)
    }


# =============================================================================
# Effect Sizes
# =============================================================================

def cohens_d(a: np.ndarray, b: np.ndarray, paired: bool = False) -> float:
    """
    Compute Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        a: First sample
        b: Second sample
        paired: If True, use paired formula

    Returns:
        Cohen's d value
    """
    a, b = np.asarray(a), np.asarray(b)

    if paired:
        diff = a - b
        return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
    else:
        n1, n2 = len(a), len(b)
        var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d value."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# Formatting for Papers
# =============================================================================

def format_mean_std(mean: float, std: float, precision: int = 2) -> str:
    """Format as 'mean ± std'."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def format_mean_ci(
    mean: float,
    ci_lower: float,
    ci_upper: float,
    precision: int = 2
) -> str:
    """Format as 'mean [ci_lower, ci_upper]'."""
    return f"{mean:.{precision}f} [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value for papers."""
    if p < threshold:
        return f"p < {threshold}"
    else:
        return f"p = {p:.3f}"


def format_result_with_stats(
    values: np.ndarray,
    name: str = "",
    precision: int = 2,
    include_ci: bool = True,
    confidence: float = 0.95
) -> Dict[str, Union[str, float]]:
    """
    Compute and format full statistics for a result.

    Args:
        values: Array of measurements (e.g., from different seeds)
        name: Name of the metric
        precision: Decimal places
        include_ci: If True, include confidence interval
        confidence: Confidence level

    Returns:
        Dict with formatted strings and raw values
    """
    values = np.asarray(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0
    se = std / np.sqrt(n) if n > 0 else 0

    result = {
        'name': name,
        'n': n,
        'mean': mean,
        'std': std,
        'se': se,
        'min': np.min(values),
        'max': np.max(values),
        'formatted_mean_std': format_mean_std(mean, std, precision)
    }

    if include_ci and n > 1:
        _, ci_lower, ci_upper = mean_ci(values, confidence)
        result['ci_lower'] = ci_lower
        result['ci_upper'] = ci_upper
        result['formatted_mean_ci'] = format_mean_ci(mean, ci_lower, ci_upper, precision)

    return result


def compare_methods(
    method_a_values: np.ndarray,
    method_b_values: np.ndarray,
    method_a_name: str = "A",
    method_b_name: str = "B",
    paired: bool = True,
    precision: int = 2
) -> Dict:
    """
    Full statistical comparison between two methods.

    Args:
        method_a_values: Results from method A
        method_b_values: Results from method B
        method_a_name: Name of method A
        method_b_name: Name of method B
        paired: If True, use paired tests (same seeds)
        precision: Decimal places

    Returns:
        Dict with all comparison statistics
    """
    a = np.asarray(method_a_values)
    b = np.asarray(method_b_values)

    # Basic stats
    stats_a = format_result_with_stats(a, method_a_name, precision)
    stats_b = format_result_with_stats(b, method_b_name, precision)

    # Comparison
    if paired:
        ttest = paired_ttest(a, b)
        nonparam = wilcoxon_test(a, b) if len(a) >= 6 else None
    else:
        ttest = independent_ttest(a, b)
        nonparam = mannwhitney_test(a, b)

    d = cohens_d(a, b, paired=paired)

    result = {
        method_a_name: stats_a,
        method_b_name: stats_b,
        'difference': {
            'mean_diff': stats_a['mean'] - stats_b['mean'],
            'direction': f"{method_a_name} > {method_b_name}" if stats_a['mean'] > stats_b['mean'] else f"{method_b_name} > {method_a_name}"
        },
        'ttest': ttest,
        'effect_size': {
            'cohens_d': d,
            'interpretation': effect_size_interpretation(d)
        },
        'significant_at_05': ttest['p_value'] < 0.05,
        'significant_at_01': ttest['p_value'] < 0.01,
        'formatted_comparison': f"{method_a_name}: {stats_a['formatted_mean_std']}, {method_b_name}: {stats_b['formatted_mean_std']}, {format_pvalue(ttest['p_value'])}, d={d:.2f}"
    }

    if nonparam:
        result['nonparametric'] = nonparam

    return result


# =============================================================================
# Table Generation
# =============================================================================

def generate_results_table(
    results: Dict[str, np.ndarray],
    baseline: str = None,
    precision: int = 2,
    show_ci: bool = True
) -> str:
    """
    Generate a markdown table from results dictionary.

    Args:
        results: Dict mapping method name to array of values
        baseline: Name of baseline method for comparison
        precision: Decimal places
        show_ci: If True, show confidence intervals

    Returns:
        Markdown table string
    """
    lines = []

    # Header
    if show_ci:
        lines.append("| Method | Mean [95% CI] | vs Baseline |")
        lines.append("|--------|---------------|-------------|")
    else:
        lines.append("| Method | Mean ± Std | vs Baseline |")
        lines.append("|--------|------------|-------------|")

    baseline_vals = np.asarray(results.get(baseline, [])) if baseline else None

    for method, values in results.items():
        values = np.asarray(values)
        stats = format_result_with_stats(values, method, precision, show_ci)

        if show_ci and 'formatted_mean_ci' in stats:
            val_str = stats['formatted_mean_ci']
        else:
            val_str = stats['formatted_mean_std']

        # Comparison to baseline
        if baseline and method != baseline and baseline_vals is not None and len(baseline_vals) > 0:
            if len(values) == len(baseline_vals):
                comparison = paired_ttest(values, baseline_vals)
            else:
                comparison = independent_ttest(values, baseline_vals)

            p = comparison['p_value']
            d = comparison['cohens_d']
            direction = "↑" if np.mean(values) > np.mean(baseline_vals) else "↓"
            comp_str = f"{direction} {format_pvalue(p)}"
        else:
            comp_str = "-" if method == baseline else "N/A"

        lines.append(f"| {method} | {val_str} | {comp_str} |")

    return "\n".join(lines)


# =============================================================================
# Convenience Functions for Common Comparisons
# =============================================================================

def kd_vs_hard_labels_comparison(
    hard_labels_results: Dict[str, np.ndarray],
    kd_results: Dict[str, np.ndarray],
    metrics: List[str] = None
) -> Dict[str, Dict]:
    """
    Compare KD vs hard labels across multiple metrics.

    Args:
        hard_labels_results: Dict mapping metric name to array of values
        kd_results: Dict mapping metric name to array of values
        metrics: List of metrics to compare (default: all)

    Returns:
        Dict of comparisons per metric
    """
    if metrics is None:
        metrics = list(set(hard_labels_results.keys()) & set(kd_results.keys()))

    comparisons = {}
    for metric in metrics:
        if metric in hard_labels_results and metric in kd_results:
            comparisons[metric] = compare_methods(
                kd_results[metric],
                hard_labels_results[metric],
                "KD", "Hard Labels",
                paired=True
            )

    return comparisons


if __name__ == '__main__':
    # Example usage
    print("Statistical Utilities Demo")
    print("=" * 50)

    # Simulated data
    np.random.seed(42)
    hard_labels_acc = np.array([73.2, 74.1, 72.8, 73.5, 74.0])
    kd_acc = np.array([76.5, 77.2, 76.1, 76.8, 77.0])

    # Basic stats
    print("\nHard Labels Accuracy:")
    stats_hard = format_result_with_stats(hard_labels_acc, "Hard Labels")
    print(f"  {stats_hard['formatted_mean_ci']}")

    print("\nKD Accuracy:")
    stats_kd = format_result_with_stats(kd_acc, "KD")
    print(f"  {stats_kd['formatted_mean_ci']}")

    # Comparison
    print("\nComparison:")
    comparison = compare_methods(kd_acc, hard_labels_acc, "KD", "Hard Labels", paired=True)
    print(f"  {comparison['formatted_comparison']}")
    print(f"  Significant at α=0.05: {comparison['significant_at_05']}")
    print(f"  Effect size: {comparison['effect_size']['interpretation']}")

    # Generate table
    print("\n" + "=" * 50)
    print("Results Table:")
    results = {
        "Hard Labels": hard_labels_acc,
        "KD (τ=3)": kd_acc,
        "KD (τ=5)": kd_acc + np.random.randn(5) * 0.3
    }
    print(generate_results_table(results, baseline="Hard Labels"))
