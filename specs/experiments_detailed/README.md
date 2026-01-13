# Experiments: Understanding the Validation Suite

> **Purpose**: Document the experimental validation of Theorems 1-3
> **Location**: Implementations at `/kdmech/engineer/experiments/`
> **Target**: NeurIPS 2026 submission

---

## Overview

This directory contains documentation for understanding the experimental validation suite. The experiments validate our three main theorems using synthetic multi-view data where **ground truth is exactly known**.

### The Three Theorems

| Theorem | Statement | Key Prediction |
|---------|-----------|----------------|
| **Theorem 1** | Views = Pathways | Output decomposes as $f(x) = \sum_m R_{y,m}$ |
| **Theorem 2** | Race Dynamics | Hard labels → $C(f) \approx 1/M$ |
| **Theorem 3** | KD Circumvents Race | Soft labels → $C(S) \approx C(T)$ |

---

## Experiment Structure

### Priority Levels

| Priority | Experiments | Purpose |
|----------|-------------|---------|
| **P1** | 2.1, 2.3, 3.1, 3.3 | Core validation (paper figures) |
| **P2** | 1.1-1.3, 2.2, 2.4-2.6, 3.2, 3.4-3.7 | Supporting evidence |
| **P3** | Ablations | Robustness checks |

### Implemented Experiments

| File | Experiment | Theorem | Description |
|------|------------|---------|-------------|
| `exp_2_1_single_view.py` | 2.1 | Thm 2 | Single-view convergence |
| `exp_2_3_race_dynamics.py` | 2.3 | Thm 2 | Race dynamics visualization |
| `exp_3_1_kd_coverage.py` | 3.1 | Thm 3 | KD coverage transfer |
| `exp_3_3_pathway_evolution.py` | 3.3 | Thm 3 | Pathway evolution comparison |
| `run_all.py` | All P1 | All | Master experiment runner |

---

## Documentation Files

| File | Contents |
|------|----------|
| [README.md](README.md) | This overview |
| [configuration.md](configuration.md) | Default parameters and rationale |
| [exp_2_1.md](exp_2_1.md) | Single-view convergence details |
| [exp_2_3.md](exp_2_3.md) | Race dynamics visualization |
| [exp_3_1.md](exp_3_1.md) | KD coverage transfer |
| [exp_3_3.md](exp_3_3.md) | Pathway evolution comparison |
| [expected_results.md](expected_results.md) | Predictions and interpretation |
| [running_experiments.md](running_experiments.md) | How to run experiments |

---

## Quick Start

### Running All P1 Experiments

```bash
cd /kdmech/engineer
python experiments/run_all.py --priority P1
```

### Quick Test (Reduced Parameters)

```bash
python experiments/run_all.py --priority P1 --quick
```

### Individual Experiments

```bash
# Experiment 2.1: Single-view convergence
python experiments/exp_2_1_single_view.py --seeds 30 --epochs 100

# Experiment 2.3: Race dynamics
python experiments/exp_2_3_race_dynamics.py --seed 0 --class 0 --epochs 200

# Experiment 3.1: KD coverage
python experiments/exp_3_1_kd_coverage.py --teachers 5 --trials 10

# Experiment 3.3: Pathway evolution
python experiments/exp_3_3_pathway_evolution.py --seed 0 --class 0
```

---

## Synthetic Data Design

### Slot Structure

```
Input x ∈ ℝ^150 where d = 3 × 50

Each view occupies a disjoint "slot":
- Slot 0: dimensions [0, 50)    → View 0
- Slot 1: dimensions [50, 100)  → View 1
- Slot 2: dimensions [100, 150) → View 2
```

### View Features

For class $y$ and view $m$:
- $\phi_{y,m}$ is a random unit vector in slot $m$
- Views are **exactly orthogonal** (non-overlapping slots)
- Each view is **sufficient** for classification

### Sample Generation

```python
def generate_sample(y, view_prob=0.5):
    x = zeros(150)
    active_views = []

    for m in [0, 1, 2]:
        if random() < view_prob:
            x += phi[(y, m)]
            active_views.append(m)

    # Ensure at least one view
    if len(active_views) == 0:
        m = random_choice([0, 1, 2])
        x += phi[(y, m)]
        active_views.append(m)

    x += noise_std * randn(150)
    return x, y, active_views
```

---

## Key Metrics

### View Coverage

$$C(f) = \frac{1}{KM} \sum_{y=1}^{K} \sum_{m=1}^{M} \mathbb{1}[\text{network detects view } (y,m)]$$

A view is "detected" if:
1. Network classifies $\phi_{y,m}$ correctly (argmax = y)
2. With confidence > threshold (default 0.5)

### Pathway Strength

$$s_{y,m} = \|R_{y,m}(f)\| = \|f(\phi_{y,m})\|$$

The L2 norm of the network's output when given only view $(y,m)$.

### Surviving Pathways

A pathway "survives" if:
$$s_{y,m} > \theta \cdot \max_{m'} s_{y,m'}$$

where $\theta = 0.1$ (default threshold ratio).

---

## Expected Results Summary

| Experiment | Metric | Expected | Validates |
|------------|--------|----------|-----------|
| 2.1 | Mean coverage | $\approx 0.33$ | $C(f) = 1/M$ |
| 2.3 | Winner ratio | $> 0.8$ | Winner-take-all |
| 3.1 | KD vs Hard | KD $>$ Hard by 0.3+ | KD benefit |
| 3.1 | KD vs Ensemble | Difference $< 0.1$ | Coverage inheritance |
| 3.3 | Hard surviving | 1 pathway | Winner-take-all |
| 3.3 | KD surviving | 2-3 pathways | Multiple survive |

---

## Output Structure

```
results/
├── results_exp_2_1.json     # Coverage statistics
├── results_exp_2_3.json     # Race dynamics data
├── results_exp_3_1.json     # KD comparison data
├── results_exp_3_3.json     # Pathway evolution data
├── experiment_summary.json  # Overall summary
├── experiment_summary.md    # Markdown report
└── figures/
    ├── race_dynamics_hard.png
    ├── kd_coverage_comparison.png
    └── pathway_comparison.png
```

---

## Compute Requirements

| Experiment | Seeds/Trials | Epochs | Est. Time |
|------------|--------------|--------|-----------|
| 2.1 | 30 seeds | 100 | ~30 min |
| 2.3 | 1 seed | 200 | ~5 min |
| 3.1 | 10 trials × 5 teachers | 100 | ~1 hour |
| 3.3 | 1 seed × 5 teachers | 200 | ~15 min |

**Total P1 experiments**: ~2 hours on single GPU

### Hardware

- **Minimum**: CPU-only (slower)
- **Recommended**: 1 GPU (RTX 3080 or equivalent)
- **Optimal**: Multi-GPU for parallel seeds

---

## Connection to Paper

| Paper Section | Experiment | Figure |
|---------------|------------|--------|
| §4 Race Dynamics | 2.1, 2.3 | Fig 3 |
| §5 Why KD Works | 3.1, 3.3 | Fig 4 |
| Appendix | All P2 | Supplementary |
