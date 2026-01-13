# Expected Results Guide

> **Purpose**: Comprehensive reference for interpreting experiment outcomes
> **Structure**: Predictions, pass criteria, and troubleshooting

---

## Summary Table

### Theorem 1 Predictions (P2 Experiments)

| Experiment | Metric | Expected | Validates |
|------------|--------|----------|-----------|
| 1.1 | Output decomposition correlation | > 0.95 | $f(x) = \sum_m R_{y,m}$ |
| 1.2 | Min gating Hamming distance | > 0 | Views → distinct gating |
| 1.3 | Response spectrum | Sparse | Few views detected per network |

### Theorem 2 Predictions

| Experiment | Metric | Expected | Validates |
|------------|--------|----------|-----------|
| **2.1** (P1) | Mean coverage | $\approx 1/M = 0.33$ | Single-view convergence |
| 2.2 | View diversity across seeds | > 0.6 | Different seeds → different views |
| **2.3** (P1) | Winner dominance ratio | > 0.8 | Winner-take-all dynamics |
| 2.4 | Winner prediction accuracy | > 80% | Init determines winner |
| 2.5 | Early phase $R^2$ (exponential fit) | > 0.9 | Exponential growth |
| 2.6 | Win rate ordering | Strong > Medium > Weak | Correlation effect |

### Theorem 3 Predictions

| Experiment | Metric | Expected | Validates |
|------------|--------|----------|-----------|
| **3.1** (P1) | KD coverage | $\approx$ ensemble | Multi-view via KD |
| **3.1** (P1) | KD - Hard difference | > 0.3 | KD benefit |
| 3.2 | Gradient Gini | KD < Hard | Distributed gradients |
| **3.3** (P1) | Hard surviving pathways | 1 | Winner-take-all |
| **3.3** (P1) | KD surviving pathways | 2-3 | Multiple survive |
| 3.4 | $C(S)$ vs $C(T)$ correlation | > 0.9 | Coverage inheritance |
| 3.5 | Coverage vs τ | Increasing then plateau | Temperature effect |
| 3.6 | Gen n+1 vs Gen n coverage | Increasing | Self-distillation |
| 3.7 | KD(single) vs Hard | $\approx$ Equal | Single teacher fails |

---

## Detailed Expectations

### Experiment 2.1: Single-View Convergence

**What we measure**: View coverage $C(f)$ across 30 random seeds

**Expected distribution**:
```
Coverage values: [0.30, 0.33, 0.37, 0.30, 0.33, ...]

Mean: 0.333
Std:  0.045 (approximately)
Range: [0.27, 0.40]
```

**Why this value**:
- Each network learns ~1 view per class
- With K=10 classes and M=3 views, $C(f) = K/KM = 1/M = 0.33$
- Variance from: (a) detection threshold effects, (b) some classes learning 2 views

**Pass if**:
```python
abs(mean_coverage - 1/M) < 0.05
```

---

### Experiment 2.3: Race Dynamics

**What we measure**: Pathway strengths over training epochs

**Expected trajectory**:
```
Epoch   s_0     s_1     s_2     Notes
──────────────────────────────────────────
0       0.03    0.02    0.04    Random init
25      0.15    0.10    0.25    All growing
50      0.40    0.20    0.80    View 2 leading
100     0.30    0.10    2.50    View 2 dominant
150     0.15    0.05    4.00    Losers decaying
200     0.08    0.04    4.50    Winner saturated
```

**Key metrics**:
- Winner dominance ratio: $\frac{\max s_m}{\sum s_m} > 0.8$
- Loser decay: Final strengths < 0.1 × winner

**Pass if**:
- One clear winner emerges
- Dynamics show expected sigmoidal shape
- Different seeds produce different winners

---

### Experiment 3.1: KD Coverage Transfer

**What we measure**: Coverage under hard labels vs KD

**Expected values** (10 trials):

| Condition | Mean | Std | 95% CI |
|-----------|------|-----|--------|
| Hard labels | 0.33 | 0.05 | [0.30, 0.37] |
| KD | 0.89 | 0.06 | [0.85, 0.93] |
| Ensemble | 0.93 | 0.05 | [0.90, 0.97] |

**Key comparisons**:
- KD vs Hard: Difference > 0.3 (usually ~0.55)
- KD vs Ensemble: Difference < 0.1

**Pass if**:
```python
(mean_kd - mean_hard > 0.1) and (abs(mean_kd - mean_ensemble) < 0.1)
```

---

### Experiment 3.3: Pathway Evolution

**What we measure**: Number of surviving pathways, dominance ratio

**Expected final state**:

| Condition | Surviving | Dominance Ratio |
|-----------|-----------|-----------------|
| Hard labels | 1 | 0.85-0.98 |
| KD | 2-3 | 0.35-0.50 |

**Pathway strength distribution**:

Hard labels:
```
View 0: 0.12  (8%)
View 1: 0.08  (5%)
View 2: 4.85  (87%)  ← Winner
```

KD:
```
View 0: 2.45  (41%)
View 1: 1.98  (33%)
View 2: 1.52  (26%)  ← More balanced
```

**Pass if**:
```python
(hard_dominance_ratio > 0.7) and (kd_dominance_ratio < 0.6)
```

---

## Quantitative Benchmarks

### Coverage Benchmarks

| Scenario | Expected Coverage |
|----------|-------------------|
| Hard labels, M=3 | 0.33 ± 0.05 |
| Hard labels, M=5 | 0.20 ± 0.04 |
| KD from 5-teacher ensemble, M=3 | 0.90 ± 0.06 |
| KD from 10-teacher ensemble, M=3 | 0.95 ± 0.04 |
| KD from single teacher, M=3 | 0.35 ± 0.05 |

### Ensemble Coverage Scaling

| Num Teachers | Expected Ensemble Coverage |
|--------------|----------------------------|
| 1 | 0.33 |
| 3 | 0.70 |
| 5 | 0.90 |
| 10 | 0.97 |
| ∞ | 1.00 |

Formula: $C(T^N) \approx 1 - (1 - 1/M)^N$

### Temperature Effect

| Temperature τ | Expected KD Coverage |
|---------------|----------------------|
| 1.0 | 0.40 (near hard labels) |
| 2.0 | 0.65 |
| 4.0 | 0.90 |
| 8.0 | 0.93 |
| 16.0 | 0.92 (slight decrease) |

---

## Failure Modes

### Coverage Too High (Hard Labels)

**Observation**: Mean coverage > 0.45 under hard labels

**Possible causes**:
1. Views not orthogonal (overlap in slots)
2. Network too wide (can represent multiple views)
3. Training not converged (check loss curve)
4. Detection threshold too low

**Remedies**:
- Run `verify_dataset()` to check orthogonality
- Reduce hidden width
- Increase training epochs
- Raise detection threshold to 0.6

### Coverage Too Low (KD)

**Observation**: KD coverage < 0.7 (should be ~0.9)

**Possible causes**:
1. Temperature too low (hard label-like)
2. Teachers not diverse (all learned same view)
3. Training not converged
4. Ensemble too small

**Remedies**:
- Increase temperature to 4 or 8
- Use more teachers (try 10)
- Increase training epochs
- Verify teacher diversity

### No Winner-Take-All

**Observation**: Hard label training shows multiple pathways surviving

**Possible causes**:
1. Training not converged
2. Views have very different correlation strengths
3. Learning rate too low
4. Early stopping

**Remedies**:
- Increase epochs (try 300-400)
- Check data generation (views should be symmetric)
- Increase learning rate
- Train until loss plateaus

---

## Statistical Considerations

### Sample Sizes

| Experiment | Recommended N | Minimum N |
|------------|---------------|-----------|
| 2.1 | 30 seeds | 10 seeds |
| 2.3 | 5 seeds | 1 seed |
| 3.1 | 10 trials | 5 trials |
| 3.3 | 3 classes | 1 class |

### Confidence Intervals

For N=30 seeds, 95% CI width:
$$\text{CI width} \approx 2 \times 1.96 \times \frac{\sigma}{\sqrt{N}} \approx 0.017$$

For coverage std ≈ 0.045:
$$\text{CI} = \text{mean} \pm 0.016$$

### Effect Sizes

| Comparison | Expected Cohen's d |
|------------|-------------------|
| KD vs Hard coverage | > 3.0 (very large) |
| Hard dominance vs KD dominance | > 2.0 (large) |
| Ensemble vs single teacher | > 2.0 (large) |

---

## What Constitutes "Passing"

### Strict Criteria (for Paper)

1. **Exp 2.1**: $|C - 1/M| < 0.03$
2. **Exp 2.3**: Winner ratio > 0.85, clear sigmoidal dynamics
3. **Exp 3.1**: $C_{KD} - C_{Hard} > 0.4$, $|C_{KD} - C_{Ens}| < 0.08$
4. **Exp 3.3**: Hard survives = 1, KD survives ≥ 2

### Relaxed Criteria (for Initial Validation)

1. **Exp 2.1**: $|C - 1/M| < 0.05$
2. **Exp 2.3**: One pathway clearly dominant
3. **Exp 3.1**: $C_{KD} > C_{Hard} + 0.1$
4. **Exp 3.3**: Hard dominance > KD dominance

---

## Reporting Results

### Summary Statistics to Report

For each coverage metric:
```
Mean: 0.333
Std:  0.045
95% CI: [0.317, 0.349]
N: 30
```

### Key Figures for Paper

1. **Figure 3**: Race dynamics (Exp 2.3)
   - Single seed showing winner-take-all
   - Multiple seeds showing diversity

2. **Figure 4**: KD effect (Exp 3.1, 3.3)
   - Bar chart: Hard vs KD vs Ensemble coverage
   - Side-by-side dynamics: Hard vs KD
