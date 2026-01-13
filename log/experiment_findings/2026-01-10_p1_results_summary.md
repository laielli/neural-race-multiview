# P1 Experiment Results Summary

**Date**: 2026-01-10
**From**: ML Engineer
**To**: Advisor
**Priority**: High - Requires Theoretical Review

---

## Executive Summary

All P1 infrastructure and experiments have been implemented and executed. However, **the experimental results do not match the theoretical predictions**. This is a critical finding that requires discussion before proceeding.

---

## Implementation Status

### Completed Components

| Component | File | Status |
|-----------|------|--------|
| Multi-view data generator | `src/data.py` | Done |
| Network architecture | `src/model.py` | Done |
| Measurement functions | `src/metrics.py` | Done |
| Training loops (hard + KD) | `src/train.py` | Done |
| Exp 2.1: Single-view convergence | `experiments/exp_2_1_single_view.py` | Done |
| Exp 2.3: Race dynamics | `experiments/exp_2_3_race_dynamics.py` | Done |
| Exp 3.1: KD coverage | `experiments/exp_3_1_kd_coverage.py` | Done |
| Exp 3.3: Pathway evolution | `experiments/exp_3_3_pathway_evolution.py` | Done |
| Master runner | `experiments/run_all.py` | Done |

---

## Experimental Results

### Summary Table

| Experiment | Metric | Expected | Actual | Status |
|------------|--------|----------|--------|--------|
| 2.1 | Coverage (hard labels) | ~0.33 (1/M) | **1.000** | MISMATCH |
| 2.3 | Winner dominance ratio | >0.7 | **0.344** | MISMATCH |
| 3.1 | KD coverage vs Hard | KD >> Hard | KD = Hard = 1.0 | MISMATCH |
| 3.3 | Surviving pathways | Hard=1, KD>1 | Hard=3, KD=3 | MISMATCH |

### Detailed Findings

#### Experiment 2.1: Single-View Convergence
- **Expected**: After hard label training, C(f) ≈ 1/M ≈ 0.33
- **Actual**: C(f) = 1.0 (all views correctly classified with >99% confidence)
- **Interpretation**: The network learns ALL views, not just 1/M

#### Experiment 2.3: Race Dynamics
- **Expected**: One pathway dominates (winner-take-all), dominance ratio >0.7
- **Actual**: All pathways have equal strength, dominance ratio ≈ 0.34 (which is exactly 1/M)
- **Interpretation**: No winner-take-all dynamics observed

#### Experiment 3.1: KD Coverage Transfer
- **Expected**: C(student_KD) ≈ C(ensemble) >> C(student_hard)
- **Actual**: All three have coverage = 1.0
- **Interpretation**: Cannot differentiate KD benefit when baseline already perfect

#### Experiment 3.3: Pathway Evolution
- **Expected**: Hard labels → 1 surviving pathway; KD → multiple surviving pathways
- **Actual**: Both conditions have 3 surviving pathways (all views)
- **Interpretation**: No difference in pathway survival between training methods

---

## Root Cause Analysis

After extensive debugging, I identified why the experiments don't show winner-take-all dynamics:

### 1. Orthogonal View Slots Don't Create Competition
The current data generator places each view in a **disjoint slot** (separate dimensions):
- View 0: dimensions [0:50]
- View 1: dimensions [50:100]
- View 2: dimensions [100:150]

Because views occupy different input dimensions, the network weights for each slot are **completely independent**. There is no competition - the network can learn all views without any interference.

### 2. Network Has Excess Capacity
With hidden=200 neurons and only 30 total views (10 classes × 3 views), the network has more than enough capacity to represent all views. Even reducing hidden to 10 neurons didn't force winner-take-all.

### 3. Coverage Metric Threshold Too Low
The threshold of 0.5 is too low. All views are classified correctly with >99% confidence, so coverage = 100% regardless of which views the network "specializes" in.

---

## Diagnostic Experiments Performed

I ran additional tests to understand the behavior:

1. **Varied hidden layer size** (200 → 10): No effect on coverage or dominance
2. **Single-view training samples**: Same result - all views learned equally
3. **Overlapping views in same dimension**: Still no winner-take-all
4. **Added weight decay**: No significant effect

---

## Generated Artifacts

### Figures (in `results/figures/`)
1. `race_dynamics_hard.png` - Shows all 3 pathways growing equally (not winner-take-all)
2. `kd_coverage_comparison.png` - Bar chart showing all conditions at 100% coverage
3. `pathway_comparison.png` - Side-by-side comparison (hard vs KD) - both show equal pathways

### Data Files (in `results/`)
- `results_exp_2_1.json` - Raw coverage data across 3 seeds
- `results_exp_2_3.json` - Pathway strength evolution over training
- `results_exp_3_1.json` - KD vs hard label comparison
- `results_exp_3_3.json` - Pathway evolution comparison

---

## Questions for Advisor

1. **Is the orthogonal slot structure correct?** The current setup ensures views don't interfere, but this may prevent competition. Should views share input dimensions?

2. **What mechanism should drive winner-take-all?** In the current setup, there's no reason for the network to prefer one view over another - it can learn all of them.

3. **Should we modify the coverage metric?** Perhaps coverage should measure something other than classification accuracy (e.g., pathway strength ratio)?

4. **Are there specific hyperparameters needed?** Should we try different learning rates, longer training, or specific initialization schemes?

---

## Recommendations

### Option A: Modify Data Generator
Create views that share dimensions and must compete:
```python
# Instead of disjoint slots, all views in same d-dimensional space
for y in range(K):
    for m in range(M):
        phi[(y, m)] = random_unit_vector(d)  # All in same space
```

### Option B: Constrain Network Further
Use an extremely narrow bottleneck that forces view selection:
```python
# e.g., hidden = K (one neuron per class)
model = MultiViewNet(d=150, hidden=10, K=10)
```

### Option C: Change Training Dynamics
Add explicit sparsity pressure or use a different optimizer that encourages specialization.

### Option D: Reframe the Theory
If the theory assumes views compete, the experimental setup needs to create that competition explicitly.

---

## Next Steps

Awaiting guidance on how to proceed. Options:

1. **Revise data generator** to create competing views
2. **Adjust experimental parameters** based on theoretical requirements
3. **Proceed to P2 experiments** despite mismatched P1 results
4. **Revisit theoretical framework** to align with experimental observations

Please advise on preferred direction.

---

## Code Usage

To re-run experiments:
```bash
# Quick test (3 seeds, 50 epochs)
python experiments/run_all.py --priority P1 --quick

# Full run (30 seeds, 100 epochs)
python experiments/run_all.py --priority P1
```

All code is documented and tested. Infrastructure is ready for parameter adjustments once we determine the correct experimental setup.
