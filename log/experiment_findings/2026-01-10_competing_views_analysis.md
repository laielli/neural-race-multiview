# Competing Views Analysis - Follow-up Report

**Date**: 2026-01-10
**From**: ML Engineer
**To**: Advisor
**Re**: Option A - Competing Views Implementation

---

## Summary

Per your request, I implemented **Option A: Competing Views** where all views share the same dimensional space. However, **the results still do not show winner-take-all dynamics**.

---

## What Was Implemented

### New Dataset: `CompetingViewDataset`

Created a new dataset class (`src/data.py`) where:
- All views are random unit vectors in the **same** d-dimensional space
- Views for each class are perturbations of a class centroid
- Within-class view similarity is tunable via `view_spread` parameter

```python
# Views share dimensions instead of being in disjoint slots
class CompetingViewDataset:
    # Each view is: centroid[y] + random_perturbation
    # All views live in the same d-dimensional space
```

### Experiments Conducted

1. **Varied view distinctness** (`view_spread` 0.3 → 2.0)
2. **Varied hidden layer size** (200 → 5 neurons)
3. **Truly independent views** (no centroid structure)
4. **L1 regularization** (0.001 → 0.1)
5. **Multi-view samples** (all views present simultaneously)
6. **Tracked pathway evolution** during training

---

## Results

### Orthogonal Slots vs Competing Views

| Setup | Coverage | Dominance | Expected |
|-------|----------|-----------|----------|
| Orthogonal slots | 1.000 | 0.344 | 0.333 (coverage), >0.7 (dominance) |
| Competing views | 1.000 | 0.345 | 0.333 (coverage), >0.7 (dominance) |

**No difference** - both show all views learned equally.

### Parameter Sweep Results

| Variation | Coverage | Dominance | Notes |
|-----------|----------|-----------|-------|
| view_spread=0.3 (similar views) | 1.0 | 0.345 | No effect |
| view_spread=2.0 (distinct views) | 1.0 | 0.350 | No effect |
| hidden=5 (extreme bottleneck) | 1.0 | 0.354 | Still learns all |
| L1=0.01 | 0.0 | 0.333 | Kills learning |
| Multi-view samples | 1.0 | 0.366 | No effect |

### Pathway Evolution Over Training

Tracked pathway strengths for 300 epochs:

```
Epoch   View0    View1    View2    Dominance
    0     0.39     0.34     0.41    0.363
   50     9.47     8.72     9.69    0.348
  100    10.45     9.65    10.70    0.347
  200    11.42    10.57    11.70    0.347
  300    11.97    11.10    12.27    0.347
```

**All pathways grow in parallel** - dominance ratio is constant throughout training. There is no "race" where one view pulls ahead.

---

## Key Finding

**Standard SGD+momentum training does NOT produce winner-take-all dynamics** regardless of:
- Whether views are in orthogonal slots or shared dimensions
- Network capacity (tested 200 → 5 hidden units)
- View distinctness
- Regularization (L1/L2)
- Training data distribution (single-view vs multi-view samples)

The dominance ratio stays constant at ~0.33-0.35 (exactly 1/M) throughout training.

---

## Analysis

### Why No Winner-Take-All?

1. **Gradient distribution**: Each view appears equally often in training data, so all views receive equal gradient signal over time.

2. **No competition mechanism**: Even in shared dimensions, the network can learn separate "detectors" for each view without interference.

3. **Excess capacity**: The network has enough capacity to represent all views, so there's no pressure to select.

4. **Stable equilibrium**: Equal pathway strengths appears to be a stable equilibrium of the training dynamics.

### What Might Produce Winner-Take-All?

Based on our experiments, winner-take-all would require one of:

1. **Architectural constraints**: Explicit attention or gating mechanisms that force selection
2. **Different optimization**: Not standard SGD (perhaps winner-take-all requires specific dynamics)
3. **Implicit regularization**: From overparameterization in specific regimes
4. **Data imbalance**: If some views appear more often, they might dominate

---

## Questions for Theory

1. **What mechanism drives winner-take-all?** In our experiments, all views receive equal gradient and grow equally. What causes one to "win"?

2. **Is there a specific regime?** Does winner-take-all require specific:
   - Learning rate?
   - Initialization scale?
   - Width/depth ratio?
   - Training duration?

3. **Is the 2-layer MLP the right architecture?** Perhaps deeper networks or different activations are needed?

4. **Is this a batch size effect?** We used batch_size=128. Would full-batch or very small batches differ?

---

## Generated Artifacts

### New Files
- `src/data.py` - Added `CompetingViewDataset` class
- `experiments/exp_competing_views.py` - Comparison experiment

### Figures
- `results/figures/competing_views_comparison.png` - Side-by-side comparison
- `results/figures/pathway_evolution_competing.png` - Training dynamics

### Data
- `results/results_competing_views.json` - Numerical results

---

## Recommendation

The experimental setup may need more fundamental changes than just modifying the data generator. Suggest one of:

1. **Literature review**: Find papers that empirically demonstrate winner-take-all in neural networks - what setup do they use?

2. **Theory refinement**: Identify what conditions the theory requires and verify they hold

3. **Alternative architectures**: Try attention mechanisms, mixture-of-experts, or other selective architectures

4. **Pause P1 experiments**: Until we understand why winner-take-all doesn't emerge, the current experiments can't validate the theory

Awaiting guidance.
