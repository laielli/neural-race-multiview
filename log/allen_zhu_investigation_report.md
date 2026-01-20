# Allen-Zhu Discrepancy Investigation Report

**Date**: 2026-01-20
**Status**: Investigation Complete - Key Finding Identified

---

## The Discrepancy

**Allen-Zhu et al. claim**: Different teachers (trained with different random seeds) naturally learn different "views" → ensemble has diversity → KD transfers this diversity to student.

**Our finding**: All teachers with different seeds converge to **identical** representations (V0 std < 0.003 across all conditions tested).

---

## Investigation Summary

We systematically tested 6 hypotheses for why teachers don't develop diversity:

### 1. Signal Strength Asymmetry

**Hypothesis**: Decaying signal strengths create a clear "best" view.

| Condition | V0 std | V0 mean | Effective Rank |
|-----------|--------|---------|----------------|
| Decaying [1, 0.5, 0.25, ...] | 0.0008 | 0.504 | 3.59 (WTA) |
| Equal [1, 1, 1, 1, 1] | 0.0007 | 0.198 | 5.00 (balanced) |
| Reverse [0.06, ..., 1] | 0.0004 | 0.045 | 3.58 (WTA) |

**Finding**: Signal asymmetry determines WHICH view dominates, but all teachers converge to the SAME solution (either same WTA or same balanced).

### 2. Network Architecture

| Architecture | V0 std | V0 mean |
|--------------|--------|---------|
| 2-layer Linear | 0.0007 | 0.198 |
| 2-layer ReLU | 0.0019 | 0.200 |
| 4-layer Linear | 0.0023 | 0.200 |
| 4-layer ReLU | 0.0023 | 0.198 |

**Finding**: Neither nonlinearity nor depth creates diversity. Slightly more variance in deeper/nonlinear nets, but still negligible.

### 3. Loss Function

| Loss | V0 std | V0 mean | Avg Rank |
|------|--------|---------|----------|
| Cross-Entropy | 0.0007 | 0.198 | 5.00 |
| MSE | 0.0009 | 0.198 | 5.00 |

**Finding**: Loss function doesn't affect diversity when signals are equal.

### 4. Network Capacity

| Hidden Size | V0 std | V0 mean |
|-------------|--------|---------|
| 16 | 0.0006 | 0.198 |
| 64 | 0.0007 | 0.198 |
| 256 | 0.0003 | 0.198 |
| 512 | 0.0002 | 0.198 |

**Finding**: Larger networks actually have LESS variance (more deterministic convergence).

### 5. Initialization Scale

| Scale | V0 std | V0 mean |
|-------|--------|---------|
| 0.01 | 0.0004 | 0.200 |
| 0.1 | 0.0007 | 0.198 |
| 1.0 | 0.0020 | 0.199 |
| 2.0 | 0.0012 | 0.200 |

**Finding**: Larger init scale creates slightly more variance, but still negligible.

### 6. Learning Rate

| LR | V0 std | V0 mean |
|----|--------|---------|
| 0.01 | 0.0017 | 0.199 |
| 0.1 | 0.0007 | 0.198 |
| 0.5 | 0.0007 | 0.198 |
| 1.0 | 0.0007 | 0.198 |

**Finding**: Learning rate doesn't significantly affect diversity.

---

## Key Insight: The Loss Landscape Has a Single Basin

The investigation reveals that our synthetic setting has a **single global optimum** that all teachers converge to, regardless of initialization:

1. **With asymmetric signals**: Single WTA solution where the strongest view dominates
2. **With equal signals**: Single balanced solution where all views contribute equally

This is fundamentally different from Allen-Zhu's implicit assumption that different initializations can lead to different local minima with different view preferences.

---

## Why Allen-Zhu's Setting Might Be Different

Our synthetic multi-view data may be too "clean" compared to real-world settings:

1. **Perfect view-class alignment**: Each view perfectly predicts the class
2. **Independent views**: No correlation between view features
3. **Simple data structure**: Linear separability in each view
4. **No noise in labels**: Perfect training signal

In real data (e.g., CIFAR, ImageNet), multiple imperfect views may create:
- Local minima where different features are preferred
- Trade-offs between accuracy and computational efficiency
- Path-dependent solutions based on early learning

---

## Implications for the Paper

### What This Means

1. **Random seed diversity is not guaranteed**: Simply training multiple teachers doesn't create diverse ensembles in clean synthetic settings.

2. **Diversity must be explicitly forced**: Use view masking, data augmentation, or architectural constraints.

3. **Allen-Zhu's results may depend on real-data complexity**: Their theory may require implicit diversity mechanisms present in natural data.

### Recommended Paper Framing

> "While Allen-Zhu & Li (2023) suggest that independently trained teachers naturally learn different views, we find this depends on the loss landscape structure. In our synthetic setting with clean multi-view data, all teachers converge to identical solutions regardless of random seed. Diversity must be explicitly induced through mechanisms such as view masking or data partitioning. This explains why naive ensemble KD in our setting provides no benefit over a single teacher."

---

## Connection to Other Findings

This investigation complements our other findings:

| Finding | Implication |
|---------|-------------|
| No natural diversity | Must force diversity via masking |
| Signal asymmetry → WTA | Strongest view always wins |
| Equal signals → balanced | No WTA when views are equally informative |
| Single global optimum | Loss landscape has one basin in synthetic data |

---

## Future Directions

To bridge to Allen-Zhu's setting:

1. **Test on real data (CIFAR, ImageNet)**: May naturally have multiple local minima
2. **Add label noise**: Create uncertainty that different teachers resolve differently
3. **Add view correlation**: Make views partially redundant
4. **Reduce view quality**: Make each view only partially predictive
5. **Test with SGD mini-batches**: May introduce stochasticity that creates different paths

---

## Conclusion

The Allen-Zhu discrepancy arises because our synthetic setting has a unique global optimum that all teachers find. This is a **setting-specific limitation**, not a flaw in our methodology. The paper should acknowledge this limitation and clarify that:

1. Natural diversity requires specific data/architecture conditions
2. Our experiments validly test the KD mechanism once diversity is established
3. Forcing diversity (via masking) recovers the expected behavior
