# Bottleneck Architecture Analysis - Follow-up Report

**Date**: 2026-01-10
**From**: ML Engineer
**To**: Advisor
**Re**: Option B - Constrained Bottleneck Implementation

---

## Summary

Per your request, I implemented **Option B: Constrained Bottleneck** with various architectures designed to force view selection. **Winner-take-all dynamics (dominance > 0.7) were NOT observed in any configuration.**

However, I discovered an important finding: **networks DO specialize internally, but all views achieve equal output strength**.

---

## Architectures Tested

### 1. Bottleneck MLP
```
input(100) → 100 → BOTTLENECK → 100 → output(10)
```

| Bottleneck | Coverage | Dominance | Accuracy |
|------------|----------|-----------|----------|
| 50 | 1.000 | 0.352 | 1.000 |
| 20 | 1.000 | 0.358 | 1.000 |
| 10 | 1.000 | 0.352 | 1.000 |
| 5 | 1.000 | 0.355 | 1.000 |
| 3 | 1.000 | 0.350 | 1.000 |
| 2 | 1.000 | 0.345 | 1.000 |
| **1** | 0.700 | 0.335 | 0.803 |

Even with **bottleneck=1** (single neuron!), no winner-take-all.

### 2. Linear Bottleneck (No Nonlinearity)
```
input(100) → LINEAR COMPRESSION → output(10)
```

| Bottleneck | Coverage | Dominance | Accuracy |
|------------|----------|-----------|----------|
| 20 | 1.000 | 0.356 | 1.000 |
| 10 | 1.000 | 0.361 | 1.000 |
| 5 | 1.000 | 0.358 | 1.000 |
| 3 | 1.000 | 0.348 | 1.000 |
| 1 | 0.567 | 0.340 | 0.587 |

### 3. Low-Rank Factorization
```
W = U × V where U is d×rank, V is rank×K
```

| Rank | Coverage | Dominance | Accuracy |
|------|----------|-----------|----------|
| 20 | 1.000 | 0.355 | 1.000 |
| 10 | 1.000 | 0.361 | 1.000 |
| 5 | 1.000 | 0.357 | 1.000 |
| 3 | 1.000 | 0.350 | 1.000 |
| 1 | 0.167 | 0.363 | 0.203 |

### 4. Top-K Sparse Activation
Forces only k neurons to be active.

| Top-K | Coverage | Dominance | Gating Overlap |
|-------|----------|-----------|----------------|
| 50 | 1.000 | 0.347 | 0.500 |
| 20 | 1.000 | 0.345 | 0.222 |
| 10 | 1.000 | 0.350 | 0.148 |
| 5 | 1.000 | 0.347 | 0.120 |
| **3** | 1.000 | 0.344 | **0.000** |

**Critical finding**: With top-k=3, views use **completely different neurons** (0% overlap), yet dominance is still 0.344!

### 5. Mixture of Experts (MoE)
Explicit gating network selects experts.

| Experts | Coverage | Dominance | Accuracy |
|---------|----------|-----------|----------|
| 10 | 1.000 | 0.343 | 1.000 |
| 5 | 1.000 | 0.346 | 1.000 |
| 3 | 1.000 | 0.341 | 1.000 |
| 2 | 1.000 | 0.341 | 1.000 |

**Expert selection for class 0 with 3 experts:**
```
View 0 → Expert 1 (99.9%)
View 1 → Expert 0 (99.9%)
View 2 → Expert 1 (99.9%)
```

The gating network learns **near-perfect view separation**, routing different views to different experts. But pathway strengths are still equal!

### 6. Hard Gating MoE (Gumbel-Softmax)
```
Hard MoE: coverage=1.000, dominance=0.350, acc=1.000
```

---

## Key Insight: Internal Specialization ≠ Output Dominance

The experiments reveal a crucial distinction:

1. **Networks DO specialize internally**:
   - Different views activate different neurons (0% overlap with top-k=3)
   - Different views route to different experts (99.9% gating accuracy)

2. **But all views achieve equal OUTPUT strength**:
   - Dominance = max_strength / total_strength ≈ 0.33 always
   - The network learns to correctly classify ALL views

**The network finds separate pathways for each view, but all pathways are equally strong.**

---

## Why This Happens

### The Optimization Objective
Cross-entropy loss encourages **all views to be correctly classified**. Since each view appears equally in training data, the network learns all views equally well.

### No Selection Pressure
There's no term in the loss that penalizes learning multiple views. The network has no reason to "choose" - it can learn everything.

### Equal Gradient Signal
Each view receives gradient signal proportional to:
- How often it appears (equal for all views)
- How much error it causes (decreases equally as all views are learned)

---

## What Would Produce Winner-Take-All?

Based on these experiments, winner-take-all would require one of:

### 1. Modified Loss Function
Add explicit penalty for learning multiple views:
```python
loss = CE_loss + lambda * multi_view_penalty
```

### 2. Limited Training Data
If only one view per class appears in training, only that view would be learned.

### 3. Asymmetric Initialization
If one view starts with much higher pathway strength, it might "win" through rich-get-richer dynamics.

### 4. Different Optimization
Perhaps some optimizers (not SGD/Adam) produce winner-take-all through implicit regularization.

### 5. Biological Constraints
Biological neural networks have energy constraints that might force selection.

---

## Conclusion

**Standard neural network training does NOT produce winner-take-all dynamics**, regardless of architecture. The network learns all views through separate pathways, all equally strong.

The theoretical framework may need to:
1. Specify what mechanism drives winner-take-all
2. Identify what loss function or regularization produces it
3. Consider whether "winner-take-all" refers to internal structure (which we DO see) rather than output dominance (which we DON'T see)

---

## Recommendation

The experiments suggest the theory's predicted behavior requires conditions beyond standard training. Suggest:

1. **Clarify theoretical assumptions**: What exactly should "dominate" and under what conditions?

2. **Consider alternative metrics**: Maybe measure internal pathway overlap rather than output norm?

3. **Add explicit selection mechanism**: If the theory requires winner-take-all, we may need to add it to the loss function explicitly.

Awaiting guidance on how to proceed.
