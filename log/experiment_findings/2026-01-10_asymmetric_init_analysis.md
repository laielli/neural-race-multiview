# Asymmetric Initialization Analysis - Critical Finding

**Date**: 2026-01-10
**From**: ML Engineer
**To**: Advisor
**Re**: Option C - Asymmetric Initialization Experiments

---

## Summary

Per your request to investigate whether initial advantages produce winner-take-all dynamics, I tested **asymmetric initialization** where the network starts with one view strongly biased.

**Critical finding**: SGD training actively **ERODES** initial advantages. Dominance converges to ~0.35 regardless of initial bias strength.

---

## Experiment Design

### Asymmetric Initialization

Modified network initialization to bias pathway strength toward a specific view:

```python
def init_weights_asymmetric(model, dataset, biased_views, bias_strength):
    # Standard Xavier initialization
    nn.init.xavier_normal_(model.fc1.weight)
    nn.init.xavier_normal_(model.fc2.weight)

    # Boost biased view's output pathway
    for y in range(K):
        m = biased_views[y]  # View to bias for class y
        phi = dataset.get_view_feature(y, m)
        h_response = F.relu(model.fc1(phi))
        model.fc2.weight[y] += bias_strength * h_response
```

This creates initial dominance of ~0.60 for the biased view.

---

## Results

### 1. Dominance Evolution Over Training

Tracking pathway strengths every 20 epochs with `bias_strength=5.0`:

| Epoch | View 0 | View 1 | View 2 | Dominance | V0 Ratio |
|-------|--------|--------|--------|-----------|----------|
| 0 | 2.79 | 0.99 | 0.83 | **0.605** | 0.606 |
| 20 | 3.64 | 1.61 | 1.39 | 0.547 | 0.548 |
| 40 | 4.83 | 2.67 | 2.46 | 0.485 | 0.485 |
| 60 | 5.77 | 3.64 | 3.46 | 0.448 | 0.448 |
| 100 | 6.96 | 5.05 | 4.93 | 0.411 | 0.411 |
| 200 | 8.46 | 6.98 | 6.96 | **0.378** | 0.378 |

**All pathways grow, but non-biased views grow FASTER**, catching up to the biased view.

### 2. Bias Strength Has No Effect

Tested bias strengths from 2x to 100x:

| Bias | Initial Dom | Final Dom | Final Acc |
|------|-------------|-----------|-----------|
| 2.0 | 0.607 | 0.381 | 0.997 |
| 5.0 | 0.605 | 0.378 | 0.997 |
| 10.0 | 0.604 | 0.367 | 0.998 |
| 20.0 | 0.604 | 0.357 | 0.999 |
| 50.0 | 0.604 | 0.351 | 1.000 |
| 100.0 | 0.604 | 0.359 | 1.000 |

**Regardless of initial bias (even 100x), training converges to dominance ~0.35-0.38**

### 3. Training Configuration Effects

#### Learning Rate:
| LR | Init Dom | Final Dom | Accuracy |
|----|----------|-----------|----------|
| 0.001 | 0.604 | 0.510 | 0.814 |
| 0.01 | 0.604 | 0.367 | 0.996 |
| 0.1 | 0.604 | 0.342 | 1.000 |
| 0.5 | 0.604 | 0.346 | 1.000 |

**Lower LR preserves dominance but sacrifices accuracy**

#### Early Stopping:
| Epochs | Init Dom | Final Dom | Accuracy |
|--------|----------|-----------|----------|
| 5 | 0.604 | 0.589 | 0.610 |
| 10 | 0.604 | 0.556 | 0.695 |
| 20 | 0.604 | 0.479 | 0.883 |
| 50 | 0.604 | 0.389 | 0.987 |
| 100 | 0.604 | 0.367 | 0.996 |

**Early stopping preserves dominance but with incomplete learning**

#### Momentum:
| Momentum | Init Dom | Final Dom | Accuracy |
|----------|----------|-----------|----------|
| 0.0 | 0.604 | 0.505 | 0.823 |
| 0.5 | 0.604 | 0.454 | 0.925 |
| 0.9 | 0.604 | 0.367 | 0.996 |

**Less momentum preserves dominance but slows convergence**

---

## Key Insight: Loss Landscape Favors Equalization

The fundamental issue is that **cross-entropy loss encourages learning ALL views**:

1. **Equal data distribution**: Each view appears equally often, generating equal gradient signal
2. **No selection pressure**: Loss decreases when ANY view is correctly classified
3. **Rich-get-richer doesn't apply**: SGD doesn't amplify initial advantages - it **reduces** them

The only ways to maintain dominance are:
- **Under-train** (early stopping, low LR) → poor accuracy
- **Explicitly penalize** multiple views in loss function

---

## Comparison: All Approaches Tested

| Approach | Winner-Take-All? | Notes |
|----------|------------------|-------|
| Orthogonal slots (baseline) | NO | Dominance = 0.34 |
| Competing views (shared dims) | NO | Dominance = 0.35 |
| Bottleneck MLP (down to 1 neuron) | NO | Dominance = 0.34 |
| Top-K sparse (k=3, 0% overlap) | NO | Internal specialization but equal output |
| Mixture of Experts | NO | Perfect gating but equal output |
| Asymmetric init (2x-100x bias) | NO | Initial advantage eroded by training |
| Early stopping | PARTIAL | But accuracy suffers |

---

## Implications for Theory

### What This Tells Us

1. **Standard gradient descent does NOT produce winner-take-all**
   - Equal data → equal gradients → equal learning
   - Initial advantages are actively eroded

2. **Winner-take-all requires explicit mechanism**
   - Modified loss function (penalty for multiple views)
   - Data imbalance (some views appear more often)
   - Architectural constraints (attention, hard selection)

3. **The theory may need refinement**
   - Predicted behavior does not emerge from standard training
   - Need to specify what mechanism drives selection

### Possible Theory Adjustments

1. **Reinterpret "winner-take-all"**: Maybe it refers to internal pathway structure (which we DO see) rather than output dominance?

2. **Add selection mechanism**: Perhaps KD naturally creates selection pressure that hard labels don't?

3. **Different optimization regime**: Maybe winner-take-all only appears with specific (non-standard) training protocols?

---

## Recommendation

The experiments consistently show:
- **NO** standard training configuration produces winner-take-all
- Internal specialization (neurons/experts) does occur
- But all views achieve equal output strength

Suggest one of:
1. **Pivot to internal metrics**: Measure pathway overlap/separation rather than output dominance
2. **Add explicit selection**: If theory requires winner-take-all, add it to loss function
3. **Re-examine theory**: What condition produces the predicted dynamics?

---

## Next Steps

Awaiting guidance on how to proceed. Possible directions:
1. **Test KD**: Does soft label training from biased teacher produce different dynamics?
2. **Add selection loss**: Explicitly penalize learning multiple views
3. **Different architectures**: Transformers with attention?

Ready to implement whichever direction you recommend.
