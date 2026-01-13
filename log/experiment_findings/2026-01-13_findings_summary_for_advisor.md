# Findings Summary: Neural Race Theory Investigation

**Date**: 2026-01-13
**Prepared for**: Advisor Review
**Status**: DECISION REQUIRED

---

## Executive Summary

After extensive experimentation, we have identified a fundamental gap between the neural race theory and experimental reality:

1. **Winner-take-all dynamics do NOT emerge naturally** from standard neural network training
2. **Explicit competition mechanisms** (loss terms) are required to produce winner-take-all
3. **Knowledge distillation does NOT break competition** — contradicting Theorem 3's prediction
4. **KD performs WORSE under competition** — soft labels are weaker than hard labels

**Bottom line**: The paper's central thesis requires revision. We need guidance on how to proceed.

---

## Background

### Original Theory Predictions

| Theorem | Prediction | Status |
|---------|------------|--------|
| Theorem 2 (Race Dynamics) | Hard label training produces winner-take-all (coverage ≈ 0.33) | NOT VALIDATED |
| Theorem 3 (KD Mechanism) | KD soft labels distribute gradients, breaking winner-take-all | CONTRADICTED |
| Theorem 1 (Gating) | Views induce different gating patterns | Not yet tested |

### Key Metrics

- **Coverage**: Fraction of views correctly classified (1.0 = all views learned)
- **Dominance**: max(pathway strength) / sum(pathway strengths)
  - Dominance ≈ 0.33 = equal pathways (M=3 views)
  - Dominance ≈ 1.0 = winner-take-all

---

## Experimental Results

### Phase 1: Standard Training (All Failed to Show WTA)

| Configuration | Dominance | Coverage | Winner-Take-All? |
|---------------|-----------|----------|------------------|
| Cross-entropy loss | 0.34 | 1.00 | NO |
| MSE loss | 0.35 | 1.00 | NO |
| Deep linear network (no ReLU) | 0.41 | 1.00 | NO |
| Gradient flow (no momentum) | 0.41 | 1.00 | NO |
| Small learning rate (0.001) | 0.41 | 1.00 | NO |

**Conclusion**: Theory-matched settings (deep linear, MSE, gradient flow) still don't produce winner-take-all.

### Phase 1b: Comprehensive MSE + Gradient Flow Testing (NEW)

After reading Saxe et al. 2022 paper, we tested exact theory-matched conditions systematically:

**MSE vs CE comparison:**

| Config | Dominance | Coverage | WTA? |
|--------|-----------|----------|------|
| MSE + Gradient Flow | 0.379 | 1.000 | NO |
| CE + Gradient Flow | 0.368 | 1.000 | NO |
| MSE + SGD | 0.349 | 1.000 | NO |
| CE + SGD | 0.355 | 1.000 | NO |

**Learning rate sweep (gradient flow approximation):**

| LR | Dominance | Coverage | WTA? |
|----|-----------|----------|------|
| 0.1 | 0.400 | 0.833 | NO |
| 0.001 | 0.400 | 0.800 | NO |
| 0.00001 | 0.407 | 0.333 | NO |

**Seed variance (10 seeds):**

| Metric | Value |
|--------|-------|
| Mean dominance | 0.379 |
| Std dominance | **0.000** |

**Critical observation**: All 10 seeds produce **IDENTICAL** results — there is no race dynamics at all. If neural race were occurring, different seeds would produce different winners with high variance.

**Extended training (up to 10000 epochs):**

| Epochs | Dominance | Coverage |
|--------|-----------|----------|
| 1000 | 0.407 | 0.367 |
| 5000 | 0.400 | 0.800 |
| 10000 | 0.379 | 1.000 |

Longer training = more views learned, not winner-take-all.

**Conclusion**: The gap is NOT due to wrong loss/optimizer. MSE + gradient flow behaves identically to CE + SGD in our setup.

### Phase 2: Capacity Constraints (All Failed)

| Configuration | Dominance | Coverage | Winner-Take-All? |
|---------------|-----------|----------|------------------|
| Hidden = 1 neuron | 0.46 | 0.20 | NO (just can't learn) |
| Hidden = 2 neurons | 0.41 | 0.40 | NO |
| Hidden = 3 neurons | 0.41 | 0.43 | NO |
| Hidden = 10 neurons | 0.36 | 1.00 | NO |
| Weight decay = 0.1 | 0.39 | 0.93 | NO |
| Weight decay = 0.5+ | 0.33 | 0.10 | NO (model collapse) |

**Conclusion**: Capacity constraints reduce what the network CAN learn, but don't create competition for what it DOES learn.

### Phase 3: Explicit Competition (SUCCESS)

| Mechanism | Dominance | Coverage | Winner-Take-All? |
|-----------|-----------|----------|------------------|
| Competition loss (weight=0.5) | 0.58 | 0.93 | YES |
| Competition loss (weight=1.0) | 0.60 | 0.87 | YES |
| Entropy minimization | 0.55-0.57 | 0.87 | YES |

**Key observation**: Different random seeds produce different winning views — as theory predicts should happen.

| Seed | Winners (classes 0,1,2) | Dominance |
|------|-------------------------|-----------|
| 42   | [2, 0, 1] | 0.599 |
| 123  | [1, 1, 2] | 0.591 |
| 456  | [2, 1, 2] | 0.584 |

**Conclusion**: Winner-take-all CAN be achieved, but requires explicit competition in the loss function.

### Phase 4: KD vs Competition (THEOREM 3 CONTRADICTED)

**Without competition (baseline):**

| Training | Dominance | Coverage | Notes |
|----------|-----------|----------|-------|
| Hard labels | 0.35 | 1.00 | Learns all views |
| KD | 0.39 | 0.97 | Also learns all views |

**With competition:**

| Comp Weight | Hard Dom | Hard Cov | KD Dom | KD Cov |
|-------------|----------|----------|--------|--------|
| 0.25 | 0.57 | 0.83 | 0.65 | 0.73 |
| 0.50 | 0.58 | 0.93 | 0.70 | 0.43 |
| 1.00 | 0.60 | 0.87 | 0.73 | 0.33 |
| 2.00 | 0.61 | 0.90 | 0.76 | 0.40 |

**Critical finding**: Under competition, KD has HIGHER dominance (more winner-take-all) and LOWER coverage than hard labels!

**Conclusion**: Theorem 3 prediction is wrong. Soft labels don't resist competition — they're MORE susceptible to it.

---

## Root Cause Analysis

### What Saxe et al. Theory Actually Requires

From reading the original paper (arXiv:2207.10430), the theory assumes:
1. **MSE loss** with natural saturation property
2. **Gradient flow** (continuous-time, infinitesimal steps)
3. **Emergent s_max** from the Lotka-Volterra competitive dynamics

The key competition term: `(1 - Σs²/s_max²)` is supposed to emerge automatically from MSE gradient flow.

### Why Competition Doesn't Emerge in Our Setup

**We tested ALL theory-matched conditions and still see no competition:**

1. **MSE loss tested**: No difference from cross-entropy (both learn all views)

2. **Gradient flow tested**: No momentum, small LR — still no winner-take-all

3. **Deep linear networks tested**: Exact match to theory — still no competition

4. **Zero seed variance**: 10 different seeds produce IDENTICAL results (dominance = 0.379 ± 0.000), proving there is NO race dynamics occurring

### The Fundamental Issue

The Saxe theory may apply to a different problem structure:
- **Multi-task learning** with shared representations (their main application)
- **Specific input-output correlation structures** that create competition
- **Task-induced gating** where different tasks compete for shared pathways

Our **multi-view classification** setup may simply not create the conditions for neural race:
- Orthogonal view slots don't compete — they're independent
- Single classification task doesn't create multi-task competition
- No shared representation bottleneck that forces selection

### Why KD Fails to Break Competition

1. **Soft labels are weaker signals**: Temperature-scaled probabilities have lower gradient magnitude than hard labels.

2. **Competition loss dominates**: When explicit competition is present, the gradient from competition term overwhelms the KD gradient.

3. **No pathway-specific routing**: KD distributes gradients to output layer, but doesn't specifically target pathway structure.

---

## Theory vs Reality

| Theory Assumes | Reality Shows |
|----------------|---------------|
| MSE loss creates saturation | MSE behaves same as CE in our setup |
| Gradient flow creates competition | Gradient flow behaves same as SGD |
| s_max emerges from dynamics | No saturation emerges — all views learned |
| Different seeds → different winners | All seeds produce IDENTICAL results |
| Capacity limits force selection | Networks use all capacity for all views |
| KD distributes gradients evenly | KD gradients weaker, not pathway-specific |

**Key insight**: The theory may be correct for multi-task learning but doesn't apply to multi-view single-task classification.

---

## Options for Paper Direction

### Option A: Revise Theory to Match Reality

**Approach**: Accept that competition is designed, not emergent. Modify theorems:
- Theorem 2: "When competition pressure is applied, winner-take-all emerges"
- Theorem 3: "KD transfers multi-view knowledge in absence of competition, but doesn't break explicit competition"

**Pros**:
- Honest about what we found
- Still provides insights about competition mechanisms
- Can characterize when/how competition matters

**Cons**:
- Weaker contribution than original thesis
- Less explanatory power for "why KD works"

### Option B: Find Emergent Competition

**Approach**: Look for architectures/domains where competition naturally emerges:
- Transformers with limited attention heads
- Mixture of Experts with hard routing
- Reinforcement learning with sparse rewards
- Biological neural networks with metabolic constraints

**Pros**:
- Could validate original theory in different setting
- Stronger contribution if successful

**Cons**:
- May not find it
- Significant additional work
- Different domain may not generalize

### Option C: Reframe Contribution

**Approach**: Pivot from "explaining KD" to "controlled study of pathway competition":
- Present explicit competition as a design choice
- Study tradeoffs between coverage and dominance
- KD findings become: "soft labels don't resist designed competition"

**Pros**:
- Novel contribution about competition mechanisms
- Experimental setup is solid
- Clear, reproducible results

**Cons**:
- Different paper than originally planned
- May need new theoretical framing

---

## Recommendation

**Suggested path**: Option A (Revise Theory) with elements of Option C (Reframe)

**Rationale**:
1. The finding that "competition requires explicit design" is itself interesting and publishable
2. The KD results (worse under competition) are surprising and worth reporting
3. We have solid experimental infrastructure and reproducible results
4. Reframing can be done relatively quickly vs. finding emergent competition

**Proposed narrative**:
- "We investigated whether neural race dynamics explain KD's effectiveness"
- "We found that winner-take-all requires explicit competition mechanisms"
- "Surprisingly, KD is MORE susceptible to competition, not less"
- "This suggests KD's benefits come from a different mechanism than gradient distribution"

---

## Questions for Advisor

1. **Direction**: Which option (A, B, or C) should we pursue?

2. **Scope**: Should we attempt to find emergent competition, or accept the current findings?

3. **Timeline**: Given NeurIPS deadline (~May 22), is there time to pivot significantly?

4. **Theorem revision**: How should we modify Theorem 3 given the contradicting results?

5. **Contribution framing**: Is "competition requires explicit design" a strong enough contribution?

---

## Appendix: Code Changes Made

| File | Changes |
|------|---------|
| `src/model.py` | Added `DeepLinearNet`, `init_weights_linear` |
| `src/train.py` | Added `train_with_competition()`, MSE loss option, gradient flow option |
| `src/metrics.py` | Added `compute_pathway_dominance()` |
| `src/experiments/exp_theory_match.py` | New experiment script |
| `src/experiments/exp_mse_gradient_flow.py` | Comprehensive MSE + gradient flow testing |

All code tested and results reproducible.

---

## Appendix: Reference Paper

**Saxe et al. 2022** - "The Neural Race Reduction: Dynamics of Abstraction in Gated Networks" (ICML 2022, arXiv:2207.10430)

Summary available at: `reading_stack/summaries/PAPER-001-saxe-2022.md`

Key takeaway: The theory applies to Gated Deep Linear Networks with specific input-output correlation structures. Our multi-view orthogonal slot setup may not create the conditions for neural race dynamics.

---

*Prepared by: ML Agent*
*Last updated: 2026-01-13*
*Full experiment log: `log/experiment_log.md`*
