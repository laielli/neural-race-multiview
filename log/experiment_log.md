# Experiment Status Tracker

**Project**: KDMech - Mechanistic Account of Knowledge Distillation

**Last Updated**: 2026-01-13

---

## Quick Status

| Category | Total | Done | In Progress | Not Started |
|----------|-------|------|-------------|-------------|
| Infrastructure (P0) | 4 | 4 | 0 | 0 |
| Theorem 2 Experiments | 4 | 2 | 0 | 2 |
| Theorem 3 Experiments | 4 | 3 | 0 | 1 |
| Theorem 1 Experiments | 2 | 0 | 0 | 2 |
| **Follow-up Investigations** | 3 | 3 | 0 | 0 |
| **Competition Mechanisms** | 4 | 4 | 0 | 0 |

**Overall**: 16/21 complete

**Status**: AWAITING DIRECTION - Theory-experiment gap requires paper pivot decision

---

## CRITICAL FINDING

**Winner-take-all dynamics are NOT observed** in any tested configuration:
- Orthogonal slots (baseline): Dominance = 0.34
- Competing views (shared dimensions): Dominance = 0.35
- Bottleneck architectures (down to 1 neuron): Dominance = 0.34
- Mixture of Experts: Dominance = 0.34
- Asymmetric initialization (100x bias): Dominance erodes from 0.60 → 0.35

See reports in `exchange/to_advisor/` for details.

---

## Infrastructure (P0) — COMPLETE

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Multi-view data generator | `src/data.py` | Done | `MultiViewDataset` + `CompetingViewDataset` |
| Network architecture | `src/model.py` | Done | `MultiViewNet` + pathway measurement |
| Measurement functions | `src/metrics.py` | Done | Coverage, pathway strength, etc. |
| Training loops | `src/train.py` | Done | Hard labels + KD |

All infrastructure verified and tested.

---

## Theorem 2 Experiments — Race Dynamics

### Exp 2.1: Single-View Convergence (P1) — DONE

**Question**: Does hard label training lead to C(f) ≈ 1/M?

| Item | Status |
|------|--------|
| Script | `experiments/exp_2_1_single_view.py` | Done |
| Execution | Done |
| Results | **MISMATCH WITH THEORY** |

**Expected**: Mean coverage ≈ 0.33 ± 0.05

**Actual**: Coverage = 1.000 ± 0.000 (ALL views learned)

---

### Exp 2.3: Race Dynamics Visualization (P1) — DONE

**Question**: Do pathway strengths show winner-take-all pattern?

| Item | Status |
|------|--------|
| Script | `experiments/exp_2_3_race_dynamics.py` | Done |
| Figure | `results/figures/pathway_evolution_hard.png` | Done |

**Expected**: One pathway grows, others decay

**Actual**: ALL pathways grow in parallel, dominance stays ~0.34

---

### Exp 2.2: View Diversity Across Seeds (P2) — Not Started

### Exp 2.4: Winner Prediction (P2) — Not Started

---

## Theorem 3 Experiments — KD Mechanism

### Exp 3.1: KD Coverage Transfer (P1) — DONE

**Question**: Does KD from ensemble transfer coverage to student?

| Item | Status |
|------|--------|
| Script | `experiments/exp_3_1_kd_coverage.py` | Done |
| Execution | Done |

**Expected**:
- C(student_hard) ≈ 0.33
- C(student_KD) ≈ C(ensemble)

**Actual**:
- C(student_hard) = 1.000 (not 0.33!)
- C(student_KD) = 1.000
- C(ensemble) = 1.000

All approaches learn all views.

---

### Exp 3.3: Pathway Evolution Under KD (P1) — DONE

**Question**: Do multiple pathways survive under KD?

| Item | Status |
|------|--------|
| Script | `experiments/exp_3_3_pathway_evolution.py` | Done |
| Figure | `results/figures/pathway_comparison.png` | Done |

**Expected**: Multiple pathways > 0 at convergence (under KD)

**Actual**: Multiple pathways survive under BOTH hard and KD (no difference)

---

### Exp 3.2: Gradient Distribution Analysis (P2) — Not Started

### Exp 3.4: Coverage Inheritance (P2) — Not Started

---

## Follow-up Investigations — All DONE

### Option A: Competing Views — DONE

**Question**: Does shared dimensional space create winner-take-all?

**Result**: NO. Dominance = 0.35, same as baseline.

Report: `exchange/to_advisor/2026-01-10_competing_views_analysis.md`

---

### Option B: Constrained Bottleneck — DONE

**Question**: Does capacity constraint force view selection?

**Tested**: Bottleneck MLP, Linear compression, Low-rank, Top-K sparse, MoE

**Result**: NO winner-take-all. Internal specialization occurs (0% neuron overlap), but output dominance stays ~0.34.

Report: `exchange/to_advisor/2026-01-10_bottleneck_analysis.md`

---

### Option C: Asymmetric Initialization — DONE

**Question**: Does initial advantage grow into winner-take-all?

**Result**: NO. SGD training actively ERODES initial advantages. Dominance goes from 0.60 → 0.35 regardless of bias strength (tested 2x to 100x).

Report: `exchange/to_advisor/2026-01-10_asymmetric_init_analysis.md`

---

## Results Summary

### Completed Experiments

| Experiment | Date | Key Result | Matches Theory? |
|------------|------|------------|-----------------|
| 2.1 Single-view | 2026-01-10 | Coverage = 1.0 | NO (expected 0.33) |
| 2.3 Race dynamics | 2026-01-10 | All grow equally | NO (expected winner-take-all) |
| 3.1 KD coverage | 2026-01-10 | All methods = 1.0 | NO |
| 3.3 Pathway evolution | 2026-01-10 | All survive | Inconclusive |
| Competing views | 2026-01-10 | Dominance = 0.35 | NO |
| Bottleneck | 2026-01-10 | Dominance = 0.34 | NO |
| Asymmetric init | 2026-01-10 | Advantage erodes | NO |

### Figures Generated

| Figure | Experiment | Path | Status |
|--------|------------|------|--------|
| Race dynamics (hard) | 2.3 | `results/figures/pathway_evolution_hard.png` | Done |
| Pathway comparison | 3.3 | `results/figures/pathway_comparison.png` | Done |
| Competing views | Follow-up | `results/figures/competing_views_comparison.png` | Done |

### Key Unexpected Findings

| Date | Experiment | Finding | Impact |
|------|------------|---------|--------|
| 2026-01-10 | 2.1 | Coverage = 1.0, not 0.33 | Theory prediction not validated |
| 2026-01-10 | 2.3 | No winner-take-all | Race dynamics not observed |
| 2026-01-10 | Bottleneck | Internal specialization but equal output | Selection happens internally only |
| 2026-01-10 | Asymmetric | Initial advantage erodes | SGD equalizes pathways |

---

## BREAKTHROUGH: Explicit Competition Loss (2026-01-13)

### Finding

Winner-take-all dynamics **CAN be achieved** with explicit competition loss!

### What Worked

Added `train_with_competition()` function that includes a loss term encouraging pathway dominance:
```
competition_loss = -sum over classes of (max_pathway_strength / sum_pathway_strengths)
```

### Results

| Seed | Winners (classes 0,1,2) | Dominance | Accuracy |
|------|-------------------------|-----------|----------|
| 42   | [2, 0, 1]               | 0.599     | 78.5%    |
| 123  | [1, 1, 2]               | 0.591     | 79.1%    |
| 456  | [2, 1, 2]               | 0.584     | 79.7%    |
| 789  | [2, 1, 0]               | 0.585     | 78.4%    |
| 1000 | [0, 1, 2]               | 0.594     | 80.3%    |

**Key observations:**
- Different seeds → different winning views (as theory predicts!)
- Dominance ≈ 0.58-0.60 (vs 0.34 baseline) - clear winner-take-all
- Different classes have different winners - not a global collapse
- Accuracy drops slightly (80% vs 100%) - expected tradeoff

### Root Cause of Original Problem

Standard loss functions (cross-entropy, MSE) don't have selection pressure for winner-take-all:
1. CE is satisfied once correct class has highest probability
2. MSE drives outputs toward one-hot targets regardless of pathway structure
3. SGD with momentum actively equalizes pathways

The theory's race dynamics equation assumes competition emerges from saturation (`s_max`), but in discrete SGD this doesn't occur naturally.

### New Code

- `src/model.py`: Added `DeepLinearNet` (linear network matching theory)
- `src/train.py`: Added `train_with_competition()` and `loss_type='mse'` option
- `src/metrics.py`: Added `compute_pathway_dominance()`
- `src/experiments/exp_theory_match.py`: Theory-matched experiment script

### Implications

1. **Theory validation**: Race dynamics CAN produce winner-take-all when competition is enforced
2. **Paper narrative**: May need to discuss when/how competition naturally emerges vs requires explicit enforcement
3. **Next steps**: Test if KD can break competition loss dynamics (Theorem 3)

---

## Theorem 3 Testing: KD and Winner-Take-All (2026-01-13)

### Experiment: Does KD break winner-take-all dynamics?

Tested whether KD from full-coverage teachers can prevent winner-take-all.

### Results Summary

**Without competition loss (baseline behavior):**

| Condition | Dominance | Coverage | Notes |
|-----------|-----------|----------|-------|
| Hard labels | 0.345 | 1.000 | Learns all views |
| KD | 0.389 | 0.967 | Also learns all views |

**With competition loss at different strengths:**

| Comp Weight | Hard Dom | Hard Cov | KD Dom | KD Cov | KD Resists? |
|-------------|----------|----------|--------|--------|-------------|
| 0.00 | 0.371 | 1.000 | 0.398 | 0.967 | NO |
| 0.25 | 0.572 | 0.833 | 0.651 | 0.733 | NO |
| 0.50 | 0.582 | 0.933 | 0.698 | 0.433 | NO |
| 1.00 | 0.595 | 0.867 | 0.728 | 0.333 | NO |
| 2.00 | 0.613 | 0.900 | 0.756 | 0.400 | NO |

### Key Findings

1. **Without competition**: KD successfully transfers multi-view knowledge (coverage 96.7%)
   - ✓ Soft labels encode information about all views
   - ✓ Student learns from teacher's full coverage

2. **With competition**: KD does NOT resist winner-take-all better than hard labels
   - ✗ KD shows HIGHER dominance than hard labels
   - ✗ KD shows LOWER coverage than hard labels
   - The explicit competition loss overrides KD's gradient distribution

### Interpretation

**Partial Theorem 3 validation:**
- When there's no competition pressure, KD transfers multi-view knowledge ✓
- When explicit competition is added, KD doesn't break it ✗

**Why this might differ from theory:**
1. Our competition is explicit (loss term), not emergent (saturation)
2. The theory assumes competition from capacity constraints (s_max), not external loss
3. KL divergence loss may interact poorly with competition term

### Paper Implications

The narrative should distinguish between:
1. **Emergent competition** (from saturation/capacity) - theory's assumption
2. **Explicit competition** (from loss term) - our experimental proxy

KD's ability to break winner-take-all may depend on HOW competition arises.
Additional experiments needed to test natural competition scenarios.

---

## Emergent Competition Experiments (2026-01-13)

### Goal
Create winner-take-all dynamics through capacity constraints rather than explicit competition loss.

### Approaches Tested

**1. Hidden Layer Size (Capacity Bottleneck)**

| Hidden | Linear Dom | Linear Cov | ReLU Dom | ReLU Cov |
|--------|------------|------------|----------|----------|
| 1 | 0.460 | 0.200 | 0.355 | 0.167 |
| 2 | 0.409 | 0.400 | 0.337 | 0.133 |
| 3 | 0.412 | 0.433 | 0.349 | 0.233 |
| 10 | 0.357 | 1.000 | 0.361 | 0.833 |
| 50 | 0.347 | 1.000 | 0.364 | 1.000 |

**Result**: Small capacity reduces coverage but doesn't create winner-take-all (dominance stays ~0.35-0.46)

**2. Weight Decay (L2 Regularization)**

| Weight Decay | Dominance | Coverage | Accuracy |
|--------------|-----------|----------|----------|
| 0.0 | 0.347 | 1.000 | 99.8% |
| 0.1 | 0.386 | 0.933 | 92.6% |
| 0.5+ | 0.333 | 0.100 | 10% |

**Result**: High weight decay collapses to random; doesn't create winner-take-all

**3. Pathway Budget Constraints**
- Per-class budget: Σ_m ||R_{y,m}||² ≤ s_max²
- Forces total pathway strength to stay within budget

**Result**: Budget is distributed equally among pathways, not winner-take-all

**4. Entropy-Based Competition** ✓ SUCCESS

| Temp | Entropy Weight | Dominance | Coverage | WTA? |
|------|----------------|-----------|----------|------|
| 0.1 | 5.0 | 0.498 | 1.000 | NO |
| 0.5 | 1.0 | 0.553 | 0.867 | YES |
| 0.5 | 5.0 | 0.555 | 0.867 | YES |
| 1.0 | 5.0 | 0.568 | 0.867 | YES |

**Result**: Entropy minimization creates winner-take-all!

### KD vs Entropy Competition

| Condition | Dominance | Coverage |
|-----------|-----------|----------|
| Hard labels + entropy | 0.566 | 0.867 |
| KD + entropy | 0.732 | 0.433 |

**Result**: KD performs WORSE under entropy competition (higher dominance, lower coverage)

### Key Insight

**Winner-take-all requires explicit competition pressure** in our setup:
- Capacity constraints (small hidden, weight decay) don't create WTA
- Entropy minimization creates WTA but isn't "emergent" - it's a loss term
- The theory's saturation mechanism (s_max) doesn't emerge naturally from discrete SGD

**KD consistently fails to break competition**:
- Under explicit competition loss: KD worse
- Under entropy competition: KD worse
- The soft label gradient signal is weaker than hard labels, making KD MORE susceptible to competition

### Implications for Paper

1. **Competition is not emergent** in our setup - requires explicit mechanism
2. **KD doesn't break competition** - contradicts Theorem 3 prediction
3. **May need to revise theory** to account for:
   - Discrete SGD vs gradient flow differences
   - When/how competition naturally emerges
   - Why KD soft labels don't distribute gradients as predicted

---

## MSE + Gradient Flow Experiment (2026-01-13)

### Goal

Test if Saxe theory-matched conditions produce emergent winner-take-all:
- MSE loss (not cross-entropy)
- Gradient flow (no momentum)
- Deep linear networks
- Small initialization

### Results

**TEST 1: MSE vs Cross-Entropy**

| Config | Dominance | Coverage | WTA? |
|--------|-----------|----------|------|
| MSE + GradFlow | 0.379 | 1.000 | NO |
| CE + GradFlow | 0.368 | 1.000 | NO |
| MSE + SGD | 0.349 | 1.000 | NO |
| CE + SGD | 0.355 | 1.000 | NO |

**No difference between MSE and CE** - both learn all views equally.

**TEST 2: Learning Rate Sweep**

| LR | Dominance | Coverage | WTA? |
|----|-----------|----------|------|
| 0.1 | 0.400 | 0.833 | NO |
| 0.01 | 0.400 | 0.800 | NO |
| 0.001 | 0.400 | 0.800 | NO |
| 0.0001 | 0.400 | 0.800 | NO |
| 0.00001 | 0.407 | 0.333 | NO |

**Smaller LR doesn't help** - just slower learning.

**TEST 3: Seed Variance**

| Metric | Value |
|--------|-------|
| Mean dominance | 0.379 |
| Std dominance | 0.000 |
| Min | 0.379 |
| Max | 0.379 |

**All seeds produce IDENTICAL results** - no race dynamics at all!

**TEST 4: Extended Training (10000 epochs)**

| Epochs | Dominance | Coverage |
|--------|-----------|----------|
| 100 | 0.408 | 0.233 |
| 1000 | 0.407 | 0.367 |
| 5000 | 0.400 | 0.800 |
| 10000 | 0.379 | 1.000 |

**Longer training = more views learned**, not winner-take-all.

### Key Finding

**Even under theory-matched conditions (MSE + gradient flow + deep linear + small init), winner-take-all does NOT emerge.**

The Saxe theory may require additional conditions not captured in our setup:
1. Specific input-output correlation structure
2. Task-specific gating patterns
3. Multi-task learning context (not just multi-view classification)

### Implication

The gap between theory and experiment is NOT due to:
- Wrong loss function (MSE doesn't help)
- Wrong optimizer (gradient flow doesn't help)
- Wrong architecture (deep linear doesn't help)

The gap may be more fundamental - the theory's assumptions about when competition emerges may not apply to our multi-view classification setup.

---

## Final Conclusions (2026-01-13)

### Summary of Experimental Findings

| Finding | Implication |
|---------|-------------|
| Standard training (CE/MSE) learns all views | No emergent competition |
| Explicit competition loss creates WTA | Competition requires loss term |
| Capacity constraints don't create WTA | Saturation mechanism doesn't emerge |
| KD doesn't break competition | Contradicts Theorem 3 prediction |
| KD performs WORSE under competition | Soft labels weaker than hard labels |

### Theory-Experiment Gap

The theory assumes:
1. **Saturation mechanism** (s_max) creates competition naturally
2. **Gradient flow** dynamics where pathways compete continuously
3. **KD distributes gradients** to prevent winner-take-all

Our experiments show:
1. **No emergent saturation** - capacity constraints don't create WTA
2. **Discrete SGD** actively equalizes pathways (asymmetric init erodes)
3. **KD soft labels are weaker** - more susceptible to competition, not less

### Root Cause Analysis

The fundamental issue is that **standard neural network training does not naturally produce winner-take-all dynamics**:

1. **Cross-entropy loss**: Satisfied once correct class has highest probability, regardless of pathway structure
2. **MSE loss**: Drives outputs toward targets equally across all pathways
3. **SGD with momentum**: Actively equalizes pathway contributions over time
4. **Gradient flow approximation**: Poor - discrete SGD behaves differently from continuous gradient flow

### Implications for Paper

**Option A: Revise Theory**
- Acknowledge that competition requires explicit enforcement
- Focus on "when competition is present" rather than "competition emerges"
- KD's role is different from predicted: transfers coverage without competition, but doesn't break competition

**Option B: Find True Emergent Competition**
- Investigate other architectures (attention, transformers)
- Look for domains where competition emerges naturally
- Consider biological neural networks where resources are truly limited

**Option C: Reframe Contribution**
- Shift focus from "explaining KD" to "controlled study of competition"
- Present explicit competition as a design choice, not emergent property
- KD findings become: soft labels don't resist competition

---

## Issues & Blockers

| Issue | Severity | Status | Resolution |
|-------|----------|--------|------------|
| Theory predictions don't match experiments | HIGH | **REQUIRES THEORY REVISION** | Competition not emergent; explicit loss works |
| Winner-take-all not observed naturally | HIGH | **RESOLVED (QUALIFIED)** | Requires explicit competition mechanism |
| KD doesn't break competition | HIGH | **NEW FINDING** | Contradicts Theorem 3 - may need revision |
| Coverage metric may need revision | MEDIUM | RESOLVED | Use threshold=0.1 or dominance metric |

---

## Next Actions

### Awaiting Advisor Guidance on Paper Direction

**Critical Decision Required**:
The experiments reveal a fundamental gap between theory and practice. Three options:

1. **Revise theory** to match experimental reality (competition is designed, not emergent)
2. **Find different experimental setup** where competition emerges naturally
3. **Reframe the paper contribution** around explicit competition mechanisms

### If Revising Theory
- [ ] Update theorem statements to condition on explicit competition
- [ ] Revise KD mechanism explanation (Theorem 3 needs modification)
- [ ] Document gap between gradient flow theory and discrete SGD practice

### If Seeking Emergent Competition
- [ ] Try transformer architectures with limited attention heads
- [ ] Investigate dropout as implicit competition
- [ ] Look at reinforcement learning sparse reward settings

### Remaining Experiments (Lower Priority)
- [ ] Exp 2.2 (view diversity) - still valid
- [ ] Exp 2.4 (winner prediction) - depends on theory revision
- [ ] Exp 3.2 (gradient analysis) - may explain KD failure
- [ ] Exp 3.4 (coverage inheritance) - still valid
- [ ] Theorem 1 experiments - blocked on theory

---

## Reference

### Hyperparameters (Default)

| Parameter | Value | Notes |
|-----------|-------|-------|
| K (classes) | 10 | |
| M (views/class) | 3 | |
| d_view | 50 | Total d = 150 |
| view_prob | 0.5 | Balanced |
| noise_std | 0.1 | Low noise |
| hidden | 200 | Network width |
| epochs | 100 | Training |
| lr | 0.01 | SGD with momentum=0.9 |
| batch_size | 128 | |
| temperature | 4.0 | KD temperature |

### Reports Sent to Advisor

1. `2026-01-10_p1_results_summary.md` - Initial P1 results (mismatch with theory)
2. `2026-01-10_competing_views_analysis.md` - Option A results
3. `2026-01-10_bottleneck_analysis.md` - Option B results
4. `2026-01-10_asymmetric_init_analysis.md` - Option C results

---

*Last updated: 2026-01-13*
