# Experiment Status Tracker

**Project**: KDMech - Mechanistic Account of Knowledge Distillation

**Last Updated**: 2026-01-10

---

## Quick Status

| Category | Total | Done | In Progress | Not Started |
|----------|-------|------|-------------|-------------|
| Infrastructure (P0) | 4 | 4 | 0 | 0 |
| Theorem 2 Experiments | 4 | 2 | 0 | 2 |
| Theorem 3 Experiments | 4 | 2 | 0 | 2 |
| Theorem 1 Experiments | 2 | 0 | 0 | 2 |
| **Follow-up Investigations** | 3 | 3 | 0 | 0 |

**Overall**: 11/17 complete

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

## Issues & Blockers

| Issue | Severity | Status | Resolution |
|-------|----------|--------|------------|
| Theory predictions don't match experiments | HIGH | OPEN | Awaiting advisor guidance |
| Winner-take-all not observed | HIGH | OPEN | Multiple approaches tried, none work |
| Coverage metric may need revision | MEDIUM | OPEN | Consider internal metrics? |

---

## Next Actions

### Awaiting Advisor Guidance
- Should we add explicit selection mechanism to loss?
- Should we pivot to internal pathway metrics?
- Should we test KD with biased teachers?
- Should we pause experiments until theory is refined?

### If Proceeding
- [ ] Exp 2.2 (view diversity)
- [ ] Exp 2.4 (winner prediction)
- [ ] Exp 3.2 (gradient analysis)
- [ ] Exp 3.4 (coverage inheritance)
- [ ] Theorem 1 experiments

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

*Last updated: 2026-01-10*
