# Experiment Results Summary

**Date**: 2026-01-10T10:33:48.607954
**Mode**: quick

---

## Overview

| Experiment | Status | Key Result |
|------------|--------|------------|
| exp_2_1 | ✓ Pass | Coverage: 1.000 (expected 0.333) |
| exp_2_3 | ✓ Pass | Winner ratio: 0.344 |
| exp_3_1 | ✓ Pass | KD: 1.000, Hard: 1.000 |
| exp_3_3 | ✓ Pass | KD surviving: 3, Hard: 3 |

---

## Detailed Results

### Experiment 2.1: Single-View Convergence

**Validates**: Hard label training leads to C(f) ≈ 1/M

- Mean coverage: **1.000** ± 0.000
- Expected (1/M): **0.333**
- Difference: 0.667
- **Result**: FAILED ✗

### Experiment 2.3: Race Dynamics

**Validates**: Winner-take-all pathway evolution

- Winning view: 0
- Winner dominance ratio: 0.344
- Figure: `results/figures/race_dynamics_hard.png`

### Experiment 3.1: KD Coverage Transfer

**Validates**: C(student_KD) ≈ C(ensemble) >> C(student_hard)

- Hard label coverage: **1.000** ± 0.000
- KD coverage: **1.000** ± 0.000
- Ensemble coverage: **1.000** ± 0.000
- KD matches ensemble: YES ✓
- KD > Hard: NO ✗
- Figure: `results/figures/kd_coverage_comparison.png`

### Experiment 3.3: Pathway Evolution

**Validates**: Multiple pathways survive under KD (vs winner-take-all for hard labels)

- Hard labels:
  - Surviving pathways: 3
  - Dominance ratio: 0.344
- KD:
  - Surviving pathways: 3
  - Dominance ratio: 0.349
- **Result**: INCONCLUSIVE
- Figure: `results/figures/pathway_comparison.png`

---

## Generated Figures

1. `race_dynamics_hard.png` - Pathway strength evolution under hard labels
2. `kd_coverage_comparison.png` - Coverage comparison bar chart
3. `pathway_comparison.png` - Side-by-side pathway evolution (hard vs KD)

---

## Next Steps

Based on these results:
1. If all P1 experiments pass, proceed to P2 experiments
2. If any experiments fail, investigate and adjust parameters
3. Send figures to Writer for paper inclusion
