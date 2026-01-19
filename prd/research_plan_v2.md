# Research Plan v2: What Can GDLNs Teach Us About Knowledge Distillation?

**Status**: Core Hypothesis Validated
**Last Updated**: 2026-01-19
**Target**: NeurIPS 2026

---

## Validated Results

**Experiment**: `exp_kd_breaks_wta.py`

| Metric | Hard Labels | KD | Change |
|--------|-------------|-----|--------|
| View 0 Contribution | 0.407 ± 0.003 | 0.257 ± 0.003 | **37% reduction** |
| Effective Rank | 4.37 | 4.95 | **More balanced** |
| Accuracy | 100% | 99.2% | Both high |

**Conclusion**: KD breaks winner-take-all dynamics while maintaining task performance.

---

## New Research Direction

The validated finding ("race winners = learned views" + "KD breaks WTA") is *one* insight from the GDLN framework. The broader question:

> **What else can the GDLN framework teach us about Knowledge Distillation?**

---

## Key Theoretical Predictions to Test

### 1. Temperature's Mechanistic Role

**Theory**: Soft labels distribute gradient proportional to teacher's confidence per pathway. Higher τ → more uniform distribution → more balanced pathways.

**Experiment**: Sweep τ ∈ {1.0, 2.0, 3.0, 5.0, 10.0}, measure effective rank vs τ.

**Prediction**: Effective rank increases monotonically with τ (up to saturation).

### 2. Alpha (Soft Label Weight) Phase Transition

**Theory**: The gradient has two components:
- External (KD): α · teacher_signal
- Self-reinforcing (hard): (1-α) · s_m · competition_term

**Experiment**: Sweep α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, measure dominance.

**Prediction**: Below critical α*, KD fails to break WTA (hard label term dominates).

### 3. Learning Speed Acceleration

**Theory**: External gradient term should accelerate slow pathways:
- Hard: ds_m/dt = σ_1(Σ_m) · s_m  (rich get richer)
- KD: ds_m/dt = α_m(teacher) + γ · σ_1(Σ_m) · s_m  (constant boost + rich get richer)

**Experiment**: Track per-pathway strength curves. Compare hard vs KD.

**Prediction**: Weak pathways grow faster under KD (constant term matters more when s_m small).

### 4. Minimum Ensemble Diversity

**Theory**: KD from single teacher ≈ hard labels (teacher learned one view). Multi-teacher ensemble provides gradient for multiple views.

**Experiment**: Train student from N ∈ {1, 2, 3, 5, 10} teachers.

**Prediction**: Effective rank of student grows with N, saturates at teacher ensemble coverage.

### 5. Student Capacity Requirements

**Theory**: Narrower student has fewer pathways to distribute learning across. Must trade off pathway breadth vs pathway depth.

**Experiment**: Fix teacher (wide), vary student width ∈ {32, 64, 128, 256}.

**Prediction**: Narrow students can't inherit multi-view knowledge (capacity bottleneck).

### 6. SVD Spectrum as Knowledge Diagnostic

**Theory**: Effective rank = exp(entropy of normalized singular values). Higher effective rank = more balanced "knowledge" across modes.

**Experiment**: Compare SVD spectra: hard labels vs KD vs multi-task.

**Prediction**: SVD spectrum of KD models is "flatter" (more uniform).

---

## Experiments to Implement

| Experiment | File | Priority |
|------------|------|----------|
| Temperature sweep | `exp_temperature_sweep.py` | 1 |
| Alpha sweep | `exp_alpha_sweep.py` | 2 |
| Learning curves | `exp_learning_curves.py` | 3 |
| Ensemble size | `exp_ensemble_size.py` | 4 |
| Student width | `exp_student_width.py` | 5 |
| SVD spectrum | `exp_svd_spectrum.py` | 6 |

---

## Success Criteria

| Experiment | Passes If |
|------------|-----------|
| Temperature | Effective rank increases with τ (r > 0.8 correlation) |
| Alpha | Clear phase transition (dominance jumps at some α*) |
| Learning curves | Weak pathways grow faster under KD |
| Ensemble size | Student rank ≈ ensemble coverage (within 0.1) |
| Student width | Narrow students (32) have lower rank than wide (256) |
| SVD spectrum | KD spectrum flatter than hard (entropy diff > 0.5) |

---

## Expected Paper Contributions

1. **Winner prediction formula** (validated): A_{y,m} = σ_1(Σ_{y,m}) · s_{y,m}(0)
2. **KD breaks WTA mechanism** (validated): Soft labels provide external gradient
3. **Temperature role** (to test): τ controls gradient distribution breadth
4. **Critical α threshold** (to test): Phase transition for when KD works
5. **Learning speed effect** (to test): KD accelerates slow pathways
6. **Ensemble size requirements** (to test): Minimum N for coverage transfer
7. **Capacity bottleneck** (to test): Student width limits knowledge inheritance
8. **SVD diagnostic** (to test): Effective rank as knowledge breadth measure

---

## Paper Positioning

> "A Mechanistic Account of Knowledge Distillation Through the Lens of Neural Race Dynamics"

**Contributions**:
- **Theory**: Gradient decomposition into external + self-reinforcing terms
- **Predictions**: Temperature, alpha, ensemble size, capacity effects
- **Diagnostics**: SVD spectrum as knowledge breadth measure
- **Practice**: Guidelines for KD hyperparameter selection
