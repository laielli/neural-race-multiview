# Experimental Findings Summary

**Last Updated**: 2026-01-19
**Status**: Two breakthroughs achieved — (1) hierarchical structure enables 5.3x speedup, (2) forced diverse ensemble with weighting achieves rank > hard labels

---

## Core Validated Findings

### Finding 1: KD Breaks WTA in Flat-Class Settings

| Metric | Hard Labels | KD (τ=3, α=0.7) |
|--------|-------------|-----------------|
| View 0 Contribution | 0.407 | 0.257 |
| Effective Rank | 4.37 | 4.95 |
| Accuracy | 100% | 99.2% |

**Interpretation**: Under hard labels, View 0 captures ~41% of representational weight. Under KD, contributions are more balanced (~20% each). In flat-class settings, soft labels act as "brakes" — they suppress dominant pathways.

### Finding 2: KD Accelerates Learning in Hierarchical Settings ⭐ BREAKTHROUGH

| Metric | Hard Labels | KD (τ=3, α=0.7) |
|--------|-------------|-----------------|
| **Epoch to 90%** | 160 | **30** |
| Final Accuracy | 99.998% | **100%** |
| Effective Rank | 4.74 | 3.75 |
| View 0 Contribution | 0.33 | 0.49 |

**Interpretation**: When there's latent subclass structure, soft labels provide **additional learning signal** — a 5.3x speedup. The teacher's soft labels encode inter-class similarity (17.7% same-superclass confusion) that hard labels cannot provide.

---

## Key Mechanistic Insights

### Insight 1: Soft Labels as "Brakes" (Flat Classes)
In flat-class settings with no latent structure:
- Hard labels provide the learning signal (gradient magnitude)
- Soft labels suppress dominant pathways (differential braking)
- Result: More balanced representations, but slower overall learning

### Insight 2: Soft Labels as "Gas" (Hierarchical Classes) ⭐ NEW
In hierarchical settings with latent subclass structure:
- Hard labels provide coarse task signal ("this is a dog")
- Soft labels provide **additional fine-grained signal** ("this dog looks like a wolf")
- Result: Faster learning (5.3x speedup), teacher's representation transferred to student

---

## 1. Temperature Sweep (FULL RUN) ✓

**Result**: No strong temperature effect found in GDLN setting

| Temperature | Effective Rank | View 0 | Accuracy |
|-------------|----------------|--------|----------|
| Hard labels | 4.371 | 0.407 | 100% |
| τ = 1.0 | 4.947 | 0.258 | 99.2% |
| τ = 5.0 | 4.950 | 0.256 | 99.2% |
| τ = 20.0 | 4.950 | 0.256 | 99.2% |

**Takeaway**: Temperature has marginal effect in our setting. The switch from hard to soft labels matters (4.37 → 4.95), but temperature within KD doesn't (4.947 → 4.950).

**Missing conditions**: Temperature effects may require richer settings where soft labels carry more information.

---

## 2. Alpha Sweep (FULL RUN) ✓

**Result**: Smooth transition with accuracy degradation at high α

| Alpha | Dominance | Eff Rank | Accuracy |
|-------|-----------|----------|----------|
| 0.0 (hard only) | 0.407 | 4.37 | 100.0% |
| 0.3 | 0.350 | 4.67 | 100.0% |
| 0.5 | 0.302 | 4.84 | 99.8% |
| 0.7 | 0.257 | 4.95 | 99.2% |
| 0.9 | 0.224 | 4.99 | 96.3% |
| **1.0 (soft only)** | 0.213 | 5.00 | **86.1%** |

**Key findings**:
1. **Smooth transition**: Dominance decreases monotonically with α (no sharp phase transition)
2. **Accuracy degrades at high α**: Pure soft labels (α=1.0) achieve only 86.1% accuracy
3. **Sweet spot**: α ∈ [0.5, 0.7] provides good balance + high accuracy

**Interpretation**: Hard labels provide essential learning signal. Soft labels alone can eventually learn (86% > random) but are insufficient for full performance. This confirms soft labels act as "brakes" not "gas".

---

## 3. Learning Curves (FULL RUN) ✓

**Result**: KD suppresses ALL pathways, especially dominant ones

### Growth Rates (early phase)
| View | Hard Rate | KD Rate | KD/Hard |
|------|-----------|---------|---------|
| 0 (strong) | 0.000618 | 0.000110 | **0.18** |
| 1 | 0.000159 | 0.000022 | 0.14 |
| 2 | 0.000044 | 0.000006 | 0.13 |
| 3 | 0.000010 | 0.000001 | 0.08 |
| 4 (weak) | 0.000004 | 0.000001 | 0.13 |

### Final Weight Norms
| View | Hard Final | KD Final | KD/Hard |
|------|------------|----------|---------|
| 0 (strong) | 1.339 | 0.510 | **38%** |
| 1 | 0.719 | 0.398 | 55% |
| 2 | 0.474 | 0.364 | 77% |
| 3 | 0.390 | 0.358 | 92% |
| 4 (weak) | 0.368 | 0.358 | **97%** |

### Final Dominance
- Hard labels: View 0 = **40.7%**
- KD: View 0 = **25.7%**

**Mechanistic insight**:

The original theory predicted "KD accelerates weak pathways."

**Actual mechanism**: KD **differentially suppresses** pathways:
- Strong views suppressed to ~38% of hard label growth
- Weak views retain ~97% of hard label growth
- Net effect: More balanced final distribution

**The pattern**: KD doesn't accelerate losers, it suppresses winners. Weak views end up at nearly the same absolute level under both conditions, but strong views are heavily suppressed under KD.

---

## 4. Ensemble Size (Quick Run)

**Result**: Single teacher suffices in GDLN setting

| N Teachers | Student Eff Rank | Accuracy |
|------------|------------------|----------|
| 1 | 4.999 | 96.7% |
| 5 | 4.999 | 96.8% |

**Finding**: Even N=1 teacher produces a balanced student. Ensemble diversity is not critical in our setting.

---

## 5. Student Width (Quick Run)

**Result**: Expected — wider students have better accuracy

| Width | Effective Rank | Accuracy |
|-------|----------------|----------|
| 32 | 4.997 | 94.8% |
| 128 | 4.999 | 98.8% |

---

## 6. SVD Spectrum (Quick Run)

**Result**: KD has flatter spectrum (confirms view contribution results)

---

## 7. Hierarchical Class Structure (FULL RUN) ✓ ⭐ BREAKTHROUGH

**Setup**:
- K_super = 10 superclasses, K_sub = 3 subclasses each (30 fine classes)
- Teacher trained on 30-class problem (with cross-entropy loss)
- Student trained on 10-class problem
- Subclass similarity = 0.5 (subclasses share 50% of prototype with superclass)

**Result**: Soft labels provide 5.3x learning speedup!

| Metric | Hard Labels | KD | Difference |
|--------|-------------|-----|------------|
| **Epoch to 90%** | 160 | 30 | **5.3x faster** |
| Final Accuracy | 99.998% | 100% | +0.002% |
| Effective Rank | 4.74 | 3.75 | -0.99 |
| View 0 Contribution | 0.33 | 0.49 | +0.16 |

**Soft Label Information Content**:
- Avg entropy: 1.59 / 2.30 (max) — meaningful spread, not peaked
- Avg true superclass prob: 55% — teacher is confident but not certain
- **Same-superclass confusion: 17.7%** — teacher encodes that "huskies look like wolves"

**Key Findings**:
1. **5.3x learning speedup**: KD reaches 90% accuracy in 30 epochs vs 160 for hard labels
2. **Perfect final accuracy**: KD achieves 100% vs 99.998% for hard labels
3. **Teacher bias transfer**: Student inherits teacher's view preferences (View 0: 33% → 49%)
4. **Information content**: Soft labels encode inter-class similarity that hard labels cannot

**Mechanism**: The teacher is confused between subclasses within the same superclass. When it sees a "husky" (fine class), it places:
- ~55% probability on the true superclass ("dog")
- ~18% on other subclasses within "dog" (wolves, poodles)
- ~27% on other superclasses

This encodes: "this is a dog, but it looks somewhat like a wolf" — information that hard labels cannot provide!

---

## Summary of Findings

| Finding | Status | Implication |
|---------|--------|-------------|
| KD breaks WTA (flat) | ✓ Confirmed | But due to MSE teachers being balanced |
| KD accelerates (hierarchical) | ✓ **CONFIRMED** | 5.3x speedup with latent structure |
| KD transfers teacher repr | ✓ **CONFIRMED** | Student inherits teacher's view biases |
| CE creates WTA teachers | ✓ **CONFIRMED** | All CE teachers converge to same WTA |
| No natural ensemble diversity | ✓ **CONFIRMED** | V0 std = 0.000 with random seeds |
| Forced diversity + weighting works | ✓ **NEW** | Single-view teachers + inv weights → rank 4.95 > 4.73 |
| Uniform weighting fails | ✓ **NEW** | Confident teachers dominate → more WTA |
| **Balance-speed trade-off** | ✓ **NEW** | Inv weights: best rank (4.95) but slowest (250 ep); uniform: fast (90 ep) but more WTA |
| Temperature effect | ✗ Not found (flat) | May matter in hierarchical settings |
| α phase transition | ✓ Smooth, not sharp | Use α ∈ [0.5, 0.7] |
| Pure soft labels fail (flat) | ✓ 86% accuracy | Hard labels essential for flat tasks |
| KD accelerates weak (flat) | ✗ Rejected | KD suppresses strong in flat settings |
| Soft labels encode structure | ✓ **CONFIRMED** | 17.7% same-superclass confusion |

**Resolution of Allen-Zhu Discrepancy**: Our initial finding that all teachers converge to identical representations was due to not forcing diversity. When we:
1. Force diversity via single-view masking
2. Weight teachers inversely to signal strength

KD achieves **higher rank than hard labels** (4.95 vs 4.73), consistent with Allen-Zhu et al.

---

## Critical Open Question: ANSWERED ✓

### ~~We have NOT found conditions where soft labels provide additional learning signal~~

### ANSWER: Hierarchical Class Structure Creates Acceleration

**Condition found**: When the teacher knows more than the student (finer-grained labels), soft labels provide **additional learning signal** that accelerates learning.

**Mechanism**:
1. Teacher trained on fine-grained task (30 classes)
2. Student trained on coarse task (10 superclasses)
3. Teacher's soft labels encode inter-class similarity within superclasses
4. This is information hard labels CANNOT provide
5. Result: 5.3x learning speedup

**Why it works**:
- In flat-class settings: teacher and student have the same label granularity → soft labels only redistribute existing signal → suppression
- In hierarchical settings: teacher has richer labels → soft labels contain NEW information → acceleration

---

## Paper Narrative

### Complete Story (Final)

"KD's effect on view balance depends on **teacher diversity** and **ensemble weighting**:

1. **Without forced diversity**: All CE teachers converge to identical WTA representations. KD transfers this WTA to the student, resulting in **lower** effective rank than hard labels.

2. **With forced diversity + proper weighting**: When teachers are forced to learn different views (via masking) and weighted inversely to their confidence, KD achieves **higher** effective rank than hard labels.

3. **Hierarchical structure**: When teacher has finer-grained labels than student, soft labels encode inter-class similarity that provides **5.3x learning speedup**, regardless of view balance.

The key insight: KD is a **transfer mechanism**. It transfers whatever the teacher ensemble collectively knows. To get balanced students, you need:
- Diverse teachers (each learning different views)
- Proper weighting (so weak teachers aren't dominated by strong ones)"

### Key Contributions
1. **Transfer mechanism**: KD students inherit teacher ensemble's collective representation
2. **Diversity requirement**: Random seeds don't create diversity; explicit forcing (view masking) is required
3. **Weighting requirement**: Uniform weighting fails; inverse-signal weighting achieves rank > hard labels
4. **Acceleration condition**: KD accelerates when teacher has richer knowledge (5.3x speedup)
5. **Loss function effect**: CE creates WTA teachers; MSE creates balanced but uninformative teachers

### Resolution of Allen-Zhu Discrepancy
Allen-Zhu et al. claim diverse teachers learn different views. Our experiments show:
- **Without intervention**: All CE teachers converge to same solution (no diversity)
- **With forced diversity**: Single-view masking creates verified diverse teachers
- **With proper weighting**: KD achieves higher rank than hard labels (4.95 vs 4.73)

The discrepancy was due to **implicit vs explicit diversity**. Allen-Zhu's setting may have implicit diversity mechanisms (nonlinear networks, harder tasks) that our linear setting lacks. We can recover the expected behavior by explicitly forcing diversity.

---

## 8. Investigation: Why Does KD Have Lower Effective Rank in Hierarchical Settings?

The hierarchical experiment showed a counter-intuitive result: KD students have **lower** effective rank (more WTA) than hard-label students. This contradicts our hypothesis that KD preserves more views.

### Root Cause: Loss Function Determines Teacher Representation

| Teacher Loss | Teacher Rank | Teacher V0 | Soft Labels |
|--------------|--------------|------------|-------------|
| MSE | 4.99 (balanced) | 0.23 | Uniform (useless) |
| Cross-Entropy | 3.41 (WTA) | 0.54 | Informative |

**Trade-off**: To get informative soft labels (needed for hierarchical speedup), we must use cross-entropy. But CE creates WTA in teachers.

### No Ensemble Diversity in Our Setting

Despite using 5 teachers with different random seeds, all converge to **identical** view contributions:

| Teacher | Effective Rank | View 0 Contribution |
|---------|----------------|---------------------|
| Teacher 0 | 3.415 | 0.542 |
| Teacher 1 | 3.418 | 0.541 |
| Teacher 2 | 3.413 | 0.542 |
| Teacher 3 | 3.412 | 0.543 |
| Teacher 4 | 3.415 | 0.542 |

**V0 std = 0.000** — all teachers learn the same representation!

### KD Transfers Teacher Representation to Student

| Student Type | Eff Rank | V0 | Epoch@90% |
|--------------|----------|-----|-----------|
| Hard Labels | 4.73 | 0.33 | 150 |
| KD (CE teachers) | 3.75 | 0.49 | 30 |
| KD (Balanced teachers*) | 3.97 | 0.41 | 30 |

*Balanced teachers created via View-0 dropout during training (rank 4.11, V0 0.27)

**Key finding**: Even with balanced teachers, KD students are less balanced than hard-label students. Any α > 0 pulls student toward teacher's representation.

### Alpha Has Diminishing Effect on Balance

| α | Student Rank | Epoch@90% |
|---|--------------|-----------|
| 0.0 (hard only) | 4.73 | 150 |
| 0.3 | 3.95 | 50 |
| 0.5 | 3.96 | 30 |
| 0.9 | 3.96 | 20 |

Even 30% soft labels dominates the representation. More soft labels = faster learning but similar (lower) rank.

### Corrected Understanding

**Original claim**: "KD breaks WTA"

**Corrected claim**: "KD transfers the teacher's representation to the student"

- Flat-class experiments used MSE teachers (balanced) → KD transferred balance
- Hierarchical experiments used CE teachers (WTA) → KD transferred WTA

KD doesn't inherently break or preserve WTA — it transfers whatever the teacher learned.

### ⚠️ CAVEAT: Contradiction with Allen-Zhu et al.

**Our finding**: All CE teachers with different initializations converge to identical representations (V0 std = 0.000).

**Allen-Zhu et al. finding**: Different teachers learn different "views" or features, and ensemble distillation combines these diverse perspectives.

**Possible explanations for discrepancy** (needs investigation):
1. **Linear vs nonlinear networks**: Our GDLNs may have simpler loss landscapes with single basins
2. **Task complexity**: Our synthetic task may be too easy, with one dominant solution
3. **Network capacity**: Our networks may be too small to support multiple solutions
4. **Data structure**: Our multi-view data with decaying signal strengths creates clear "best" view
5. **Training dynamics**: CE + SGD on our data may have strong implicit bias toward View 0

---

## 9. Forced Diverse Ensemble with Weighted KD ✓ ⭐ RESOLUTION

To test the Allen-Zhu hypothesis properly, we need:
1. **Guaranteed teacher diversity**: Each teacher must learn a different view
2. **Proper weighting**: Weak teachers must be upweighted to have equal influence

### Creating Verified Diverse Teachers

**Method**: Train single-view teachers — each teacher only sees ONE view (others zeroed out)

| Teacher | Sees Only | Accuracy | Dominant View Contribution |
|---------|-----------|----------|---------------------------|
| Teacher 0 | View 0 | 99.9% | V0 = 0.874 |
| Teacher 1 | View 1 | 84.3% | V1 = 0.862 |
| Teacher 2 | View 2 | 35.8% | V2 = 0.564 |
| Teacher 3 | View 3 | 21.2% | V3 = 0.308 |
| Teacher 4 | View 4 | 10.1% | V4 = 0.236 |

**Diagonal dominance verified**: Each teacher has highest contribution on its designated view ✓

### The Weighting Problem

With **uniform weighting**, confident teachers dominate:
- Teachers 0-1 have high accuracy (99.9%, 84.3%)
- Teachers 2-4 have low accuracy (35.8%, 21.2%, 10.1%)
- Uniform average → confident predictions from T0/T1 dominate → student favors V0/V1

### Solution: Inverse Signal Weighting

Weight teachers inversely to their signal strength (or accuracy):
```
weights = [1/signal_strength[m] for m in range(M)]
```

### Results: Weighted Diverse Ensemble

| Method | Eff Rank | V0 | View Contributions |
|--------|----------|-----|-------------------|
| Hard Labels | 4.73 | 0.333 | [0.333, 0.206, 0.163, 0.150, 0.147] |
| KD (uniform weights) | 4.03 | 0.442 | [0.442, 0.260, 0.127, 0.091, 0.080] |
| KD (inv accuracy) | 4.92 | 0.272 | [0.272, 0.206, 0.177, 0.173, 0.172] |
| **KD (inv signal)** | **4.95** | **0.252** | [0.252, 0.205, 0.183, 0.180, 0.179] |

### Key Finding: Weighted Diverse Ensemble KD > Hard Labels ✓

**KD with properly weighted diverse ensemble achieves HIGHER effective rank than hard labels!**
- Hard Labels: rank = 4.73
- KD (inv signal weights): rank = **4.95**

This validates the Allen-Zhu hypothesis when:
1. Teachers are **truly diverse** (each forced to learn different view via masking)
2. Ensemble is **properly weighted** (weak teachers upweighted to compensate)

### Why Uniform Weighting Fails

With uniform weights, KD actually **increases** WTA (rank 4.03 < 4.73) because:
1. Teachers with stronger views produce more confident predictions
2. Confident predictions dominate the averaged soft labels
3. Student learns to focus on the views that produce confident predictions

### Learning Speed Trade-off ⭐ NEW

There's a fundamental trade-off between view balance and learning speed:

| Method | Eff Rank | @90% Acc | @95% Acc |
|--------|----------|----------|----------|
| Hard Labels | 4.73 | 150 | 180 |
| KD Diverse (uniform weights) | 4.03 | **90** | 100 |
| KD Diverse (inv signal weights) | **4.95** | 250 | 290 |

**Key insight**:
- **Uniform weighting**: Fast (90 epochs, 1.7x speedup) but creates MORE WTA (rank 4.03 < 4.73)
- **Inverse signal weighting**: Highest rank (4.95) but SLOWER (250 epochs, 0.6x)
- **Hard labels**: Middle ground (150 epochs, rank 4.73)

**Interpretation**: Confident teachers (those with strong views) provide better learning signal. When you upweight weak teachers to achieve balance, you're trading signal quality for balance — resulting in slower learning.

### Recipe for Diverse Ensemble KD

1. **Force diversity**: Train each teacher on a subset of views (e.g., single-view masking)
2. **Verify diversity**: Check diagonal dominance in view contribution matrix
3. **Choose your priority**:
   - For **speed**: Use uniform weights (accept more WTA)
   - For **balance**: Use inverse signal weights (accept slower learning)
4. **Result**: KD student achieves higher rank than hard-label baseline (with proper weighting)

---

## Practical Guidelines (Final)

### For Learning Speedup (Hierarchical Settings)
1. **Use KD when teacher has richer knowledge than student** — 5.3x speedup observed
2. **α ∈ [0.5, 0.7]** — balance between soft (for signal) and hard (for task)
3. **Accept teacher bias transfer** — speedup comes with inheriting teacher's view preferences

### For View Balance (Breaking WTA)
4. **Don't rely on random seeds for diversity** — all teachers converge to same solution
5. **Force diversity explicitly** — train single-view teachers (each sees only one view)
6. **Verify diversity** — check diagonal dominance in view contribution matrix
7. **Choose weighting based on priority**:
   - **Inverse signal weights**: Best balance (rank 4.95 > hard 4.73) but slower (1.7x)
   - **Uniform weights**: Faster than hard labels (1.7x speedup) but more WTA (rank 4.03)
8. **Recognize the trade-off**: You cannot get both maximum balance AND maximum speed with diverse ensemble KD

### General
9. **Measure soft label entropy** — higher entropy (but not uniform) = more information
10. **CE loss creates WTA teachers** — needed for informative soft labels, but creates bias
11. **MSE loss creates balanced teachers** — but produces uniform (useless) soft labels
