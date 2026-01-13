# Writing Status Tracker

**Paper**: "Unifying Neural Race Reduction with Multi-View Theory: A Mechanistic Account of Knowledge Distillation"

**Target**: NeurIPS 2026 (Deadline: ~May 22, 2026)

**Last Updated**: 2026-01-10

---

## Quick Status

| Section | Status | Words | Figures | Tables |
|---------|--------|-------|---------|--------|
| Abstract | Not Started | 0/150 | — | — |
| 1. Introduction | Not Started | 0/800 | 0 | 0 |
| 2. Background | Not Started | 0/500 | 0 | 0 |
| 3. Theorem 1 | Not Started | 0/700 | 0 | 0 |
| 4. Theorem 2 | Not Started | 0/700 | 0/1 | 0 |
| 5. Theorem 3 | Not Started | 0/700 | 0 | 0 |
| 6. Experiments | Not Started | 0/900 | 0/4 | 0/3 |
| 7. Discussion | Not Started | 0/300 | 0 | 0 |
| **Total** | **0%** | **0/4750** | **0/5** | **0/3** |

**Draft Version**: Not started

---

## Section Details

### Abstract

| Item | Status | Notes |
|------|--------|-------|
| File | `paper/sections/abstract.tex` | Not created |
| Draft | Not Started | |
| Word count | 0/150 | Target: ~150 words |

**Structure**:
- [ ] Gap statement (1-2 sentences)
- [ ] Our approach (1-2 sentences)
- [ ] Key results (2-3 sentences)
- [ ] Implications (1 sentence)

**Key phrases to include**:
- "no existing theory predicts which view will be learned"
- "neural race dynamics provide this mechanism"
- "knowledge distillation circumvents the race"

---

### 1. Introduction

| Item | Status | Notes |
|------|--------|-------|
| File | `paper/sections/introduction.tex` | Not created |
| Draft | Not Started | |
| Word count | 0/800 | ~1.5 pages |

**Structure**:
- [ ] Opening hook — KD puzzle
- [ ] Multi-view theory background + gap
- [ ] Our contribution (4 bullet points)
- [ ] Paper roadmap

**Contribution bullets**:
1. [ ] View-Pathway Correspondence (Theorem 1)
2. [ ] Predictive Theory for View Selection (Theorem 2)
3. [ ] KD Mechanism (Theorem 3)
4. [ ] Experimental Validation

---

### 2. Background

| Item | Status | Notes |
|------|--------|-------|
| File | `paper/sections/background.tex` | Not created |
| Draft | Not Started | |
| Word count | 0/500 | ~1 page |

**Structure**:
- [ ] 2.1 Multi-View Theory (~250 words)
  - [ ] Define multi-view data
  - [ ] Key results from Allen-Zhu & Li
  - [ ] Open question: "which view wins?"
- [ ] 2.2 Neural Race Reduction (~250 words)
  - [ ] Define GDLN
  - [ ] Key results from Saxe et al.
  - [ ] Note: not previously connected to multi-view

---

### 3. Setup & Theorem 1

| Item | Status | Notes |
|------|--------|-------|
| File | `paper/sections/theorem1.tex` | Not created |
| Draft | Not Started | |
| Word count | 0/700 | ~1.5 pages |

**Structure**:
- [ ] 3.1 Problem Setup (~300 words)
  - [ ] Multi-view distribution definition
  - [ ] GDLN architecture
  - [ ] Assumptions A1-A5
- [ ] 3.2 Theorem 1 (~400 words)
  - [ ] Theorem box (informal statement)
  - [ ] Intuition
  - [ ] Proof sketch reference

**Theorem box**:
```
[PLACEHOLDER - THEOREM 1]
View-Pathway Correspondence
f(x) = Σ R_{y,m}(f) + error
```

---

### 4. Theorem 2

| Item | Status | Notes |
|------|--------|-------|
| File | `paper/sections/theorem2.tex` | Not created |
| Draft | Not Started | |
| Word count | 0/700 | ~1.5 pages |
| Figures | 0/1 | Race dynamics visualization |

**Structure**:
- [ ] 4.1 Pathway Strength Dynamics (~250 words)
  - [ ] Define s_{y,m}(t)
  - [ ] Dynamics equation
- [ ] 4.2 Winner Prediction (~250 words)
  - [ ] Initial advantage formula
  - [ ] Factors determining winner
- [ ] 4.3 Implications (~200 words)
  - [ ] Explains single-view convergence
  - [ ] C(f) ≈ 1/M prediction

**Theorem box**:
```
[PLACEHOLDER - THEOREM 2]
Race Dynamics
ds/dt = σ₁·s·(1-competition)
Winner = argmax A_{y,m}
```

**Figure needed**: Race dynamics plot (from Engineer)

---

### 5. Theorem 3

| Item | Status | Notes |
|------|--------|-------|
| File | `paper/sections/theorem3.tex` | Not created |
| Draft | Not Started | |
| Word count | 0/700 | ~1.5 pages |

**Structure**:
- [ ] 5.1 The Puzzle (~150 words)
  - [ ] Why does KD help if race causes single-view?
- [ ] 5.2 Gradient Analysis (~350 words)
  - [ ] Hard vs soft label gradient
  - [ ] External signal α_m from teacher
- [ ] 5.3 Special Cases (~200 words)
  - [ ] Single-view teacher: no benefit
  - [ ] Ensemble teacher: full transfer
  - [ ] Self-distillation: exceeds teacher

**Theorem box**:
```
[PLACEHOLDER - THEOREM 3]
KD Breaks the Race
Gradient ∝ α_m(T) + γ·s_{y,m}
C(S) ≈ C(T)
```

---

### 6. Experiments

| Item | Status | Notes |
|------|--------|-------|
| File | `paper/sections/experiments.tex` | Not created |
| Draft | Not Started | |
| Word count | 0/900 | ~2 pages |
| Figures | 0/4 | Awaiting Engineer results |
| Tables | 0/3 | Awaiting Engineer results |

**Structure**:
- [ ] 6.1 Synthetic Data (~500 words)
  - [ ] Data setup description
  - [ ] Exp 2.1: Single-view convergence [TABLE 1]
  - [ ] Exp 2.3: Race visualization [FIGURE 1]
  - [ ] Exp 3.1: KD coverage [TABLE 2]
  - [ ] Exp 3.3: Pathway comparison [FIGURE 2]
- [ ] 6.2 Real Data (~400 words)
  - [ ] CIFAR setup
  - [ ] Results [TABLE 3, FIGURES 3-4]

**Figures needed** (from Engineer):
- [ ] Figure 1: Race dynamics (hard labels)
- [ ] Figure 2: Pathway comparison (hard vs KD)
- [ ] Figure 3: CIFAR view diversity (optional)
- [ ] Figure 4: Coverage bar chart (optional)

**Tables needed**:
- [ ] Table 1: Single-view convergence (30 seeds)
- [ ] Table 2: KD coverage comparison
- [ ] Table 3: Real data results (optional)

---

### 7. Discussion

| Item | Status | Notes |
|------|--------|-------|
| File | `paper/sections/discussion.tex` | Not created |
| Draft | Not Started | |
| Word count | 0/300 | ~0.5 pages |

**Structure**:
- [ ] 7.1 Limitations (~150 words)
  - [ ] Simplified assumptions
  - [ ] GDLN approximation
  - [ ] Unknown view structure in real data
- [ ] 7.2 Future Work (~100 words)
  - [ ] Optimal teacher design
  - [ ] Progressive distillation
- [ ] 7.3 Conclusion (~50 words)

---

### Appendix

| Item | Status | Notes |
|------|--------|-------|
| File | `paper/appendix.tex` | Not created |
| Draft | Not Started | |

**Structure**:
- [ ] A. Proof of Theorem 1
- [ ] B. Proof of Theorem 2
- [ ] C. Proof of Theorem 3
- [ ] D. Additional Experiments
- [ ] E. Hyperparameter Details

---

## Figures Inventory

| Figure | Source | File | Status |
|--------|--------|------|--------|
| Fig 1: Race dynamics (hard) | Engineer Exp 2.3 | `figures/race_dynamics_hard.png` | Not received |
| Fig 2: Pathway comparison | Engineer Exp 3.3 | `figures/pathway_comparison.png` | Not received |
| Fig 3: Data illustration | Create in paper | — | Not started |
| Fig 4: Coverage bar chart | Engineer Exp 3.1 | — | Not received |

---

## Tables Inventory

| Table | Source | Status |
|-------|--------|--------|
| Table 1: Single-view convergence | Engineer Exp 2.1 | Not received |
| Table 2: KD coverage comparison | Engineer Exp 3.1 | Not received |
| Table 3: Real data results | Engineer (optional) | Not started |

---

## Draft History

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| v0.0 | — | Not started | Current |
| v0.1 | TBD | Initial skeleton with placeholders | Pending |
| v0.2 | TBD | Full prose draft | Pending |
| v1.0 | TBD | Review-ready draft | Pending |

---

## Review Checklist (for v1.0)

### Content
- [ ] All theorems stated clearly
- [ ] All proofs referenced (appendix)
- [ ] All experiments described
- [ ] All figures/tables included
- [ ] Limitations discussed honestly

### Writing
- [ ] Clear narrative arc
- [ ] No undefined notation
- [ ] Consistent terminology
- [ ] Proper citations
- [ ] Grammar/spelling check

### Formatting
- [ ] Within page limit (9 pages)
- [ ] Figures legible at print size
- [ ] References complete
- [ ] NeurIPS style compliance

---

## Communication Log

| Date | From | To | Summary |
|------|------|-----|---------|
| 2026-01-10 | Advisor | Writer | Paper sketch instructions received |

---

## Blockers

| Blocker | Severity | Status | Resolution |
|---------|----------|--------|------------|
| No experiment figures | Medium | Active | Awaiting Engineer |
| No experiment tables | Medium | Active | Awaiting Engineer |

---

## Next Actions

### Immediate (This Week)
- [ ] Read all theorem setup documents in `../advisor/initial_plan/`
- [ ] Create section files in `paper/sections/`
- [ ] Draft Introduction (skeleton)
- [ ] Draft Background (skeleton)
- [ ] Create placeholder theorem boxes

### Next
- [ ] Complete prose for all sections
- [ ] Integrate figures from Engineer
- [ ] Create main.tex with proper includes

---

## Reference Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Paper sketch instructions | `exchange/from_advisor/paper_sketch_instructions.md` | Current task |
| Paper Development Plan | `../advisor/initial_plan/Paper Development Plan.md` | Strategy |
| Theorem 1 setup | `../advisor/initial_plan/Theorem 1 - Formal Mathematical Setup.md` | Content |
| Theorem 2 setup | `../advisor/initial_plan/Theorem 2 - Race Dynamics Setup.md` | Content |
| Theorem 3 setup | `../advisor/initial_plan/Theorem 3 - Why KD Circumvents the Race.md` | Content |

---

*Last updated: 2026-01-10*
