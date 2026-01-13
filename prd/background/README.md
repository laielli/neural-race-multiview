# Foundations: Onboarding Materials

This directory contains educational materials for onboarding researchers onto the KDMech project. The content explains the foundational theories we build upon and our novel contributions.

---

## Target Audience

- ML researchers with strong deep learning background
- Familiarity with neural network training, backpropagation, optimization
- **No prior knowledge required** of multi-view theory or neural race reduction

---

## Module Overview

| Module | Topic | Time | Description |
|--------|-------|------|-------------|
| [01_multi_view_theory](01_multi_view_theory/) | Multi-View Learning | 70 min | Allen-Zhu & Li (2023) — why networks learn one view |
| [02_neural_race_reduction](02_neural_race_reduction/) | Learning Dynamics | 95 min | Saxe et al. (2014, 2022) — pathways and races |
| [03_connecting_theories](03_connecting_theories/) | Our Contribution | 80 min | The unified framework and KD mechanism |

**Total reading time**: ~4 hours

---

## Recommended Learning Path

### Path 1: Full Understanding (Recommended)

Complete all three modules in order:

```
Module 1 → Module 2 → Module 3
  (70m)     (95m)      (80m)
```

### Path 2: Quick Overview

Read only the `concepts.md` from each module:

```
01/concepts.md → 02/concepts.md → 03/the_gap.md
     (15m)           (20m)            (15m)
```

### Path 3: Focus on Our Contribution

If familiar with background:

```
03/the_gap.md → 03/view_pathway_correspondence.md → 03/why_kd_works.md
    (15m)                (20m)                          (25m)
```

---

## Key Concepts by Module

### Module 1: Multi-View Theory

- **View**: Feature subset sufficient for classification
- **Single-view convergence**: Networks learn ~1 view per class
- **Ensemble diversity**: Different seeds → different views
- **KD coverage transfer**: Students inherit teacher's view knowledge

### Module 2: Neural Race Reduction

- **Pathway**: Computational route through network
- **Race dynamics**: Pathways compete during training
- **Winner-take-all**: One pathway dominates per class
- **Initial advantage**: $A_m = \sigma_1 \cdot s_m(0)$ predicts winner

### Module 3: Connecting Theories

- **View-Pathway Correspondence**: Views map to pathways (Theorem 1)
- **Predictive Formula**: Initial advantage determines winner (Theorem 2)
- **KD Mechanism**: External signal breaks the race (Theorem 3)

---

## Key Equations

### The Initial Advantage Formula

$$A_{y,m} = \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)$$

The view with highest initial advantage wins.

### Race Dynamics

$$\frac{ds_{y,m}}{dt} = \sigma_1 \cdot s_{y,m} \cdot \left(1 - \frac{\sum_{m'} s_{y,m'}^2}{s_{\max}^2}\right)$$

### KD Gradient Decomposition

$$\nabla \mathcal{L}_{\text{KD}} \propto \underbrace{\alpha_m(T)}_{\text{external (teacher)}} + \underbrace{\gamma \sigma_1 s_{y,m}}_{\text{internal (self)}}$$

---

## Connection to Paper

| Paper Section | Corresponding Module |
|---------------|---------------------|
| §2 Background | Modules 1 & 2 |
| §3 Setup & Theorem 1 | Module 3 (view_pathway_correspondence) |
| §4 Race Dynamics | Module 2 (race_dynamics) |
| §5 Why KD Works | Module 3 (why_kd_works) |
| §6 Experiments | worked_examples in each module |

---

## File Counts

```
foundations/
├── 01_multi_view_theory/     (5 files, ~46 KB)
├── 02_neural_race_reduction/ (6 files, ~56 KB)
├── 03_connecting_theories/   (6 files, ~65 KB)
└── README.md                 (this file)

Total: 18 files, ~170 KB of educational content
```

---

## Quick Reference: All Files

### Module 1
- `README.md` — Module overview
- `concepts.md` — Intuitive introduction
- `formal_setup.md` — Mathematical definitions
- `key_results.md` — Main theorems
- `worked_examples.md` — Concrete calculations

### Module 2
- `README.md` — Module overview
- `concepts.md` — Race intuition
- `deep_linear_networks.md` — Saxe 2014 foundations
- `gated_networks.md` — GDLN formalism
- `race_dynamics.md` — Winner-take-all mechanism
- `worked_examples.md` — Concrete calculations

### Module 3
- `README.md` — Module overview
- `the_gap.md` — What existing theories miss
- `view_pathway_correspondence.md` — Theorem 1
- `why_kd_works.md` — Theorem 3
- `worked_examples.md` — Concrete calculations
- `reading_guide.md` — **Annotated bibliography of key papers**
