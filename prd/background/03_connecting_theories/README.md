# Module 3: Connecting the Theories

**Our contribution**: Unifying Multi-View Theory with Neural Race Reduction to explain knowledge distillation mechanistically.

---

## Overview

This module presents our novel theoretical contribution: connecting multi-view learning to neural race dynamics, explaining **why** KD works at the gradient level.

### The Three Theorems

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| **Theorem 1** | Views ↔ Pathways | Bridge between frameworks |
| **Theorem 2** | Winner = $\arg\max A_m$ | First predictive formula |
| **Theorem 3** | KD adds external signal | Mechanistic explanation |

## Reading Order

| Order | File | Time | Description |
|-------|------|------|-------------|
| 1 | [the_gap.md](the_gap.md) | 15 min | What existing theories don't explain |
| 2 | [view_pathway_correspondence.md](view_pathway_correspondence.md) | 20 min | Theorem 1: Views = Pathways |
| 3 | [why_kd_works.md](why_kd_works.md) | 25 min | Theorem 3: External signal breaks race |
| 4 | [worked_examples.md](worked_examples.md) | 20 min | Concrete calculations |
| 5 | [reading_guide.md](reading_guide.md) | Reference | Annotated guide to key papers |

**Total reading time**: ~80 minutes (+ paper reading)

## Prerequisites

- Module 1 (Multi-View Theory)
- Module 2 (Neural Race Reduction)

## Key Equations

### View-Pathway Decomposition (Theorem 1)

$$f(x) = \sum_{m \in \mathcal{S}} R_{y,m}(f) + E(x, \mathcal{S})$$

### Initial Advantage (Theorem 2)

$$m^*(y) = \arg\max_m \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)$$

### KD Gradient Decomposition (Theorem 3)

$$\nabla \mathcal{L}_{\text{KD}} \propto \underbrace{\alpha_m(T)}_{\text{external}} + \underbrace{\gamma \sigma_1 s_{y,m}}_{\text{internal}}$$

## What You'll Understand After This Module

1. **Why views are pathways**: Different features activate different computational routes
2. **Why hard labels fail**: Self-reinforcement creates winner-take-all
3. **How KD succeeds**: External signal prevents pathway death
4. **The coverage result**: $C(S) \approx C(T)$ because all teacher views get signal

## The Unified Picture

```
Multi-View Data → View-Pathway Correspondence → Race Dynamics
                        (Theorem 1)              (Theorem 2)
                                                     ↓
Hard Labels ────────────────────────────────→ Winner-Take-All
                                                C(f) = 1/M

Soft Labels → External Signal ──────────────→ Multiple Winners
  (KD)          (Theorem 3)                     C(S) ≈ C(T)
```

## Connection to Paper

This module covers:
- Section 3: Setup and Theorem 1
- Section 4: Race Dynamics (Theorem 2)
- Section 5: Why KD Works (Theorem 3)

The worked examples provide intuition for the experimental validation (Section 6).
