# Module 1: Multi-View Theory

**Source**: Allen-Zhu & Li (2023), ICLR — [arXiv:2012.09816](https://arxiv.org/abs/2012.09816)

---

## Overview

Multi-view theory explains why knowledge distillation works by showing that:
1. Neural networks learn only one "view" (feature subset) per class
2. Ensembles cover more views through random diversity
3. Soft labels transfer multi-view knowledge to students

## Reading Order

| Order | File | Time | Description |
|-------|------|------|-------------|
| 1 | [concepts.md](concepts.md) | 15 min | Intuitive introduction, no math |
| 2 | [formal_setup.md](formal_setup.md) | 20 min | Mathematical definitions |
| 3 | [key_results.md](key_results.md) | 15 min | Main theorems |
| 4 | [worked_examples.md](worked_examples.md) | 20 min | Concrete calculations |

**Total reading time**: ~70 minutes

## Prerequisites

- Familiarity with neural network training and classification
- Basic probability and linear algebra
- No prior knowledge of multi-view theory required

## Key Concepts Covered

- What is a "view" and why multiple views exist
- View coverage metric $C(f)$
- Single-view convergence ($C(f) \approx 1/M$)
- Ensemble diversity and coverage
- Knowledge distillation coverage transfer

## The Open Questions

This module explains *what* multi-view theory predicts, but leaves open:

1. **Which view wins?** — Given initialization, predict the learned view
2. **Why winner-take-all?** — The gradient-level mechanism
3. **How does KD break the race?** — Gradient decomposition under soft labels

These questions are addressed in subsequent modules.

## Next Module

After completing this module, proceed to:

**[Module 2: Neural Race Reduction](../02_neural_race_reduction/)** — Understanding learning dynamics as pathway competition
