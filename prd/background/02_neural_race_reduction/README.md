# Module 2: Neural Race Reduction

**Sources**:
- Saxe, McClelland, & Ganguli (2014), ICLR — Deep linear network dynamics
- Saxe, Sodhani, & Gershman (2022), ICML — Neural Race Reduction

---

## Overview

The Neural Race Reduction framework explains **why** networks learn only one view per class and **predicts which view will win**.

Key insight: Learning is a race between computational pathways. The pathway with highest **initial advantage** wins:

$$A_{y,m} = \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)$$

## Reading Order

| Order | File | Time | Description |
|-------|------|------|-------------|
| 1 | [concepts.md](concepts.md) | 20 min | Intuitive introduction to the race |
| 2 | [deep_linear_networks.md](deep_linear_networks.md) | 20 min | Saxe et al. 2014 foundations |
| 3 | [gated_networks.md](gated_networks.md) | 15 min | Extension to ReLU via gating |
| 4 | [race_dynamics.md](race_dynamics.md) | 20 min | Full mathematical treatment |
| 5 | [worked_examples.md](worked_examples.md) | 20 min | Concrete calculations |

**Total reading time**: ~95 minutes

## Prerequisites

- Module 1 (Multi-View Theory)
- Linear algebra (SVD)
- Basic differential equations

## Key Concepts Covered

- Deep linear networks and their exact solutions
- Gated Deep Linear Networks (GDLN)
- ReLU ↔ GDLN equivalence
- Pathways as computational routes
- Winner-take-all dynamics
- The initial advantage formula

## The Core Result

**Theorem (Race Dynamics)**:

Pathway strengths evolve according to:
$$\frac{ds_{y,m}}{dt} = \sigma_1(\Sigma_{y,m}) \cdot s_{y,m} \cdot \left(1 - \frac{\sum_{m'} s_{y,m'}^2}{s_{\max}^2}\right)$$

The winner is: $m^*(y) = \arg\max_m A_{y,m}$

## Connection to Multi-View Theory

| Multi-View Theory | Neural Race Explanation |
|-------------------|------------------------|
| Single-view convergence | Winner-take-all dynamics |
| Ensemble diversity | Different initializations → different winners |
| Coverage ≈ 1/M | One pathway dominates per class |

## What This Module Enables

After completing this module, you understand:
1. **Why** networks learn one view (race mechanism)
2. **How** to predict which view wins (initial advantage)
3. **The dynamics** of pathway competition

This sets up Module 3: understanding why KD breaks the race.

## Next Module

**[Module 3: Connecting the Theories](../03_connecting_theories/)** — Why Knowledge Distillation circumvents the race
