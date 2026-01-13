# Race Dynamics: Winner-Take-All

> **Target audience**: ML researchers ready for the full mathematical treatment
> **Prerequisites**: [Gated Networks](gated_networks.md)
> **Core result**: Initial advantage formula predicts which view wins

---

## Overview

This document presents the **race dynamics** that govern pathway competition during training. We derive the governing equations, analyze their fixed points, and establish the **initial advantage formula** that predicts which view will be learned.

---

## Setup

### The Learning Problem

- **Data**: $(K, M, \mathbf{p})$-multi-view distribution $\mathcal{D}$
- **Network**: GDLN with $L = 2$ layers, widths $(d, n, K)$
- **Loss**: Cross-entropy with hard labels
- **Training**: Gradient flow (continuous-time gradient descent)

### Key Quantities

| Quantity | Definition | Meaning |
|----------|------------|---------|
| $\mathbf{P}_{y,m}(t)$ | $W_2(t) D_m W_1(t)$ | Pathway matrix at time $t$ |
| $s_{y,m}(t)$ | $\|\mathbf{P}_{y,m}(t)\|_F$ | Pathway strength |
| $\Sigma_{y,m}$ | $\frac{p_m}{K} e_y \phi_{y,m}^T$ | View correlation matrix |
| $\sigma_1(\Sigma_{y,m})$ | $\frac{p_m}{K}\|\phi_{y,m}\|$ | Correlation strength |
| $A_{y,m}$ | $\sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)$ | Initial advantage |

---

## The Governing Equation

### Theorem 2, Part A: Pathway Strength Dynamics

Under gradient flow on cross-entropy loss, pathway strengths evolve according to:

$$\boxed{\frac{ds_{y,m}}{dt} = \sigma_1(\Sigma_{y,m}) \cdot s_{y,m} \cdot \left(1 - \frac{\sum_{m'=1}^{M} s_{y,m'}^2}{s_{\max}^2}\right) + E_{y,m}(t)}$$

where:
- $\sigma_1(\Sigma_{y,m})$ = correlation strength for view $(y,m)$
- $s_{\max}$ = saturation strength (maximum achievable)
- $E_{y,m}(t)$ = error term with $|E_{y,m}| \leq O(\delta + \sigma_0^2)$

### Interpretation

**Term 1: Growth** — $\sigma_1(\Sigma_{y,m}) \cdot s_{y,m}$

Pathway grows proportionally to:
- Its current strength $s_{y,m}$ (multiplicative/exponential)
- Its correlation strength $\sigma_1$ (how useful is this view?)

**Term 2: Competition** — $(1 - \sum_{m'} s_{y,m'}^2 / s_{\max}^2)$

- Shared across all views of the same class
- Approaches 0 as total pathway strength grows
- Creates competition: as one pathway grows, others slow down

**Term 3: Error** — $E_{y,m}(t)$

- Cross-pathway interactions (small under orthogonality)
- Initialization effects
- Negligible for analysis

---

## Phase Analysis

### Phase 1: Early Training (Exponential Growth)

**Condition**: All pathway strengths small: $\sum_m s_{y,m}^2 \ll s_{\max}^2$

**Dynamics**: Competition term $\approx 1$, so:
$$\frac{ds_{y,m}}{dt} \approx \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}$$

**Solution**:
$$s_{y,m}(t) = s_{y,m}(0) \cdot \exp\left(\sigma_1(\Sigma_{y,m}) \cdot t\right)$$

**Key insight**: Exponential growth with rate proportional to correlation strength.

### Phase 2: Competition Phase

**Condition**: Total strength becomes comparable to $s_{\max}$

**Dynamics**: Competition term shrinks, slowing growth for all pathways

**Key insight**: The leader maintains its advantage while others are squeezed out.

### Phase 3: Saturation (Winner-Take-All)

**Condition**: One pathway dominates: $s_{y,m^*}^2 \approx s_{\max}^2$

**Dynamics**:
- Winning pathway: $s_{y,m^*} \to s_{\max}$
- Losing pathways: $s_{y,m} \to 0$ for $m \neq m^*$

**Key insight**: The race is over. One view wins, others lose.

---

## The Initial Advantage Formula

### Definition

$$\boxed{A_{y,m} = \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)}$$

This is the product of:
1. **Correlation strength**: How predictive is view $m$?
2. **Initial pathway strength**: How strong at initialization?

### Theorem 2, Part C: Winner Determination

**With high probability over random initialization:**

For each class $y$, there exists a unique winning view:
$$m^*(y) = \arg\max_{m \in [M]} A_{y,m}$$

And the winner dominates:
$$\lim_{t \to \infty} \frac{s_{y,m}(t)}{s_{y,m^*(y)}(t)} = 0 \quad \text{for all } m \neq m^*(y)$$

### Why Initial Advantage Determines Winner

**During early phase**:
$$\frac{s_{y,m}(t)}{s_{y,m'}(t)} = \frac{s_{y,m}(0)}{s_{y,m'}(0)} \cdot \exp\left((\sigma_1^{(m)} - \sigma_1^{(m')}) \cdot t\right)$$

The ratio evolves exponentially based on the difference in correlation strengths.

**Case 1**: If $\sigma_1^{(m)} > \sigma_1^{(m')}$, the ratio grows exponentially → $m$ wins.

**Case 2**: If $\sigma_1^{(m)} = \sigma_1^{(m')}$ (symmetric views), the ratio is constant at $s_{y,m}(0) / s_{y,m'}(0)$. The pathway with higher initial strength wins.

**General case**: $A_{y,m} = \sigma_1^{(m)} \cdot s_{y,m}(0)$ captures both effects.

---

## Symmetric Views: The Common Case

### Setup

When views are symmetric:
- $\|\phi_{y,m}\| = c$ for all $m$ (equal norms)
- $p_m = p$ for all $m$ (equal activation probabilities)

Then $\sigma_1(\Sigma_{y,m}) = \sigma_1$ is the same for all views.

### Initial Advantage Simplifies

$$A_{y,m} = \sigma_1 \cdot s_{y,m}(0) \propto s_{y,m}(0)$$

**The winner is determined purely by initialization.**

### Winner Probability

**Theorem 2, Part D (Symmetric Case)**:

Under symmetric random initialization:
$$\mathbb{P}(m^*(y) = m) = \frac{1}{M}$$

Each view is equally likely to win.

### Implication: Explains Ensemble Diversity

Different random seeds → different $s_{y,m}(0)$ → different winners.

This is exactly what multi-view theory observed: different seeds learn different views.

---

## The Competitive Dynamical System

### Lotka-Volterra Form

The race dynamics are a **competitive Lotka-Volterra system**:

$$\frac{ds_m}{dt} = r_m s_m \left(1 - \frac{\sum_{m'} s_{m'}^2}{K_{\text{cap}}}\right)$$

where:
- $r_m = \sigma_1(\Sigma_{y,m})$ is the growth rate
- $K_{\text{cap}} = s_{\max}^2$ is the shared carrying capacity

### Fixed Points

Setting $\dot{s}_m = 0$ for all $m$, the system has $M$ stable fixed points:

**Fixed point $m^*$**:
$$s_{m^*} = s_{\max}, \quad s_m = 0 \text{ for } m \neq m^*$$

Each fixed point corresponds to "view $m^*$ wins."

### Basins of Attraction

The basin of attraction for fixed point $m^*$ is approximately:
$$\mathcal{B}_{m^*} = \left\{ \mathbf{s}(0) : A_{m^*} > A_m \text{ for all } m \neq m^* \right\}$$

Initial conditions where $m^*$ has the highest initial advantage.

### Phase Portrait (M = 2)

```
       s₂
        ↑
        │╲                    Fixed point 1: (s_max, 0)
  s_max │  ╲                  Fixed point 2: (0, s_max)
        │    ╲
        │      ╲              Basin boundary: A₁ = A₂
        │        ╲            i.e., σ₁s₁(0) = σ₂s₂(0)
        │          ╲
        │            ╲
        │              ╲
        │                ╲
        │                  ╲
        └────────────────────→ s₁
                           s_max

Above the line: View 2 wins
Below the line: View 1 wins
```

---

## Factors Determining Initial Advantage

### Correlation Strength $\sigma_1(\Sigma_{y,m})$

$$\sigma_1(\Sigma_{y,m}) = \frac{p_m}{K} \cdot \|\phi_{y,m}\|$$

| Factor | Effect on $\sigma_1$ | Intuition |
|--------|---------------------|-----------|
| $p_m$ ↑ | $\sigma_1$ ↑ | More training examples with this view |
| $\|\phi_{y,m}\|$ ↑ | $\sigma_1$ ↑ | Stronger signal in input |
| $K$ ↑ | $\sigma_1$ ↓ | Signal diluted across classes |

### Initial Pathway Strength $s_{y,m}(0)$

$$s_{y,m}(0) = \|W_2(0) \cdot D_m \cdot W_1(0)\|_F$$

| Factor | Effect on $s(0)$ | Intuition |
|--------|-----------------|-----------|
| More active neurons | $s(0)$ ↑ | More parameters in pathway |
| Larger init scale $\sigma_0$ | $s(0)$ ↑ | Larger initial weights |
| Favorable weight alignment | $s(0)$ ↑ | Lucky initialization |

### At Random Initialization

Under standard Gaussian initialization:
$$s_{y,m}(0) \approx \sigma_0^2 \cdot \sqrt{|D_m|_1 \cdot K \cdot d}$$

where $|D_m|_1 = \sum_j [g_m]_j$ is the number of active neurons for view $m$.

---

## View Coverage Result

### Theorem 2, Part E

After training:
$$C(f) = \frac{1}{M} + O\left(\frac{1}{\sqrt{K}}\right)$$

**Interpretation**: Approximately one view per class is learned.

### Derivation

- Each class has one winning view (with high probability)
- The winner is determined by $\arg\max_m A_{y,m}$
- Under symmetric views, each view wins with probability $1/M$
- Coverage = fraction of views detected ≈ $1/M$

---

## Summary: The Race Mechanism

### In Words

1. **Initialization**: All pathways start weak, with random strengths
2. **Early training**: Exponential growth proportional to $\sigma_1 \cdot s(0) = A$
3. **Initial advantage compounds**: Small differences become large
4. **Competition phase**: Leader captures the learning budget
5. **Saturation**: Winner dominates, losers decay to zero
6. **Result**: Single-view convergence, coverage ≈ $1/M$

### In Equations

| Phase | Dynamics | Solution |
|-------|----------|----------|
| Early | $\dot{s}_m = \sigma_1 s_m$ | $s_m(t) = s_m(0) e^{\sigma_1 t}$ |
| Competition | $\dot{s}_m = \sigma_1 s_m (1 - \text{comp})$ | Ratio preserved |
| Saturation | $\sum_m s_m^2 \to s_{\max}^2$ | $s_{m^*} \to s_{\max}$, others → 0 |

### The Key Formula

$$\boxed{m^*(y) = \arg\max_{m \in [M]} \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)}$$

**The view with highest initial advantage wins.**

---

## Testable Predictions

1. **Exponential early growth**: $\log s_m(t)$ vs $t$ is linear initially

2. **Winner prediction accuracy**: >80% agreement between $\arg\max A_m$ and actual winner

3. **Initialization sensitivity**: Changing seed changes winner

4. **Correlation dominance**: Views with higher $\|\phi\|$ win more often

5. **Coverage**: $C(f) \approx 1/M$ after training

6. **Ratio decay**: $s_m(t) / s_{m^*}(t) \to 0$ exponentially for losers

---

## What Changes Under KD?

### Preview of Module 3

Under knowledge distillation, the dynamics become:

$$\frac{ds_{y,m}}{dt} = \left(\underbrace{\alpha_m(T)}_{\text{external}} + \underbrace{\gamma \sigma_1 s_{y,m}}_{\text{internal}}\right) \cdot (1 - \text{comp})$$

The **external signal** $\alpha_m(T)$ from the teacher:
- Provides gradient to all pathways the teacher knows
- Doesn't depend on student's current state
- Breaks the winner-take-all dynamics

**Result**: Multiple pathways can survive → multi-view learning.

---

## What's Next

- **[Worked Examples](worked_examples.md)**: Concrete calculations with K=10, M=3
- **Module 3**: [Connecting Theories](../03_connecting_theories/) — Why KD breaks the race

---

## References

- Saxe, A. M., Sodhani, S., & Gershman, S. (2022). *The Neural Race Reduction: Dynamics of abstraction in gated networks*. ICML 2022.
