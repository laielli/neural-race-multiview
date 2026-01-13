# Multi-View Theory: Formal Mathematical Setup

> **Target audience**: ML researchers comfortable with mathematical notation
> **Prerequisites**: Linear algebra, probability theory, neural network basics
> **Builds on**: [concepts.md](concepts.md)

---

## Overview

This document provides the formal mathematical definitions underlying multi-view theory. We follow the formalization from Allen-Zhu & Li (2023), with notation aligned to our unified framework.

---

## 1. Multi-View Data Distribution

### 1.1 Basic Setup

**Definition 1.1 (Multi-View Distribution)**

A $(K, M, \mathbf{p})$-multi-view distribution $\mathcal{D}$ over $\mathbb{R}^d \times [K]$ is defined by:

| Parameter | Type | Meaning |
|-----------|------|---------|
| $K$ | integer | Number of classes |
| $M$ | integer | Number of views per class |
| $d$ | integer | Input dimension |
| $\mathbf{p} = (p_1, \ldots, p_M)$ | vector | View activation probabilities, $p_m \in (0, 1)$ |

### 1.2 View Features

**Definition 1.2 (View Feature Vectors)**

Each view $(y, m)$ has an associated feature vector:

$$\phi_{y,m} \in \mathbb{R}^d$$

where:
- $y \in [K] = \{1, 2, \ldots, K\}$ is the class label
- $m \in [M] = \{1, 2, \ldots, M\}$ is the view index

The view feature matrix for class $y$:

$$\Phi_y = [\phi_{y,1} \mid \phi_{y,2} \mid \cdots \mid \phi_{y,M}] \in \mathbb{R}^{d \times M}$$

### 1.3 Sampling Process

**Definition 1.3 (Sample Generation)**

To sample $(x, y) \sim \mathcal{D}$:

1. **Draw class**: $y \sim \text{Uniform}([K])$

2. **Draw active views**: For each $m \in [M]$, independently:
   $$S_m \sim \text{Bernoulli}(p_m)$$

   Let $\mathcal{S} = \{m : S_m = 1\}$ be the set of active views.

   *Constraint*: Resample if $\mathcal{S} = \emptyset$ (at least one view must be active)

3. **Generate input**:
   $$x = \sum_{m \in \mathcal{S}} \phi_{y,m} + \varepsilon$$

   where $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_d)$ is isotropic Gaussian noise

**Notation**: We write $x \sim \mathcal{D}_y^{\mathcal{S}}$ to denote a sample from class $y$ with active views $\mathcal{S}$.

---

## 2. Structural Assumptions

Multi-view theory relies on several structural assumptions about the data distribution.

### Assumption A1: View Sufficiency

> Each view is independently sufficient for correct classification.

**Formal statement**: For each $(y, m)$, there exists a linear classifier $w_{y,m} \in \mathbb{R}^d$ such that:

$$\langle w_{y,m}, \phi_{y,m} \rangle > \gamma$$

and for all $y' \neq y$ and all $m'$:

$$\langle w_{y,m}, \phi_{y',m'} \rangle < -\gamma$$

for some margin $\gamma > 0$.

**Intuition**: You could correctly classify using *only* view $(y, m)$, ignoring all other features.

### Assumption A2: View Orthogonality

> Different views are approximately orthogonal.

**Formal statement**: There exists $\delta \geq 0$ (small) such that for all $(y, m) \neq (y', m')$:

$$|\langle \phi_{y,m}, \phi_{y',m'} \rangle| \leq \delta \cdot \|\phi_{y,m}\| \cdot \|\phi_{y',m'}\|$$

**Special cases**:
- $\delta = 0$: Views are exactly orthogonal
- $\delta$ small: Views are approximately orthogonal

**Intuition**: Different views use largely non-overlapping features. This is what allows the network to "choose" among them.

### Assumption A3: View Normalization

> All views have comparable magnitude.

**Formal statement**: There exist constants $0 < c_{\min} \leq c_{\max}$ such that:

$$c_{\min} \leq \|\phi_{y,m}\| \leq c_{\max} \quad \forall y \in [K], m \in [M]$$

**Simplification**: We often assume $\|\phi_{y,m}\| = 1$ for all $(y, m)$ (unit-normalized views).

**Intuition**: No single view is inherently "louder" in the input signal. This ensures the race between views is determined by network dynamics, not input magnitude.

---

## 3. View Response and Coverage

### 3.1 Network Output

Consider a neural network $f: \mathbb{R}^d \to \mathbb{R}^K$ mapping inputs to class logits.

**Definition 3.1 (View Response)**

The response of network $f$ to view $(y, m)$ is:

$$R_{y,m}(f) := f(\phi_{y,m}) \in \mathbb{R}^K$$

This is the network's output when presented with the "pure" view feature (no noise, no other views).

### 3.2 View Detection

**Definition 3.2 (View Detection)**

View $(y, m)$ is **detected** by network $f$ if:

$$\arg\max_{k \in [K]} [R_{y,m}(f)]_k = y$$

In other words, the network correctly classifies inputs containing only view $(y, m)$.

**Definition 3.3 (Detection Margin)**

The detection margin for view $(y, m)$ is:

$$\rho_{y,m}(f) := [R_{y,m}(f)]_y - \max_{k \neq y} [R_{y,m}(f)]_k$$

View $(y, m)$ is detected if and only if $\rho_{y,m}(f) > 0$.

### 3.3 View Coverage

**Definition 3.4 (View Coverage)**

The **view coverage** of network $f$ is:

$$C(f) := \frac{1}{KM} \sum_{y=1}^{K} \sum_{m=1}^{M} \mathbb{1}\left[ \rho_{y,m}(f) > 0 \right]$$

This measures the fraction of all views the network can correctly classify.

**Range**: $C(f) \in [0, 1]$
- $C(f) = 1$: Network detects all views (perfect multi-view learning)
- $C(f) = 1/M$: Network detects one view per class (single-view convergence)
- $C(f) = 0$: Network detects no views (failure)

---

## 4. Ensemble Coverage

### 4.1 Ensemble Definition

**Definition 4.1 (Ensemble)**

An ensemble $\mathcal{E} = \{f_1, f_2, \ldots, f_N\}$ is a collection of $N$ networks.

The ensemble prediction is:

$$f_{\mathcal{E}}(x) = \frac{1}{N} \sum_{i=1}^{N} f_i(x)$$

### 4.2 Ensemble View Detection

**Definition 4.2 (Ensemble Detection)**

View $(y, m)$ is detected by ensemble $\mathcal{E}$ if:

$$\arg\max_{k \in [K]} [f_{\mathcal{E}}(\phi_{y,m})]_k = y$$

Equivalently, if at least one member detects the view with sufficient margin:

$$\exists i \in [N]: \rho_{y,m}(f_i) > \epsilon$$

(for some threshold $\epsilon > 0$, accounting for averaging effects)

### 4.3 Expected Ensemble Coverage

**Proposition 4.1 (Ensemble Coverage)**

If individual networks learn each view with probability $1/M$ (uniform random), then:

$$\mathbb{E}[C(\mathcal{E})] = 1 - \left(1 - \frac{1}{M}\right)^N$$

**Proof sketch**: For each view, the probability that *no* network learns it is $(1-1/M)^N$. Therefore, the probability that *at least one* network learns it is $1 - (1-1/M)^N$. By linearity of expectation over all $KM$ views, this equals the expected coverage.

---

## 5. Knowledge Distillation Setup

### 5.1 Teacher and Student

**Definition 5.1 (Teacher-Student Setup)**

- **Teacher** $T$: A trained network (or ensemble) with view coverage $C(T)$
- **Student** $S$: A network to be trained

### 5.2 Distillation Loss

**Definition 5.2 (Knowledge Distillation Loss)**

The KD loss at temperature $\tau$ is:

$$\mathcal{L}_{\text{KD}}(S; T, x) = \tau^2 \cdot \text{KL}\left( p_T^\tau(x) \,\|\, p_S^\tau(x) \right)$$

where the softened probabilities are:

$$p_T^\tau(x) = \text{softmax}\left( \frac{f_T(x)}{\tau} \right), \quad p_S^\tau(x) = \text{softmax}\left( \frac{f_S(x)}{\tau} \right)$$

**Components**:
- $\tau > 1$: Temperature parameter (higher = softer distributions)
- $\text{KL}(p \| q) = \sum_k p_k \log(p_k / q_k)$: Kullback-Leibler divergence
- The $\tau^2$ factor ensures gradient magnitudes are comparable across temperatures

### 5.3 Combined Training Objective

**Definition 5.3 (Combined Loss)**

In practice, distillation is often combined with hard label training:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{KD}}(S; T, x) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(S; y, x)$$

where:
- $\mathcal{L}_{\text{CE}}$ is cross-entropy loss with ground-truth label $y$
- $\alpha \in [0, 1]$ balances the two objectives

For theoretical analysis, we often consider pure distillation: $\alpha = 1$.

---

## 6. Input-Output Correlation

A key quantity for analyzing learning dynamics.

### 6.1 Definition

**Definition 6.1 (View Correlation Matrix)**

The input-output correlation for view $(y, m)$ is:

$$\Sigma_{y,m} = \frac{p_m}{K} \cdot e_y \cdot \phi_{y,m}^T \in \mathbb{R}^{K \times d}$$

where $e_y \in \mathbb{R}^K$ is the one-hot vector for class $y$.

### 6.2 Correlation Strength

**Definition 6.2 (Correlation Strength)**

The correlation strength is the largest singular value:

$$\sigma_1(\Sigma_{y,m}) = \frac{p_m}{K} \|\phi_{y,m}\|$$

Under unit-normalized views ($\|\phi_{y,m}\| = 1$) and uniform activation ($p_m = p$):

$$\sigma_1(\Sigma_{y,m}) = \frac{p}{K} \quad \text{(same for all views)}$$

### 6.3 Role in Learning Dynamics

The correlation strength determines how "visible" a view is to gradient descent:
- Higher $\sigma_1(\Sigma_{y,m})$ → stronger gradient signal for view $(y, m)$
- Under symmetric assumptions, all views have equal correlation strength
- The "winner" is then determined by initialization (covered in Module 2)

---

## 7. Notation Summary

| Symbol | Type | Meaning |
|--------|------|---------|
| $K$ | $\mathbb{Z}^+$ | Number of classes |
| $M$ | $\mathbb{Z}^+$ | Number of views per class |
| $d$ | $\mathbb{Z}^+$ | Input dimension |
| $[K]$ | set | $\{1, 2, \ldots, K\}$ |
| $[M]$ | set | $\{1, 2, \ldots, M\}$ |
| $\phi_{y,m}$ | $\mathbb{R}^d$ | Feature vector for view $(y, m)$ |
| $p_m$ | $(0,1)$ | Activation probability for view $m$ |
| $\mathcal{S}$ | $\subseteq [M]$ | Set of active views in a sample |
| $\sigma$ | $\mathbb{R}^+$ | Noise standard deviation |
| $\varepsilon$ | $\mathbb{R}^d$ | Noise vector, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ |
| $f$ | $\mathbb{R}^d \to \mathbb{R}^K$ | Neural network |
| $R_{y,m}(f)$ | $\mathbb{R}^K$ | View response |
| $\rho_{y,m}(f)$ | $\mathbb{R}$ | Detection margin |
| $C(f)$ | $[0,1]$ | View coverage |
| $\tau$ | $\mathbb{R}^+$ | Temperature for distillation |
| $\Sigma_{y,m}$ | $\mathbb{R}^{K \times d}$ | Input-output correlation matrix |
| $\sigma_1(\cdot)$ | $\mathbb{R}^+$ | Largest singular value |
| $\delta$ | $\mathbb{R}^+$ | Orthogonality violation bound |
| $\gamma$ | $\mathbb{R}^+$ | Classification margin |

---

## 8. Assumptions Summary

| ID | Name | Statement | Role |
|----|------|-----------|------|
| A1 | Sufficiency | Each view enables correct classification | Views are meaningful |
| A2 | Orthogonality | $\langle \phi_{y,m}, \phi_{y',m'} \rangle \approx 0$ for $(y,m) \neq (y',m')$ | Views are separable |
| A3 | Normalization | $\|\phi_{y,m}\| \in [c_{\min}, c_{\max}]$ | Balanced signal strength |

---

## What's Next

- **[Key Results](key_results.md)**: The main theorems from Allen-Zhu & Li (2023)
- **[Worked Examples](worked_examples.md)**: Concrete calculations with our synthetic setup

---

## References

- Allen-Zhu, Z., & Li, Y. (2023). *Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning*. ICLR 2023.
