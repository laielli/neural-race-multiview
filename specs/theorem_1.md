# Theorem 1: View-Pathway Correspondence — Formal Mathematical Setup

## Overview

This document develops the formal mathematical framework for Theorem 1, which establishes the correspondence between "views" (from Allen-Zhu & Li's multi-view theory) and "pathways" (from Saxe et al.'s neural race reduction).

**Goal**: Show that under multi-view data, each view corresponds to a distinct computational pathway, and the network's output decomposes as a sum over active view pathways.

---

## Part 1: Multi-View Data Model

### 1.1 Basic Setup

**Definition 1.1 (Multi-View Distribution)**: A $(K, M, \mathbf{p})$-multi-view distribution $\mathcal{D}$ over $\mathbb{R}^d \times [K]$ is defined by:

- $K$ classes, indexed by $y \in [K] = \{1, \ldots, K\}$
- $M$ views per class, indexed by $m \in [M] = \{1, \ldots, M\}$
- View activation probabilities $\mathbf{p} = (p_1, \ldots, p_M)$ with $p_m \in (0, 1)$

**Definition 1.2 (View Features)**: Each view $(y, m)$ has an associated feature vector:
$$\phi_{y,m} \in \mathbb{R}^d$$

We collect these into the view feature matrix for class $y$:
$$\Phi_y = [\phi_{y,1} | \phi_{y,2} | \cdots | \phi_{y,M}] \in \mathbb{R}^{d \times M}$$

### 1.2 Sample Generation

**Definition 1.3 (Sampling Process)**: To sample $(x, y) \sim \mathcal{D}$:

1. **Draw class**: $y \sim \text{Uniform}([K])$

2. **Draw active views**: For each $m \in [M]$, independently:
   $$S_m \sim \text{Bernoulli}(p_m)$$
   Let $\mathcal{S} = \{m : S_m = 1\}$ be the set of active views.

   *Constraint*: Resample if $\mathcal{S} = \emptyset$ (at least one view must be active).

3. **Generate input**:
   $$x = \sum_{m \in \mathcal{S}} \phi_{y,m} + \varepsilon$$
   where $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_d)$ is isotropic Gaussian noise.

**Notation**: We write $x \sim \mathcal{D}_y^{\mathcal{S}}$ to denote a sample from class $y$ with active views $\mathcal{S}$.

### 1.3 Structural Assumptions

**Assumption A1 (View Sufficiency)**: Each view is independently sufficient for correct classification.

*Formally*: For each $(y, m)$, there exists a linear classifier $w_{y,m} \in \mathbb{R}^d$ such that:
$$\langle w_{y,m}, \phi_{y,m} \rangle > \gamma \quad \text{and} \quad \langle w_{y,m}, \phi_{y',m'} \rangle < -\gamma \quad \forall y' \neq y, \forall m'$$

for some margin $\gamma > 0$.

**Assumption A2 (View Orthogonality)**: Different views are approximately orthogonal.

*Formally*: There exists $\delta \geq 0$ (small) such that for all $(y, m) \neq (y', m')$:
$$|\langle \phi_{y,m}, \phi_{y',m'} \rangle| \leq \delta \cdot \|\phi_{y,m}\| \cdot \|\phi_{y',m'}\|$$

When $\delta = 0$, views are exactly orthogonal.

**Assumption A3 (View Normalization)**: All views have comparable norm.

*Formally*: There exist constants $0 < c_{\min} \leq c_{\max}$ such that:
$$c_{\min} \leq \|\phi_{y,m}\| \leq c_{\max} \quad \forall y, m$$

For simplicity, we often take $\|\phi_{y,m}\| = 1$ for all $(y, m)$.

---

## Part 2: Gated Deep Linear Network Model

### 2.1 Architecture

**Definition 2.1 (GDLN Architecture)**: A Gated Deep Linear Network with $L$ layers consists of:

- **Weight matrices**: $W_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ for $\ell = 1, \ldots, L$
  - Where $n_0 = d$ (input dimension) and $n_L = K$ (number of classes)

- **Gating functions**: $G_\ell : \mathbb{R}^d \to \{0, 1\}^{n_\ell}$ for $\ell = 1, \ldots, L-1$
  - Each $G_\ell(x)$ is a binary vector depending on input $x$

**Definition 2.2 (Forward Pass)**: The network output is:
$$f(x; W) = W_L \cdot D_{L-1}(x) \cdot W_{L-1} \cdots D_1(x) \cdot W_1 \cdot x$$

where $D_\ell(x) = \text{diag}(G_\ell(x)) \in \mathbb{R}^{n_\ell \times n_\ell}$ is the diagonal gating matrix.

**Expanded form**:
$$f(x; W) = W_L \prod_{\ell=L-1}^{1} \left( D_\ell(x) \cdot W_\ell \right) x$$

### 2.2 Connection to ReLU Networks

**Proposition 2.1 (ReLU-GDLN Equivalence)**: Any ReLU network can be written as a GDLN where the gating functions are:
$$[G_\ell(x)]_j = \mathbb{1}\left[ h_\ell^{(j)}(x) > 0 \right]$$

where $h_\ell^{(j)}(x)$ is the pre-activation of neuron $j$ at layer $\ell$.

*Proof*: The ReLU function $\text{ReLU}(z) = \max(0, z) = z \cdot \mathbb{1}[z > 0]$ is exactly multiplication by a binary gate. $\square$

### 2.3 Key Property: Conditional Linearity

**Lemma 2.1 (Conditional Linearity)**: For any fixed gating pattern $g = (g_1, \ldots, g_{L-1})$, the function:
$$f_g(x) := W_L \cdot \text{diag}(g_{L-1}) \cdot W_{L-1} \cdots \text{diag}(g_1) \cdot W_1 \cdot x$$

is **linear** in $x$.

*Proof*: Composition of linear maps (with fixed diagonal scalings) is linear. $\square$

This is the crucial property that enables the decomposition in Theorem 1.

---

## Part 3: View-Induced Gating Patterns

### 3.1 Canonical Gating Patterns

**Definition 3.1 (View Gating Pattern)**: The canonical gating pattern for view $(y, m)$ is:
$$G^{(y,m)} := \left( G_1(\phi_{y,m}), G_2(\phi_{y,m}), \ldots, G_{L-1}(\phi_{y,m}) \right)$$

This is the gating pattern induced by the "pure" view feature $\phi_{y,m}$ (without noise or other views).

### 3.2 View-Gating Alignment Assumption

**Assumption A4 (View-Gating Distinguishability)**: Different views induce different gating patterns.

*Formally*: For all $(y, m) \neq (y', m')$, there exists at least one layer $\ell$ such that:
$$G_\ell(\phi_{y,m}) \neq G_\ell(\phi_{y',m'})$$

**Remark**: This assumption is natural because:
1. Under A2, different views point in different directions
2. Random initialization creates different neuron responses to orthogonal inputs
3. ReLU gating partitions input space into regions; orthogonal views typically fall in different regions

### 3.3 Gating Pattern Space

**Definition 3.2 (Gating Pattern)**: A gating pattern is a tuple:
$$g = (g_1, g_2, \ldots, g_{L-1}) \in \{0,1\}^{n_1} \times \{0,1\}^{n_2} \times \cdots \times \{0,1\}^{n_{L-1}}$$

**Definition 3.3 (Active Gating Patterns)**: The set of gating patterns induced by the data distribution is:
$$\mathcal{G}_{\text{active}} = \left\{ G^{(y,m)} : y \in [K], m \in [M] \right\}$$

Under A4, $|\mathcal{G}_{\text{active}}| = KM$ (all view gating patterns are distinct).

---

## Part 4: Pathways

### 4.1 Pathway Definition

**Definition 4.1 (Pathway)**: A pathway $P_g$ associated with gating pattern $g = (g_1, \ldots, g_{L-1})$ is the linear map:
$$P_g : \mathbb{R}^d \to \mathbb{R}^K$$
$$P_g(x) = W_L \cdot \text{diag}(g_{L-1}) \cdot W_{L-1} \cdots \text{diag}(g_1) \cdot W_1 \cdot x$$

**Equivalent matrix form**: $P_g$ can be written as multiplication by the matrix:
$$\mathbf{P}_g = W_L \cdot \text{diag}(g_{L-1}) \cdot W_{L-1} \cdots \text{diag}(g_1) \cdot W_1 \in \mathbb{R}^{K \times d}$$

### 4.2 View Pathways

**Definition 4.2 (View Pathway)**: The pathway for view $(y, m)$ is:
$$P_{y,m} := P_{G^{(y,m)}}$$

with associated matrix $\mathbf{P}_{y,m} \in \mathbb{R}^{K \times d}$.

### 4.3 Pathway Strength

**Definition 4.3 (Pathway Strength)**: The strength of pathway $P_g$ is:
$$\|P_g\| := \|\mathbf{P}_g\|_F = \left\| W_L \cdot \text{diag}(g_{L-1}) \cdot W_{L-1} \cdots \text{diag}(g_1) \cdot W_1 \right\|_F$$

**Alternative (Spectral)**: $\|P_g\|_{\text{op}} = \sigma_{\max}(\mathbf{P}_g)$

---

## Part 5: View Response

### 5.1 Definition

**Definition 5.1 (View Response)**: The response of network $f$ to view $(y, m)$ is:
$$R_{y,m}(f) := P_{y,m}(\phi_{y,m}) = \mathbf{P}_{y,m} \cdot \phi_{y,m} \in \mathbb{R}^K$$

This is a $K$-dimensional vector representing the network's output when presented with only view $(y, m)$.

### 5.2 View Detection

**Definition 5.2 (View Detection)**: View $(y, m)$ is detected by network $f$ if:
$$\arg\max_{k \in [K]} [R_{y,m}(f)]_k = y$$

i.e., the network correctly classifies inputs containing only view $(y, m)$.

**Definition 5.3 (Detection Margin)**: The detection margin for view $(y, m)$ is:
$$\rho_{y,m}(f) := [R_{y,m}(f)]_y - \max_{k \neq y} [R_{y,m}(f)]_k$$

View $(y, m)$ is detected iff $\rho_{y,m}(f) > 0$.

### 5.3 View Coverage

**Definition 5.4 (View Coverage)**: The view coverage of network $f$ is:
$$C(f) := \frac{1}{KM} \sum_{y=1}^{K} \sum_{m=1}^{M} \mathbb{1}\left[ \rho_{y,m}(f) > 0 \right]$$

This measures the fraction of views the network can correctly classify.

---

## Part 6: The Decomposition Theorem

### 6.1 Gating Under Multiple Views

**Key Question**: When input $x$ contains multiple views, how does the gating pattern relate to individual view gating patterns?

**Definition 6.1 (Gating Aggregation)**: Define the aggregated gating for view set $\mathcal{S}$ at layer $\ell$:
$$G_\ell^{(\mathcal{S})}(x) := G_\ell\left( \sum_{m \in \mathcal{S}} \phi_{y,m} + \varepsilon \right)$$

**Assumption A5 (Approximate Gating Additivity)**: In the low-noise regime ($\sigma \to 0$) and under view orthogonality (A2 with small $\delta$):
$$G_\ell^{(\mathcal{S})}(x) \approx \bigvee_{m \in \mathcal{S}} G_\ell^{(y,m)}$$

where $\vee$ denotes component-wise OR.

*Intuition*: If views are orthogonal, they activate different neurons. The union of active neurons under multiple views is approximately the OR of individual activations.

### 6.2 Main Decomposition

**Theorem 1 (View-Pathway Correspondence)**:

Let $\mathcal{D}$ be a $(K, M, \mathbf{p})$-multi-view distribution satisfying A1-A5. Let $f$ be a GDLN trained on $\mathcal{D}$.

For any sample $x \sim \mathcal{D}_y^{\mathcal{S}}$ (class $y$, active views $\mathcal{S}$):

**(Part A — Structural Decomposition)**:
$$f(x) = \sum_{m \in \mathcal{S}} R_{y,m}(f) + E(x, \mathcal{S})$$

where the error term satisfies:
$$\|E(x, \mathcal{S})\| \leq O\left( \delta \cdot |\mathcal{S}|^2 \cdot \max_{m} \|R_{y,m}\| + \sigma \cdot \sqrt{d} \cdot \|P_{\max}\| \right)$$

with $\|P_{\max}\| = \max_{y,m} \|\mathbf{P}_{y,m}\|_{\text{op}}$.

**(Part B — View-Pathway Bijection)**:
Under A4, the map $(y, m) \mapsto P_{y,m}$ is injective. That is, different views have different pathways.

**(Part C — Learning Equivalence)**:
View $(y, m)$ is detected by $f$ if and only if:
1. $\|P_{y,m}\| > 0$ (pathway is non-degenerate)
2. $\mathbf{P}_{y,m} \cdot \phi_{y,m}$ has its maximum component at index $y$

---

## Part 7: Proof Sketch

### 7.1 Proof of Part A

**Step 1**: Write the network output using the GDLN formula:
$$f(x) = W_L \cdot D_{L-1}(x) \cdot W_{L-1} \cdots D_1(x) \cdot W_1 \cdot x$$

**Step 2**: Substitute $x = \sum_{m \in \mathcal{S}} \phi_{y,m} + \varepsilon$:
$$f(x) = W_L \cdot D_{L-1}(x) \cdots D_1(x) \cdot W_1 \cdot \left( \sum_{m \in \mathcal{S}} \phi_{y,m} + \varepsilon \right)$$

**Step 3**: Under A5 (gating additivity), the effective gating for the sum is approximately the OR of individual gatings. This means the network output can be written as:
$$f(x) \approx \sum_{m \in \mathcal{S}} f(\phi_{y,m}) + \text{cross-terms} + \text{noise terms}$$

**Step 4**: By definition, $f(\phi_{y,m}) = R_{y,m}(f)$ (the view response).

**Step 5**: Bound the cross-terms using A2 (orthogonality). When $\delta$ is small:
$$\|\text{cross-terms}\| \leq O(\delta \cdot |\mathcal{S}|^2 \cdot \max_m \|R_{y,m}\|)$$

**Step 6**: Bound the noise terms using standard concentration:
$$\|\text{noise terms}\| \leq O(\sigma \cdot \sqrt{d} \cdot \|P_{\max}\|)$$

### 7.2 Proof of Part B

**Direct from A4**: By assumption, $G^{(y,m)} \neq G^{(y',m')}$ for $(y,m) \neq (y',m')$. Since pathways are defined by their gating patterns, different views have different pathways. $\square$

### 7.3 Proof of Part C

**Forward direction**: If view $(y,m)$ is detected, then $[R_{y,m}(f)]_y > [R_{y,m}(f)]_k$ for all $k \neq y$. This requires $R_{y,m}(f) \neq 0$, hence $\|P_{y,m}\| > 0$.

**Backward direction**: If $\|P_{y,m}\| > 0$ and the maximum component of $\mathbf{P}_{y,m} \cdot \phi_{y,m}$ is at index $y$, then by definition, view $(y,m)$ is detected. $\square$

---

## Part 8: Simplified Case — Two Views ($M = 2$)

For initial proof development, consider the symmetric two-view case.

### 8.1 Setup

- $M = 2$ views per class
- Views perfectly orthogonal: $\langle \phi_{y,1}, \phi_{y,2} \rangle = 0$ for all $y$
- Views normalized: $\|\phi_{y,m}\| = 1$
- No noise: $\sigma = 0$
- Symmetric view probability: $p_1 = p_2 = p$

### 8.2 Simplified Decomposition

For $x = \phi_{y,1} + \phi_{y,2}$ (both views active):

$$f(x) = R_{y,1}(f) + R_{y,2}(f) + E$$

where $E = 0$ under perfect orthogonality and the OR-gating assumption.

### 8.3 Two-Layer Network ($L = 2$)

Further simplification: single hidden layer with width $n$.

$$f(x) = W_2 \cdot \text{diag}(G_1(x)) \cdot W_1 \cdot x$$

**Gating**: $[G_1(x)]_j = \mathbb{1}[w_1^{(j)} \cdot x > 0]$ where $w_1^{(j)}$ is row $j$ of $W_1$.

**View pathways**:
$$\mathbf{P}_{y,m} = W_2 \cdot \text{diag}(G_1(\phi_{y,m})) \cdot W_1$$

**Orthogonality implication**: Under random initialization, with high probability:
- Some neurons activate for $\phi_{y,1}$ only
- Some neurons activate for $\phi_{y,2}$ only
- Some neurons activate for both
- Some neurons activate for neither

The pathways $P_{y,1}$ and $P_{y,2}$ differ in which neurons are active.

---

## Part 9: Connection to Race Dynamics

### 9.1 Setting Up Theorem 2

Theorem 1 establishes the **static** correspondence: views ↔ pathways.

Theorem 2 will analyze the **dynamics**: how pathway strengths evolve during training.

**Key quantities for dynamics**:
- $s_{y,m}(t) := \|P_{y,m}(t)\|$ — pathway strength at time $t$
- $\Sigma_{y,m} := \phi_{y,m} \phi_{y,m}^T$ — input correlation for view $(y,m)$

### 9.2 Preview of Race Dynamics

From Saxe et al.'s analysis of deep linear networks, pathway strengths evolve according to:
$$\frac{d s_{y,m}}{dt} \propto \sigma_1(\Sigma_{y,m}) \cdot s_{y,m} \cdot \left(1 - \frac{\sum_{m'} s_{y,m'}^2}{s_{\max}^2}\right)$$

This is a **winner-take-all** dynamic: the pathway with highest initial $\sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)$ grows fastest and eventually dominates.

**Connection to Theorem 1**:
- Learning view $(y,m)$ ⟺ $s_{y,m}$ becomes large
- Standard training ⟹ one $s_{y,m}$ dominates ⟹ one view learned
- This is the mechanism for single-view convergence

---

## Part 10: Notation Summary

| Symbol | Type | Meaning |
|--------|------|---------|
| $K$ | scalar | Number of classes |
| $M$ | scalar | Number of views per class |
| $d$ | scalar | Input dimension |
| $L$ | scalar | Number of layers |
| $n_\ell$ | scalar | Width of layer $\ell$ |
| $\phi_{y,m}$ | $\mathbb{R}^d$ | Feature vector for view $(y,m)$ |
| $W_\ell$ | $\mathbb{R}^{n_\ell \times n_{\ell-1}}$ | Weight matrix at layer $\ell$ |
| $G_\ell(x)$ | $\{0,1\}^{n_\ell}$ | Gating vector at layer $\ell$ |
| $D_\ell(x)$ | $\mathbb{R}^{n_\ell \times n_\ell}$ | Diagonal gating matrix |
| $G^{(y,m)}$ | tuple | Canonical gating pattern for view $(y,m)$ |
| $P_{y,m}$ | linear map | Pathway for view $(y,m)$ |
| $\mathbf{P}_{y,m}$ | $\mathbb{R}^{K \times d}$ | Pathway matrix |
| $R_{y,m}(f)$ | $\mathbb{R}^K$ | View response |
| $\rho_{y,m}(f)$ | scalar | Detection margin |
| $C(f)$ | $[0,1]$ | View coverage |
| $\mathcal{S}$ | subset of $[M]$ | Active views in sample |
| $\delta$ | scalar | Orthogonality violation bound |
| $\sigma$ | scalar | Noise standard deviation |

---

## Part 11: Key Assumptions Summary

| Assumption | Statement | Role |
|------------|-----------|------|
| **A1** (Sufficiency) | Each view enables correct classification | Ensures views are meaningful |
| **A2** (Orthogonality) | $\langle \phi_{y,m}, \phi_{y',m'} \rangle \approx 0$ | Enables decomposition |
| **A3** (Normalization) | $\|\phi_{y,m}\| \in [c_{\min}, c_{\max}]$ | Technical convenience |
| **A4** (Distinguishability) | Different views induce different gating | View-pathway bijection |
| **A5** (Gating Additivity) | OR aggregation of gating patterns | Enables sum decomposition |

---

## Part 12: Open Questions for Proof Development

1. **Tightness of error bounds**: Can we get sharper bounds on $\|E(x, \mathcal{S})\|$?

2. **A5 verification**: Under what conditions on $W_1$ initialization does A5 hold? Can we prove it rather than assume it?

3. **Extension to soft gating**: Real networks have "soft" transitions at ReLU boundaries. How does this affect the decomposition?

4. **Deep networks**: How do the error terms scale with depth $L$?

5. **Relaxing orthogonality**: What happens when $\delta$ is not small? Can we still get a useful (approximate) decomposition?

---

## Next Steps

1. **Prove Theorem 1 rigorously for $M=2$, $L=2$, $\delta=0$, $\sigma=0$**
2. **Verify A5 under random initialization** (probabilistic argument)
3. **Extend to $M > 2$ and $L > 2$**
4. **Develop Theorem 2** (race dynamics) building on this foundation
