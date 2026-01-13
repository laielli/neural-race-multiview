# Gated Deep Linear Networks

> **Target audience**: ML researchers comfortable with ReLU networks
> **Prerequisites**: [Deep Linear Networks](deep_linear_networks.md)
> **Key insight**: ReLU networks = Gated Deep Linear Networks

---

## The Bridge: From Linear to ReLU

### The Problem with Deep Linear Networks

Deep linear networks provide analytical tractability but miss a key aspect of real networks:

**Nonlinearity creates input-dependent computation.**

In a ReLU network, different inputs activate different neurons. The same weights compute different functions depending on which neurons fire.

### The Solution: Input-Dependent Gating

A **Gated Deep Linear Network (GDLN)** introduces binary gates that depend on the input:

$$f(x) = W_L \cdot D_{L-1}(x) \cdot W_{L-1} \cdots D_1(x) \cdot W_1 \cdot x$$

where $D_\ell(x) = \text{diag}(G_\ell(x))$ is a diagonal matrix of binary gates.

**Key point**: For any *fixed* input $x$, the function is linear in $x$. But different inputs may have different gates active.

---

## ReLU-GDLN Equivalence

### ReLU as Gated Linear

The ReLU function:
$$\text{ReLU}(z) = \max(0, z) = z \cdot \mathbb{1}[z > 0]$$

This is multiplication by a binary gate that depends on the sign of $z$.

### Formal Equivalence

**Proposition**: Any ReLU network can be written as a GDLN where:
$$[G_\ell(x)]_j = \mathbb{1}\left[ h_\ell^{(j)}(x) > 0 \right]$$

Here $h_\ell^{(j)}(x)$ is the pre-activation of neuron $j$ at layer $\ell$.

**Proof**:
$$\text{ReLU}(W_\ell h_{\ell-1}(x))_j = [W_\ell h_{\ell-1}(x)]_j \cdot \mathbb{1}[[W_\ell h_{\ell-1}(x)]_j > 0]$$

The indicator function is exactly the gate $[G_\ell(x)]_j$. $\square$

### Implications

| ReLU Network | GDLN View |
|--------------|-----------|
| ReLU activations | Binary gates $G_\ell(x)$ |
| Active neurons | Gates = 1 |
| Inactive neurons | Gates = 0 |
| Network output | $W_L D_{L-1}(x) \cdots D_1(x) W_1 x$ |

---

## Pathways: The Central Concept

### Definition

**Definition (Gating Pattern)**: A gating pattern is a tuple of binary vectors:
$$g = (g_1, g_2, \ldots, g_{L-1})$$
where $g_\ell \in \{0, 1\}^{n_\ell}$ specifies which neurons are active at layer $\ell$.

**Definition (Pathway)**: The pathway associated with gating pattern $g$ is the linear map:
$$P_g(x) = W_L \cdot \text{diag}(g_{L-1}) \cdot W_{L-1} \cdots \text{diag}(g_1) \cdot W_1 \cdot x$$

Or equivalently, the matrix:
$$\mathbf{P}_g = W_L \cdot \text{diag}(g_{L-1}) \cdot W_{L-1} \cdots \text{diag}(g_1) \cdot W_1 \in \mathbb{R}^{K \times d}$$

### Intuition

A pathway is the **effective linear function** computed by the network for inputs that produce gating pattern $g$.

**Example** (2-layer network, 4 hidden neurons):

```
Gating pattern g = [1, 0, 1, 0]  (neurons 1 and 3 active)

Input x → W₁x → [h₁, h₂, h₃, h₄]
                  ↓    ↓    ↓    ↓
         gates:  [1]  [0]  [1]  [0]
                  ↓         ↓
              [h₁, 0, h₃, 0]  → W₂ → output

Pathway matrix: P_g = W₂ · diag([1,0,1,0]) · W₁
             = only rows 1,3 of W₁ and cols 1,3 of W₂ matter
```

### Pathway Strength

**Definition (Pathway Strength)**:
$$s_g = \|\mathbf{P}_g\|_F = \left\| W_L \prod_{\ell=L-1}^{1} \text{diag}(g_\ell) W_\ell \right\|_F$$

This measures how "powerful" the pathway is—how much it can affect the output.

---

## View Pathways

### Connecting to Multi-View Theory

In multi-view data, each view $(y, m)$ has a characteristic feature vector $\phi_{y,m}$.

This feature induces a **canonical gating pattern**:
$$G^{(y,m)} = \left( G_1(\phi_{y,m}), G_2(\phi_{y,m}), \ldots, G_{L-1}(\phi_{y,m}) \right)$$

This is the gating that occurs when the input is the "pure" view feature.

### View Pathway Definition

**Definition (View Pathway)**: The pathway for view $(y, m)$ is:
$$P_{y,m} := P_{G^{(y,m)}}$$

with matrix $\mathbf{P}_{y,m} \in \mathbb{R}^{K \times d}$.

**Definition (View Response)**: The response of the network to view $(y, m)$ is:
$$R_{y,m}(f) = P_{y,m}(\phi_{y,m}) = \mathbf{P}_{y,m} \cdot \phi_{y,m} \in \mathbb{R}^K$$

### Why Different Views Have Different Pathways

Under the **View-Gating Distinguishability** assumption (A4):

Different views point in different directions in input space, so they activate different neurons.

**Example**:
- View 1: feature in dimensions 0-49 → activates neurons sensitive to those dimensions
- View 2: feature in dimensions 50-99 → activates different neurons

Therefore: $G^{(y,1)} \neq G^{(y,2)}$ → different pathways.

---

## The Two-Layer Case

### Explicit Formulas

For $L = 2$:
- $W_1 \in \mathbb{R}^{n \times d}$ (input → hidden)
- $W_2 \in \mathbb{R}^{K \times n}$ (hidden → output)
- Gating $g = G_1(x) \in \{0, 1\}^n$

**Pathway matrix**:
$$\mathbf{P}_g = W_2 \cdot \text{diag}(g) \cdot W_1$$

**View pathway**:
$$\mathbf{P}_{y,m} = W_2 \cdot \text{diag}(G_1(\phi_{y,m})) \cdot W_1$$

### Decomposition by Active Neurons

Let $\mathcal{A}(g) = \{j : g_j = 1\}$ be the set of active neurons.

$$\mathbf{P}_g = \sum_{j \in \mathcal{A}(g)} w_2^{(j)} \cdot (w_1^{(j)})^T$$

where:
- $w_1^{(j)}$ = row $j$ of $W_1$ (input weights to neuron $j$)
- $w_2^{(j)}$ = column $j$ of $W_2$ (output weights from neuron $j$)

**Interpretation**: The pathway is a sum of rank-1 contributions from each active neuron.

### Pathway Strength Formula

$$s_g = \|\mathbf{P}_g\|_F = \left\| \sum_{j \in \mathcal{A}(g)} w_2^{(j)} (w_1^{(j)})^T \right\|_F$$

At initialization (small random weights):
$$s_g(0) \approx \sigma_0^2 \cdot \sqrt{|\mathcal{A}(g)| \cdot K \cdot d}$$

where $\sigma_0$ is the initialization scale.

**Key insight**: Pathways with more active neurons have higher initial strength.

---

## Conditional Linearity: The Key Property

### Statement

**Lemma (Conditional Linearity)**: For any fixed gating pattern $g$, the function $f_g(x) = P_g(x)$ is **linear** in $x$.

**Proof**: It's a composition of matrix multiplications (with fixed diagonal matrices). $\square$

### Why This Matters

1. **Analysis**: For any specific input, we can use linear algebra tools
2. **Decomposition**: The network output is a sum of pathway contributions
3. **Gradients**: Gradient computation simplifies when gating is fixed

### The Approximation

In practice, gating patterns change during training as weights change. The GDLN analysis assumes **fixed gating** (or slowly-changing gating).

This is a good approximation when:
- Views are well-separated (different views activate clearly different neurons)
- Training is analyzed in phases (early dynamics with stable gating)

---

## Output Decomposition Theorem

### Setup

Input $x$ from class $y$ with active views $\mathcal{S} \subseteq [M]$:
$$x = \sum_{m \in \mathcal{S}} \phi_{y,m} + \varepsilon$$

### The Decomposition

**Theorem 1 (View-Pathway Correspondence), Part A**:

$$f(x) = \sum_{m \in \mathcal{S}} R_{y,m}(f) + E(x, \mathcal{S})$$

where:
- $R_{y,m}(f) = \mathbf{P}_{y,m} \cdot \phi_{y,m}$ is the response to view $m$
- $E(x, \mathcal{S})$ is an error term

**Error bound**:
$$\|E\| \leq O\left(\delta \cdot |\mathcal{S}|^2 \cdot \max_m \|R_{y,m}\| + \sigma \cdot \sqrt{d} \cdot \|\mathbf{P}_{\max}\|_{\text{op}}\right)$$

where $\delta$ is the orthogonality violation and $\sigma$ is noise level.

### Interpretation

The network's output is approximately a **sum of view responses**.

When views are orthogonal and noise is low, the error is small, and:
$$f(x) \approx \sum_{m \in \mathcal{S}} R_{y,m}(f)$$

**This connects multi-view data structure to pathway decomposition.**

---

## Gating Under Multiple Views

### The Challenge

When input contains multiple views, which gating pattern is used?

$$x = \phi_{y,1} + \phi_{y,2} + \varepsilon$$

Does the network use $G^{(y,1)}$, $G^{(y,2)}$, or something else?

### The Approximation: OR-Gating

**Assumption A5 (Gating Additivity)**: In the low-noise regime with orthogonal views:

$$G_\ell^{(\mathcal{S})}(x) \approx \bigvee_{m \in \mathcal{S}} G_\ell^{(y,m)}$$

where $\vee$ is component-wise OR.

**Intuition**: If views are orthogonal, they activate non-overlapping sets of neurons. The union of active neurons is the OR of individual activations.

### When This Holds

The OR-gating approximation is good when:
- Views use disjoint input dimensions (our slot structure)
- Noise is small compared to view signal
- Views don't have overlapping "sensitive" neurons

---

## Summary: The GDLN Framework

### Key Objects

| Object | Notation | Meaning |
|--------|----------|---------|
| Gating pattern | $g = (g_1, \ldots, g_{L-1})$ | Which neurons active |
| Pathway | $P_g$ | Linear map for gating $g$ |
| Pathway matrix | $\mathbf{P}_g$ | Matrix representation |
| Pathway strength | $s_g = \|\mathbf{P}_g\|_F$ | How powerful |
| View pathway | $P_{y,m}$ | Pathway for view $(y,m)$ |
| View response | $R_{y,m} = P_{y,m}(\phi_{y,m})$ | Network response to view |

### Key Results

1. **ReLU = GDLN**: Any ReLU network is a GDLN with input-dependent gates
2. **Conditional linearity**: For fixed gating, the network is linear
3. **View-pathway correspondence**: Different views → different pathways
4. **Output decomposition**: $f(x) \approx \sum_{m \in \mathcal{S}} R_{y,m}(f)$

### Why This Matters

The GDLN framework lets us:
- **Decompose** network computation into pathway contributions
- **Track** individual pathway strengths during training
- **Analyze** competition between pathways (the race)
- **Predict** which pathway (view) will dominate

---

## What's Next

- **[Race Dynamics](race_dynamics.md)**: How pathways compete during training
- **[Worked Examples](worked_examples.md)**: Concrete calculations with our setup

---

## References

- Saxe, A. M., Sodhani, S., & Gershman, S. (2022). *The Neural Race Reduction: Dynamics of abstraction in gated networks*. ICML 2022.
- Jarvis, D., Klein, R., & Rosman, B. (2025). *On the relationship between ReLU networks and gated linear networks*. ICML 2025.
