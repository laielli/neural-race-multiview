# View-Pathway Correspondence: Theorem 1

> **Target audience**: Researchers ready for the formal connection
> **Prerequisites**: Modules 1 and 2
> **Key result**: Views = Pathways under multi-view data

---

## Overview

Theorem 1 establishes the formal correspondence between:
- **Views** (from multi-view theory): Feature subsets $\phi_{y,m}$
- **Pathways** (from neural race reduction): Computational routes $P_{y,m}$

This correspondence is the foundation that allows us to use race dynamics to explain multi-view learning.

---

## The Setup

### Multi-View Data (from Module 1)

- $(K, M, \mathbf{p})$-multi-view distribution $\mathcal{D}$
- View features $\phi_{y,m} \in \mathbb{R}^d$
- Sample: $x = \sum_{m \in \mathcal{S}} \phi_{y,m} + \varepsilon$

### GDLN (from Module 2)

- Network: $f(x) = W_L D_{L-1}(x) W_{L-1} \cdots D_1(x) W_1 x$
- Gating: $D_\ell(x) = \text{diag}(G_\ell(x))$ with $G_\ell(x) \in \{0,1\}^{n_\ell}$
- Pathway: $P_g(x) = W_L \text{diag}(g_{L-1}) \cdots \text{diag}(g_1) W_1 x$

### The Key Question

How do views and pathways relate?

---

## The Required Assumptions

For the correspondence to hold, we need:

### A1: View Sufficiency
Each view enables correct classification.

### A2: View Orthogonality
Views are approximately orthogonal: $|\langle \phi_{y,m}, \phi_{y',m'} \rangle| \leq \delta \|\phi_{y,m}\| \|\phi_{y',m'}\|$

### A3: View Normalization
Views have comparable norms: $c_{\min} \leq \|\phi_{y,m}\| \leq c_{\max}$

### A4: View-Gating Distinguishability (New)

**Statement**: Different views induce different gating patterns.

Formally: For $(y, m) \neq (y', m')$, there exists layer $\ell$ such that:
$$G_\ell(\phi_{y,m}) \neq G_\ell(\phi_{y',m'})$$

**Why this holds**: Under random initialization, orthogonal inputs activate different neurons with high probability.

### A5: Gating Additivity (New)

**Statement**: Multi-view gating is approximately OR of individual view gatings.

Formally: For input $x = \sum_{m \in \mathcal{S}} \phi_{y,m} + \varepsilon$:
$$G_\ell(x) \approx \bigvee_{m \in \mathcal{S}} G_\ell(\phi_{y,m})$$

**Why this holds**: Orthogonal views activate disjoint neuron sets; union is OR.

---

## Theorem 1: View-Pathway Correspondence

### Statement

Let $\mathcal{D}$ be a $(K, M, \mathbf{p})$-multi-view distribution satisfying A1-A5. Let $f$ be a GDLN.

For any sample $x \sim \mathcal{D}_y^{\mathcal{S}}$ (class $y$, active views $\mathcal{S}$):

---

**Part A — Structural Decomposition**:

$$\boxed{f(x) = \sum_{m \in \mathcal{S}} R_{y,m}(f) + E(x, \mathcal{S})}$$

where:
- $R_{y,m}(f) = \mathbf{P}_{y,m} \cdot \phi_{y,m}$ is the view response
- $\mathbf{P}_{y,m} = W_L D_{L-1}^{(y,m)} \cdots D_1^{(y,m)} W_1$ is the view pathway matrix
- $E(x, \mathcal{S})$ is an error term

**Error bound**:
$$\|E(x, \mathcal{S})\| \leq O\left(\delta \cdot |\mathcal{S}|^2 \cdot \max_m \|R_{y,m}\| + \sigma \sqrt{d} \|\mathbf{P}_{\max}\|_{\text{op}}\right)$$

---

**Part B — View-Pathway Bijection**:

Under A4, the map $(y, m) \mapsto P_{y,m}$ is **injective**.

Different views have different pathways.

---

**Part C — Learning Equivalence**:

View $(y, m)$ is detected by $f$ if and only if:
1. Pathway is non-degenerate: $\|\mathbf{P}_{y,m}\| > 0$
2. Correct classification: $\arg\max_k [\mathbf{P}_{y,m} \cdot \phi_{y,m}]_k = y$

---

## Interpretation

### Part A: Decomposition

The network's output is a **sum of view responses**.

```
Input x = φ_{y,1} + φ_{y,3} + ε  (views 1 and 3 active)
              ↓
         ┌────────────┐
         │  Network f │
         └────────────┘
              ↓
f(x) ≈ R_{y,1}(f) + R_{y,3}(f)
       └─────┬─────┘   └─────┬─────┘
         response to      response to
          view 1           view 3
```

**Key insight**: The network processes each view through its own pathway, then sums the results.

### Part B: Bijection

Each view has a **unique** pathway.

```
View (y=3, m=1)  ←→  Pathway P_{3,1}  (neurons {A, C, E} active)
View (y=3, m=2)  ←→  Pathway P_{3,2}  (neurons {B, D, F} active)
View (y=3, m=3)  ←→  Pathway P_{3,3}  (neurons {A, D, G} active)
```

No two views share the same pathway.

### Part C: Equivalence

**Learning a view = Strengthening its pathway**.

$$\text{View detected} \Leftrightarrow \text{Pathway strong and correctly aligned}$$

This is why race dynamics (pathway competition) explain multi-view learning.

---

## Proof Sketch

### Part A: Decomposition

**Step 1**: Start with GDLN formula:
$$f(x) = W_L D_{L-1}(x) \cdots D_1(x) W_1 x$$

**Step 2**: Substitute $x = \sum_{m \in \mathcal{S}} \phi_{y,m} + \varepsilon$

**Step 3**: Under A5 (gating additivity), the gating for the sum is:
$$D_\ell(x) \approx \text{diag}\left(\bigvee_{m \in \mathcal{S}} G_\ell(\phi_{y,m})\right)$$

**Step 4**: Expand using distributivity of matrix multiplication over addition:
$$f(x) \approx \sum_{m \in \mathcal{S}} W_L D_{L-1}^{(m)} \cdots D_1^{(m)} W_1 \phi_{y,m} + \text{cross terms} + \text{noise}$$

**Step 5**: The pure view term is exactly $R_{y,m}(f)$. Cross terms are small by A2 (orthogonality).

### Part B: Bijection

**Direct from A4**: Different views have different gating patterns by assumption. Since pathways are defined by gating patterns, different views have different pathways. $\square$

### Part C: Equivalence

**Forward**: If view detected, then $R_{y,m}$ has max at index $y$, so $\mathbf{P}_{y,m} \neq 0$.

**Backward**: If $\mathbf{P}_{y,m} \phi_{y,m}$ has max at $y$, view is detected by definition. $\square$

---

## The Two-Layer Case (L = 2)

### Explicit Formulas

Pathway matrix:
$$\mathbf{P}_{y,m} = W_2 \cdot \text{diag}(g_{y,m}) \cdot W_1$$

where $g_{y,m} = G_1(\phi_{y,m}) \in \{0,1\}^n$ is the hidden layer gating.

View response:
$$R_{y,m}(f) = W_2 \cdot \text{diag}(g_{y,m}) \cdot W_1 \cdot \phi_{y,m}$$

### Decomposition in 2-Layer Network

For input with views $\mathcal{S} = \{1, 2\}$:

$$f(x) = W_2 \cdot \text{diag}(g_1 \vee g_2) \cdot W_1 \cdot (\phi_1 + \phi_2 + \varepsilon)$$

Under orthogonality (views in disjoint dimensions):
$$\approx W_2 \text{diag}(g_1) W_1 \phi_1 + W_2 \text{diag}(g_2) W_1 \phi_2 + \text{noise}$$
$$= R_{y,1}(f) + R_{y,2}(f) + E$$

---

## Why A4 (Distinguishability) Holds

### Intuition

Different views point in different directions. Under random initialization, different directions activate different neurons.

### Formal Argument (Sketch)

Let $w_j \in \mathbb{R}^d$ be the weight vector for hidden neuron $j$.

Neuron $j$ is active for view $(y,m)$ iff $\langle w_j, \phi_{y,m} \rangle > 0$.

Under random $w_j \sim \mathcal{N}(0, I/d)$ and orthogonal views:
- $\langle w_j, \phi_{y,1} \rangle$ and $\langle w_j, \phi_{y,2} \rangle$ are independent
- Probability both have same sign: 50%
- Over $n$ neurons, expected difference: $n/2$ neurons differ

**Conclusion**: With high probability, orthogonal views have different gating patterns.

---

## Why A5 (Additivity) Holds

### Intuition

If views activate disjoint neuron sets, their union is simply OR.

### Our Slot Structure

In our synthetic data:
- View 1 features in dims 0-49
- View 2 features in dims 50-99
- View 3 features in dims 100-149

Neurons primarily sensitive to:
- Dims 0-49 → activated by View 1
- Dims 50-99 → activated by View 2
- Dims 100-149 → activated by View 3

**Result**: $G(\phi_1 + \phi_2) = G(\phi_1) \vee G(\phi_2)$ exactly.

### General Case

When views are only approximately orthogonal ($\delta > 0$), some neurons may respond to multiple views. The error term captures this.

---

## Implications of Theorem 1

### Implication 1: Reduced Complexity

Instead of analyzing the full network, we analyze individual pathways.

**Before**: $f: \mathbb{R}^d \to \mathbb{R}^K$ (complex nonlinear function)

**After**: $\{P_{y,m}\}_{y,m}$ (collection of linear maps)

### Implication 2: Race Framework Applies

Since views = pathways, the race dynamics from Module 2 directly apply.

- Pathway strength $s_{y,m}(t) = \|\mathbf{P}_{y,m}(t)\|_F$
- Race equation: $\dot{s}_{y,m} = \sigma_1 s_{y,m} (1 - \text{comp})$
- Winner: $\arg\max_m A_{y,m}$

### Implication 3: View Coverage = Pathway Strength Distribution

$$C(f) = \frac{1}{KM} \sum_{y,m} \mathbb{1}[s_{y,m} > \epsilon \text{ and correctly aligned}]$$

Single-view convergence means one pathway per class is strong; others are weak.

---

## Summary

### Theorem 1 in One Sentence

Under multi-view data, each view maps to a distinct computational pathway, and the network output decomposes as a sum of view responses.

### The Three Parts

| Part | Statement | Significance |
|------|-----------|--------------|
| A | $f(x) = \sum_m R_{y,m} + E$ | Output decomposes by view |
| B | Views ↔ Pathways bijection | Different views, different pathways |
| C | Detection ↔ Strong pathway | Learning view = strengthening pathway |

### Why It Matters

Theorem 1 is the **bridge** between multi-view theory and neural race reduction. Without it, we couldn't apply race dynamics to explain multi-view learning.

---

## What's Next

- **[Why KD Works](why_kd_works.md)**: Theorem 3 — How soft labels break the race
- **[Worked Examples](worked_examples.md)**: Concrete decomposition calculations
