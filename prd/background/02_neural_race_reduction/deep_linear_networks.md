# Deep Linear Networks: Foundation

> **Target audience**: ML researchers ready for mathematical treatment
> **Prerequisites**: Linear algebra (SVD), differential equations basics
> **Source**: Saxe, McClelland, & Ganguli (2014)

---

## Why Study Deep Linear Networks?

### The Paradox

A deep linear network with weights $W_1, W_2, \ldots, W_L$ computes:

$$f(x) = W_L W_{L-1} \cdots W_1 x = W_{\text{eff}} x$$

where $W_{\text{eff}} = W_L \cdots W_1$ is just a single matrix.

**So why not use a single layer?**

The answer: **the learning dynamics are fundamentally different**.

### What Deep Linear Networks Reveal

| Phenomenon | Single Layer | Deep Linear | Deep Nonlinear |
|------------|--------------|-------------|----------------|
| Representational power | $W_{\text{eff}}$ | Same | Higher |
| Learning dynamics | Convex | Non-convex, structured | Non-convex, complex |
| Feature learning | No | Emergent modes | Yes |
| Sequential learning | No | Yes | Yes |

Deep linear networks capture the essence of **depth-induced learning phenomena** while remaining mathematically tractable.

---

## Setup: The Learning Problem

### Architecture

**Input**: $x \in \mathbb{R}^d$
**Output**: $y \in \mathbb{R}^K$ (typically one-hot for classification)
**Network**: $L$ layers with weight matrices $W_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$

$$f(x) = W_L W_{L-1} \cdots W_1 x$$

where $n_0 = d$ and $n_L = K$.

### Loss Function

**Mean Squared Error**:
$$\mathcal{L}(W) = \frac{1}{2} \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \|y - f(x)\|^2 \right]$$

### Input-Output Correlation

**Definition**: The input-output correlation matrix is:
$$\Sigma_{yx} = \mathbb{E}[y \cdot x^T] \in \mathbb{R}^{K \times d}$$

This matrix captures what the network needs to learn—the relationship between inputs and targets.

---

## The Key Tool: Singular Value Decomposition

### SVD of Correlation Matrix

Decompose the correlation matrix:
$$\Sigma_{yx} = U S V^T$$

where:
- $U \in \mathbb{R}^{K \times r}$ — left singular vectors (output modes)
- $S = \text{diag}(s_1, s_2, \ldots, s_r)$ — singular values, $s_1 \geq s_2 \geq \cdots > 0$
- $V \in \mathbb{R}^{d \times r}$ — right singular vectors (input modes)
- $r = \text{rank}(\Sigma_{yx})$

### Interpretation of Modes

Each singular mode $\alpha$ represents a **learnable pattern**:
- Input direction: $v_\alpha$ (the $\alpha$-th column of $V$)
- Output direction: $u_\alpha$ (the $\alpha$-th column of $U$)
- Strength: $s_\alpha$ (how important is this mode?)

**Learning goal**: The network should map input direction $v_\alpha$ to output direction $u_\alpha$ with strength $s_\alpha$.

---

## Saxe et al.'s Key Results

### Result 1: Mode-by-Mode Decomposition

The effective weight matrix can be written in the SVD basis:
$$W_{\text{eff}} = \sum_{\alpha=1}^{r} a_\alpha(t) \cdot u_\alpha v_\alpha^T$$

where $a_\alpha(t)$ is the network's learned "alignment" with mode $\alpha$ at time $t$.

**Interpretation**: Learning is about adjusting the coefficients $a_\alpha(t)$ to match the target singular values $s_\alpha$.

### Result 2: Decoupled Dynamics

**The modes evolve independently!**

Under gradient flow, each mode satisfies its own differential equation:

$$\frac{da_\alpha}{dt} = s_\alpha \cdot a_\alpha \cdot \left(1 - \frac{a_\alpha^2}{s_\alpha^2}\right)$$

This is a **scalar ODE** for each mode—much simpler than the full matrix dynamics.

### Result 3: Sigmoidal Learning Curves

**Exact solution**:
$$a_\alpha(t) = \frac{s_\alpha}{\sqrt{1 + \left(\frac{s_\alpha^2}{a_\alpha(0)^2} - 1\right) e^{-2 s_\alpha t}}}$$

This is a **sigmoidal** curve: slow start, rapid middle phase, saturation at $a_\alpha \to s_\alpha$.

```
                a_α(t)
                  ↑
             s_α  │────────────────────────────╮
                  │                          ╱ │
                  │                        ╱   │
                  │                      ╱     │
                  │                    ╱       │
                  │                  ╱         │
                  │                ╱           │
                  │              ╱             │
                  │            ╱               │
           a_α(0) │──────────╱                 │
                  └─────────────────────────────→ time
                       slow    rapid    saturation
```

### Result 4: Sequential Learning

**Larger singular values are learned first.**

Mode $\alpha$ reaches half-maximum at time:
$$t_\alpha^{1/2} \approx \frac{1}{2 s_\alpha} \log\left(\frac{s_\alpha^2}{a_\alpha(0)^2}\right)$$

Since $t_\alpha^{1/2} \propto 1/s_\alpha$, modes with larger $s_\alpha$ are learned earlier.

**Implication**: The network learns the "coarse" structure first (high-variance modes), then refines with "fine" details (low-variance modes).

---

## The Two-Layer Case (L = 2)

### Setup

For maximum clarity, consider $L = 2$:
- $W_1 \in \mathbb{R}^{n \times d}$ (input → hidden)
- $W_2 \in \mathbb{R}^{K \times n}$ (hidden → output)
- $W_{\text{eff}} = W_2 W_1 \in \mathbb{R}^{K \times d}$

### Gradient Flow Equations

Under gradient descent with infinitesimal learning rate:

$$\frac{dW_1}{dt} = W_2^T (U S V^T - W_2 W_1) \Sigma_{xx}$$

$$\frac{dW_2}{dt} = (U S V^T - W_2 W_1) (W_1 \Sigma_{xx} W_1^T)$$

where $\Sigma_{xx} = \mathbb{E}[x x^T]$ is the input covariance.

### Whitened Inputs

For simplicity, assume whitened inputs: $\Sigma_{xx} = I$.

Then:
$$\frac{dW_1}{dt} = W_2^T (\Sigma_{yx} - W_2 W_1)$$
$$\frac{dW_2}{dt} = (\Sigma_{yx} - W_2 W_1) W_1^T$$

### Mode Dynamics Derivation (Sketch)

**Step 1**: Project weights onto SVD basis:
$$W_1 = \sum_\alpha b_\alpha v_\alpha q_\alpha^T, \quad W_2 = \sum_\alpha c_\alpha p_\alpha u_\alpha^T$$

where $q_\alpha, p_\alpha$ are hidden layer modes.

**Step 2**: The effective alignment is $a_\alpha = b_\alpha c_\alpha$.

**Step 3**: Under gradient flow, the dynamics reduce to:
$$\frac{da_\alpha}{dt} = s_\alpha \cdot a_\alpha - a_\alpha^3 / s_\alpha + \text{lower order terms}$$

Simplifying:
$$\frac{da_\alpha}{dt} = s_\alpha \cdot a_\alpha \cdot \left(1 - \frac{a_\alpha^2}{s_\alpha^2}\right)$$

---

## Properties of the Dynamics

### Fixed Points

Setting $\frac{da_\alpha}{dt} = 0$:

1. $a_\alpha = 0$ — **unstable** fixed point (learning hasn't started)
2. $a_\alpha = s_\alpha$ — **stable** fixed point (mode fully learned)
3. $a_\alpha = -s_\alpha$ — **stable** fixed point (mode learned with wrong sign)

In practice, initialization determines which stable fixed point is reached.

### Phase Portrait

```
     da/dt
       ↑
       │      ╱╲
       │     ╱  ╲
       │    ╱    ╲
     0 │───●──────●──────●───→ a
      -s_α  0     s_α
       │        ╱    ╲
       │       ╱      ╲
       │      ╱        ╲

● = fixed points
Arrows show direction of flow
```

For positive initialization ($a_\alpha(0) > 0$), the system flows toward $a_\alpha = s_\alpha$.

### Learning Rate

The characteristic time for mode $\alpha$ is $\tau_\alpha \sim 1/s_\alpha$.

**Modes with larger singular values learn faster.**

---

## From Modes to Features

### What Do Modes Represent?

In the input space:
- $v_\alpha$ is a direction that the network should respond to
- Think of it as a "feature template"

In the output space:
- $u_\alpha$ is the corresponding output direction
- Think of it as a "response pattern"

### Connection to Features in Deep Learning

| Deep Linear | Deep ReLU |
|-------------|-----------|
| Mode $v_\alpha$ | Learned feature |
| Alignment $a_\alpha$ | Feature strength |
| Sequential learning | Progressive feature learning |

The modes in deep linear networks are analogous to learned features in deep ReLU networks.

---

## Implications for Multi-View Learning

### Multi-View as Multi-Mode

In a multi-view setting:
- Each view corresponds to a distinct mode
- View $(y, m)$ → mode with $v_\alpha \propto \phi_{y,m}$

### Why Single-View Convergence?

If multiple views have **equal singular values** (symmetric data):
- All views start with same "learning speed" $s_\alpha$
- Initial condition $a_\alpha(0)$ determines winner
- Whichever view has higher initial alignment wins

### The Race Emerges

The dynamics $\frac{da_\alpha}{dt} \propto s_\alpha \cdot a_\alpha$ create a race:
- Multiplicative growth (exponential early)
- Small initial advantages compound
- Winner-take-all when modes compete for shared resources

---

## Summary: Key Equations

| Quantity | Definition | Interpretation |
|----------|------------|----------------|
| $\Sigma_{yx}$ | $\mathbb{E}[y x^T]$ | What to learn |
| $s_\alpha$ | $\alpha$-th singular value | Mode strength |
| $a_\alpha(t)$ | Network alignment with mode $\alpha$ | What's learned |
| Dynamics | $\dot{a}_\alpha = s_\alpha a_\alpha (1 - a_\alpha^2/s_\alpha^2)$ | How learning happens |
| Solution | $a_\alpha(t) = s_\alpha / \sqrt{1 + (s_\alpha^2/a_\alpha(0)^2 - 1)e^{-2s_\alpha t}}$ | Sigmoidal curve |
| Time scale | $\tau_\alpha \sim 1/s_\alpha$ | Larger modes faster |

---

## Limitations

1. **Linear assumption**: Real networks have nonlinearities
2. **Continuous time**: Real training uses discrete steps
3. **MSE loss**: Classification typically uses cross-entropy
4. **Independent modes**: Assumes orthogonal structure

These limitations are addressed by the **Gated Deep Linear Network** extension (next document).

---

## What's Next

- **[Gated Networks](gated_networks.md)**: Extension to input-dependent gating (ReLU equivalent)
- **[Race Dynamics](race_dynamics.md)**: Full treatment of pathway competition
- **[Worked Examples](worked_examples.md)**: Concrete calculations

---

## References

- Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). *Exact solutions to the nonlinear dynamics of learning in deep linear networks*. ICLR 2014. [arXiv:1312.6120](https://arxiv.org/abs/1312.6120)
