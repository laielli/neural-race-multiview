# Theorem 2: Race Dynamics Under Multi-View Data — Formal Setup

## Overview

This document develops the formal mathematical framework for Theorem 2, which analyzes how pathway strengths evolve during training and explains why one view "wins" per class.

**Building on**: Theorem 1 (View-Pathway Correspondence) establishes that views map to pathways. Theorem 2 analyzes the *dynamics* of these pathways during training.

**Goal**: Show that training dynamics create a "race" where one pathway per class dominates, explaining single-view convergence.

---

## Part 1: Background — Deep Linear Network Dynamics

### 1.1 Saxe et al. (2014) Key Results

For a **deep linear network** $f(x) = W_L W_{L-1} \cdots W_1 x$ trained on MSE loss:

**Input-Output Correlation**: Define
$$\Sigma_{yx} = \mathbb{E}[y \cdot x^T] \in \mathbb{R}^{K \times d}$$

**SVD Decomposition**:
$$\Sigma_{yx} = U S V^T$$
where $S = \text{diag}(s_1, \ldots, s_r)$ with $s_1 \geq s_2 \geq \cdots \geq s_r > 0$.

**Key Theorem (Saxe et al.)**: Under gradient flow, each singular mode $\alpha$ evolves according to:
$$\frac{da_\alpha}{dt} = s_\alpha \cdot a_\alpha \cdot \left(1 - \frac{a_\alpha^2}{s_\alpha^2}\right)$$

where $a_\alpha(t)$ is the network's "alignment" with mode $\alpha$.

**Implications**:
1. Modes with larger singular values $s_\alpha$ learn faster
2. Learning is **sequential**: larger modes learned first
3. Each mode follows a **sigmoidal** trajectory

### 1.2 Extension to Gated Networks (Saxe et al. 2022)

For **Gated Deep Linear Networks**, the key insight is:
- Different gating patterns define different **pathways**
- Each pathway has its own effective correlation structure
- Pathways **race** to explain the data
- The pathway with highest correlation strength wins

---

## Part 2: Multi-View Correlation Structure

### 2.1 Setup

Recall from Theorem 1:
- Multi-view distribution $\mathcal{D}$ with $K$ classes, $M$ views per class
- View features $\phi_{y,m} \in \mathbb{R}^d$ for class $y$, view $m$
- View pathways $P_{y,m}$ with gating patterns $G^{(y,m)}$

### 2.2 Per-View Correlation Matrix

**Definition 2.1 (View Correlation)**: The input-output correlation contributed by view $(y,m)$ is:
$$\Sigma_{y,m} = \mathbb{E}_{x \sim \mathcal{D}}\left[ \mathbb{1}[(y,m) \text{ active in } x] \cdot e_y \cdot x^T \right]$$

where $e_y \in \mathbb{R}^K$ is the one-hot encoding of class $y$.

**Lemma 2.1**: Under the multi-view sampling model:
$$\Sigma_{y,m} = \frac{p_m}{K} \cdot e_y \cdot \mathbb{E}[\phi_{y,m} + \text{other views} + \varepsilon]^T$$

**Simplification (Single Active View)**: When views are sparse (typically one active per sample):
$$\Sigma_{y,m} \approx \frac{p_m}{K} \cdot e_y \cdot \phi_{y,m}^T$$

This is a **rank-1 matrix** with:
- Singular value: $\sigma_1(\Sigma_{y,m}) = \frac{p_m}{K} \cdot \|\phi_{y,m}\|$
- Left singular vector: $e_y / \|e_y\| = e_y$
- Right singular vector: $\phi_{y,m} / \|\phi_{y,m}\|$

### 2.3 Total Correlation Decomposition

**Proposition 2.1**: The total input-output correlation decomposes by view:
$$\Sigma_{yx} = \sum_{y=1}^{K} \sum_{m=1}^{M} \Sigma_{y,m} + \Sigma_{\text{cross}}$$

where $\Sigma_{\text{cross}}$ captures cross-view interactions (small under orthogonality assumption A2).

---

## Part 3: Pathway Strength Dynamics

### 3.1 Pathway Strength Definition

**Definition 3.1 (Pathway Strength)**: The strength of pathway $(y,m)$ at time $t$ is:
$$s_{y,m}(t) := \|\mathbf{P}_{y,m}(t)\|_F$$

where $\mathbf{P}_{y,m}(t) = W_L(t) \cdot D_{L-1}^{(y,m)} \cdot W_{L-1}(t) \cdots D_1^{(y,m)} \cdot W_1(t)$.

**Note**: The gating matrices $D_\ell^{(y,m)} = \text{diag}(G_\ell(\phi_{y,m}))$ are **fixed** (determined by initialization and view features). Only the weights $W_\ell(t)$ evolve.

### 3.2 Gradient Flow

**Training Dynamics**: Under gradient flow on loss $\mathcal{L}$:
$$\frac{dW_\ell}{dt} = -\nabla_{W_\ell} \mathcal{L}$$

**Loss Function**: Cross-entropy loss
$$\mathcal{L} = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ -\log \frac{\exp(f_y(x))}{\sum_k \exp(f_k(x))} \right]$$

### 3.3 Induced Dynamics on Pathway Strength

**Chain Rule**:
$$\frac{ds_{y,m}}{dt} = \sum_{\ell=1}^{L} \left\langle \frac{\partial s_{y,m}}{\partial W_\ell}, \frac{dW_\ell}{dt} \right\rangle$$

**Key Challenge**: Computing $\frac{\partial s_{y,m}}{\partial W_\ell}$ for the Frobenius norm of a matrix product.

---

## Part 4: Two-Layer Analysis ($L = 2$)

### 4.1 Setup

For tractability, consider $L = 2$:
- $W_1 \in \mathbb{R}^{n \times d}$ (input to hidden)
- $W_2 \in \mathbb{R}^{K \times n}$ (hidden to output)
- Gating $g_{y,m} = G_1(\phi_{y,m}) \in \{0,1\}^n$

**Pathway matrix**:
$$\mathbf{P}_{y,m} = W_2 \cdot \text{diag}(g_{y,m}) \cdot W_1 \in \mathbb{R}^{K \times d}$$

**Pathway strength**:
$$s_{y,m} = \|\mathbf{P}_{y,m}\|_F = \|W_2 \cdot \text{diag}(g_{y,m}) \cdot W_1\|_F$$

### 4.2 Gradient Computation

**Notation**: Let $D_m = \text{diag}(g_{y,m})$ for brevity.

**Gradient of loss w.r.t. $W_2$**:
$$\nabla_{W_2} \mathcal{L} = \mathbb{E}\left[ (p(x) - e_y) \cdot h(x)^T \right]$$

where:
- $p(x) = \text{softmax}(f(x)) \in \mathbb{R}^K$ is the predicted distribution
- $h(x) = \text{ReLU}(W_1 x) \in \mathbb{R}^n$ is the hidden activation
- $e_y$ is the one-hot target

**Gradient of loss w.r.t. $W_1$**:
$$\nabla_{W_1} \mathcal{L} = \mathbb{E}\left[ D(x) \cdot W_2^T \cdot (p(x) - e_y) \cdot x^T \right]$$

where $D(x) = \text{diag}(\mathbb{1}[W_1 x > 0])$ is the ReLU gating.

### 4.3 Decomposition by Pathway

**Key Insight**: The gradient decomposes by which gating pattern is active.

For input $x$ with active views $\mathcal{S}$:
$$D(x) \approx \bigvee_{m \in \mathcal{S}} D_m$$

Under the sparse activation regime (one view per sample):
$$\nabla_{W_\ell} \mathcal{L} \approx \sum_{y,m} w_{y,m} \cdot \nabla_{W_\ell}^{(y,m)}$$

where:
- $w_{y,m}$ = weight (frequency of view $(y,m)$)
- $\nabla_{W_\ell}^{(y,m)}$ = gradient contribution from view $(y,m)$

### 4.4 Pathway-Specific Gradient

**Definition 4.1**: The gradient contribution from pathway $(y,m)$ is:
$$\nabla_{W_2}^{(y,m)} = \mathbb{E}_{x \sim \mathcal{D}_y^{\{m\}}} \left[ (p(x) - e_y) \cdot (D_m W_1 x)^T \right]$$

$$\nabla_{W_1}^{(y,m)} = \mathbb{E}_{x \sim \mathcal{D}_y^{\{m\}}} \left[ D_m \cdot W_2^T \cdot (p(x) - e_y) \cdot x^T \right]$$

### 4.5 Dynamics of Pathway Strength

**Proposition 4.1 (Pathway Strength Dynamics)**: Under gradient flow:
$$\frac{ds_{y,m}}{dt} = \underbrace{\sigma_1(\Sigma_{y,m}) \cdot s_{y,m}}_{\text{growth}} \cdot \underbrace{\left(1 - \frac{\sum_{m'} s_{y,m'}^2}{s_{\max}^2}\right)}_{\text{competition}} + \underbrace{O(\delta + \sigma_0^2)}_{\text{interactions}}$$

where:
- $\sigma_1(\Sigma_{y,m})$ = correlation strength for view $(y,m)$
- $s_{\max}$ = maximum achievable pathway strength (saturation)
- The competition term is shared across views of the same class

**Interpretation**:
1. **Growth term**: Pathway grows proportionally to its current strength and correlation
2. **Competition term**: As total pathway strength increases, growth slows
3. **Interaction term**: Cross-pathway effects (small under assumptions)

---

## Part 5: The Race — Winner-Take-All Dynamics

### 5.1 Early Phase Analysis

**Definition 5.1 (Early Phase)**: The time interval $t \in [0, T_{\text{early}}]$ where all pathway strengths are small:
$$s_{y,m}(t) \ll s_{\max} \quad \forall y, m$$

**Proposition 5.1 (Exponential Growth)**: In the early phase:
$$s_{y,m}(t) \approx s_{y,m}(0) \cdot \exp\left(\sigma_1(\Sigma_{y,m}) \cdot t\right)$$

*Proof Sketch*: When $\sum_{m'} s_{y,m'}^2 \ll s_{\max}^2$, the competition term $\approx 1$, giving:
$$\frac{ds_{y,m}}{dt} \approx \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}$$
which has solution $s_{y,m}(t) = s_{y,m}(0) \cdot e^{\sigma_1 t}$. $\square$

### 5.2 Initial Advantage

**Definition 5.2 (Initial Advantage)**: The initial advantage of pathway $(y,m)$ is:
$$A_{y,m} := \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)$$

**Components**:
1. **Correlation strength** $\sigma_1(\Sigma_{y,m}) = \frac{p_m}{K} \cdot \|\phi_{y,m}\|$
   - Larger view norm → stronger correlation → faster learning
   - Higher view probability → more training signal

2. **Initial pathway strength** $s_{y,m}(0) = \|W_2(0) D_m W_1(0)\|_F$
   - Depends on random initialization
   - Depends on gating pattern (how many neurons active)

### 5.3 Winner Determination

**Theorem 5.1 (Race Outcome)**: Consider two pathways $(y, m_1)$ and $(y, m_2)$ for the same class. If:
$$A_{y,m_1} > A_{y,m_2}$$

then as $t \to \infty$:
$$\frac{s_{y,m_2}(t)}{s_{y,m_1}(t)} \to 0$$

*Proof Sketch*:

**Step 1**: In early phase, both grow exponentially:
$$s_{y,m_1}(t) \approx s_{y,m_1}(0) \cdot e^{\sigma_1^{(1)} t}$$
$$s_{y,m_2}(t) \approx s_{y,m_2}(0) \cdot e^{\sigma_1^{(2)} t}$$

**Step 2**: The ratio evolves as:
$$\frac{s_{y,m_2}(t)}{s_{y,m_1}(t)} = \frac{s_{y,m_2}(0)}{s_{y,m_1}(0)} \cdot e^{(\sigma_1^{(2)} - \sigma_1^{(1)}) t}$$

If $\sigma_1^{(1)} > \sigma_1^{(2)}$, this ratio decays exponentially.

**Step 3**: Even if $\sigma_1^{(1)} = \sigma_1^{(2)}$ but $s_{y,m_1}(0) > s_{y,m_2}(0)$, pathway 1 maintains its lead. The competition term then favors the leader.

**Step 4**: As pathway 1 grows, it "consumes" the available learning signal, leaving less for pathway 2. This creates a rich-get-richer dynamic. $\square$

### 5.4 Formal Race Dynamics

**Definition 5.3 (Competitive System)**: The pathway strengths satisfy the competitive dynamical system:

$$\frac{ds_{y,m}}{dt} = \sigma_1^{(y,m)} \cdot s_{y,m} \cdot \left(1 - \frac{\sum_{m'=1}^{M} s_{y,m'}^2}{s_{\max}^2}\right)$$

for each class $y$ and view $m$.

**Proposition 5.2**: This system has $M$ stable fixed points (for each class):
$$s_{y,m^*} = s_{\max}, \quad s_{y,m} = 0 \text{ for } m \neq m^*$$

for each $m^* \in [M]$. The basin of attraction of fixed point $m^*$ is approximately:
$$\mathcal{B}_{m^*} = \left\{ (s_1, \ldots, s_M) : A_{y,m^*} > A_{y,m} \text{ for all } m \neq m^* \right\}$$

---

## Part 6: Formal Statement of Theorem 2

### Theorem 2 (Race Dynamics Under Multi-View Data)

Let $\mathcal{D}$ be a $(K, M, \mathbf{p})$-multi-view distribution satisfying assumptions A1-A5. Let $f$ be a GDLN with $L=2$ layers, trained via gradient flow on cross-entropy loss. Initialize weights i.i.d. from $\mathcal{N}(0, \sigma_0^2/n)$.

---

**(Part A — Pathway Strength Dynamics)**:

The pathway strengths evolve according to:
$$\frac{ds_{y,m}}{dt} = \sigma_1(\Sigma_{y,m}) \cdot s_{y,m} \cdot \left(1 - \frac{\sum_{m'} s_{y,m'}^2}{s_{\max}^2}\right) + E_{y,m}(t)$$

where the error $|E_{y,m}(t)| \leq O(\delta \cdot s_{\max} + \sigma_0^2)$.

---

**(Part B — Early Phase Exponential Growth)**:

For $t \in [0, T_{\text{early}}]$ where $T_{\text{early}} = \Theta\left(\frac{1}{\max_{y,m} \sigma_1(\Sigma_{y,m})} \log \frac{s_{\max}}{s_{y,m}(0)}\right)$:

$$s_{y,m}(t) = s_{y,m}(0) \cdot \exp\left(\sigma_1(\Sigma_{y,m}) \cdot t \cdot (1 + O(\sigma_0))\right)$$

---

**(Part C — Winner Determination)**:

Define the initial advantage $A_{y,m} := \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)$.

With probability $\geq 1 - \exp(-\Omega(n))$ over random initialization, for each class $y$:

1. There exists a unique winning view:
   $$m^*(y) = \arg\max_{m \in [M]} A_{y,m}$$

2. The winning pathway dominates:
   $$\lim_{t \to \infty} \frac{s_{y,m}(t)}{s_{y,m^*(y)}(t)} = 0 \quad \text{for all } m \neq m^*(y)$$

---

**(Part D — Predictive Formula)**:

The probability that view $m$ wins for class $y$ is:
$$\mathbb{P}(m^*(y) = m) = \mathbb{P}\left( \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0) > \max_{m' \neq m} \sigma_1(\Sigma_{y,m'}) \cdot s_{y,m'}(0) \right)$$

Under symmetric views ($\sigma_1(\Sigma_{y,m})$ equal for all $m$):
$$\mathbb{P}(m^*(y) = m) \approx \frac{\mathbb{E}[s_{y,m}(0)]}{\sum_{m'} \mathbb{E}[s_{y,m'}(0)]}$$

---

**(Part E — View Coverage)**:

The trained network satisfies:
$$C(f) = \frac{1}{M} + O\left(\frac{1}{\sqrt{K}}\right)$$

i.e., approximately one view per class is learned.

---

## Part 7: Factors Determining the Winner

### 7.1 Correlation Strength $\sigma_1(\Sigma_{y,m})$

$$\sigma_1(\Sigma_{y,m}) = \frac{p_m}{K} \cdot \|\phi_{y,m}\|$$

| Factor | Effect | Intuition |
|--------|--------|-----------|
| View probability $p_m$ ↑ | $\sigma_1$ ↑ | More training examples containing this view |
| View norm $\|\phi_{y,m}\|$ ↑ | $\sigma_1$ ↑ | Stronger signal in input |
| Number of classes $K$ ↑ | $\sigma_1$ ↓ | Signal diluted across classes |

### 7.2 Initial Pathway Strength $s_{y,m}(0)$

$$s_{y,m}(0) = \|W_2(0) \cdot \text{diag}(g_{y,m}) \cdot W_1(0)\|_F$$

| Factor | Effect | Intuition |
|--------|--------|-----------|
| More active neurons in $g_{y,m}$ | $s_{y,m}(0)$ ↑ | More parameters in pathway |
| Larger $\sigma_0$ | $s_{y,m}(0)$ ↑ | Larger initial weights |
| Alignment of $W_1$ rows with $\phi_{y,m}$ | $s_{y,m}(0)$ ↑ | Initial structure favors view |

### 7.3 Initialization Distribution

**Proposition 7.1**: Under random Gaussian initialization:
$$s_{y,m}(0) \approx \sigma_0^2 \cdot \sqrt{|g_{y,m}|_1 \cdot K \cdot d}$$

where $|g_{y,m}|_1 = \sum_j [g_{y,m}]_j$ is the number of active neurons for view $(y,m)$.

**Corollary**: Views that activate more neurons have higher initial pathway strength.

### 7.4 Depth Effect (for $L > 2$)

For deeper networks:
$$s_{y,m}(0) \propto \sigma_0^L \cdot \prod_{\ell=1}^{L-1} \sqrt{|g_\ell^{(y,m)}|_1}$$

The initial strength **decreases exponentially with depth** (for small $\sigma_0$).

**Implication**: In deep networks, shallower pathways (fewer active layers) have an advantage.

---

## Part 8: Proof Sketch for Theorem 2

### 8.1 Step 1: Gradient Decomposition

**Claim**: The gradient of the loss decomposes as:
$$\nabla_{W_\ell} \mathcal{L} = \sum_{y,m} \nu_{y,m} \cdot \nabla_{W_\ell}^{(y,m)} + \nabla^{\text{cross}}$$

where:
- $\nu_{y,m} = \mathbb{P}[\text{view } (y,m) \text{ is active}] = p_m / K$
- $\nabla_{W_\ell}^{(y,m)}$ = gradient from view $(y,m)$ samples
- $\nabla^{\text{cross}}$ = cross-view terms (small under A2)

### 8.2 Step 2: Project onto Pathway

**Claim**: The change in pathway strength satisfies:
$$\frac{ds_{y,m}}{dt} = -\left\langle \frac{\partial \mathbf{P}_{y,m}}{\partial W}, \nabla_W \mathcal{L} \right\rangle_F$$

Using the gradient decomposition:
$$\frac{ds_{y,m}}{dt} = \underbrace{-\nu_{y,m} \left\langle \frac{\partial \mathbf{P}_{y,m}}{\partial W}, \nabla_W^{(y,m)} \right\rangle_F}_{\text{self-reinforcement}} + \underbrace{\text{other terms}}_{\text{competition + cross}}$$

### 8.3 Step 3: Self-Reinforcement Analysis

**Claim**: The self-reinforcement term equals:
$$-\nu_{y,m} \left\langle \frac{\partial \mathbf{P}_{y,m}}{\partial W}, \nabla_W^{(y,m)} \right\rangle_F = \sigma_1(\Sigma_{y,m}) \cdot s_{y,m} \cdot (1 + O(\text{error}))$$

*Proof*: This follows from the alignment between:
- The pathway direction $\mathbf{P}_{y,m}$
- The correlation direction $e_y \phi_{y,m}^T$

The gradient pushes the pathway toward the correlation direction with strength proportional to $\sigma_1$.

### 8.4 Step 4: Competition Analysis

**Claim**: The competition arises from:
1. **Shared weights**: Multiple pathways use the same $W_1, W_2$
2. **Softmax normalization**: Pushing one class up pushes others down
3. **Saturation**: Total pathway strength is bounded

The net effect is the factor $(1 - \sum_{m'} s_{y,m'}^2 / s_{\max}^2)$.

### 8.5 Step 5: Convergence Analysis

**Claim**: The system converges to a corner fixed point.

*Proof*: The competitive Lotka-Volterra system has corner equilibria as stable fixed points. The basin of attraction is determined by initial advantages. $\square$

---

## Part 9: Special Case — Symmetric Views

### 9.1 Setup

Assume views are symmetric:
- $\|\phi_{y,m}\| = 1$ for all $y, m$
- $p_m = 1/M$ for all $m$ (equal view probabilities)
- $|g_{y,m}|_1 = n/M$ for all $y, m$ (equal gating density)

Then $\sigma_1(\Sigma_{y,m}) = \sigma_1$ is constant, and:
$$A_{y,m} = \sigma_1 \cdot s_{y,m}(0)$$

### 9.2 Winner is Determined by Initialization

**Proposition 9.1**: Under symmetric views, the winning view is:
$$m^*(y) = \arg\max_{m} s_{y,m}(0)$$

The winner is **purely determined by random initialization**.

### 9.3 Distribution of Winners

**Proposition 9.2**: Under symmetric random initialization:
$$\mathbb{P}(m^*(y) = m) = \frac{1}{M}$$

Each view is equally likely to win.

**Corollary**: Across $K$ classes with independent initializations:
- Expected number of classes learning view $m$: $K/M$
- Different classes learn different views with high probability

---

## Part 10: Connection to Theorems 1 and 3

### 10.1 Building on Theorem 1

Theorem 1 establishes: $f(x) = \sum_{m \in \mathcal{S}} R_{y,m}(f) + E$

Theorem 2 shows: After training, $\|R_{y,m^*}\| \gg \|R_{y,m}\|$ for $m \neq m^*$

**Combined**: $f(x) \approx R_{y,m^*}(f) \cdot \mathbb{1}[m^* \in \mathcal{S}]$

The network effectively only "sees" the winning view.

### 10.2 Setting Up Theorem 3

**Key Question for Theorem 3**: How do soft labels change the race dynamics?

**Preview**: Under KD with teacher $T$:
$$\frac{ds_{y,m}}{dt} \propto C_m(T) \cdot \sigma_1(\Sigma_{y,m}) \cdot s_{y,m} \cdot (\text{competition})$$

where $C_m(T)$ is teacher's coverage of view $m$.

If teacher covers multiple views ($C_m(T) > 0$ for multiple $m$), all pathways receive gradient, preventing winner-take-all.

---

## Part 11: Notation Summary

| Symbol | Definition | Meaning |
|--------|------------|---------|
| $s_{y,m}(t)$ | $\|\mathbf{P}_{y,m}(t)\|_F$ | Pathway strength at time $t$ |
| $\Sigma_{y,m}$ | $\frac{p_m}{K} e_y \phi_{y,m}^T$ | View correlation matrix |
| $\sigma_1(\Sigma_{y,m})$ | $\frac{p_m}{K}\|\phi_{y,m}\|$ | Correlation strength |
| $A_{y,m}$ | $\sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)$ | Initial advantage |
| $m^*(y)$ | $\arg\max_m A_{y,m}$ | Winning view for class $y$ |
| $T_{\text{early}}$ | End of exponential phase | Transition time |
| $s_{\max}$ | Saturation strength | Maximum pathway strength |

---

## Part 12: Key Predictions

### Testable Predictions from Theorem 2

1. **Exponential early growth**: Plot $\log s_{y,m}(t)$ vs $t$ should be linear initially

2. **Winner prediction**: The view with highest $A_{y,m}$ should win (>90% accuracy)

3. **Initialization sensitivity**: Changing initialization changes winner

4. **Correlation dominance**: Views with higher $\|\phi_{y,m}\|$ more likely to win

5. **Final coverage**: $C(f) \approx 1/M$ after training

6. **Ratio decay**: $s_{y,m}(t) / s_{y,m^*}(t) \to 0$ exponentially for losers

---

## Part 13: Open Questions

1. **Gating changes**: How do results change if gating patterns evolve during training?

2. **Deep networks**: Exact dynamics for $L > 2$?

3. **Finite learning rate**: Discrete gradient descent vs. continuous flow?

4. **Cross-class interactions**: How do races in different classes interact?

5. **Multiple winners**: Can two views tie? Under what conditions?

---

## Next Steps

1. **Rigorous proof of Proposition 4.1** (pathway dynamics equation)
2. **Verify predictions on synthetic data**
3. **Extend to $L > 2$ layers**
4. **Develop Theorem 3** (KD changes the dynamics)
