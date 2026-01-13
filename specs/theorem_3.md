# Theorem 3: Why Knowledge Distillation Circumvents the Race — Formal Setup

## Overview

This document develops the formal mathematical framework for Theorem 3, which explains WHY knowledge distillation enables students to learn multiple views when direct training cannot.

**Building on**:
- Theorem 1: Views ↔ Pathways correspondence
- Theorem 2: Race dynamics lead to single-view convergence under hard labels

**Goal**: Show that soft labels from a multi-view teacher distribute gradients across pathways, breaking the winner-take-all dynamics.

---

## Part 1: The Core Question

### 1.1 The Puzzle

From Theorem 2, we know:
- Hard label training → winner-take-all → one view per class
- Coverage: $C(f) \approx 1/M$

Yet empirically:
- KD from ensemble → student matches ensemble accuracy
- Student coverage: $C(S) \approx C(T_{\text{ensemble}})$

**Question**: What is it about soft labels that enables multi-view learning?

### 1.2 The Answer (Preview)

**Hard labels**: Gradient signal is dominated by the winning pathway's self-reinforcement.

**Soft labels**: Gradient signal includes explicit information about multiple views from the teacher. This external signal overrides the self-reinforcement dynamics.

---

## Part 2: Hard Label Gradient Analysis

### 2.1 Loss Function

**Cross-Entropy Loss**:
$$\mathcal{L}_{\text{hard}}(f, y) = -\log p_y(x) = -\log \frac{\exp(f_y(x))}{\sum_k \exp(f_k(x))}$$

where $p(x) = \text{softmax}(f(x))$.

### 2.2 Gradient Structure

**Gradient w.r.t. logits**:
$$\frac{\partial \mathcal{L}_{\text{hard}}}{\partial f_k(x)} = p_k(x) - \mathbb{1}[k = y] = p_k(x) - e_y^{(k)}$$

**Gradient w.r.t. weights** (via backprop):
$$\nabla_W \mathcal{L}_{\text{hard}} = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ (p(x) - e_y) \cdot \nabla_W f(x)^T \right]$$

### 2.3 Decomposition by Pathway

Using Theorem 1, for input $x$ from class $y$ with active view $m$:
$$f(x) \approx R_{y,m}(f) = \mathbf{P}_{y,m} \cdot \phi_{y,m}$$

The gradient decomposes:
$$\nabla_W \mathcal{L}_{\text{hard}} \approx \sum_{y,m} \nu_{y,m} \cdot \nabla_W^{(y,m)} \mathcal{L}_{\text{hard}}$$

where $\nu_{y,m} = p_m / K$ is the frequency of view $(y,m)$.

### 2.4 The Self-Reinforcement Problem

**Key Observation**: The gradient for pathway $(y,m)$ under hard labels is:

$$\nabla_{P_{y,m}} \mathcal{L}_{\text{hard}} \propto (p_y(x) - 1) \cdot \frac{\partial f_y}{\partial P_{y,m}} \propto -(1 - p_y(x)) \cdot \phi_{y,m}$$

The magnitude depends on:
1. **Classification error** $(1 - p_y(x))$: Decreases as network improves
2. **Pathway contribution**: Implicitly through $\frac{\partial f_y}{\partial P_{y,m}}$

**The Problem**: Once one pathway $P_{y,m^*}$ dominates:
- $p_y(x) \to 1$ for samples with view $m^*$
- Gradient $\to 0$ for these samples
- Other pathways $P_{y,m}$ (for $m \neq m^*$) receive minimal signal

This creates the winner-take-all dynamics of Theorem 2.

---

## Part 3: Soft Label Gradient Analysis

### 3.1 KD Loss Function

**Knowledge Distillation Loss**:
$$\mathcal{L}_{\text{KD}}(f_S, f_T) = \mathbb{E}_x \left[ \tau^2 \cdot KL(p_T^\tau(x) \| p_S^\tau(x)) \right]$$

where:
- $p_T^\tau(x) = \text{softmax}(f_T(x) / \tau)$ is teacher's soft prediction
- $p_S^\tau(x) = \text{softmax}(f_S(x) / \tau)$ is student's soft prediction
- $\tau$ is the temperature
- $\tau^2$ scaling preserves gradient magnitude

**Expanded form**:
$$\mathcal{L}_{\text{KD}} = \mathbb{E}_x \left[ -\tau^2 \sum_k p_T^{\tau,k}(x) \log p_S^{\tau,k}(x) \right] + \text{const}$$

### 3.2 Gradient Structure

**Gradient w.r.t. student logits**:
$$\frac{\partial \mathcal{L}_{\text{KD}}}{\partial f_S^k(x)} = \tau \cdot (p_S^{\tau,k}(x) - p_T^{\tau,k}(x))$$

**Key Difference from Hard Labels**:
- Hard: target is $e_y$ (one-hot)
- Soft: target is $p_T^\tau(x)$ (distribution encoding teacher's knowledge)

**Gradient w.r.t. weights**:
$$\nabla_W \mathcal{L}_{\text{KD}} = \tau \cdot \mathbb{E}_x \left[ (p_S^\tau(x) - p_T^\tau(x)) \cdot \nabla_W f_S(x)^T \right]$$

### 3.3 What Information is in Soft Labels?

**Teacher's output decomposition** (from Theorem 1):

For teacher $T$ and input $x$ with active views $\mathcal{S}$:
$$f_T(x) = \sum_{m \in \mathcal{S}} R_{y,m}^T + E_T$$

where $R_{y,m}^T = \mathbf{P}_{y,m}^T \cdot \phi_{y,m}$ is teacher's response to view $m$.

**Soft label**:
$$p_T^\tau(x) = \text{softmax}\left( \frac{1}{\tau} \sum_{m \in \mathcal{S}} R_{y,m}^T \right)$$

**Critical Insight**: The soft label $p_T^\tau(x)$ depends on WHICH views are active in $x$.

### 3.4 View-Dependent Soft Labels

Consider two samples from class $y$:
- $x_1$: only view 1 active → $f_T(x_1) \approx R_{y,1}^T$
- $x_2$: only view 2 active → $f_T(x_2) \approx R_{y,2}^T$

If teacher learned both views ($R_{y,1}^T \neq 0$ and $R_{y,2}^T \neq 0$):
$$p_T^\tau(x_1) \neq p_T^\tau(x_2)$$

even though both have the same hard label $y$.

**Example** (car classification):
- $x_1$ shows wheels: $p_T(x_1) = (0.90_{\text{car}}, 0.05_{\text{truck}}, 0.03_{\text{motorcycle}}, ...)$
- $x_2$ shows headlights: $p_T(x_2) = (0.85_{\text{car}}, 0.08_{\text{cat}}, 0.04_{\text{owl}}, ...)$

The "secondary" probabilities encode which view is present!

---

## Part 4: Gradient Distribution Across Pathways

### 4.1 The Key Proposition

**Proposition 4.1 (Gradient Distribution)**: Under KD, the expected gradient for student pathway $(y,m)$ is:

$$\mathbb{E}\left[ \nabla_{P_{y,m}^S} \mathcal{L}_{\text{KD}} \right] = \underbrace{\alpha_m(T, \tau)}_{\text{teacher signal}} \cdot \nabla_m^{\text{ext}} + \underbrace{\beta(s_{y,m}^S)}_{\text{self term}} \cdot \nabla_m^{\text{self}}$$

where:
- $\alpha_m(T, \tau) \propto \|R_{y,m}^T\|^2 / \tau > 0$ if teacher responds to view $m$
- $\beta(s) \propto s$ is the self-reinforcement factor
- $\nabla_m^{\text{ext}}$ pushes pathway toward teacher's response
- $\nabla_m^{\text{self}}$ is the self-reinforcement direction

**Crucial Difference**:
- Hard labels: only $\beta \cdot \nabla^{\text{self}}$ term (winner-take-all)
- Soft labels: additional $\alpha \cdot \nabla^{\text{ext}}$ term (teacher-guided)

### 4.2 Derivation of Teacher Signal Term

**Setup**: Consider samples where only view $m$ is active.

**Teacher output**: $f_T(x) \approx R_{y,m}^T$

**Student output**: $f_S(x) \approx R_{y,m}^S = \mathbf{P}_{y,m}^S \cdot \phi_{y,m}$

**Soft label mismatch**:
$$p_S^\tau(x) - p_T^\tau(x) = \text{softmax}(R_{y,m}^S/\tau) - \text{softmax}(R_{y,m}^T/\tau)$$

**Gradient**:
$$\nabla_{P_{y,m}^S} \mathcal{L}_{\text{KD}} \propto (p_S^\tau - p_T^\tau) \cdot \phi_{y,m}^T$$

**Key Point**: This gradient is NON-ZERO whenever $R_{y,m}^S \neq R_{y,m}^T$.

Even if student's pathway $P_{y,m}^S$ is weak (small $s_{y,m}^S$), the gradient pushes it toward teacher's response $R_{y,m}^T$.

### 4.3 The External Signal

**Definition 4.1 (Teacher Signal Strength)**: For view $(y,m)$:
$$\alpha_m(T, \tau) := \frac{\nu_{y,m}}{\tau} \cdot \left\| \frac{\partial}{\partial R_{y,m}^S} KL(p_T^\tau \| p_S^\tau) \right\|_{R^S = 0}$$

**Proposition 4.2**: Under mild conditions:
$$\alpha_m(T, \tau) \geq c \cdot \frac{\|R_{y,m}^T\|^2}{\tau \cdot K}$$

for some constant $c > 0$.

**Interpretation**: If teacher has non-zero response to view $m$, there is a non-zero external gradient signal for the student's pathway $m$.

---

## Part 5: Modified Race Dynamics Under KD

### 5.1 The New Dynamical System

**Theorem 5.1 (KD Race Dynamics)**: Under KD from teacher $T$, student pathway strengths evolve as:

$$\frac{ds_{y,m}^S}{dt} = \left( \underbrace{\alpha_m(T, \tau)}_{\text{external}} + \underbrace{\gamma \cdot \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}^S}_{\text{internal}} \right) \cdot \left(1 - \frac{\sum_{m'} (s_{y,m'}^S)^2}{s_{\max}^2}\right)$$

where $\gamma < 1$ is a damping factor from the soft label smoothing.

### 5.2 Comparison with Hard Label Dynamics

**Hard labels** (Theorem 2):
$$\frac{ds_{y,m}}{dt} = \sigma_1(\Sigma_{y,m}) \cdot s_{y,m} \cdot (1 - \text{competition})$$

- Only internal (self-reinforcement) term
- Initial advantage determines winner
- Convergence to single dominant pathway

**Soft labels** (KD):
$$\frac{ds_{y,m}^S}{dt} = (\alpha_m + \gamma \sigma_1 s_{y,m}^S) \cdot (1 - \text{competition})$$

- External term $\alpha_m$ from teacher
- Teacher signal can override initial disadvantage
- Multiple pathways can reach positive equilibrium

### 5.3 Equilibrium Analysis

**Proposition 5.2 (Multi-View Equilibrium)**: Consider class $y$ with views $\{1, \ldots, M\}$.

If $\alpha_m(T, \tau) > 0$ for views $m \in \mathcal{M}_T \subseteq [M]$, then there exists a stable equilibrium where:
$$s_{y,m}^S(\infty) > 0 \quad \forall m \in \mathcal{M}_T$$

*Proof Sketch*:

At equilibrium, $\frac{ds_{y,m}^S}{dt} = 0$ for all $m$. This requires either:
1. Competition term = 0 (saturation), or
2. Growth term = 0

For $m \in \mathcal{M}_T$: $\alpha_m > 0$, so growth term = 0 only if competition term = 0 (saturation).

At saturation: $\sum_{m'} (s_{y,m'}^S)^2 = s_{\max}^2$

The equilibrium distributes strength across pathways with $\alpha_m > 0$. $\square$

### 5.4 Equilibrium Distribution

**Proposition 5.3**: At equilibrium, the pathway strengths satisfy:
$$s_{y,m}^S(\infty) \propto \sqrt{\alpha_m(T, \tau)} \propto \|R_{y,m}^T\| / \sqrt{\tau}$$

**Interpretation**: Student's pathway strengths mirror teacher's view responses.

---

## Part 6: Formal Statement of Theorem 3

### Theorem 3 (KD Circumvents Race Dynamics)

Let $\mathcal{D}$ be a $(K, M, \mathbf{p})$-multi-view distribution satisfying A1-A5. Let $T$ be a teacher network with view responses $\{R_{y,m}^T\}_{y,m}$ and view coverage:
$$\mathcal{M}_T(y) := \{m \in [M] : \|R_{y,m}^T\| > \epsilon_{\text{detect}}\}$$

Let $S$ be a student trained via KD loss at temperature $\tau$:
$$\mathcal{L}_{\text{KD}} = \tau^2 \cdot \mathbb{E}_x [KL(p_T^\tau(x) \| p_S^\tau(x))]$$

---

**(Part A — Gradient Decomposition)**:

The gradient for student pathway $(y,m)$ decomposes as:
$$\nabla_{P_{y,m}^S} \mathcal{L}_{\text{KD}} = \alpha_m(T, \tau) \cdot \nabla^{\text{ext}}_{y,m} + \beta_m(S) \cdot \nabla^{\text{self}}_{y,m}$$

where:
- **External signal**: $\alpha_m(T, \tau) = \Theta\left(\frac{\|R_{y,m}^T\|^2}{\tau \cdot K}\right)$
- **Self-reinforcement**: $\beta_m(S) = \Theta\left(\gamma \cdot \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}^S\right)$
- $\gamma < 1$ depends on temperature (higher $\tau$ → lower $\gamma$)

---

**(Part B — Modified Race Dynamics)**:

Student pathway strengths evolve as:
$$\frac{ds_{y,m}^S}{dt} = \left(\alpha_m(T, \tau) + \beta_m(S)\right) \cdot \left(1 - \frac{\|\mathbf{s}_y^S\|^2}{s_{\max}^2}\right) + O(\delta)$$

where $\mathbf{s}_y^S = (s_{y,1}^S, \ldots, s_{y,M}^S)$.

---

**(Part C — Breaking Winner-Take-All)**:

Under hard labels: The system has $M$ stable fixed points, each with one dominant pathway.

Under KD: If $|\mathcal{M}_T(y)| > 1$, the system has a unique stable fixed point with multiple non-zero pathway strengths.

---

**(Part D — Coverage Inheritance)**:

The student's view coverage satisfies:
$$C(S) \geq \frac{1}{KM} \sum_{y=1}^K |\mathcal{M}_T(y)| - O\left(\frac{1}{\sqrt{\tau}}\right)$$

In particular, if teacher covers all views ($\mathcal{M}_T(y) = [M]$ for all $y$):
$$C(S) \geq 1 - O\left(\frac{1}{\sqrt{\tau}}\right)$$

---

**(Part E — Coverage Bound)**:

The student cannot exceed teacher's coverage:
$$C(S) \leq C(T) + O(\epsilon_{\text{detect}})$$

---

## Part 7: Special Cases

### 7.1 Single-View Teacher

If teacher learned only view $m^*$ per class:
- $\mathcal{M}_T(y) = \{m^*(y)\}$
- $\alpha_m = 0$ for $m \neq m^*$
- Student dynamics revert to race (Theorem 2)
- $C(S) \approx 1/M$

**Conclusion**: Single-view teacher provides no benefit over hard labels.

### 7.2 Ensemble Teacher

Ensemble of $N$ independent teachers:
- Teacher $i$ learned view $m_i(y)$ for class $y$
- Ensemble output: $f_T^{\text{ens}}(x) = \frac{1}{N} \sum_{i=1}^N f_{T_i}(x)$
- Ensemble covers: $\mathcal{M}_T^{\text{ens}}(y) = \bigcup_{i=1}^N \{m_i(y)\}$

With $N \geq M$ independent teachers:
$$|\mathcal{M}_T^{\text{ens}}(y)| \approx M \cdot (1 - (1-1/M)^N) \to M$$

**Conclusion**: Ensemble covers all views → student learns all views.

### 7.3 Self-Distillation

Teacher and student have same architecture.

**Setup**:
- Teacher (first generation) learned view $m_1$ via hard labels
- Student initialized differently, would learn view $m_2$ if trained from scratch

**Under KD**:
- External signal $\alpha_{m_1} > 0$ from teacher
- Student's initialization favors $m_2$ (internal advantage)
- Result: Student learns BOTH $m_1$ and $m_2$

**Coverage**: $C(S) > C(T)$ is possible (student beats teacher!)

This explains why self-distillation improves performance.

---

## Part 8: The Role of Temperature

### 8.1 Temperature Effect on Teacher Signal

Recall:
$$\alpha_m(T, \tau) \propto \frac{\|R_{y,m}^T\|^2}{\tau}$$

**High $\tau$**:
- Softer distributions (more entropy)
- Smaller $\alpha_m$ (weaker teacher signal)
- But also smaller $\gamma$ (weaker self-reinforcement)
- Net effect: More balanced across pathways

**Low $\tau$**:
- Harder distributions (lower entropy)
- Larger $\alpha_m$ for dominant view
- Approaches hard label behavior
- Self-reinforcement dominates

### 8.2 Temperature Effect on Self-Reinforcement

The damping factor $\gamma$ satisfies:
$$\gamma(\tau) \approx \frac{1}{1 + \tau/\tau_0}$$

for some characteristic temperature $\tau_0$.

**High $\tau$**: $\gamma \to 0$, self-reinforcement suppressed
**Low $\tau$**: $\gamma \to 1$, full self-reinforcement

### 8.3 Optimal Temperature

**Trade-off**:
- Need $\tau$ high enough to preserve secondary view information
- Need $\tau$ low enough to maintain class discrimination

**Proposition 8.1**: The optimal temperature for view transfer is:
$$\tau^* = \Theta\left( \frac{\max_m \|R_{y,m}^T\|}{\min_{m \in \mathcal{M}_T} \|R_{y,m}^T\|} \right)$$

Scales with the ratio of strongest to weakest teacher view response.

---

## Part 9: Information-Theoretic Perspective

### 9.1 Dark Knowledge as View Information

**Definition 9.1 (Dark Knowledge Content)**: The dark knowledge in teacher's soft labels is:
$$DK(T, \tau) := \mathbb{E}_x \left[ H(p_T^\tau(x)) - H(e_y) \right] = \mathbb{E}_x \left[ H(p_T^\tau(x)) \right]$$

(since hard label entropy is 0).

### 9.2 View Information in Soft Labels

**Proposition 9.1**: The mutual information between soft labels and views is:
$$I(p_T^\tau(X); M | Y) \geq \sum_{m \in \mathcal{M}_T} \Omega\left( \frac{\|R_{y,m}^T\|^2}{\tau^2} \right)$$

**Interpretation**: Soft labels contain information about which view is present, proportional to teacher's view responses.

### 9.3 Information Transfer Bound

**Proposition 9.2**: The student can learn at most the view information present in soft labels:
$$C(S) \leq C(T) + O\left( \sqrt{\frac{\tau}{I(p_T^\tau; M | Y)}} \right)$$

**Interpretation**: You can't teach what you don't know. Student's coverage is bounded by information in teacher's outputs.

---

## Part 10: Proof Sketch for Theorem 3

### 10.1 Step 1: Compute KD Gradient

**KD Loss gradient**:
$$\nabla_W \mathcal{L}_{\text{KD}} = \tau \cdot \mathbb{E}_x \left[ (p_S^\tau(x) - p_T^\tau(x)) \cdot \nabla_W f_S(x)^T \right]$$

### 10.2 Step 2: Condition on Active View

For samples with view $m$ active:
$$\nabla_W^{(m)} \mathcal{L}_{\text{KD}} = \tau \cdot \mathbb{E}_{x \sim \mathcal{D}_y^{\{m\}}} \left[ (p_S^\tau(x) - p_T^\tau(x)) \cdot \nabla_W f_S(x)^T \right]$$

### 10.3 Step 3: Project onto Pathway

The gradient for pathway $(y,m)$ is:
$$\nabla_{P_{y,m}^S} \mathcal{L}_{\text{KD}} = \left\langle \nabla_W \mathcal{L}_{\text{KD}}, \frac{\partial \mathbf{P}_{y,m}^S}{\partial W} \right\rangle$$

### 10.4 Step 4: Decompose into External and Self Terms

**External term**: Arises from mismatch $(p_S^\tau - p_T^\tau)$ when student's pathway is weak.

**Self term**: Arises from the fact that changing $P_{y,m}^S$ affects $p_S^\tau$ through $f_S(x)$.

### 10.5 Step 5: Derive Modified Dynamics

Combine external and self terms to get:
$$\frac{ds_{y,m}^S}{dt} = (\alpha_m + \gamma \sigma_1 s_{y,m}^S) \cdot (1 - \text{competition})$$

### 10.6 Step 6: Analyze Equilibrium

Show that multiple pathways can have $s_{y,m}^S > 0$ at equilibrium when $\alpha_m > 0$ for multiple $m$.

---

## Part 11: Summary — Why KD Works

### The Mechanism in One Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HARD LABELS                                  │
│                                                                  │
│   Gradient ∝ s_{y,m} (current strength)                         │
│                                                                  │
│   Strong pathway → more gradient → stronger → DOMINATES         │
│   Weak pathway  → less gradient  → weaker  → DIES               │
│                                                                  │
│   Result: Winner-take-all, C(f) ≈ 1/M                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     SOFT LABELS (KD)                             │
│                                                                  │
│   Gradient ∝ α_m(T) + γ·s_{y,m}                                 │
│              ↑            ↑                                      │
│         external      internal                                   │
│         (teacher)     (self)                                     │
│                                                                  │
│   External signal provides "floor" for all teacher's views      │
│   Even weak pathways receive gradient if teacher knows view     │
│                                                                  │
│   Result: Multiple views learned, C(S) ≈ C(T)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Equations Comparison

| Quantity | Hard Labels | Soft Labels (KD) |
|----------|-------------|------------------|
| Target | $e_y$ (one-hot) | $p_T^\tau(x)$ (soft) |
| Gradient for pathway $m$ | $\propto s_{y,m}$ | $\propto \alpha_m(T) + \gamma s_{y,m}$ |
| Dynamics | $\dot{s} = \sigma_1 s (1-\text{comp})$ | $\dot{s} = (\alpha + \gamma \sigma_1 s)(1-\text{comp})$ |
| Equilibrium | One pathway dominates | Multiple pathways coexist |
| Coverage | $\approx 1/M$ | $\approx C(T)$ |

### The Three Key Insights

1. **Soft labels encode view information**: Different views produce different soft distributions from teacher.

2. **External gradient signal**: Teacher's response to each view creates gradient for the corresponding student pathway, independent of student's current state.

3. **Breaking positive feedback**: The external signal $\alpha_m$ provides a "floor" that prevents weak pathways from dying out.

---

## Part 12: Experimental Predictions

### Testable Predictions from Theorem 3

1. **Gradient distribution**: Under KD, measure gradient magnitude for each pathway. Should be more uniform than under hard labels.

2. **Pathway strength evolution**: Plot $s_{y,m}(t)$ during training. Under KD, multiple pathways should grow; under hard labels, one dominates.

3. **Coverage inheritance**: $C(S) \approx C(T)$ across different teachers.

4. **Temperature effect**: Higher $\tau$ → more uniform pathway strengths.

5. **Single-view teacher failure**: KD from single-view teacher ≈ hard labels.

6. **Self-distillation improvement**: Second generation learns more views than first.

---

## Part 13: Open Questions

1. **Optimal teacher design**: Given student capacity, what teacher coverage maximizes student performance?

2. **Progressive distillation**: Can we iteratively improve coverage through multiple generations?

3. **Feature distillation**: Does intermediate layer matching provide additional view signal?

4. **Capacity constraints**: What if student can't represent all teacher's views?

5. **Online KD**: How do dynamics change with evolving teacher?

---

## Next Steps

1. **Verify gradient decomposition** (Part A) with explicit computation
2. **Simulate dynamics** on synthetic multi-view data
3. **Measure pathway strengths** during training on real data
4. **Compare coverage** for hard labels vs. KD vs. ensemble KD
