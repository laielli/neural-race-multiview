# Why Knowledge Distillation Works: Theorem 3

> **Target audience**: Researchers ready for the mechanistic explanation
> **Prerequisites**: Modules 1-2, Theorem 1
> **Key result**: Soft labels break winner-take-all via external gradient signal

---

## The Puzzle

From Theorems 1 and 2, we know:
- Views = Pathways (Theorem 1)
- Hard labels → Winner-take-all → One view per class (Theorem 2)

Yet empirically:
- Students trained via KD achieve $C(S) \approx C(T)$
- A single student can match an ensemble!

**Question**: What is it about soft labels that enables multi-view learning?

---

## The Answer in One Sentence

> Soft labels provide an **external gradient signal** proportional to the teacher's response to each view, which **breaks the winner-take-all dynamics** by ensuring all teacher-known pathways receive gradient regardless of their current strength.

---

## Hard Labels: The Self-Reinforcement Problem

### Gradient Structure

Under cross-entropy loss with hard labels $e_y$:

$$\nabla_{P_{y,m}} \mathcal{L}_{\text{hard}} \propto (p_y(x) - 1) \cdot \phi_{y,m}$$

where $p_y(x) = \text{softmax}(f(x))_y$ is the predicted probability for class $y$.

### The Problem

As training progresses:
1. Strong pathway $P_{y,m^*}$ makes $p_y(x)$ close to 1
2. Gradient $(p_y - 1) \to 0$
3. All pathways receive vanishing gradient
4. But the **strong pathway benefited first** and locked in its lead

**Result**: Self-reinforcing winner-take-all dynamics.

### Visualization

```
Hard Label Gradient for Pathway m:

Gradient ∝ (1 - p_y(x)) × pathway_contribution

Strong pathway:  High contribution → Got gradient early → Grew → Dominated
                      ↓
                 Now p_y ≈ 1 → Gradient ≈ 0 for everyone
                      ↓
Weak pathways:   Never got enough gradient → Stayed weak → Died
```

---

## Soft Labels: The External Signal

### KD Loss Function

$$\mathcal{L}_{\text{KD}} = \tau^2 \cdot \text{KL}(p_T^\tau(x) \| p_S^\tau(x))$$

where:
- $p_T^\tau = \text{softmax}(f_T(x)/\tau)$ — teacher's soft prediction
- $p_S^\tau = \text{softmax}(f_S(x)/\tau)$ — student's soft prediction
- $\tau > 1$ — temperature (softens distributions)

### Gradient Structure

$$\nabla_{P_{y,m}^S} \mathcal{L}_{\text{KD}} \propto \tau \cdot (p_S^\tau(x) - p_T^\tau(x)) \cdot \phi_{y,m}$$

The gradient pushes student to **match teacher's output**, not just get the class right.

### The Key Insight: Soft Labels Encode View Information

**Hard label**: "This is class $y$" (one-hot)
- Same for all samples of class $y$
- No information about which view is present

**Soft label from multi-view teacher**: $p_T^\tau(x)$
- Different for different samples (depending on which views active)
- Encodes teacher's response to each view

**Example**:
```
Sample 1: View 1 active → Teacher output: [0.03, 0.02, 0.90, 0.03, 0.02]
Sample 2: View 2 active → Teacher output: [0.04, 0.03, 0.85, 0.05, 0.03]
                                           └── slight differences encode view info
```

---

## Theorem 3: KD Circumvents Race Dynamics

### Statement

Let $T$ be a teacher with view coverage:
$$\mathcal{M}_T(y) = \{m \in [M] : \|R_{y,m}^T\| > \epsilon\}$$

Let $S$ be a student trained via KD at temperature $\tau$.

---

**Part A — Gradient Decomposition**:

$$\boxed{\nabla_{P_{y,m}^S} \mathcal{L}_{\text{KD}} = \alpha_m(T, \tau) \cdot \nabla^{\text{ext}}_{y,m} + \beta_m(S) \cdot \nabla^{\text{self}}_{y,m}}$$

where:
- **External signal**: $\alpha_m(T, \tau) = \Theta\left(\frac{\|R_{y,m}^T\|^2}{\tau \cdot K}\right)$
- **Self-reinforcement**: $\beta_m(S) = \Theta(\gamma \cdot \sigma_1 \cdot s_{y,m}^S)$ with $\gamma < 1$

---

**Part B — Modified Race Dynamics**:

$$\boxed{\frac{ds_{y,m}^S}{dt} = \left(\alpha_m(T, \tau) + \gamma \sigma_1 s_{y,m}^S\right) \cdot \left(1 - \frac{\|\mathbf{s}_y^S\|^2}{s_{\max}^2}\right)}$$

---

**Part C — Breaking Winner-Take-All**:

If $|\mathcal{M}_T(y)| > 1$ (teacher knows multiple views), the system has a **unique stable equilibrium** with multiple non-zero pathway strengths:

$$s_{y,m}^S(\infty) \propto \sqrt{\alpha_m(T, \tau)} \propto \|R_{y,m}^T\| / \sqrt{\tau}$$

---

**Part D — Coverage Inheritance**:

$$\boxed{C(S) \geq \frac{1}{KM} \sum_{y=1}^K |\mathcal{M}_T(y)| - O(1/\sqrt{\tau})}$$

Student inherits teacher's view coverage.

---

## The Two Terms Explained

### External Signal: $\alpha_m(T, \tau)$

**Source**: Mismatch between student and teacher when view $m$ is active.

**Properties**:
- Depends on **teacher's** knowledge: $\propto \|R_{y,m}^T\|^2$
- Does **not** depend on student's current state
- Non-zero whenever teacher responds to view $m$

**Effect**: Provides a "floor" of gradient to pathway $m$, preventing it from dying.

### Self-Reinforcement: $\beta_m(S) = \gamma \sigma_1 s_{y,m}^S$

**Source**: Student's own pathway contributing to the output.

**Properties**:
- Depends on **student's** current strength: $\propto s_{y,m}^S$
- Same type of term as in hard label training
- Damped by factor $\gamma < 1$ (temperature smoothing)

**Effect**: Provides additional growth for already-strong pathways.

---

## Comparison: Hard vs Soft Labels

### Dynamics Comparison

| Aspect | Hard Labels | Soft Labels (KD) |
|--------|-------------|------------------|
| Gradient structure | $\propto s_{y,m}$ | $\propto \alpha_m(T) + \gamma s_{y,m}$ |
| Zero-strength pathway | Zero gradient | Non-zero gradient (if teacher knows) |
| Dynamics | $\dot{s} = \sigma_1 s (1-\text{comp})$ | $\dot{s} = (\alpha + \gamma\sigma_1 s)(1-\text{comp})$ |
| Equilibrium type | Corner (one dominant) | Interior (multiple non-zero) |
| Coverage | $\approx 1/M$ | $\approx C(T)$ |

### Visualization

```
HARD LABELS                          SOFT LABELS (KD)
─────────────                        ─────────────────
Gradient:                            Gradient:
│                                    │         ←── external α_m
│       ╱                            │────────────────────
│     ╱                              │         ╱
│   ╱                                │       ╱
│ ╱                                  │     ╱
│╱                                   │   ╱
└──────────→ s_{y,m}                 └───────────→ s_{y,m}
 ↑                                    ↑
 Gradient = 0 when s = 0              Gradient > 0 even when s = 0
 (weak pathways get nothing)          (weak pathways still get α_m)
```

---

## Why External Signal Breaks Winner-Take-All

### Hard Labels: Corner Equilibria

The system $\dot{s}_m = \sigma_1 s_m (1 - \text{comp})$ has equilibria only at corners:
- $(s_{\max}, 0, 0)$ — View 1 wins
- $(0, s_{\max}, 0)$ — View 2 wins
- $(0, 0, s_{\max})$ — View 3 wins

**No interior equilibrium exists.** One pathway must dominate.

### Soft Labels: Interior Equilibrium

The system $\dot{s}_m = (\alpha_m + \gamma\sigma_1 s_m)(1 - \text{comp})$ has an interior equilibrium.

**Finding it**: At equilibrium, $\dot{s}_m = 0$ for all $m$. This requires:
$$\alpha_m + \gamma\sigma_1 s_m = 0 \quad \text{OR} \quad 1 - \text{comp} = 0$$

Since $\alpha_m > 0$ and $s_m \geq 0$, the first option is impossible.

So we need $\sum_m s_m^2 = s_{\max}^2$ (saturation).

At saturation, each pathway has:
$$s_m(\infty) \propto \sqrt{\alpha_m} \propto \|R_{y,m}^T\|/\sqrt{\tau}$$

**Multiple pathways are non-zero** (interior point, not corner).

---

## Special Cases

### Case 1: Single-View Teacher

Teacher learned only $m^*$: $\mathcal{M}_T(y) = \{m^*\}$

Then:
- $\alpha_{m^*} > 0$, $\alpha_m = 0$ for $m \neq m^*$
- External signal only for $m^*$
- Dynamics revert to race among $m \neq m^*$, plus forcing toward $m^*$

**Result**: Student learns $m^*$ (teacher's view), maybe one other (from its own initialization).

$C(S) \approx C(T) = 1/M$ — **No benefit over hard labels**.

### Case 2: Ensemble Teacher

Ensemble of $N$ teachers:
- Teacher $i$ learned view $m_i(y)$
- Ensemble covers $\mathcal{M}_T^{\text{ens}}(y) = \bigcup_i \{m_i(y)\}$

With $N \geq M$ independent teachers:
- Expected $|\mathcal{M}_T^{\text{ens}}| \approx M(1 - (1-1/M)^N) \to M$
- All views have $\alpha_m > 0$
- All pathways receive external signal

**Result**: Student learns all views. $C(S) \approx 1$.

### Case 3: Self-Distillation

Teacher and student same architecture, but different seeds.

- Teacher learned view $m_T$
- Student's initialization favors view $m_S$ (possibly different)

Under KD:
- External signal pushes toward $m_T$
- Internal advantage pushes toward $m_S$
- Both can win!

**Result**: $C(S) \geq C(T)$ — student can **beat** teacher by learning both views.

---

## The Role of Temperature

### How Temperature Affects $\alpha_m$

$$\alpha_m(T, \tau) \propto \frac{\|R_{y,m}^T\|^2}{\tau}$$

**High $\tau$**:
- Softer distributions
- Smaller $\alpha_m$ for all views
- More equal gradient distribution
- But also weaker overall signal

**Low $\tau$**:
- Sharper distributions
- Larger $\alpha_m$ for teacher's dominant view
- Approaches hard label behavior

### How Temperature Affects $\gamma$

The damping factor:
$$\gamma(\tau) \approx \frac{1}{1 + \tau/\tau_0}$$

**High $\tau$**: $\gamma \to 0$, self-reinforcement suppressed
**Low $\tau$**: $\gamma \to 1$, full self-reinforcement

### Optimal Temperature

**Trade-off**:
- Need $\tau$ high enough to reveal secondary view information
- Need $\tau$ low enough to maintain class discrimination

Roughly: $\tau^* \sim$ ratio of strongest to weakest teacher view response.

---

## Information-Theoretic View

### Dark Knowledge = View Information

The "dark knowledge" in soft labels is precisely information about which view is present.

**Hard labels**: $H(e_y) = 0$ (no entropy, no information beyond class)

**Soft labels**: $H(p_T^\tau(x)) > 0$ (entropy encodes view information)

### Bound on Student Coverage

$$C(S) \leq C(T) + O(\epsilon)$$

Student cannot learn views that teacher doesn't know. Information must come from somewhere.

---

## Summary: The Mechanism

### Why KD Enables Multi-View Learning

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE KD MECHANISM                             │
│                                                                  │
│  1. Teacher knows multiple views (via ensemble or luck)          │
│     └── $|M_T(y)| > 1$                                          │
│                                                                  │
│  2. Soft labels encode view-specific responses                   │
│     └── Different views → different $p_T^\tau(x)$               │
│                                                                  │
│  3. KD gradient has external signal for each teacher view        │
│     └── $\alpha_m > 0$ for $m \in M_T$                          │
│                                                                  │
│  4. External signal provides "floor" preventing pathway death    │
│     └── Weak pathways still receive gradient                     │
│                                                                  │
│  5. Multiple pathways reach non-zero equilibrium                 │
│     └── $s_m(\infty) \propto ||R_{y,m}^T||$                      │
│                                                                  │
│  6. Student inherits teacher's view coverage                     │
│     └── $C(S) ≈ C(T)$                                           │
└─────────────────────────────────────────────────────────────────┘
```

### The Key Equation

$$\nabla \mathcal{L}_{\text{KD}} \propto \underbrace{\alpha_m(T)}_{\substack{\text{external}\\\text{(teacher)}}} + \underbrace{\gamma \sigma_1 s_{y,m}}_{\substack{\text{internal}\\\text{(self)}}}$$

The external term is what hard labels lack and what KD provides.

---

## Testable Predictions

1. **Gradient distribution**: Under KD, gradient more uniform across pathways
2. **Pathway evolution**: Multiple pathways grow (vs one under hard labels)
3. **Coverage inheritance**: $C(S) \approx C(T)$
4. **Single-view teacher fails**: KD from single-view teacher ≈ hard labels
5. **Temperature effect**: Higher $\tau$ → more balanced pathways
6. **Self-distillation gains**: Student can exceed teacher coverage

---

## What's Next

- **[Worked Examples](worked_examples.md)**: Concrete gradient calculations and dynamics
