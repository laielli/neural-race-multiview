# The Gap: What Existing Theories Don't Explain

> **Target audience**: Researchers ready to understand our contribution
> **Prerequisites**: Modules 1 and 2
> **Purpose**: Motivate the unified framework

---

## Two Powerful but Disconnected Theories

### Multi-View Theory (Allen-Zhu & Li 2023)

**Explains**:
- Networks learn one view per class
- Ensembles cover more views
- KD transfers coverage from teacher to student

**Doesn't explain**:
- ❌ *Which* view will be learned
- ❌ *Why* winner-take-all happens
- ❌ *How* soft labels enable multi-view learning at the gradient level

### Neural Race Reduction (Saxe et al. 2022)

**Explains**:
- Pathway competition during learning
- Winner-take-all dynamics
- How initial conditions determine winners

**Doesn't explain**:
- ❌ Connection to multi-view data
- ❌ How soft labels change the dynamics
- ❌ Why KD succeeds where hard labels fail

---

## The Gap We Fill

### Question 1: Which View Wins?

**Multi-view theory says**: "One view wins, uniformly at random."

**We want**: A formula predicting the winner given:
- Network architecture
- Initialization $\theta_0$
- Data distribution $\mathcal{D}$

**Our answer** (Theorem 2):
$$m^*(y) = \arg\max_{m \in [M]} \underbrace{\sigma_1(\Sigma_{y,m})}_{\text{correlation}} \cdot \underbrace{s_{y,m}(0)}_{\text{init strength}}$$

### Question 2: Why Does Single-View Convergence Happen?

**Multi-view theory says**: "Networks converge to one view."

**We want**: The gradient-level mechanism explaining *why*.

**Our answer** (Theorems 1 & 2):
- Views correspond to pathways
- Pathways race with winner-take-all dynamics
- The race is governed by:
$$\frac{ds_{y,m}}{dt} = \sigma_1 \cdot s_{y,m} \cdot (1 - \text{competition})$$

### Question 3: How Does KD Break the Race?

**Multi-view theory says**: "Soft labels encode view information."

**We want**: Explicit gradient decomposition showing *how* soft labels help.

**Our answer** (Theorem 3):
$$\nabla_{P_{y,m}} \mathcal{L}_{\text{KD}} = \underbrace{\alpha_m(T)}_{\text{external}} \cdot \nabla^{\text{ext}} + \underbrace{\gamma \sigma_1 s_{y,m}}_{\text{internal}} \cdot \nabla^{\text{self}}$$

The **external signal** $\alpha_m(T)$ from the teacher prevents winner-take-all.

---

## The Key Insight: Views Are Pathways

### The Bridge

| Multi-View Concept | Neural Race Concept |
|--------------------|---------------------|
| View $(y, m)$ | Pathway $P_{y,m}$ |
| View feature $\phi_{y,m}$ | Pathway gating pattern |
| View response $R_{y,m}$ | Pathway output |
| View coverage $C(f)$ | Fraction of strong pathways |
| Single-view convergence | One pathway dominates |

### Why This Connection Matters

**Before**: Two separate frameworks, each with blind spots.

**After**: Unified theory that:
1. Uses multi-view data structure to define pathways
2. Uses race dynamics to explain learning
3. Uses gradient analysis to explain KD

---

## What Each Theory Contributes

### From Multi-View Theory

1. **Data model**: $(K, M, \mathbf{p})$-multi-view distribution
2. **Orthogonality structure**: Views are approximately orthogonal
3. **Sufficiency**: Each view enables correct classification
4. **Coverage metric**: $C(f) = \frac{1}{KM}\sum_{y,m} \mathbb{1}[\text{view detected}]$

### From Neural Race Reduction

1. **GDLN formalism**: Networks as gated linear maps
2. **Pathway decomposition**: $f(x) = \sum_g P_g(x) \cdot \mathbb{1}[g \text{ active}]$
3. **Race dynamics**: Winner-take-all equations
4. **Initial advantage**: $A = \sigma_1 \cdot s(0)$ predicts winner

### Our Novel Contributions

1. **View-Pathway Correspondence** (Theorem 1):
   - Different views induce different gating patterns
   - Therefore, different views = different pathways
   - Network output decomposes by view

2. **Predictive Formula** (Theorem 2):
   - Initial advantage predicts winning view
   - First quantitative prediction in multi-view learning

3. **KD Mechanism** (Theorem 3):
   - Soft labels add external gradient signal
   - External signal breaks winner-take-all
   - Student inherits teacher's view coverage

---

## Visualizing the Gap

### Before: Two Separate Pictures

```
Multi-View Theory:                    Neural Race Reduction:
┌─────────────────────┐               ┌─────────────────────┐
│  View 1  View 2     │               │  Pathway A          │
│    ↓       ↓        │               │      ↓              │
│ Network learns      │               │  Competes with      │
│ one view (which?)   │               │  Pathway B          │
│        ?            │               │      ↓              │
│ KD helps (how?)     │               │  Winner takes all   │
│        ?            │               │                     │
└─────────────────────┘               └─────────────────────┘
        GAP                                   GAP
        ↓                                     ↓
   No prediction                      No KD analysis
   No mechanism                       No multi-view link
```

### After: Unified Picture

```
┌─────────────────────────────────────────────────────────────┐
│                   UNIFIED FRAMEWORK                          │
│                                                              │
│  Multi-View Data        View-Pathway           Race         │
│  ┌──────────────┐      Correspondence      ┌─────────────┐  │
│  │ View 1 (y,1) │ ════════════════════════►│ Pathway P₁  │  │
│  │ View 2 (y,2) │ ════════════════════════►│ Pathway P₂  │  │
│  │ View 3 (y,3) │ ════════════════════════►│ Pathway P₃  │  │
│  └──────────────┘      (Theorem 1)         └──────┬──────┘  │
│                                                   │         │
│                                                   ▼         │
│                                            Race Dynamics    │
│                                            (Theorem 2)      │
│                                                   │         │
│                                                   ▼         │
│  Hard Labels ──────────────────────────► Winner-Take-All   │
│       │                                   Coverage = 1/M    │
│       │                                                     │
│  Soft Labels ──► External Signal ──────► Multiple Winners  │
│  (KD)             (Theorem 3)             Coverage ≈ C(T)   │
└─────────────────────────────────────────────────────────────┘
```

---

## The Three Theorems at a Glance

### Theorem 1: View-Pathway Correspondence

**Statement**: Under multi-view data, each view maps to a distinct pathway.

$$f(x) = \sum_{m \in \mathcal{S}} R_{y,m}(f) + E(x, \mathcal{S})$$

**Significance**: Reduces multi-view learning to pathway analysis.

### Theorem 2: Race Dynamics

**Statement**: Pathways race with winner-take-all dynamics. Winner determined by:

$$m^*(y) = \arg\max_m \sigma_1(\Sigma_{y,m}) \cdot s_{y,m}(0)$$

**Significance**: First quantitative prediction for which view wins.

### Theorem 3: KD Breaks the Race

**Statement**: Soft labels add external gradient signal:

$$\nabla \mathcal{L}_{\text{KD}} \propto \alpha_m(T) + \gamma \sigma_1 s_{y,m}$$

**Significance**: Explains mechanistically why KD enables multi-view learning.

---

## Why This Matters

### Theoretical Contribution

- **Fills explicit gap**: Allen-Zhu & Li acknowledge "no predictive theory for which view wins"
- **Unifies frameworks**: First connection between multi-view and race dynamics
- **Provides mechanism**: Not just *what* but *why* and *how*

### Practical Implications

1. **Predict learning outcomes**: Know which features network will learn before training
2. **Design better teachers**: Understand what makes a good KD teacher
3. **Improve single-network learning**: Interventions to slow the race

---

## What's Next

- **[View-Pathway Correspondence](view_pathway_correspondence.md)**: Our Theorem 1 in detail
- **[Why KD Works](why_kd_works.md)**: Our Theorem 3 and gradient analysis
- **[Worked Examples](worked_examples.md)**: Concrete calculations showing the unified theory
