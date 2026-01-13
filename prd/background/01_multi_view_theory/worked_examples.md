# Multi-View Theory: Worked Examples

> **Target audience**: ML researchers who learn best through concrete calculations
> **Prerequisites**: [formal_setup.md](formal_setup.md)
> **Goal**: Ground abstract concepts in our specific experimental setup

---

## Our Synthetic Setup

Throughout this project, we use a controlled synthetic dataset that satisfies multi-view assumptions exactly. This section walks through concrete calculations using these parameters.

### Default Configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $K$ | 10 | Number of classes |
| $M$ | 3 | Views per class |
| $d$ | 150 | Input dimension (50 per view) |
| $d_{\text{view}}$ | 50 | Dimensions per view slot |
| $p_m$ | 0.5 | View activation probability (all views) |
| $\sigma$ | 0.1 | Noise standard deviation |

### Orthogonal Slot Structure

To ensure exact orthogonality ($\delta = 0$), we use disjoint dimensional "slots":

```
Input vector x ∈ ℝ¹⁵⁰:

Dimensions:   [0-49]     [50-99]    [100-149]
              ───────    ────────   ──────────
View 1 slot   ███████    ........   ..........
View 2 slot   ........   ████████   ..........
View 3 slot   ........   ........   ██████████
```

Each view $(y, m)$ has features only in its designated slot:

$$\phi_{y,m} = [0, \ldots, 0, \underbrace{v_{y,m}}_{\text{slot } m}, 0, \ldots, 0]$$

where $v_{y,m} \in \mathbb{R}^{50}$ is the actual feature pattern for view $(y, m)$.

---

## Example 1: Sample Generation

### Setup

Class $y = 3$, with three views:
- $\phi_{3,1}$: feature pattern in dimensions 0-49
- $\phi_{3,2}$: feature pattern in dimensions 50-99
- $\phi_{3,3}$: feature pattern in dimensions 100-149

### Step 1: Draw Active Views

Each view activates independently with probability $p = 0.5$.

**Example draw**: $\mathcal{S} = \{1, 3\}$ (views 1 and 3 active, view 2 inactive)

### Step 2: Sum Active View Features

$$x_{\text{signal}} = \phi_{3,1} + \phi_{3,3}$$

Schematically:
```
φ_{3,1} = [███...█ | 0...0 | 0...0]     (50 dims | 50 dims | 50 dims)
φ_{3,3} = [0...0   | 0...0 | ███...█]
─────────────────────────────────────
x_signal= [███...█ | 0...0 | ███...█]   (views 1 and 3 present)
```

### Step 3: Add Noise

$$x = x_{\text{signal}} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 0.01 \cdot I_{150})$$

### Final Sample

```python
# Pseudocode
x = np.zeros(150)
x[0:50] = phi_3_1 + noise[0:50]      # View 1 present + noise
x[50:100] = noise[50:100]             # View 2 absent (only noise)
x[100:150] = phi_3_3 + noise[100:150] # View 3 present + noise
label = 3
```

---

## Example 2: View Coverage Calculation

### Setup

Trained network $f: \mathbb{R}^{150} \to \mathbb{R}^{10}$

We need to compute $C(f)$ by testing all 30 view-class pairs.

### Step-by-Step Calculation

**For each $(y, m) \in [10] \times [3]$:**

1. Compute view response: $R_{y,m}(f) = f(\phi_{y,m}) \in \mathbb{R}^{10}$

2. Check if detected: $\arg\max_k [R_{y,m}(f)]_k \stackrel{?}{=} y$

**Example: View $(y=3, m=1)$**

```
R_{3,1}(f) = f(φ_{3,1}) = [-1.2, 0.3, -0.8, 4.7, -0.5, 0.1, -0.9, 0.2, -1.1, 0.0]
                           ^     ^     ^     ^     ^     ^     ^     ^     ^     ^
                          k=0   k=1   k=2   k=3   k=4   k=5   k=6   k=7   k=8   k=9

argmax = 3 = y  ✓  View (3,1) is DETECTED
```

**Example: View $(y=3, m=2)$**

```
R_{3,2}(f) = f(φ_{3,2}) = [-0.8, 0.1, -0.5, 0.9, -0.3, 0.2, 2.1, -0.4, 0.3, -0.1]

argmax = 6 ≠ 3  ✗  View (3,2) is NOT DETECTED
```

### Coverage Computation

After testing all 30 views:

| Class y | View 1 | View 2 | View 3 | Views Detected |
|---------|--------|--------|--------|----------------|
| 0 | ✓ | ✗ | ✗ | 1 |
| 1 | ✗ | ✓ | ✗ | 1 |
| 2 | ✓ | ✗ | ✗ | 1 |
| 3 | ✓ | ✗ | ✗ | 1 |
| 4 | ✗ | ✗ | ✓ | 1 |
| 5 | ✗ | ✓ | ✗ | 1 |
| 6 | ✓ | ✗ | ✗ | 1 |
| 7 | ✗ | ✗ | ✓ | 1 |
| 8 | ✗ | ✓ | ✗ | 1 |
| 9 | ✓ | ✗ | ✗ | 1 |
| **Total** | 5 | 3 | 2 | **10** |

$$C(f) = \frac{10}{30} = 0.333 \approx \frac{1}{M} = \frac{1}{3}$$

**Result**: Exactly one view detected per class — single-view convergence confirmed.

---

## Example 3: Ensemble Coverage

### Setup

Train 5 networks $\{f_1, f_2, f_3, f_4, f_5\}$ with different random seeds.

### Individual Network Results

| Network | Views Learned per Class (majority) | Coverage |
|---------|-----------------------------------|----------|
| $f_1$ | {1,2,1,1,3,2,1,3,2,1} | 0.333 |
| $f_2$ | {2,1,2,3,1,1,2,1,3,2} | 0.333 |
| $f_3$ | {1,3,1,2,2,3,1,2,1,3} | 0.333 |
| $f_4$ | {3,2,3,1,1,2,3,1,2,1} | 0.333 |
| $f_5$ | {2,1,2,1,3,1,2,3,1,2} | 0.333 |

### Ensemble Coverage Calculation

For each view $(y, m)$, check if ANY network detects it:

**Class 0:**
| View | $f_1$ | $f_2$ | $f_3$ | $f_4$ | $f_5$ | Ensemble |
|------|-------|-------|-------|-------|-------|----------|
| m=1 | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ |
| m=2 | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ |
| m=3 | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |

All 3 views covered for class 0!

**Aggregated Results:**

| Class | Views Covered | Coverage |
|-------|---------------|----------|
| 0 | 3/3 | 100% |
| 1 | 3/3 | 100% |
| 2 | 3/3 | 100% |
| 3 | 3/3 | 100% |
| 4 | 3/3 | 100% |
| 5 | 3/3 | 100% |
| 6 | 2/3 | 67% |
| 7 | 3/3 | 100% |
| 8 | 3/3 | 100% |
| 9 | 3/3 | 100% |
| **Total** | **29/30** | **96.7%** |

### Comparison to Theory

**Theoretical prediction** for N=5, M=3:

$$C(\mathcal{E}) = 1 - \left(1 - \frac{1}{3}\right)^5 = 1 - \left(\frac{2}{3}\right)^5 = 1 - 0.132 = 0.868$$

**Observed**: 0.967

The observation exceeds prediction because some classes happened to have all views covered by chance. With more classes, the empirical coverage would converge to 0.868.

---

## Example 4: Knowledge Distillation

### Setup

- **Teacher**: Ensemble $\mathcal{E}$ from Example 3 (coverage ≈ 0.97)
- **Student**: Single network $S$ with same architecture as individual teachers
- **Training**: KD loss with temperature $\tau = 4$

### Training Process

**For each training sample $(x, y)$:**

1. Get teacher soft labels: $p_T(x) = \text{softmax}(f_\mathcal{E}(x) / \tau)$

2. Compute KD loss: $\mathcal{L}_{KD} = \tau^2 \cdot \text{KL}(p_T \| p_S)$

3. Update student via gradient descent

### Why Soft Labels Help: A Concrete Example

**Sample**: Class 3, views {1, 3} active

**Hard label**:
```
y_hard = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (one-hot for class 3)
```

**Teacher ensemble output** (ensemble knows views 1, 2, 3 for class 3):
```
f_E(x) = [-1.5, -0.8, -1.2, 6.2, -0.9, -1.1, -0.7, -1.0, -0.6, -1.3]
```

**Soft labels** ($\tau = 4$):
```
p_T(x) = softmax(f_E(x) / 4)
       = [0.04, 0.06, 0.05, 0.52, 0.05, 0.04, 0.06, 0.05, 0.07, 0.04]
```

**Key observation**: The soft label is NOT a sharp peak at class 3. It has:
- Strong signal for class 3 (0.52) — the correct answer
- Residual probability mass spread across other classes

This residual structure encodes information about *which views contributed* to the prediction. The student learns to reproduce this structure, implicitly learning multiple views.

### Student Result

After training:

$$C(S) = 0.90 \approx C(\mathcal{E}) = 0.97$$

The student (single network) achieves coverage comparable to the 5-network ensemble!

**Comparison**:
| Model | # Networks | Coverage |
|-------|------------|----------|
| Individual (hard labels) | 1 | 0.33 |
| Ensemble | 5 | 0.97 |
| **Student (KD from ensemble)** | **1** | **0.90** |

---

## Example 5: Input-Output Correlation

### Computing $\Sigma_{y,m}$

For view $(y=3, m=1)$ with $p_1 = 0.5$, $K = 10$:

$$\Sigma_{3,1} = \frac{p_1}{K} \cdot e_3 \cdot \phi_{3,1}^T = \frac{0.5}{10} \cdot e_3 \cdot \phi_{3,1}^T$$

**Dimensions**: $\Sigma_{3,1} \in \mathbb{R}^{10 \times 150}$

**Structure** (sparse):
```
Σ_{3,1} = 0.05 ×
          [  0    0    ...   0  ]  ← row 0
          [  0    0    ...   0  ]  ← row 1
          [  0    0    ...   0  ]  ← row 2
          [φ_{3,1}^T        0  ]  ← row 3 (only non-zero row)
          [  0    0    ...   0  ]  ← row 4
          [  ...              ]
          [  0    0    ...   0  ]  ← row 9
```

Only row 3 is non-zero, containing the feature vector $\phi_{3,1}$.

### Correlation Strength

$$\sigma_1(\Sigma_{3,1}) = \frac{p_1}{K} \|\phi_{3,1}\| = \frac{0.5}{10} \cdot 1 = 0.05$$

(assuming unit-normalized views)

**Key point**: Under symmetric assumptions (all $p_m$ equal, all $\|\phi_{y,m}\|$ equal), the correlation strength is the same for all views:

$$\sigma_1(\Sigma_{y,m}) = \frac{p}{K} = 0.05 \quad \forall (y, m)$$

This means the "winner" is determined entirely by initialization—the view whose pathway happens to be strongest at $t=0$.

---

## Example 6: Detection Margin Dynamics

### Tracking a Single View During Training

Consider view $(y=3, m=1)$. Track its detection margin $\rho_{3,1}(f_t)$ during training:

**Definition**:
$$\rho_{3,1}(f_t) = [R_{3,1}(f_t)]_3 - \max_{k \neq 3} [R_{3,1}(f_t)]_k$$

### Typical Trajectory (View That Wins)

```
Epoch |  R_{3,1}[3]  |  max R_{3,1}[k≠3]  |  ρ_{3,1}  | Detected?
------|--------------|--------------------|-----------|-----------
    0 |    0.12      |       0.08         |    0.04   |    ✓
   10 |    0.45      |       0.15         |    0.30   |    ✓
   50 |    1.82      |       0.23         |    1.59   |    ✓
  100 |    3.94      |       0.18         |    3.76   |    ✓
  200 |    5.21      |       0.09         |    5.12   |    ✓
```

The margin grows monotonically — this view is winning the race.

### Typical Trajectory (View That Loses)

Consider view $(y=3, m=2)$ which loses to view 1:

```
Epoch |  R_{3,2}[3]  |  max R_{3,2}[k≠3]  |  ρ_{3,2}  | Detected?
------|--------------|--------------------|-----------|-----------
    0 |    0.09      |       0.11         |   -0.02   |    ✗
   10 |    0.18      |       0.31         |   -0.13   |    ✗
   50 |    0.22      |       0.89         |   -0.67   |    ✗
  100 |    0.15      |       1.42         |   -1.27   |    ✗
  200 |    0.08      |       1.95         |   -1.87   |    ✗
```

The margin becomes increasingly negative — this view is being suppressed.

### Winner-Take-All Visualization

```
                   View 1 (winner)
Detection         ╱
Margin     ──────╱
                ╱
           ────╱─────────────────────────────────  0
              ╲
               ╲────
                    ╲────────  View 2 (loser)
                         ╲────────────────  View 3 (loser)

           └───────────────────────────────────────→
             0      50     100     150     200   Epoch
```

---

## Summary

These examples demonstrate:

1. **Sample generation**: How views combine in inputs with our slot structure
2. **Coverage calculation**: Testing all view-class pairs
3. **Ensemble diversity**: Different seeds covering different views
4. **KD mechanism**: Soft labels carrying multi-view information
5. **Correlation structure**: Why symmetric views create seed-dependent outcomes
6. **Margin dynamics**: How winning and losing views diverge

The calculations use our specific setup (K=10, M=3, d=150) but the principles generalize to any multi-view distribution.

---

## Code Reference

The synthetic data implementation is in:
- `/Users/michaellaielli/workspace/kdmech/engineer/src/data.py` — `MultiViewDataset` class
- `/Users/michaellaielli/workspace/kdmech/engineer/src/metrics.py` — Coverage calculation

---

## What's Next

You now have a concrete understanding of multi-view theory. Next:

- **Module 2**: [Neural Race Reduction](../02_neural_race_reduction/) — Why single-view convergence happens
- **Module 3**: [Connecting the Theories](../03_connecting_theories/) — Our unified framework
