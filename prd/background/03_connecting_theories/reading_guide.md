# Annotated Reading Guide: Key Papers

> **Purpose**: Guide researchers through the foundational literature with specific reading recommendations
> **Time commitment**: ~15-20 hours total for thorough reading
> **Recommendation**: Read in the order presented

---

## Overview

This guide covers five essential papers:

| Priority | Paper | Year | Why Essential |
|----------|-------|------|---------------|
| **P1** | Allen-Zhu & Li | 2023 | Multi-view theory foundation |
| **P1** | Saxe et al. | 2014 | Deep linear network dynamics |
| **P2** | Saxe et al. | 2022 | Neural race reduction |
| **P2** | Hinton et al. | 2015 | KD original formulation |
| **P3** | Jarvis et al. | 2025 | ReLU ↔ GDLN equivalence |

**P1** = Essential, read thoroughly
**P2** = Important, read key sections
**P3** = Supporting, skim or reference as needed

---

## Paper 1: Multi-View Theory

### Citation

> Allen-Zhu, Z., & Li, Y. (2023). **Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning**. *International Conference on Learning Representations (ICLR)*. [arXiv:2012.09816](https://arxiv.org/abs/2012.09816)

### Reading Time
- Full paper: ~4 hours
- Key sections only: ~2 hours

### What This Paper Does

Introduces a theoretical framework explaining why:
1. Neural networks learn only a subset of available features
2. Ensembles capture more features than individual networks
3. Knowledge distillation transfers this broader coverage to students

### Key Sections to Read

#### Must Read (Core)

**Section 2: Problem Setup** (~20 min)
- Definition of multi-view data distribution
- The "view" abstraction
- Mathematical notation you'll need throughout

*Focus on*: Definition 2.1 (multi-view distribution), the sampling process

**Section 3: Main Results** (~45 min)
- Theorem 3.1: Single-view convergence
- Theorem 3.2: Ensemble coverage
- Theorem 3.3: Distillation transfers coverage

*Focus on*: The theorem statements themselves; proof sketches are optional on first read

**Section 4: Why Single-View?** (~30 min)
- The "feature competition" intuition
- Why gradient descent favors one feature subset

*Focus on*: The intuitive explanation in 4.1

#### Recommended (Supporting)

**Section 5: Experiments** (~30 min)
- Synthetic data validation
- Real data (CIFAR) results

*Note*: Their synthetic setup is similar to ours

#### Optional (Deep Dive)

**Appendix A-C: Full Proofs**
- Only if you need proof techniques
- Can skip on first read

### Key Concepts to Extract

| Concept | Definition | Where Used |
|---------|------------|------------|
| View | Feature subset sufficient for classification | Throughout our work |
| View coverage C(f) | Fraction of views network can classify | Our main metric |
| Single-view convergence | C(f) ≈ 1/M after training | Theorem 2 explains why |
| Ensemble diversity | Different seeds → different views | Our experiments |

### Critical Quotes

> "The key observation is that neural networks trained with hard labels tend to learn only a subset of the available features, while ensembles of such networks collectively cover more features."

> "The mechanism by which networks select among equally predictive features remains unclear."
— **This is the gap we fill**

### What This Paper Doesn't Explain

1. ❌ **Which** view will be learned (no predictive formula)
2. ❌ **Why** winner-take-all at gradient level
3. ❌ **How** soft labels enable multi-view learning mechanistically

### Connection to Our Work

| Their Result | Our Extension |
|--------------|---------------|
| C(f) ≈ 1/M | Explained by race dynamics (Thm 2) |
| "Random" view selection | Predicted by initial advantage |
| KD transfers coverage | Explained by external signal (Thm 3) |

---

## Paper 2: Deep Linear Network Dynamics

### Citation

> Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). **Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Networks**. *International Conference on Learning Representations (ICLR)*. [arXiv:1312.6120](https://arxiv.org/abs/1312.6120)

### Reading Time
- Full paper: ~5 hours (math-heavy)
- Key sections only: ~2 hours

### What This Paper Does

Derives exact analytical solutions for learning dynamics in deep linear networks, revealing:
1. Learning is structured by input-output correlations (SVD)
2. Modes are learned sequentially (larger singular values first)
3. Depth creates non-trivial dynamics despite linear function class

### Key Sections to Read

#### Must Read (Core)

**Section 2: Setup** (~15 min)
- Deep linear network definition
- MSE loss formulation
- Gradient flow equations

*Focus on*: Equations (1)-(3), the basic setup

**Section 3: Two-Layer Dynamics** (~45 min)
- The key derivation
- Mode-by-mode decomposition
- The sigmoidal learning curve

*Focus on*: Equation (7) — the central dynamics equation:
$$\frac{da_\alpha}{dt} = s_\alpha \cdot a_\alpha \cdot (1 - a_\alpha^2/s_\alpha^2)$$

**Section 4: Implications** (~30 min)
- Sequential learning (larger modes first)
- The "learning time" formula
- Connection to real networks

*Focus on*: Figure 2 (learning curves), the scaling of learning time with singular value

#### Recommended (Supporting)

**Section 5: Deep Networks (L > 2)** (~30 min)
- Extension to arbitrary depth
- How depth affects dynamics

*Note*: We primarily use L=2, but this provides context

**Section 6: Experiments** (~20 min)
- Validation on real networks
- Comparison to nonlinear networks

#### Optional (Deep Dive)

**Appendix: Full Derivations**
- Detailed mathematics
- Only if reproducing proofs

### Key Equations to Understand

**1. Mode Dynamics**
$$\frac{da_\alpha}{dt} = s_\alpha \cdot a_\alpha \cdot \left(1 - \frac{a_\alpha^2}{s_\alpha^2}\right)$$

- $a_\alpha$: network's alignment with mode $\alpha$
- $s_\alpha$: singular value (strength of mode in data)
- This is a **sigmoidal** ODE

**2. Solution**
$$a_\alpha(t) = \frac{s_\alpha}{\sqrt{1 + (s_\alpha^2/a_\alpha(0)^2 - 1)e^{-2s_\alpha t}}}$$

**3. Learning Time**
$$\tau_\alpha \sim \frac{1}{s_\alpha} \log(s_\alpha / a_\alpha(0))$$

### Critical Insights

> "Despite computing only a linear function, deep linear networks exhibit nonlinear learning dynamics that capture important aspects of deep learning."

> "Learning proceeds in a stage-wise manner, with higher singular value modes learned before lower singular value modes."

### What This Paper Provides for Us

| Their Contribution | How We Use It |
|-------------------|---------------|
| Exact ODE for mode dynamics | Basis for pathway dynamics |
| Sequential learning | Explains why one view dominates |
| SVD structure | Correlation strength σ₁ |

### Limitations for Our Context

1. Linear networks only (no gating)
2. MSE loss (not cross-entropy)
3. No input-dependent computation

→ These are addressed by Saxe et al. 2022

---

## Paper 3: Neural Race Reduction

### Citation

> Saxe, A. M., Sodhani, S., & Gershman, S. (2022). **The Neural Race Reduction: Dynamics of Abstraction in Gated Networks**. *International Conference on Machine Learning (ICML)*. [arXiv:2207.10430](https://arxiv.org/abs/2207.10430)

### Reading Time
- Full paper: ~4 hours
- Key sections only: ~2 hours

### What This Paper Does

Extends deep linear network analysis to **gated** networks, showing:
1. Different gating patterns define different "pathways"
2. Pathways "race" during training
3. Winner-take-all dynamics emerge naturally

### Key Sections to Read

#### Must Read (Core)

**Section 2: Gated Deep Linear Networks** (~20 min)
- GDLN definition
- Connection to ReLU networks
- Pathway concept

*Focus on*: Definition of gating matrices, Equation (2)

**Section 3: Race Dynamics** (~45 min)
- How pathways compete
- The race equations
- Winner determination

*Focus on*: Theorem 1 (race dynamics), the competitive Lotka-Volterra form

**Section 4: Abstraction and Compositionality** (~30 min)
- How races create structured representations
- The "initial advantage" concept

*Focus on*: The intuition for why initial conditions matter

#### Recommended (Supporting)

**Section 5: Experiments** (~30 min)
- Empirical validation of race dynamics
- Visualization of pathway competition

*Note*: Their pathway strength tracking is what we replicate

#### Optional (Deep Dive)

**Appendix: Proofs and Details**
- Full mathematical treatment
- Reference as needed

### Key Concepts

**1. Gated Deep Linear Network (GDLN)**
$$f(x) = W_L \cdot D_{L-1}(x) \cdot W_{L-1} \cdots D_1(x) \cdot W_1 \cdot x$$

where $D_\ell(x) = \text{diag}(G_\ell(x))$ are input-dependent gates.

**2. Pathway**
A pathway is the linear map associated with a specific gating pattern:
$$P_g(x) = W_L \text{diag}(g_{L-1}) \cdots \text{diag}(g_1) W_1 x$$

**3. Race Equation**
$$\frac{ds_g}{dt} = r_g \cdot s_g \cdot (1 - \text{competition})$$

Pathways with higher initial $r_g \cdot s_g(0)$ win.

### Critical Quotes

> "Different gating patterns partition the input space into regions, each with its own effective linear function."

> "Learning dynamics become a race between pathways to explain the data."

### What This Paper Provides for Us

| Their Contribution | How We Use It |
|-------------------|---------------|
| GDLN formalism | Our network model |
| Pathway definition | View pathways |
| Race dynamics | Theorem 2 foundation |
| Winner-take-all | Explains single-view convergence |

### What They Don't Cover

1. ❌ Multi-view data specifically
2. ❌ Knowledge distillation
3. ❌ Soft label effects on dynamics

→ These are our contributions (Theorems 1, 3)

---

## Paper 4: Knowledge Distillation

### Citation

> Hinton, G., Vinyals, O., & Dean, J. (2015). **Distilling the Knowledge in a Neural Network**. *NIPS Deep Learning Workshop*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

### Reading Time
- Full paper: ~1 hour (short and accessible)
- Key sections only: ~30 min

### What This Paper Does

Introduces knowledge distillation:
1. Train student on soft labels from teacher
2. Use temperature to soften distributions
3. Student can match (or approach) teacher performance

### Key Sections to Read

#### Must Read (Core)

**Section 1: Introduction** (~10 min)
- The KD idea
- "Dark knowledge" concept
- Temperature parameter

*Focus on*: The soft label formulation, intuition for why it helps

**Section 2: Distillation** (~15 min)
- Mathematical formulation
- KD loss function
- Temperature scaling

*Focus on*: Equations (1)-(3), the KD loss

#### Recommended (Supporting)

**Section 3: Experiments** (~20 min)
- MNIST results
- Speech recognition
- Ensemble distillation

### Key Equations

**Soft Labels**
$$p_i^\tau = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$$

where $\tau$ is temperature (higher = softer).

**KD Loss**
$$\mathcal{L}_{KD} = \tau^2 \cdot \text{KL}(p_T^\tau \| p_S^\tau)$$

The $\tau^2$ factor ensures gradient magnitudes are preserved.

### Critical Quote

> "When the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases."

This is the "dark knowledge" — information in the soft distribution beyond the hard label.

### What This Paper Provides for Us

| Their Contribution | How We Use It |
|-------------------|---------------|
| KD formulation | Our training objective |
| Temperature parameter | Controls view information preservation |
| "Dark knowledge" intuition | We formalize as view information |

### What They Don't Explain

1. ❌ Why dark knowledge helps mechanistically
2. ❌ What information is in soft labels
3. ❌ Gradient-level analysis

→ Theorem 3 provides the mechanistic explanation

---

## Paper 5: ReLU-GDLN Equivalence

### Citation

> Jarvis, D., Klein, R., & Rosman, B. (2025). **On the Relationship Between ReLU Networks and Gated Linear Networks**. *International Conference on Machine Learning (ICML)*.

### Reading Time
- Skim: ~30 min
- Reference as needed

### What This Paper Does

Formally proves that ReLU networks are exactly equivalent to GDLNs:
$$\text{ReLU}(z) = z \cdot \mathbb{1}[z > 0]$$

### Key Result

**Theorem (ReLU-GDLN Equivalence)**:
Any ReLU network can be written as a GDLN where gates are:
$$[G_\ell(x)]_j = \mathbb{1}[h_\ell^{(j)}(x) > 0]$$

### Why This Matters for Us

- Justifies analyzing ReLU networks via GDLN formalism
- Our experiments use ReLU networks
- Theory applies through this equivalence

### Reading Recommendation

- Skim Section 2 (main result)
- Skip proofs unless needed
- Reference for justifying GDLN analysis

---

## Reading Schedule Recommendation

### Week 1: Foundations

| Day | Paper | Sections | Time |
|-----|-------|----------|------|
| 1-2 | Allen-Zhu & Li | Sections 2-4 | 3 hr |
| 3-4 | Saxe et al. 2014 | Sections 2-4 | 3 hr |
| 5 | Review + Module 1-2 notes | | 2 hr |

### Week 2: Race Dynamics & KD

| Day | Paper | Sections | Time |
|-----|-------|----------|------|
| 1-2 | Saxe et al. 2022 | Sections 2-4 | 3 hr |
| 3 | Hinton et al. 2015 | Full paper | 1 hr |
| 4 | Jarvis et al. 2025 | Skim | 30 min |
| 5 | Review + Module 3 notes | | 2 hr |

---

## Cross-Paper Concept Map

```
                    ┌─────────────────────┐
                    │  Hinton et al. 2015 │
                    │  (KD Formulation)   │
                    └──────────┬──────────┘
                               │
                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│ Allen-Zhu & Li  │    │   OUR WORK       │    │ Saxe et al. 2014   │
│ 2023            │───►│                  │◄───│                    │
│ (Multi-View)    │    │ Thm 1: Views =   │    │ (Linear Dynamics)  │
│                 │    │        Pathways  │    │                    │
│ - Views         │    │                  │    │ - Mode dynamics    │
│ - Coverage      │    │ Thm 2: Race      │    │ - SVD structure    │
│ - KD transfer   │    │        Dynamics  │    │ - Sequential       │
│                 │    │                  │    │   learning         │
└─────────────────┘    │ Thm 3: KD breaks │    └────────────────────┘
                       │        the race  │              │
                       │                  │              │
                       └────────┬─────────┘              │
                                │                        │
                                ▼                        ▼
                       ┌────────────────────────────────────┐
                       │       Saxe et al. 2022             │
                       │       (Neural Race Reduction)      │
                       │                                    │
                       │       - GDLN formalism             │
                       │       - Pathway competition        │
                       │       - Winner-take-all            │
                       └───────────────────┬────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────┐
                       │       Jarvis et al. 2025           │
                       │       (ReLU = GDLN)                │
                       └────────────────────────────────────┘
```

---

## Quick Reference: Key Results by Paper

| Paper | Key Result | Our Use |
|-------|------------|---------|
| Allen-Zhu & Li | C(f) ≈ 1/M | What we explain |
| Saxe 2014 | $\dot{a} = sa(1-a^2/s^2)$ | Basis for dynamics |
| Saxe 2022 | Pathway race | Theorem 2 foundation |
| Hinton | $\mathcal{L}_{KD} = \tau^2 \text{KL}(p_T \| p_S)$ | Our objective |
| Jarvis | ReLU = GDLN | Justifies analysis |

---

## Further Reading (Optional)

### On Knowledge Distillation
- Cho & Hariharan (2019): "On the Efficacy of Knowledge Distillation"
- Mobahi et al. (2020): "Self-Distillation Amplifies Regularization"

### On Feature Learning
- Arora et al. (2019): "Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks"

### On Learning Dynamics
- Du et al. (2019): "Gradient Descent Finds Global Minima of Deep Neural Networks"
- Jacot et al. (2018): "Neural Tangent Kernel"
