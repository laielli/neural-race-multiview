# Paper Development Plan: Neural Race × Multi-View Theory

## Executive Summary

**Target**: NeurIPS 2026 (deadline ~late May 2026, conference Dec 6-12 in Sydney)

**Core Thesis**: The Neural Race Reduction provides the mechanistic foundation for Multi-View Theory in knowledge distillation, explaining WHY networks learn one view, PREDICTING which view will be learned, and EXPLAINING why distillation circumvents this limitation.

**Unique Value**: First paper to connect these two theoretical frameworks. Fills the explicit gap: "no predictive theory for which view wins."

---

## Part 1: Strategic Positioning

### Two Complementary Frameworks

| Framework | Explains | Doesn't Explain |
|-----------|----------|-----------------|
| **Multi-View Theory** (Allen-Zhu & Li 2023) | Ensembles cover more views; distillation transfers views | WHY networks learn one view; WHICH view will be learned |
| **Neural Race Reduction** (Saxe et al. 2022-2025) | Learning dynamics as pathway competition; what determines winners | Connection to KD; how soft labels affect race |

### The Gap We Fill

**Multi-View Theory's Open Questions** (from CLAUDE.md):
1. ❓ "Which view wins?" — No predictive theory exists
2. ❓ "Why does distillation succeed where direct training fails?"
3. ❓ "What determines which feature subset a network selects?"

**Neural Race Provides Answers**:
1. ✅ The pathway with highest input-output correlation + largest initialization wins
2. ✅ Soft labels distribute gradients across pathways, preventing race monopolization
3. ✅ Initialization alignment with view features determines selection

### Comparison to View Coverage Theory (Existing Direction)

| Aspect | View Coverage Theory | Neural Race × Multi-View |
|--------|---------------------|-------------------------|
| **Focus** | Quantifying coverage; optimal λ | Mechanistic explanation; predictions |
| **Novelty** | Formal C(f) definition | Unifies two frameworks |
| **Tractability** | Medium (extends Allen-Zhu & Li) | Medium-Hard (extends both) |
| **Impact** | Practical (how to improve) | Foundational (why it works) |

**Recommendation**: Pursue Neural Race × Multi-View as primary direction (higher novelty, fills bigger gap). View Coverage Theory can be incorporated or pursued separately.

---

## Part 2: Paper Scope and Contributions

### Proposed Title

**"The Neural Race Explains Multi-View Learning: Why Networks Choose One View and How Distillation Overcomes This"**

Alternative: "Racing to One View: A Unified Theory of Feature Selection and Knowledge Distillation"

### Core Contributions

#### Contribution 1: View-Pathway Correspondence
**Theorem**: Under multi-view data, each view V_i corresponds to a distinct computational pathway P_i in the network. Learning view V_i is equivalent to pathway P_i winning the neural race.

**Significance**: Formally connects two major theoretical frameworks.

#### Contribution 2: Predictive Theory for View Selection
**Theorem**: Given multi-view data D and initialization θ₀, the probability that view V_i is learned is:

$$P(V_i \text{ wins}) \propto \sigma_1(Φ_{V_i}) \cdot ||\theta_0^{(i)}||^2 \cdot \exp(-d_i / \tau)$$

where:
- $\sigma_1(Φ_{V_i})$ = largest singular value of view i's input-output correlation
- $||\theta_0^{(i)}||$ = initialization magnitude in pathway i
- $d_i$ = depth of pathway i
- $\tau$ = temperature parameter

**Significance**: First quantitative prediction for which view will be learned. Testable.

#### Contribution 3: Why KD Circumvents the Race
**Theorem**: Training with soft labels from a teacher T distributes gradients across pathways proportionally to T's view coverage. If T covers views {V_1, ..., V_k}, the student receives non-zero gradient signal for pathways {P_1, ..., P_k}, preventing race monopolization.

**Significance**: Mechanistic explanation for KD's success. Explains why students can learn what teachers know but direct training cannot discover.

#### Contribution 4: Design Principles
**Corollary**: From race dynamics, derive:
1. **Initialization schemes** that don't favor any single view
2. **Architectures** where pathways are equally competitive
3. **Training objectives** that slow dominant pathways

**Significance**: Practical guidance for multi-view learning without ensembles.

### Paper Structure (9 pages main + appendix)

| Section | Pages | Content |
|---------|-------|---------|
| **1. Introduction** | 1.5 | Motivation, gap, contributions |
| **2. Background** | 1.0 | Multi-view theory, neural race reduction |
| **3. View-Pathway Correspondence** | 1.5 | Setup, Theorem 1, proof sketch |
| **4. Predictive Theory** | 1.5 | Theorem 2, derivation, predictions |
| **5. Why KD Works** | 1.5 | Theorem 3, gradient analysis |
| **6. Experiments** | 2.0 | Synthetic + real data validation |
| **7. Discussion & Conclusion** | 0.5 | Limitations, future work |
| **Appendix** | ∞ | Full proofs, additional experiments |

---

## Part 3: Proof Strategy

### 3.1 Mathematical Setup

**Multi-View Data Model** (following Allen-Zhu & Li):
- K classes, each with M views
- Sample x from class k: $x = \sum_{v \in S_k(x)} \phi_v + \epsilon$
- Views are orthogonal: $\langle \phi_v, \phi_{v'} \rangle \approx 0$ for $v \neq v'$
- Each view independently sufficient for classification

**Gated Deep Linear Network** (following Saxe et al.):
- L layers with weights $W_1, ..., W_L$
- Gating functions $G_1(x), ..., G_{L-1}(x)$ (input-dependent)
- Output: $f(x) = W_L \cdot G_{L-1}(x) \cdot W_{L-1} \cdot ... \cdot G_1(x) \cdot W_1 \cdot x$

**Key Insight**: For multi-view data, different views activate different gating patterns. Define pathway $P_v$ as the gating pattern activated by view v.

### 3.2 Theorem 1: View-Pathway Correspondence

**Setup**:
- Let $G_v(x) = \mathbb{1}[\phi_v \in x]$ be the gating pattern when view v is present
- Define pathway $P_v = \{W_L \cdot G_v \cdot W_{L-1} \cdot ... \cdot G_v \cdot W_1\}$

**Statement**: Under multi-view data with orthogonal views, the network's output for class k can be decomposed as:
$$f_k(x) = \sum_{v \in V_k} R_v(W) \cdot \mathbb{1}[\phi_v \in x]$$

where $R_v(W)$ is the "view response" depending only on pathway $P_v$.

**Proof Approach**:
1. Use linearity of GDLN given fixed gating
2. Show that orthogonal views produce independent gating patterns
3. Each gating pattern defines a distinct pathway
4. The network's response to each view is determined by the corresponding pathway's weights

### 3.3 Theorem 2: Race Dynamics Under Multi-View Data

**Setup**:
- Initialize weights with scale $\sigma_0$
- Train with gradient descent on cross-entropy loss
- Track pathway "strengths" $s_v(t) = ||P_v(W(t))||_F$

**Statement**: The learning dynamics of pathway strengths follow:
$$\frac{ds_v}{dt} = \sigma_1(\Sigma_v) \cdot s_v \cdot (1 - \sum_{v'} s_{v'}^2 / s_{\max}^2) + O(\sigma_0^3)$$

where $\Sigma_v$ is the input-output correlation for view v.

**Key Result**: This is a "winner-take-all" dynamic. The pathway with highest initial $\sigma_1(\Sigma_v) \cdot s_v(0)$ grows fastest and eventually dominates.

**Proof Approach**:
1. Start from Saxe et al.'s exact dynamics for deep linear networks
2. Extend to GDLN with input-dependent gating
3. Show that multi-view structure creates separate "races" per pathway
4. Analyze equilibrium: one pathway dominates when $s_v \gg s_{v'}$ for all $v' \neq v$

### 3.4 Theorem 3: KD Gradient Distribution

**Setup**:
- Teacher T with soft output $p_T(y|x) = \text{softmax}(f_T(x)/\tau)$
- Student S trained with KD loss $L_{KD} = KL(p_T || p_S)$

**Statement**: The gradient of $L_{KD}$ with respect to pathway $P_v$'s weights is:
$$\nabla_{P_v} L_{KD} \propto C_v(T) \cdot (\text{pathway-specific terms})$$

where $C_v(T)$ is teacher T's coverage of view v.

**Key Insight**: Unlike hard labels (which provide gradient only to the winning pathway), soft labels distribute gradient across all pathways proportionally to teacher's view coverage.

**Proof Approach**:
1. Compute gradient of KL divergence
2. Use chain rule through student's pathway decomposition
3. Show that teacher's soft output encodes information about multiple views
4. Each view's information creates gradient signal for corresponding pathway

### 3.5 Simplifying Assumptions for Tractability

**Level 1 (Most Tractable)**:
- Two views (M = 2)
- Linear features (deep linear network)
- Orthogonal views
- Balanced view presence probability

**Level 2 (Moderate)**:
- Multiple views (M > 2)
- Deep linear network
- Approximately orthogonal views

**Level 3 (Full Generality)**:
- ReLU networks via GDLN equivalence
- Correlated views
- Varying view probabilities

**Strategy**: Prove Level 1 rigorously, extend to Level 2, argue Level 3 via GDLN equivalence + experiments.

---

## Part 4: Experimental Design

### 4.1 Tier 1: Synthetic Validation (Essential)

**Goal**: Verify theorems under controlled conditions where ground truth is known.

**Dataset**: Multi-view synthetic data
- K = 10 classes
- M = 3 views per class
- Views are orthogonal Gaussian clusters
- Control parameters: view separation, overlap probability, noise level

**Experiments**:

| Experiment | Prediction | Metric |
|------------|------------|--------|
| E1.1: Different seeds → different views | High diversity | View diversity across seeds |
| E1.2: Initialization predicts winner | Correlation > 0.8 | Corr(init alignment, learned view) |
| E1.3: Race dynamics observable | Early divergence | Pathway strength over training |
| E1.4: KD distributes gradients | Multi-pathway activation | Gradient magnitude per pathway |
| E1.5: Distilled model covers multiple views | C(student) > C(individual) | View coverage measure |

**Validation Criteria**:
- Theorem 1: View-pathway decomposition matches empirical output
- Theorem 2: Predicted winning view matches actual learned view (>80% accuracy)
- Theorem 3: Gradient distribution under KD matches prediction

### 4.2 Tier 2: Mechanism Analysis (Important)

**Goal**: Visualize and validate the race mechanism.

**Experiments**:

**E2.1: Race Visualization**
- Train network on multi-view data
- Track pathway strengths $s_v(t)$ over training
- Expected: One pathway grows while others decay

**E2.2: Gradient Distribution Comparison**
- Compare gradient flow under: (a) hard labels, (b) soft labels from single teacher, (c) soft labels from ensemble
- Expected: (a) concentrated, (b) partially distributed, (c) fully distributed

**E2.3: Initialization Intervention**
- Deliberately initialize to favor specific views
- Verify that predicted view is learned
- Test: Can we reliably control which view is learned?

**E2.4: Race Slowing Interventions**
- Test training modifications that should slow the race:
  - Smaller learning rate
  - Label smoothing
  - Intra-class contrastive loss
- Measure: View coverage C(f) and rate of race convergence

### 4.3 Tier 3: Real Data Validation (Strengthens Paper)

**Datasets**: CIFAR-10, CIFAR-100, TinyImageNet

**Experiments**:

**E3.1: View Diversity Across Seeds**
- Train 10 models with different random seeds
- Analyze learned features via Grad-CAM
- Expected: Different models focus on different object parts

**E3.2: Ensemble vs. Individual Coverage**
- Measure "view coverage" proxy via feature diversity
- Compare ensemble of 10 models vs. single model
- Expected: Ensemble has broader coverage

**E3.3: Distillation Transfer**
- Distill ensemble into single student
- Measure student's view coverage
- Expected: Student matches ensemble coverage, exceeds individual

**E3.4: Prediction Validation**
- Estimate initialization alignment with different "views" (feature clusters)
- Predict which view each model will learn
- Validate prediction accuracy

### 4.4 Tier 4: Practical Implications (Differentiating)

**E4.1: Optimal Initialization for Multi-View**
- Design initialization that doesn't favor any view
- Compare to standard initialization
- Measure: Average view coverage

**E4.2: Training Objectives That Prevent Race**
- Test: intra-class contrastive, self-distillation checkpoints, progressive training
- Identify most effective intervention

**E4.3: When Does Theory Fail?**
- Test on data that violates multi-view assumptions
- Single-view data: MNIST (simple)
- Correlated views: hierarchical classes
- Expected: Predictions fail gracefully; theory indicates limitations

---

## Part 5: Timeline for NeurIPS 2026

**Key Dates** (estimated based on NeurIPS 2025):
- Abstract deadline: ~May 15, 2026
- Full paper deadline: ~May 22, 2026
- Reviews: July-August 2026
- Decisions: ~September 2026
- Conference: December 6-12, 2026

### Phase 1: Theoretical Foundation (Jan 1 - Feb 28, 2026)

**Duration**: 8 weeks

| Week | Goal | Deliverable |
|------|------|-------------|
| 1-2 | Setup formal framework | Multi-view GDLN definition, notation |
| 3-4 | Prove Theorem 1 (view-pathway) | Proof for M=2 case |
| 5-6 | Prove Theorem 2 (race dynamics) | Proof for linear case |
| 7-8 | Analyze Theorem 3 (KD gradients) | Gradient derivation |

**Milestone**: Draft theoretical framework with proofs for simplified case.

**Risk Point**: If proofs don't work out by Week 6, consider:
- Weaker theorem statements (bounds instead of exact)
- More assumptions
- Empirical validation of conjectures

### Phase 2: Synthetic Experiments (Mar 1 - Apr 15, 2026)

**Duration**: 6 weeks

| Week | Goal | Deliverable |
|------|------|-------------|
| 1-2 | Implement synthetic data generator | Code + basic experiments |
| 3-4 | Validate Theorems 1-2 | E1.1-E1.3 results |
| 5-6 | Validate Theorem 3 + mechanisms | E1.4-E1.5, E2.x results |

**Milestone**: Experimental evidence supporting all main theorems.

**Feedback Loop**: Refine theory based on experimental findings.

### Phase 3: Real Data + Writing (Apr 16 - May 22, 2026)

**Duration**: 5 weeks

| Week | Goal | Deliverable |
|------|------|-------------|
| 1 | CIFAR experiments | E3.x results |
| 2 | Practical implications | E4.x results |
| 3 | Draft introduction + theory sections | 5 pages |
| 4 | Draft experiments + discussion | 4 pages |
| 5 | Polish, internal review, submit | Final paper |

**Milestone**: Submitted paper.

### Buffer Allocation
- 2 weeks distributed across phases
- Use for: proof difficulties, unexpected experimental results, writing revisions

### Parallel Work Streams

```
Theory Development:     [==============]
Synthetic Experiments:          [==========]
Real Experiments:                    [======]
Writing:                                [========]
                        Jan   Feb   Mar   Apr   May
```

---

## Part 6: Risk Analysis

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Proofs too hard for full generality | Medium | High | Prove simplified case; validate extensions empirically |
| GDLN approximation too coarse for ReLU | Medium | Medium | Bound approximation error; validate on real networks |
| Multi-view assumption unrealistic | Low | Medium | Explicit failure case analysis; show graceful degradation |
| Predictions don't hold in experiments | Low | High | Pilot experiments early; refine theory |

### Strategic Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scooped on Neural Race × KD connection | Low | High | Move quickly; unique framing |
| Reviewers don't see value | Medium | Medium | Strong experiments; practical implications |
| Positioned as "incremental" | Medium | Medium | Emphasize predictive power (novel); unified framework |

### Contingency Plans

**If proofs fail (by Feb 28)**:
- Option A: Weaken to "empirically validated conjecture"
- Option B: Pivot to View Coverage Theory (more tractable)
- Option C: Pure empirical paper on race dynamics in KD

**If experiments contradict theory**:
- Revise theoretical assumptions
- Characterize failure modes (still valuable contribution)
- Focus on conditions where theory holds

**If time runs short**:
- Priority 1: Theorems 1-2 + synthetic experiments (minimal viable paper)
- Priority 2: Add real data experiments
- Priority 3: Add practical implications

---

## Part 7: Anticipated Reviewer Objections

### Objection 1: "This just combines existing frameworks"

**Response**:
- The combination yields NEW predictions neither framework makes alone
- Specifically: quantitative prediction of which view will be learned
- We validate these predictions empirically
- The mechanistic explanation for KD is novel

### Objection 2: "Proofs require unrealistic assumptions"

**Response**:
- Simplified settings (linear networks, Gaussian data) are standard in theoretical ML
- Saxe et al. (2014) established this paradigm; cited 2000+ times
- We validate that conclusions hold for realistic networks empirically
- We explicitly characterize when assumptions break down

### Objection 3: "How is this different from Saxe et al.'s work?"

**Response**:
- Saxe et al. analyze race dynamics but NOT in context of multi-view data
- They focus on systematization/compositionality, not KD
- We provide the first connection to multi-view theory
- We derive predictions for KD that their work doesn't address

### Objection 4: "The practical impact is limited"

**Response**:
1. First predictive theory for view selection (currently no theory exists)
2. Explains when KD will/won't work (actionable guidance)
3. Suggests initialization and training modifications for multi-view learning
4. Unifies understanding across two major theoretical directions

### Objection 5: "How does this relate to label averaging (Jeong & Chung 2025)?"

**Response**:
- Label averaging operates in fixed-feature regime (linear probing)
- Neural race operates in feature-learning regime (training from scratch)
- These are complementary mechanisms for different settings
- We characterize when each dominates: network capacity, training paradigm, data structure

### Objection 6: "The multi-view assumption is too strong"

**Response**:
- We inherit this from Allen-Zhu & Li (2023), accepted at ICLR 2023
- We explicitly analyze failure cases (Section X)
- Theory degrades gracefully: partial multi-view structure → partial predictions
- Real data experiments validate practical applicability

---

## Part 8: Collaboration and Resources

### Required Expertise
1. **Theoretical ML**: Proof development, dynamics analysis
2. **Deep learning**: Experiments, architecture understanding
3. **Optimization**: Gradient analysis, convergence

### Compute Requirements
- Synthetic experiments: Minimal (laptop-scale)
- CIFAR experiments: Moderate (1-2 GPUs, days)
- ImageNet (if included): Significant (multi-GPU, weeks)

### Key References to Study Deeply
1. [Saxe et al. 2014](https://arxiv.org/abs/1312.6120) - Deep linear network dynamics
2. [Saxe et al. 2022](https://arxiv.org/abs/2207.10430) - Neural Race Reduction
3. [Jarvis et al. 2025](https://arxiv.org/abs/2503.06181) - ReLU ↔ GDLN equivalence
4. [Allen-Zhu & Li 2023](https://arxiv.org/abs/2012.09816) - Multi-view theory
5. [Jeong & Chung 2025](https://arxiv.org/abs/2402.10482) - Label averaging alternative

---

## Part 9: Success Criteria

### Minimum Viable Paper (Priority 1)
- [ ] Theorem 1: View-pathway correspondence (proven for M=2, linear)
- [ ] Theorem 2: Race dynamics predict view selection (proven for simplified case)
- [ ] Synthetic experiments validate predictions (E1.1-E1.5)
- [ ] Clear writing and positioning

### Strong Paper (Priority 2)
- [ ] All above, plus:
- [ ] Theorem 3: KD gradient distribution
- [ ] Real data experiments (CIFAR)
- [ ] Mechanism visualizations (race dynamics)

### Excellent Paper (Priority 3)
- [ ] All above, plus:
- [ ] Proofs extend beyond M=2
- [ ] Practical implications demonstrated
- [ ] Failure case analysis
- [ ] ImageNet or additional datasets

---

## Part 10: Immediate Next Steps

### This Week
1. **Read Saxe et al. 2014 deeply**: Understand exact dynamics, proof techniques
2. **Read Allen-Zhu & Li 2023 Appendix**: Understand multi-view formalization
3. **Sketch formal setup**: Write out multi-view GDLN model in LaTeX

### Next 2 Weeks
1. **Attempt Theorem 1 proof**: View-pathway correspondence for M=2
2. **Pilot experiment**: Generate synthetic multi-view data, verify basic predictions
3. **Literature check**: Search for any Neural Race × KD connections

### Next Month
1. **Complete Phase 1 theory**: All theorem sketches
2. **Validate with experiments**: Pilot synthetic results
3. **Decision point**: Commit to this direction or pivot

---

## Summary

**This paper unifies Neural Race Reduction and Multi-View Theory to provide:**

1. ✅ **Mechanistic explanation** for why networks learn one view (race dynamics)
2. ✅ **Predictive theory** for which view will be learned (fills major gap)
3. ✅ **Understanding** of why KD works (gradient distribution)
4. ✅ **Design principles** for multi-view learning

**Key strengths:**
- High novelty (first connection between frameworks)
- Fills explicit theoretical gap
- Testable predictions
- Practical implications

**Main challenges:**
- Proof complexity for general case
- Validating GDLN approximation
- Convincing reviewers of value

**Recommendation**: Pursue this direction with View Coverage Theory as backup. The Neural Race connection is deeper and more novel.
