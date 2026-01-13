# Paper Requirements Document: Neural Race × Multi-View Theory

## Research Question

**How does the Neural Race Reduction explain multi-view learning in deep networks, and why does knowledge distillation succeed where direct training fails?**

### Sub-questions
1. Why do networks trained on multi-view data typically learn only one view?
2. Which view will a network learn, and can we predict this from initialization and data structure?
3. How do soft labels in knowledge distillation prevent winner-take-all dynamics?

---

## Hypothesis

The Neural Race Reduction provides the mechanistic foundation for Multi-View Theory:

1. **View-Pathway Correspondence**: Each view in multi-view data corresponds to a distinct computational pathway in the network
2. **Race Dynamics**: Winner-take-all competition between pathways explains single-view convergence
3. **KD Mechanism**: Soft labels distribute gradients across pathways, preventing race monopolization and enabling multi-view learning

---

## Target Venue

**NeurIPS 2026** (Neural Information Processing Systems)
- **Abstract Deadline**: ~May 15, 2026
- **Full Paper Deadline**: ~May 22, 2026
- **Conference**: December 6-12, 2026 in Sydney, Australia

### Why NeurIPS?
- Top-tier venue for theoretical ML research
- Strong track record for unified theory papers
- Audience interested in both learning dynamics and knowledge distillation
- Timeline aligns with May deadline

---

## Success Metrics

### Theoretical Contributions
- [ ] 3 theorems formally stated with rigorous mathematical setup
- [ ] At least simplified proofs (M=2, L=2 case) for core theorems
- [ ] Novel predictive theory for view selection

### Experimental Validation
- [ ] Synthetic experiments validate race dynamics
- [ ] Demonstrate winner-take-all behavior (coverage ≈ 1/M)
- [ ] Show KD enables multi-view learning (coverage ≈ C(teacher))
- [ ] Predictions about view selection match experimental observations

### Impact Metrics
- [ ] Paper accepted at NeurIPS 2026
- [ ] Fills explicit gap in Multi-View Theory ("no predictive theory for which view wins")
- [ ] First work unifying Neural Race Reduction with Multi-View Theory
- [ ] Provides mechanistic explanation for KD's success

---

## Core Contributions

### Contribution 1: View-Pathway Correspondence
**Theorem**: Under multi-view data, each view V_i corresponds to a distinct computational pathway P_i in the network. Learning view V_i is equivalent to pathway P_i winning the neural race.

**Significance**: Formally connects two major theoretical frameworks.

### Contribution 2: Predictive Theory for View Selection
**Theorem**: Initial advantage (determined by initialization alignment and view strength) predicts which view will be learned. Networks converge to single-view solutions with C(f) ≈ 1/M.

**Significance**: First quantitative prediction for view selection. Testable and actionable.

### Contribution 3: Why KD Circumvents the Race
**Theorem**: Training with soft labels distributes gradients across pathways proportionally to teacher's view coverage, preventing race monopolization.

**Significance**: Mechanistic explanation for KD's success beyond "knowledge transfer."

### Contribution 4: Design Principles
- How to initialize networks to learn specific views
- When to use KD vs direct training
- How to diagnose view coverage in practice

---

## Timeline

| Phase | Dates | Goals | Status |
|-------|-------|-------|--------|
| **1. Theory** | Jan-Feb 2026 | Prove theorems (simplified case) | Not started |
| **2. Synthetic** | Mar-Apr 2026 | Validate with controlled experiments | **In Progress** |
| **3. Real Data** | Apr-May 2026 | CIFAR/ImageNet validation | Not started |
| **4. Writing** | Apr-May 2026 | Complete paper draft | **Sketch v0.1 Done** |

**Current**: Early Development (Theory-experiment mismatch under investigation)

---

## Current Status

### Completed
✅ **Theory Design**: All 3 theorems formally specified (see specs/)
✅ **Infrastructure**: Complete experiment codebase (~2,600 lines)
✅ **Initial Experiments**: 11/17 experiments completed with results
✅ **Paper Draft**: Sketch v0.1 complete (all sections drafted)

### Critical Blocker
❌ **Theory-Experiment Mismatch** (BLOCKER-001)
- **Expected**: Winner-take-all dynamics, coverage ≈ 0.33 (single view)
- **Observed**: All views learned equally, coverage = 1.0
- **Root Cause**: Orthogonal slot structure may eliminate competition
- **Impact**: Core Theorem 2 not validated; paper cannot proceed without resolution

**Investigated**:
- Competing views (shared dimensions) - No effect
- Bottleneck architectures - No effect
- Asymmetric initialization - Advantage erodes during training

**Awaiting**: Advisor decision on next steps (see log/historical/)

### Not Started
- Formal proofs for any theorems
- Real data experiments (CIFAR/ImageNet)
- Remaining P2 and P3 synthetic experiments
- Paper revisions beyond v0.1 sketch

---

## Key Documents

### Planning & Specs
- [Research Plan](research_plan.md) - Strategic positioning and paper scope
- [Theorem 1 Setup](../specs/theorem_1.md) - View-pathway correspondence
- [Theorem 2 Setup](../specs/theorem_2.md) - Race dynamics
- [Theorem 3 Setup](../specs/theorem_3.md) - KD mechanism
- [Validation Plan](../specs/Synthetic%20Experiments%20-%20Validation%20Plan.md)

### Background Materials
- [Multi-View Theory](background/01_multi_view_theory/) - Allen-Zhu & Li 2023
- [Neural Race Reduction](background/02_neural_race_reduction/) - Saxe et al.
- [Connecting Theories](background/03_connecting_theories/) - Novel contributions

### Progress Tracking
- [Experiment Log](../log/experiment_log.md) - Live experiment results
- [Project Status](../log/project_status.md) - Historical comprehensive status

---

## Acceptance Criteria

### For Paper Submission
- [ ] All 3 theorems proven (at least simplified case)
- [ ] Synthetic experiments validate theory predictions
- [ ] Theory-experiment mismatch resolved
- [ ] Real data experiments show generalization
- [ ] Complete paper draft with all sections
- [ ] Paper follows NeurIPS format and page limits
- [ ] Results are reproducible with provided code

### For Success
- [ ] Novel contribution clearly articulated
- [ ] Mechanistic explanation stronger than existing work
- [ ] Predictions validated experimentally
- [ ] Practical implications for KD and training
- [ ] Writing is clear and accessible

---

## Dependencies

### Technical
- PyTorch >= 2.0
- Standard ML stack (numpy, scipy, matplotlib)
- LaTeX for paper compilation

### Knowledge
- Understanding of Neural Race Reduction (Saxe et al. 2022-2025)
- Understanding of Multi-View Theory (Allen-Zhu & Li 2023)
- Knowledge distillation background

### Blockers
See [agents/context/blockers.md](../../agents/context/blockers.md) for current blockers.

---

## Risk Assessment

### High Risk
- **Theory-experiment mismatch**: Currently blocking progress. Requires resolution before proceeding.
- **Proof complexity**: Theorems may be hard to prove rigorously. Mitigation: Start with simplified cases.

### Medium Risk
- **Timeline pressure**: May deadline is tight. Mitigation: Focus on core contributions, defer extensions.
- **Real data validation**: May not generalize from synthetic. Mitigation: Start early, iterate quickly.

### Low Risk
- **Writing**: Sketch v0.1 already complete, good foundation.
- **Implementation**: Infrastructure complete and working.

---

## Next Steps

1. **Resolve BLOCKER-001**: Review advisor's proposed approaches in log/historical/
2. **Decide path forward**: Theory revision vs experimental setup changes
3. **Continue validation**: Once blocker resolved, complete P1 experiments
4. **Begin proofs**: Start with Theorem 1 for M=2, L=2 case
5. **Real data prep**: Set up CIFAR experiments while synthetic work continues

---

**Last Updated**: 2026-01-13 (after import to ai-lab)
