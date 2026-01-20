# Option A Execution Plan: Empirical Study Reframe

**Decision**: Pursue empirical study framing for NeurIPS 2026
**Rationale**: Lower risk, establishes foundation for future theory paper (ICML 2027)
**Target Deadline**: NeurIPS 2026 (~May 22, 2026)
**Created**: 2026-01-20

---

## Revised Paper Vision

### New Title
**"When Does Knowledge Distillation Help? Empirical Insights from Multi-View Learning"**

Alternative options:
- "The Transfer Mechanism of Knowledge Distillation: An Empirical Study"
- "Soft Labels as Gas and Brakes: Understanding Knowledge Distillation Dynamics"

### New Contribution Framing

**Instead of**: "A mechanistic theory unifying neural race dynamics with multi-view theory"

**Now**: "Empirical investigation revealing when and why KD helps, with practical guidelines"

### Core Claims (Revised)

1. **KD is a transfer mechanism** — students inherit teacher representations, including biases
2. **Hierarchical structure enables acceleration** — soft labels encode inter-class similarity (5.3x speedup)
3. **Balance requires explicit diversity + weighting** — with fundamental speed trade-off
4. **Natural teacher diversity is not guaranteed** — contradicting common assumptions

### What This Sets Up for Future Theory Paper

The empirical findings motivate theoretical questions:
- Why don't teachers diversify naturally in our setting?
- What architectural conditions enable race dynamics?
- Can we formalize the information content of soft labels?

These become the foundation for ICML 2027 theory submission.

---

## Execution Timeline

| Week | Dates | Focus | Deliverables |
|------|-------|-------|--------------|
| 1 | Jan 20-26 | CIFAR-10 setup | Experiment code, initial runs |
| 2 | Jan 27-Feb 2 | CIFAR-10 completion | Results, figures |
| 3 | Feb 3-9 | Statistical rigor | CI for all tables, significance tests |
| 4 | Feb 10-16 | Allen-Zhu investigation | Ablation results, explanation |
| 5 | Feb 17-23 | Paper reframe | New intro, revised framing |
| 6 | Feb 24-Mar 2 | Related work + writing | Expanded citations, polished prose |
| 7 | Mar 3-9 | Draft v0.3 | Complete revised draft |
| 8 | Mar 10-16 | External feedback | Send to 2-3 reviewers |
| 9-10 | Mar 17-30 | Revisions | Address feedback |
| 11-12 | Mar 31-Apr 13 | Final polish | Camera-ready quality |
| Buffer | Apr 14-May 22 | Submission prep | Final checks, submit |

---

## Week 1-2: CIFAR-10 Experiments

### Experiment Design

**Goal**: Validate Finding 2 (hierarchical acceleration) on real data

**Setup**:
```
Dataset: CIFAR-100 → CIFAR-10 (natural hierarchy)
- CIFAR-100: 100 fine classes grouped into 20 superclasses
- Use 20-class grouping as "coarse" labels
- Use 100-class as "fine" labels

Teacher:
- Architecture: ResNet-18 (standard)
- Training: CIFAR-100 (100 classes) with cross-entropy
- Ensemble: 5 teachers with different seeds

Student:
- Architecture: ResNet-18 (same as teacher, for controlled comparison)
- Task: CIFAR-100 superclasses (20 classes)
- Training methods: Hard labels vs KD

Metrics:
- Epochs to 90% accuracy (learning speed)
- Final accuracy (task performance)
- Representation effective rank (balance)
- Soft label entropy analysis
```

**Expected Outcomes**:
- KD should accelerate learning (target: 2-5x speedup)
- Soft labels should encode superclass similarity
- Validates synthetic finding on real data

**Files to Create**:
- `src/experiments/cifar_hierarchical.py` - Main experiment
- `src/data_cifar.py` - CIFAR data loading with hierarchical labels
- `results/cifar/` - Output directory

### Ablations

1. **Temperature sweep**: τ = 1, 2, 4, 8, 16
2. **Alpha sweep**: α = 0.0, 0.3, 0.5, 0.7, 1.0
3. **Teacher ensemble size**: n = 1, 3, 5, 10
4. **Student architecture**: ResNet-18, ResNet-34, MobileNet

---

## Week 3: Statistical Rigor

### Tables to Update

**Table 1** (KD transfers WTA):
```
| Training | Effective Rank | View 0 | Accuracy |
|----------|----------------|--------|----------|
Current: Single values
Update: Mean ± std (n=10 seeds), p-value for comparison
```

**Table 3** (Uniform weighting fails):
- Add confidence intervals
- Add paired t-test vs hard labels

**Table 4** (Inverse signal weighting):
- Add confidence intervals
- Add effect size (Cohen's d)

### Statistical Tests to Add

1. **Paired t-test**: Hard labels vs KD (same seed pairs)
2. **ANOVA**: Across weighting schemes
3. **Bootstrap CI**: For all main metrics
4. **Effect sizes**: Cohen's d for all comparisons

### Code Changes

Create `src/utils/statistics.py`:
```python
def compute_ci(values, confidence=0.95):
    """Bootstrap confidence interval."""

def paired_ttest(a, b):
    """Paired t-test with effect size."""

def format_result(mean, std, p=None):
    """Format as 'mean ± std (p=X)'"""
```

---

## Week 4: Allen-Zhu Discrepancy Investigation

### The Mystery

Allen-Zhu & Li (2023) claim:
> "Different random seeds lead to different views being learned, creating ensemble diversity"

Our experiments show:
> All 5 teachers converge to identical representations (std = 0.001)

### Hypotheses to Test

| Hypothesis | Test | Expected Outcome |
|------------|------|------------------|
| Task difficulty | Harder task (overlapping views) | More diversity |
| Network depth | Deeper networks (4, 8 layers) | More diversity |
| Nonlinearity | ReLU vs linear | Nonlinear = diversity |
| Signal strength | Equal signals (no asymmetry) | More diversity |
| Initialization | Larger init scale | More diversity |

### Experiment Design

```python
# Ablation grid
configs = {
    'depth': [2, 4, 8],
    'nonlinearity': ['linear', 'relu', 'gelu'],
    'task_difficulty': ['easy', 'medium', 'hard'],
    'signal_asymmetry': [0.0, 0.25, 0.5],  # 0 = equal, 0.5 = our default
    'init_scale': [0.01, 0.1, 1.0],
}

# For each config, train 10 teachers
# Measure: std of view contributions across teachers
# Report: which conditions produce diversity
```

### Potential Findings

1. **If depth/nonlinearity matters**: Our linear setting is special case
2. **If task difficulty matters**: Easy tasks have unique solutions
3. **If signal asymmetry matters**: Symmetric signals → multiple optima

Any of these explains the discrepancy and is worth reporting.

---

## Week 5: Paper Reframe

### New Introduction Structure

```
Paragraph 1: KD is important, widely used
Paragraph 2: Prior work explains what, not when/why
Paragraph 3: Gap - assumptions that may not hold (diversity, etc.)
Paragraph 4: Our approach - controlled experiments to understand
Paragraph 5: Our findings (preview)
Paragraph 6: Contributions (empirical insights, practical guidelines)
```

### Section Restructuring

**Current**:
1. Introduction
2. Background
3. Experimental Setup
4. Finding 1: KD Transfers
5. Finding 2: Hierarchical Acceleration
6. Finding 3: Balance-Speed Trade-off
7. Practical Implications
8. Related Work
9. Limitations
10. Conclusion

**Revised**:
1. Introduction (reframed for empirical contribution)
2. Background (trimmed, less theoretical setup)
3. Experimental Framework (setup + metrics)
4. When Does KD Help? (Finding 2 FIRST - strongest result)
5. What Does KD Transfer? (Finding 1 - the surprise)
6. How to Achieve Balance (Finding 3 - practical)
7. Why Don't Teachers Diversify? (Allen-Zhu investigation - NEW)
8. Practical Guidelines (expanded)
9. Related Work (expanded)
10. Discussion & Future Work (sets up theory paper)

### Key Reframing Changes

| Section | Change |
|---------|--------|
| Title | Remove "mechanistic", add "empirical" |
| Abstract | Lead with Finding 2, 150 words max |
| Intro | Frame as "investigating assumptions" not "unifying theories" |
| Background | Shorten, remove unsupported theoretical claims |
| Findings | Reorder: 2, 1, 3 (strongest first) |
| New section | Allen-Zhu investigation |
| Implications | Expand with clear recipes |
| Limitations | Reframe as "scope" not "weaknesses" |

---

## Week 6: Related Work Expansion

### Papers to Add

**KD Theory**:
- Menon et al. 2021 - "Statistical perspective on distillation"
- Dao et al. 2021 - "Knowledge distillation as semiparametric inference"
- Mobahi et al. 2020 - "Self-distillation amplifies regularization"
- Stanton et al. 2021 - "Does knowledge distillation really work?"
- Phuong & Lampert 2019 - "Distillation-based training for multi-exit"

**Multi-View Learning**:
- Xu et al. 2013 - "Multi-view learning overview"
- Zhao et al. 2017 - "Multi-view learning with incomplete views"

**Teacher Diversity**:
- Fort et al. 2019 - "Deep ensembles: A loss landscape perspective"
- Wilson & Izmailov 2020 - "Bayesian deep learning"

**Learning Dynamics**:
- Arora et al. 2019 - "Implicit regularization in deep learning"
- Gunasekar et al. 2017 - "Implicit bias of gradient descent"

### Related Work Structure

1. **Knowledge Distillation** (expanded from current)
2. **Multi-View Learning** (brief, context)
3. **Teacher Ensemble Diversity** (NEW - connects to our findings)
4. **Learning Dynamics** (brief, sets up speed-balance)

---

## Week 7: Draft v0.3

### Deliverables

1. Complete revised draft with all changes
2. All figures updated with error bars
3. All tables updated with statistics
4. CIFAR results integrated
5. Allen-Zhu investigation results
6. Expanded related work
7. New abstract (150 words)

### Quality Checklist

- [ ] No unsupported theoretical claims
- [ ] All results have confidence intervals
- [ ] All comparisons have significance tests
- [ ] Figures are publication quality
- [ ] Writing is clear and concise
- [ ] Contribution is clearly stated
- [ ] Limitations are honest but not self-defeating

---

## Success Criteria for NeurIPS

### Minimum Bar (Accept)

- [ ] One real-data experiment (CIFAR) validates main finding
- [ ] All synthetic results have proper statistics
- [ ] Contribution clearly framed as empirical
- [ ] Writing is clean and clear
- [ ] Practical guidelines are actionable

### Target (Strong Accept)

- [ ] CIFAR results show compelling speedup (3-5x)
- [ ] Allen-Zhu discrepancy is explained
- [ ] Multiple ablations strengthen claims
- [ ] Paper tells coherent story
- [ ] Sets up future theoretical questions

### Stretch (Oral/Spotlight)

- [ ] Additional real dataset (ImageNet, CIFAR-100 to 10)
- [ ] Novel insight about when diversity emerges
- [ ] Clear theoretical predictions for future work
- [ ] Reproducible code release

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CIFAR results don't replicate | Medium | Run pilot first, have backup metrics |
| Allen-Zhu investigation inconclusive | Medium | Report negative results honestly |
| Timeline slips | Medium | Buffer weeks built in |
| Reviewers want theory | High | Explicitly frame as empirical, cite future work |
| Real data shows different story | Low | Would still be interesting finding |

---

## Next Steps (Immediate)

1. **Today**: Set up CIFAR experiment infrastructure
2. **This week**: Run CIFAR pilot to validate setup
3. **Next week**: Full CIFAR experiments with ablations
4. **Ongoing**: Update draft as results come in

---

*Plan created: 2026-01-20*
*Status: Ready to execute*
