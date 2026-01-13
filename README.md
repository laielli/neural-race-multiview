# Neural Race × Multi-View Theory

**Title**: "Unifying Neural Race Reduction with Multi-View Theory: A Mechanistic Account of Knowledge Distillation"

**Target Venue**: NeurIPS 2026 (Deadline: ~May 22, 2026)

**Status**: Early development - Theory-experiment mismatch under investigation

---

## Quick Links

- **PRD**: [Paper Requirements Document](prd/paper_requirements.md)
- **Research Plan**: [prd/research_plan.md](prd/research_plan.md)
- **Paper Draft**: [paper/paper_sketch_v0.1.pdf](paper/paper_sketch_v0.1.pdf)
- **Experiment Log**: [log/experiment_log.md](log/experiment_log.md)
- **Project Status**: [log/project_status.md](log/project_status.md)

---

## Core Thesis

The Neural Race Reduction provides the mechanistic foundation for Multi-View Theory in knowledge distillation, explaining:
1. **WHY** networks learn one view (winner-take-all race dynamics)
2. **WHICH** view will be learned (predictable from initialization + data structure)
3. **HOW** distillation circumvents this limitation (soft labels distribute gradients)

---

## Three Core Theorems

1. **View-Pathway Correspondence** ([specs/theorem_1.md](specs/theorem_1.md))
   - Each view in multi-view data maps to a distinct computational pathway
   - Learning a view = that pathway winning the neural race

2. **Race Dynamics** ([specs/theorem_2.md](specs/theorem_2.md))
   - Initial advantage determines which view wins
   - Predicts single-view convergence (C(f) ≈ 1/M)

3. **KD Circumvents Race** ([specs/theorem_3.md](specs/theorem_3.md))
   - Soft labels distribute gradients across pathways
   - Prevents race monopolization, enables multi-view learning

---

## Project Structure

```
neural-race-multiview/
├── prd/                    # Paper requirements and background
│   ├── paper_requirements.md
│   ├── research_plan.md
│   └── background/         # Theoretical foundations
├── specs/                  # Technical specifications
│   ├── theorem_*.md        # Formal theorem setups
│   └── experiments_detailed/
├── log/                    # Experiment logs and results
│   ├── experiment_log.md   # Live experiment tracking
│   ├── project_status.md   # Historical status
│   └── results/            # Experiment outputs
├── src/                    # Experiment code
│   ├── data.py             # MultiViewDataset
│   ├── model.py            # MultiViewNet
│   ├── metrics.py          # Coverage, pathway metrics
│   ├── train.py            # Training loops
│   └── experiments/        # Experiment scripts
├── paper/                  # Paper drafts (LaTeX)
│   └── paper_sketch_v0.1.tex
└── presentation/           # Slides and notes
```

---

## Current Status

**Phase**: Early Development

**Completed**:
- ✅ 3 theorems formally specified
- ✅ Experiment infrastructure (996 lines of core code)
- ✅ 11/17 experiments completed
- ✅ Paper sketch v0.1 (all sections drafted)

**Critical Issue**:
- ❌ **Theory-experiment mismatch**: Expected winner-take-all dynamics not observed
- Networks learn all views (coverage = 1.0) instead of single view (predicted 0.33)
- Multiple follow-up investigations conducted (see log/experiment_log.md)
- **Blocker ID**: BLOCKER-001 in agents/context/blockers.md

**Next Steps**:
- Review advisor's proposed solutions (log/historical/)
- Decide path forward (revise theory vs modify experiments)
- Continue validation once resolved

---

## Running Experiments

```bash
# Install dependencies
cd src && pip install -r requirements.txt

# Run individual experiment
python -m experiments.exp_2_1_single_view

# Run all experiments
python -m experiments.run_all

# Check results
ls ../log/results/
```

---

## Building Paper

```bash
cd paper
pdflatex paper_sketch_v0.1.tex
# or
latexmk -pdf paper_sketch_v0.1.tex
```

---

## Key Documents

### Theory
- [Theorem 1 Setup](specs/theorem_1.md) - View-pathway correspondence
- [Theorem 2 Setup](specs/theorem_2.md) - Race dynamics
- [Theorem 3 Setup](specs/theorem_3.md) - KD mechanism

### Experiments
- [Validation Plan](specs/Synthetic%20Experiments%20-%20Validation%20Plan.md)
- [Experiment Log](log/experiment_log.md) - Detailed results and findings

### Background
- [Multi-View Theory](prd/background/01_multi_view_theory/) - Allen-Zhu & Li 2023
- [Neural Race Reduction](prd/background/02_neural_race_reduction/) - Saxe et al.
- [Connecting Theories](prd/background/03_connecting_theories/) - Our novel contributions

---

## Contact

For questions about this paper, see [agents/context/priorities.md](../../agents/context/priorities.md) for current focus areas or [agents/context/blockers.md](../../agents/context/blockers.md) for known issues.
