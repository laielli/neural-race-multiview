# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this paper repository.

## Paper Information

**Title**: Unifying Neural Race Reduction with Multi-View Theory: A Mechanistic Account of Knowledge Distillation

**Target**: NeurIPS 2026 (Deadline: ~May 22, 2026)

**Status**: Early development, theory-experiment mismatch under investigation

**Repository**: https://github.com/laielli/neural-race-multiview (private)

## Repository Structure

```
neural-race-multiview/
├── README.md           # Paper overview and quick links
├── prd/                # Paper requirements and background
│   ├── paper_requirements.md
│   ├── research_plan.md
│   └── background/     # Theoretical foundations (multi-view theory, neural race)
├── specs/              # Technical specifications for theorems and experiments
├── log/                # Experiment logs, results, and findings
├── src/                # Experiment source code
│   ├── data.py, model.py, metrics.py, train.py
│   ├── experiments/    # Experiment scripts
│   └── requirements.txt
├── paper/              # LaTeX paper drafts
├── presentation/       # Slides and notes
└── datasets/           # Paper-specific datasets
```

## Git Workflow

### This Paper is a Submodule

This paper repository is linked as a git submodule in the main `ai-lab` repository. It has **independent version control** from the main repo.

### Making Changes to Paper Code/Content

```bash
# 1. Make your changes to paper files
# (edit code, experiments, paper draft, etc.)

# 2. Commit changes in THIS repo
git add .
git commit -m "Description of changes"
git push

# 3. IMPORTANT: Update main repo to track new version
cd /Users/michaellaielli/workspace/ai-lab  # or wherever ai-lab is located
git add papers/neural-race-multiview
git commit -m "Update neural-race-multiview submodule"
git push
```

### Why Two Commits?

Because this is a submodule:
1. **This repo** tracks your actual paper changes (code, data, paper text)
2. **Main repo** tracks which version (commit SHA) of this paper it's using

Both commits are needed to keep everything in sync.

### Pulling Latest Changes

```bash
# Pull latest changes from GitHub
git pull
```

If someone else updated the paper, pull from the main repo too:
```bash
cd /Users/michaellaielli/workspace/ai-lab
git pull
git submodule update --remote --merge
```

## Current Project Status

### Critical Blocker: BLOCKER-001

**Issue**: Theory-experiment mismatch
- **Expected**: Winner-take-all dynamics (coverage ≈ 0.33)
- **Observed**: All views learned (coverage = 1.0)
- **Status**: Under investigation (see `log/experiment_log.md`)

### Completed
- ✅ 3 theorems formally specified (`specs/theorem_*.md`)
- ✅ Experiment infrastructure (996 lines core code)
- ✅ 11/17 experiments completed (`src/experiments/`)
- ✅ Paper sketch v0.1 (`paper/paper_sketch_v0.1.tex`)

### Next Steps
1. Resolve theory-experiment mismatch (review `log/historical/DRAFT_next_steps_pending_review.md`)
2. Complete remaining P1 experiments
3. Begin theorem proofs (simplified case)

## Running Experiments

```bash
# Install dependencies
cd src
pip install -r requirements.txt

# Run individual experiment
python -m experiments.exp_2_1_single_view

# Run all experiments
python -m experiments.run_all

# Results saved to: log/results/
```

## Building the Paper

```bash
cd paper
pdflatex paper_sketch_v0.1.tex
# or
latexmk -pdf paper_sketch_v0.1.tex
```

## Key Files to Know

### Requirements & Planning
- `prd/paper_requirements.md` - Main PRD with hypothesis and success metrics
- `prd/research_plan.md` - Strategic positioning and paper scope
- `specs/theorem_*.md` - Formal theorem specifications

### Code & Experiments
- `src/data.py` - MultiViewDataset implementation
- `src/model.py` - MultiViewNet with pathway measurement
- `src/metrics.py` - Coverage and pathway strength calculations
- `src/experiments/` - All experiment scripts

### Results & Findings
- `log/experiment_log.md` - Live experiment tracking with CRITICAL FINDING
- `log/results/` - Experiment outputs (JSON + figures)
- `log/experiment_findings/` - Detailed analysis reports

### Paper Draft
- `paper/paper_sketch_v0.1.tex` - Current paper draft
- `paper/paper_sketch_v0.1.pdf` - Compiled PDF

## Important Notes

- **Always commit and push changes** in this repo before updating the main repo
- **Check blockers** in main repo: `../../agents/context/blockers.md`
- **Paper-specific .gitignore** excludes large results files (JSON, PNG)
- **Dependencies**: PyTorch-based, see `src/requirements.txt`

## Integration with Main Repo

This paper integrates with the ai-lab framework:
- **Roadmap**: `../../execution/roadmap.md` tracks paper milestones
- **Blockers**: `../../agents/context/blockers.md` lists BLOCKER-001
- **Tasks**: `../../agents/tasks/review/TASK-001-neural-race-multiview-resolve-theory-mismatch.md`

When working on this paper, check the main repo's context files for coordination with other work.
