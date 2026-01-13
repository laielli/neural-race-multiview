# KDMech Project Status Tracker

**Paper**: "Unifying Neural Race Reduction with Multi-View Theory: A Mechanistic Account of Knowledge Distillation"

**Target**: NeurIPS 2026 (Abstract: ~May 15, Full paper: ~May 22)

**Last Updated**: 2026-01-10 (evening)

---

## Quick Status

| Component | Status | Owner | Blocking? |
|-----------|--------|-------|-----------|
| Theory (Theorems 1-3) | Designed, not proven | — | No |
| Synthetic Experiments | Designed, not implemented | Engineer | No |
| Real Data Experiments | Planned | Engineer | Yes (needs synthetic first) |
| Paper Draft | **Sketch v0.1 complete** | Writer | No |
| Proofs | Not started | — | Yes (needs theory validation) |

**Overall Phase**: Early Development (Paper Sketch Complete, Awaiting Experiments)

---

## Timeline

```
2026
Jan         Feb         Mar         Apr         May         Jun
|-----------|-----------|-----------|-----------|-----------|
[== Phase 1: Theory ===>]
            [=== Phase 2: Synthetic Experiments ===>]
                        [=== Phase 3: Real Data ====]
                                    [=== Phase 4: Writing ===]
                                                |
                                            DEADLINE
                                            May 22
```

### Phase Breakdown

| Phase | Dates | Goals | Status |
|-------|-------|-------|--------|
| **1. Theory** | Jan 1 - Feb 28 | Prove Theorems 1-3 (simplified case) | Not started |
| **2. Synthetic** | Mar 1 - Apr 15 | Validate theory with controlled experiments | Not started |
| **3. Real Data** | Apr 1 - May 10 | CIFAR/ImageNet validation | Not started |
| **4. Writing** | Apr 15 - May 22 | Complete paper draft | Not started |

---

## Workstream Status

### Theory Development

| Theorem | Formal Setup | Proof (M=2, L=2) | Proof (General) | Status |
|---------|--------------|------------------|-----------------|--------|
| **Theorem 1**: View-Pathway Correspondence | Done | Not started | Not started | Designed |
| **Theorem 2**: Race Dynamics | Done | Not started | Not started | Designed |
| **Theorem 3**: KD Circumvents Race | Done | Not started | Not started | Designed |

**Documents**:
- `initial_plan/Theorem 1 - Formal Mathematical Setup.md`
- `initial_plan/Theorem 2 - Race Dynamics Setup.md`
- `initial_plan/Theorem 3 - Why KD Circumvents the Race.md`

**Next Actions**:
- [ ] Read Saxe et al. 2014 deeply (exact dynamics, proof techniques)
- [ ] Read Allen-Zhu & Li 2023 appendix (multi-view formalization)
- [ ] Attempt Theorem 1 proof for M=2, L=2 case

---

### Experiments (Engineer)

| Experiment | Priority | Designed | Implemented | Results |
|------------|----------|----------|-------------|---------|
| **Infrastructure** | P0 | Done | Not started | — |
| Data generator | P0 | Done | Not started | — |
| Network + metrics | P0 | Done | Not started | — |
| Training loops | P0 | Done | Not started | — |
| **Theorem 2 Validation** | P1 | Done | Not started | — |
| 2.1: Single-view convergence | P1 | Done | Not started | — |
| 2.3: Race visualization | P1 | Done | Not started | — |
| 2.2: View diversity | P2 | Done | Not started | — |
| 2.4: Winner prediction | P2 | Done | Not started | — |
| **Theorem 3 Validation** | P1 | Done | Not started | — |
| 3.1: KD coverage transfer | P1 | Done | Not started | — |
| 3.3: Pathway evolution (KD) | P1 | Done | Not started | — |
| 3.2: Gradient distribution | P2 | Done | Not started | — |
| **Theorem 1 Validation** | P3 | Done | Not started | — |
| 1.1: Output decomposition | P3 | Done | Not started | — |

**Documents**:
- `initial_plan/Synthetic Experiments - Validation Plan.md`
- `exchange/to_engineer/initial_experiments_instructions.md`

**Next Actions**:
- [ ] Implement MultiViewDataset class
- [ ] Implement MultiViewNet with pathway measurement
- [ ] Run Experiment 2.1 (single-view convergence)
- [ ] Run Experiment 2.3 (race visualization)

---

### Paper Writing (Writer)

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | **Draft v0.1** | Gap → approach → results → implications |
| 1. Introduction | **Draft v0.1** | KD puzzle, gap, 3 contributions, roadmap |
| 2. Background | **Draft v0.1** | Multi-view (2.1) + Neural Race (2.2) |
| 3. Theorem 1 | **Draft v0.1** | Definitions, 5 assumptions, 3-part theorem |
| 4. Theorem 2 | **Draft v0.1** | 5-part theorem (A-E), implications |
| 5. Theorem 3 | **Draft v0.1** | 4-part theorem, special cases, comparison table |
| 6. Experiments | **Draft v0.1** | 5 experiments with placeholder tables/figures |
| 7. Discussion | **Draft v0.1** | Limitations, future work, conclusion |
| Appendix | **Draft v0.1** | Placeholder sections for proofs A-C |

**Documents**:
- `initial_plan/Paper Development Plan - Neural Race × Multi-View Theory.md`
- `initial_plan/Paper Title Analysis.md`
- `exchange/to_writer/paper_sketch_instructions.md`

**Completed**: Paper sketch v0.1 (`paper/paper_sketch_v0.1.tex`, 612 lines)

**Draft Highlights**:
- All 3 theorems formally stated with multi-part structure
- Initial advantage formula prominently featured
- Comparison table (hard vs soft labels)
- 5 synthetic experiments with predictions
- Proper placeholder conventions throughout

**Next Actions**:
- [x] Read all theorem setup documents
- [x] Draft paper skeleton with section structure
- [x] Write Introduction (gap + contributions)
- [x] Create placeholder theorem boxes
- [ ] Integrate experimental results when available
- [ ] Polish prose for v0.2
- [ ] Add real data experiments section (if results available)

---

## Dependencies

```
                    ┌─────────────────┐
                    │  Theory Setup   │ ← DONE
                    │  (Formal defs)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  Proofs    │  │ Synthetic  │  │  Paper     │
     │ (Theorems) │  │ Experiments│  │  Sketch    │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
           │               ▼               │
           │      ┌────────────────┐       │
           │      │ Theory         │       │
           └─────►│ Validation     │◄──────┘
                  │ (Iterate)      │
                  └───────┬────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │ Real Data      │
                  │ Experiments    │
                  └───────┬────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │ Final Paper    │
                  └────────────────┘
```

**Critical Path**:
1. Synthetic experiments validate theory direction
2. If experiments fail → revise theory → re-run
3. If experiments pass → proceed to proofs + real data
4. Paper writing can proceed in parallel with placeholders

---

## Risk Register

| Risk | Probability | Impact | Mitigation | Status |
|------|-------------|--------|------------|--------|
| Proofs too hard for general case | Medium | High | Prove simplified case; validate empirically | Monitoring |
| GDLN approximation too coarse | Medium | Medium | Bound approximation error; validate on real nets | Monitoring |
| Experiments contradict theory | Low | High | Early synthetic validation; revise theory | Active (running exps) |
| Scooped on Neural Race × KD | Low | High | Move quickly; unique framing | Monitoring |
| Time runs short | Medium | Medium | Prioritize MVP (Thms 1-2 + synthetic) | Active |

---

## Communication Log

| Date | From | To | Summary |
|------|------|-----|---------|
| 2026-01-10 | Advisor | Writer | Paper sketch instructions sent |
| 2026-01-10 | Advisor | Engineer | Initial experiments instructions sent |
| 2026-01-10 | Writer | Advisor | Paper sketch v0.1 complete (612 lines, all sections drafted) |

---

## Decisions Made

| Date | Decision | Rationale | Document |
|------|----------|-----------|----------|
| 2026-01-10 | Title: "Unifying Neural Race Reduction..." | Best balance of accuracy, discoverability, attribution | Paper Title Analysis.md |
| 2026-01-10 | Prioritize Theorem 2 experiments first | Race dynamics are core claim; need early validation | — |
| 2026-01-10 | Use synthetic data with slot structure | Guarantees orthogonality; ground truth known | Synthetic Experiments.md |

---

## Open Questions

1. **Proof feasibility**: Can we prove Theorem 1 for M=2, L=2 rigorously?
2. **GDLN approximation**: How tight is the GDLN ↔ ReLU correspondence?
3. **Real data views**: How do we identify "views" in CIFAR/ImageNet?
4. **Reviewer positioning**: How to frame as "foundational" vs "incremental"?

---

## Immediate Next Steps (This Week)

### Advisor
- [x] Create project status tracker
- [x] Send instructions to Writer
- [x] Send instructions to Engineer
- [x] Review paper sketch v0.1
- [ ] Review experiment results when available

### Writer
- [x] Read theorem setup documents
- [x] Draft paper sketch v0.1
- [ ] Await experiment results from Engineer
- [ ] Prepare for v0.2 revisions

### Engineer
- [ ] Implement data generator
- [ ] Implement network + metrics
- [ ] Run Experiment 2.1
- [ ] Run Experiment 2.3
- [ ] Send results to Writer for integration

---

## Milestones

| Milestone | Target Date | Criteria | Status |
|-----------|-------------|----------|--------|
| **M1**: Synthetic infrastructure | Jan 20 | Data + model + training working | Not started |
| **M2**: Theorem 2 validated | Jan 31 | Experiments 2.1, 2.3 pass | Not started |
| **M3**: Theorem 3 validated | Feb 15 | Experiments 3.1, 3.3 pass | Not started |
| **M4**: Paper sketch complete | Feb 1 | Full structure with placeholders | **COMPLETE** (Jan 10) |
| **M5**: Theorem 1 proof (M=2) | Feb 28 | Rigorous proof written | Not started |
| **M6**: Theorem 2 proof (M=2) | Mar 15 | Rigorous proof written | Not started |
| **M7**: Real data results | Apr 30 | CIFAR experiments complete | Not started |
| **M8**: Paper submitted | May 22 | NeurIPS submission | Not started |

---

## File Index

### Advisor (`/advisor/`)
| File | Purpose |
|------|---------|
| `project_status.md` | This document |
| `initial_plan/Paper Development Plan.md` | Master strategy |
| `initial_plan/Paper Title Analysis.md` | Title selection |
| `initial_plan/Theorem 1 - Formal Mathematical Setup.md` | Theorem 1 spec |
| `initial_plan/Theorem 2 - Race Dynamics Setup.md` | Theorem 2 spec |
| `initial_plan/Theorem 3 - Why KD Circumvents the Race.md` | Theorem 3 spec |
| `initial_plan/Synthetic Experiments - Validation Plan.md` | Full experiment design |
| `exchange/to_writer/paper_sketch_instructions.md` | Writer instructions |
| `exchange/to_engineer/initial_experiments_instructions.md` | Engineer instructions |

### Writer (`/writer/`)
| File | Purpose |
|------|---------|
| `CLAUDE.md` | Role definition, writing standards |
| `writing_status.md` | Section-by-section tracking |
| `paper/paper_sketch_v0.1.tex` | **Paper sketch v0.1** (612 lines) |
| `paper/paper_sketch_v0.1.pdf` | Compiled PDF |
| `exchange/to_advisor/paper_sketch_status.md` | Status report on v0.1 |

### Engineer (`/engineer/`)
| File | Purpose |
|------|---------|
| `CLAUDE.md` | Role definition, code standards |
| `experiment_status.md` | Experiment tracking |
| `requirements.txt` | Python dependencies |
| `src/` | Source code directory (implementation pending) |
| `experiments/` | Experiment scripts (implementation pending) |
| `results/` | Experiment outputs (pending) |
| `from_advisor/initial_experiments_instructions.md` | Implementation instructions |

---

*Last updated: 2026-01-10 (evening) by Advisor*
