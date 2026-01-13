# Running Experiments Guide

> **Purpose**: Step-by-step instructions for running the validation suite
> **Audience**: Researchers reproducing or extending the experiments

---

## Prerequisites

### Software Requirements

```bash
# Python 3.8+
python --version

# PyTorch
pip install torch torchvision

# Visualization
pip install matplotlib numpy scipy

# (Optional) GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Hardware Requirements

| Tier | Hardware | Est. Total Time |
|------|----------|-----------------|
| Minimum | CPU only | ~8-10 hours |
| Recommended | 1 GPU (RTX 3080) | ~2 hours |
| Optimal | Multi-GPU | ~30 min (parallel) |

### Directory Setup

```bash
cd /kdmech/engineer

# Verify source files exist
ls src/
# Should show: data.py, model.py, train.py, metrics.py

# Verify experiments exist
ls experiments/
# Should show: exp_2_1_single_view.py, exp_2_3_race_dynamics.py, ...

# Create results directory
mkdir -p results/figures
```

---

## Quick Start

### Validate Setup (Quick Test)

```bash
cd /kdmech/engineer
python experiments/run_all.py --quick
```

This runs all P1 experiments with reduced parameters (~10 min):
- 3 seeds instead of 30
- 50 epochs instead of 100
- 3 teachers instead of 5

### Full P1 Suite

```bash
python experiments/run_all.py --priority P1
```

Runs all P1 experiments with full parameters (~2 hours on GPU).

---

## Running Individual Experiments

### Experiment 2.1: Single-View Convergence

**Purpose**: Validate $C(f) \approx 1/M$

```bash
# Full run (recommended)
python experiments/exp_2_1_single_view.py --seeds 30 --epochs 100

# Quick test
python experiments/exp_2_1_single_view.py --quick

# Custom parameters
python experiments/exp_2_1_single_view.py \
    --seeds 50 \
    --epochs 150 \
    --output results/exp_2_1_extended/
```

**Output**: `results/results_exp_2_1.json`

**Time**: ~30 min (full), ~2 min (quick)

---

### Experiment 2.3: Race Dynamics

**Purpose**: Visualize winner-take-all dynamics

```bash
# Single seed visualization
python experiments/exp_2_3_race_dynamics.py --seed 0 --class 0 --epochs 200

# Multiple seeds comparison
python experiments/exp_2_3_race_dynamics.py --multi 5 --class 0

# Quick test
python experiments/exp_2_3_race_dynamics.py --quick
```

**Output**:
- `results/results_exp_2_3.json`
- `results/figures/race_dynamics_hard.png`
- `results/figures/race_dynamics_multi_seed.png` (if --multi)

**Time**: ~5 min (single), ~25 min (multi)

---

### Experiment 3.1: KD Coverage Transfer

**Purpose**: Compare hard labels vs KD

```bash
# Full run
python experiments/exp_3_1_kd_coverage.py \
    --teachers 5 \
    --trials 10 \
    --epochs 100 \
    --temperature 4.0

# Quick test
python experiments/exp_3_1_kd_coverage.py --quick

# Temperature sweep
for tau in 1 2 4 8 16; do
    python experiments/exp_3_1_kd_coverage.py \
        --temperature $tau \
        --output results/temp_$tau/
done
```

**Output**:
- `results/results_exp_3_1.json`
- `results/figures/kd_coverage_comparison.png`

**Time**: ~1 hour (full), ~10 min (quick)

---

### Experiment 3.3: Pathway Evolution

**Purpose**: Compare pathway dynamics under hard vs KD

```bash
# Single class
python experiments/exp_3_3_pathway_evolution.py \
    --seed 0 \
    --class 0 \
    --teachers 5 \
    --epochs 200

# Multiple classes
python experiments/exp_3_3_pathway_evolution.py --multi-class 3

# Quick test
python experiments/exp_3_3_pathway_evolution.py --quick
```

**Output**:
- `results/results_exp_3_3.json`
- `results/figures/pathway_comparison.png`

**Time**: ~15 min (single), ~45 min (multi-class)

---

## Using the Master Runner

### Run by Priority

```bash
# P1 experiments only (core validation)
python experiments/run_all.py --priority P1

# P2 experiments (supporting evidence)
python experiments/run_all.py --priority P2  # Not yet implemented

# All experiments
python experiments/run_all.py --priority all  # Not yet implemented
```

### Output Options

```bash
# Custom output directory
python experiments/run_all.py --output /path/to/results/

# Quick mode for testing
python experiments/run_all.py --quick
```

### Generated Files

```
results/
├── results_exp_2_1.json      # Exp 2.1 data
├── results_exp_2_3.json      # Exp 2.3 data
├── results_exp_3_1.json      # Exp 3.1 data
├── results_exp_3_3.json      # Exp 3.3 data
├── experiment_summary.json   # Machine-readable summary
├── experiment_summary.md     # Human-readable report
└── figures/
    ├── race_dynamics_hard.png
    ├── kd_coverage_comparison.png
    └── pathway_comparison.png
```

---

## GPU Configuration

### Single GPU

```bash
# Uses GPU by default if available
python experiments/run_all.py

# Force CPU
CUDA_VISIBLE_DEVICES="" python experiments/run_all.py
```

### Multi-GPU (Manual Parallelism)

Run different experiments on different GPUs:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python experiments/exp_2_1_single_view.py

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python experiments/exp_3_1_kd_coverage.py
```

### Memory Requirements

| Experiment | GPU Memory |
|------------|------------|
| 2.1 | ~2 GB |
| 2.3 | ~2 GB |
| 3.1 | ~4 GB (5 teachers) |
| 3.3 | ~4 GB (5 teachers) |

---

## Interpreting Output

### Console Output

Each experiment prints progress:

```
============================================================
Experiment 2.1: Single-View Convergence
Config: K=10, M=3, d_view=50, epochs=100
Expected coverage: 0.333
============================================================
Seed  0: coverage = 0.367, accuracy = 0.983
Seed  1: coverage = 0.300, accuracy = 0.978
...

============================================================
Results Summary
============================================================
Mean coverage: 0.333 ± 0.045
Expected (1/M): 0.333
Difference: 0.000
✓ PASSED: Coverage matches 1/M prediction
```

### JSON Results

Machine-readable results for analysis:

```json
{
  "experiment": "exp_2_1_single_view_convergence",
  "metrics": {
    "mean_coverage": 0.333,
    "passed": true
  }
}
```

### Markdown Summary

Human-readable report at `experiment_summary.md`:

```markdown
# Experiment Results Summary

| Experiment | Status | Key Result |
|------------|--------|------------|
| exp_2_1 | ✓ Pass | Coverage: 0.333 |
| exp_3_1 | ✓ Pass | KD: 0.890, Hard: 0.333 |
```

---

## Reproducing Paper Results

### Standard Configuration

For paper-quality results, use these parameters:

```bash
# Experiment 2.1
python experiments/exp_2_1_single_view.py --seeds 30 --epochs 100

# Experiment 2.3
python experiments/exp_2_3_race_dynamics.py --epochs 200 --multi 5

# Experiment 3.1
python experiments/exp_3_1_kd_coverage.py \
    --teachers 5 --trials 10 --temperature 4.0

# Experiment 3.3
python experiments/exp_3_3_pathway_evolution.py \
    --teachers 5 --epochs 200 --multi-class 3
```

### Seed Reproducibility

All experiments use deterministic seeding:

```python
# Dataset seed
dataset = MultiViewDataset(..., seed=trial_seed)

# Model seed (offset to avoid correlation)
torch.manual_seed(trial_seed + 5000)
```

To reproduce exact results, use the same seeds as reported.

---

## Troubleshooting

### Common Issues

**"No module named 'src'"**
```bash
cd /kdmech/engineer  # Run from engineer directory
```

**"CUDA out of memory"**
```bash
# Reduce batch size
python experiments/exp_3_1_kd_coverage.py  # Edit batch_size in code

# Or use CPU
CUDA_VISIBLE_DEVICES="" python experiments/exp_3_1_kd_coverage.py
```

**"File not found: results/"**
```bash
mkdir -p results/figures
```

**Experiments not passing**
- Check `expected_results.md` for expected values
- Verify dataset orthogonality
- Try increasing epochs

### Validation Checks

```python
# In Python, verify dataset
from src.data import MultiViewDataset, verify_dataset

dataset = MultiViewDataset(K=10, M=3, d_view=50)
verify_dataset(dataset)  # Should show max_dot < 0.01
```

---

## Extending Experiments

### Adding New Parameters

Experiments use argparse. Add new arguments:

```python
parser.add_argument('--new-param', type=float, default=1.0)
```

### Custom Metrics

Add to `src/metrics.py`:

```python
def my_new_metric(model, dataset):
    # Implementation
    return value
```

### New Experiments

Copy an existing experiment as template:

```bash
cp experiments/exp_2_1_single_view.py experiments/exp_new.py
# Modify as needed
```

---

## Batch Processing

### Run All and Save

```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
OUTPUT_DIR="results_${DATE}"

python experiments/run_all.py --priority P1 --output $OUTPUT_DIR

echo "Results saved to $OUTPUT_DIR"
```

### Parallel Execution

```bash
#!/bin/bash
# Run experiments in parallel on different GPUs

CUDA_VISIBLE_DEVICES=0 python experiments/exp_2_1_single_view.py &
CUDA_VISIBLE_DEVICES=1 python experiments/exp_3_1_kd_coverage.py &
wait

echo "All experiments complete"
```

---

## Output to Advisor

Results are automatically copied to the exchange directory:

```
engineer/exchange/to_advisor/YYYY-MM-DD_results_p1.md
```

This report is formatted for the advisor role to review.
