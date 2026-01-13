# Experiment 3.3: Pathway Evolution Under KD

> **Validates**: Theorem 3 — Multiple pathways survive under KD (vs winner-take-all)
> **Priority**: P1 (Core validation)
> **Implementation**: `/kdmech/engineer/experiments/exp_3_3_pathway_evolution.py`

---

## Theoretical Background

### Hard Labels: Corner Equilibrium

Under hard labels, race dynamics lead to a **corner equilibrium**:
- One pathway dominates: $s_m \to s_{\max}$
- Others die: $s_{m'} \to 0$ for $m' \neq m$

### KD: Interior Equilibrium

Under KD, external signal creates an **interior equilibrium**:
- Multiple pathways survive: $s_m(∞) \propto \sqrt{\alpha_m}$
- All teacher-known views represented

### Visual Prediction

```
Hard Labels:                    KD from Ensemble:
│                               │
│     ╱──── View 1              │   ╱──── View 1
│    ╱                          │  ╱ ╱──── View 2
│   ╱                           │ ╱ ╱ ╱──── View 3
│  ╱  ╲____ View 2              │╱ ╱ ╱
│ ╱    ╲___ View 3              │ ╱ ╱
│╱                              │╱ ╱
└──────────────                 └──────────────
     epoch                           epoch
```

---

## Experimental Protocol

### Setup

```python
def experiment_3_3_pathway_evolution_comparison(
    seed: int = 0,
    y_target: int = 0,
    num_teachers: int = 5,
    K: int = 10,
    M: int = 3,
    epochs: int = 200,
    temperature: float = 4.0,
    log_interval: int = 5,
    ...
):
```

### Procedure

1. **Train teacher ensemble** (hard labels)
   - 5 teachers with different seeds
   - Provides diverse view coverage

2. **Train student with hard labels**
   - Track pathway strengths during training
   - Record every 5 epochs

3. **Train student with KD** (same initialization!)
   - Same architecture, same initial weights
   - KD loss from teacher ensemble
   - Track pathway strengths during training

4. **Compare final states**:
   - Count surviving pathways
   - Compute dominance ratios
   - Generate side-by-side figure

---

## Code Walkthrough

### Training Both Students

```python
# Train with hard labels (with pathway tracking)
torch.manual_seed(seed + 5000)
model_hard = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
model_hard.apply(init_weights)

history_hard = train_hard_labels(
    model_hard, dataset,
    epochs=epochs,
    track_pathways=True,
    track_classes=[y_target]
)

# Train with KD (SAME initialization!)
torch.manual_seed(seed + 5000)  # Same seed!
model_kd = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
model_kd.apply(init_weights)

history_kd = train_kd(
    model_kd, teachers, dataset,
    epochs=epochs,
    temperature=temperature,
    track_pathways=True,
    track_classes=[y_target]
)
```

### Counting Surviving Pathways

```python
def count_surviving_pathways(model, dataset, threshold_ratio=0.1):
    surviving = {}

    for y in range(dataset.K):
        strengths = []
        for m in range(dataset.M):
            phi_ym = dataset.get_view_feature(y, m)
            strengths.append(model.get_pathway_strength(phi_ym))

        max_s = max(strengths)
        threshold = threshold_ratio * max_s
        surviving[y] = sum(1 for s in strengths if s > threshold)

    return surviving
```

---

## Expected Results

### Primary Predictions

| Metric | Hard Labels | KD |
|--------|-------------|-----|
| Surviving pathways | 1 | 2-3 |
| Dominance ratio | > 0.7 | < 0.5 |
| Final strengths | One high, rest ~0 | Multiple non-zero |

### Pass Criterion

```python
if hard_ratio > 0.7 and kd_ratio < 0.6:
    print("✓ PASSED: Hard labels show winner-take-all, KD preserves multiple views")
```

### Dominance Ratio Definition

$$\text{Dominance ratio} = \frac{\max_m s_m}{\sum_m s_m}$$

- Hard labels: Ratio → 1 (one winner)
- KD: Ratio → 0.3-0.5 (shared among views)

---

## Output Format

### JSON Results

```json
{
  "experiment": "exp_3_3_pathway_evolution",
  "config": {
    "seed": 0,
    "y_target": 0,
    "num_teachers": 5,
    "epochs": 200,
    "temperature": 4.0
  },
  "metrics": {
    "hard_label": {
      "final_strengths": {"0": 0.12, "1": 0.08, "2": 4.85},
      "surviving_pathways": 1,
      "dominance_ratio": 0.96
    },
    "kd": {
      "final_strengths": {"0": 2.45, "1": 1.98, "2": 1.52},
      "surviving_pathways": 3,
      "dominance_ratio": 0.41
    },
    "passed": true
  },
  "history_hard": {
    "epochs_logged": [0, 5, 10, ...],
    "pathway_strengths": {...},
    "accuracy": [...]
  },
  "history_kd": {
    "epochs_logged": [0, 5, 10, ...],
    "pathway_strengths": {...},
    "accuracy": [...]
  },
  "figure_path": "results/figures/pathway_comparison.png"
}
```

### Console Output

```
============================================================
Experiment 3.3: Pathway Evolution Comparison
Config: seed=0, target_class=0
K=10, M=3, 5 teachers
============================================================

Training teacher ensemble...
Training student with hard labels...
Epoch 50: accuracy=0.85, loss=0.42
Epoch 100: accuracy=0.95, loss=0.18
...

Training student with KD...
Epoch 50: accuracy=0.82, loss=1.15
Epoch 100: accuracy=0.94, loss=0.45
...

Surviving pathways (class 0):
  Hard labels: 1
  KD: 3

Final pathway strengths for class 0:
Hard labels: ['0.120', '0.080', '4.850']
KD:          ['2.450', '1.980', '1.520']

Dominance ratio (winner/total):
  Hard labels: 0.960
  KD: 0.410
✓ PASSED: Hard labels show winner-take-all, KD preserves multiple views
```

---

## Generated Figure

Side-by-side comparison:

```
┌──────────────────────────┬──────────────────────────┐
│  Hard Labels             │  Knowledge Distillation  │
│  (Winner-Take-All)       │  (Multiple Views)        │
│                          │                          │
│         ╱────            │     ╱────                │
│        ╱                 │    ╱  ╱────              │
│       ╱                  │   ╱  ╱  ╱────            │
│      ╱  ╲____            │  ╱  ╱  ╱                 │
│     ╱    ╲___            │ ╱  ╱  ╱                  │
│    ╱                     │╱  ╱  ╱                   │
└──────────────────────────┴──────────────────────────┘
```

Features:
- Left panel: Hard label training dynamics
- Right panel: KD training dynamics
- Same Y-axis scale for comparison
- Legend: View colors

---

## Running the Experiment

### Single Class

```bash
cd /kdmech/engineer
python experiments/exp_3_3_pathway_evolution.py \
    --seed 0 \
    --class 0 \
    --teachers 5 \
    --epochs 200
```

**Time estimate**: ~15 minutes on GPU

### Multiple Classes

```bash
python experiments/exp_3_3_pathway_evolution.py --multi-class 3
```

Runs for classes 0, 1, 2 and reports aggregate statistics.

### Quick Test

```bash
python experiments/exp_3_3_pathway_evolution.py --quick
```

Uses 3 teachers, 50 epochs (~3 minutes)

---

## Interpretation Guide

### Clear Contrast

**Expected behavior**:
- Hard labels: One pathway dominates (>80% of total)
- KD: Multiple pathways substantial (each >20%)
- Visual contrast is striking

### Weak Contrast

**Potential issues**:
- Hard labels show multiple survivors: Increase epochs
- KD shows one winner: Check temperature, verify teacher diversity

### Pathway Strength Scale

Note: Absolute values vary. Focus on:
- **Relative** strengths (ratio between pathways)
- **Dominance ratio** (summary metric)
- **Number surviving** (discrete count)

---

## Multi-Class Analysis

Running `--multi-class N` provides stronger evidence by testing across classes:

```
Summary across classes:
Hard labels - mean surviving: 1.00 ± 0.00
KD - mean surviving: 2.67 ± 0.47
```

This shows the effect is consistent, not a fluke of one class.

---

## Connection to Theory

### What This Validates

1. **Corner vs Interior equilibrium**: Different equilibria verified ✓
2. **Pathway survival**: Multiple pathways survive under KD ✓
3. **Same initialization**: Only difference is loss function ✓
4. **Visual evidence**: Dynamics look as predicted ✓

### The Key Insight

The **only difference** between the two students is the loss function:
- Hard labels: $\mathcal{L} = \text{CE}(f(x), y)$
- KD: $\mathcal{L} = \tau^2 \cdot \text{KL}(p_T \| p_S)$

Same architecture, same initialization, different outcomes. This isolates the mechanism.

### Gradient Structure Difference

Hard labels:
$$\nabla \mathcal{L} \propto s_{y,m} \quad \text{(self-reinforcing)}$$

KD:
$$\nabla \mathcal{L} \propto \alpha_m(T) + \gamma s_{y,m} \quad \text{(external + self)}$$

The external signal $\alpha_m(T)$ prevents weak pathways from dying.

---

## Troubleshooting

### Both Students Look Similar

- Increase training epochs
- Check teacher diversity (should be >50% ensemble coverage)
- Verify temperature (τ=4 is standard)

### Hard Label Student Shows Multiple Pathways

- May not have converged yet (increase epochs)
- Check that all pathways are truly distinct (not numerical noise)
- Look at dominance ratio, not just visual

### KD Student Shows One Winner

- Temperature may be too low (try τ=8)
- Teachers may all have same view (check teacher coverage)
- Training may not have converged

### Figure Not Clear

- Increase figure size
- Check Y-axis scaling (should be same for both panels)
- Try different random seed
