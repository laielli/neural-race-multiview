# Experiment 2.3: Race Dynamics Visualization

> **Validates**: Theorem 2 — Winner-take-all pathway competition
> **Priority**: P1 (Core validation)
> **Implementation**: `/kdmech/engineer/experiments/exp_2_3_race_dynamics.py`

---

## Theoretical Background

### Race Dynamics Equation

During hard label training, pathway strengths evolve according to:

$$\frac{ds_{y,m}}{dt} = \sigma_1 \cdot s_{y,m} \cdot \left(1 - \frac{\sum_{m'} s_{y,m'}^2}{s_{\max}^2}\right)$$

Key properties:
- **Exponential early growth**: $s_m(t) \approx s_m(0) \cdot e^{\sigma_1 t}$ initially
- **Winner-take-all**: Pathway with highest $A_m = \sigma_1 \cdot s_m(0)$ dominates
- **Loser suppression**: Other pathways decay to near zero

### Visual Prediction

```
Pathway Strength vs. Training Epoch
│
│     ╱────────── View 1 (winner)
│    ╱
│   ╱
│  ╱  ╲__________ View 2 (loser)
│ ╱    ╲_________ View 3 (loser)
│╱
└─────────────────────────────
  0    50   100   150   200  epoch
```

---

## Experimental Protocol

### Setup

```python
def experiment_2_3_race_visualization(
    seed: int = 0,
    y_target: int = 0,        # Class to visualize
    K: int = 10,
    M: int = 3,
    epochs: int = 200,        # Longer for dynamics
    log_interval: int = 5,    # Track every 5 epochs
    ...
):
```

### Procedure

1. **Initialize**: Create dataset and model
2. **Train with tracking**: Log pathway strengths every `log_interval` epochs
3. **Identify winner**: Find pathway with highest final strength
4. **Generate figure**: Plot all pathway strengths over time

---

## Code Walkthrough

### Training with Pathway Tracking

```python
# Train with pathway tracking
history = train_hard_labels(
    model, dataset,
    epochs=epochs,
    lr=lr,
    batch_size=batch_size,
    log_interval=log_interval,
    track_pathways=True,
    track_classes=[y_target]
)
```

### Extracting Pathway Strengths

During training, at each log interval:

```python
for y in track_classes:
    for m in range(M):
        phi_ym = dataset.get_view_feature(y, m)
        strength = model.get_pathway_strength(phi_ym)
        history['pathway_strengths'][y][m].append(strength)
```

### Figure Generation

```python
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, M))

for m in range(M):
    strengths = history['pathway_strengths'][y_target][m]
    ax.plot(epochs_logged, strengths, label=f'View {m}',
            linewidth=2.5, color=colors[m])

ax.set_xlabel('Epoch')
ax.set_ylabel('Pathway Strength')
ax.set_title(f'Race Dynamics (Class {y_target}, Hard Labels)')
ax.legend()
```

---

## Expected Results

### Primary Predictions

| Metric | Expected |
|--------|----------|
| Winner dominance | > 0.7 (ratio of winner to total) |
| Losers decay | < 0.1 × winner strength |
| Shape | Sigmoidal (exponential then saturation) |

### Visual Characteristics

1. **Early phase** (epochs 0-50): Exponential growth for all views
2. **Separation phase** (epochs 50-100): Winner pulls ahead
3. **Saturation phase** (epochs 100+): Winner plateaus, losers near zero

---

## Output Format

### JSON Results

```json
{
  "experiment": "exp_2_3_race_dynamics",
  "config": {
    "seed": 0,
    "y_target": 0,
    "K": 10,
    "M": 3,
    "epochs": 200,
    "log_interval": 5
  },
  "metrics": {
    "winning_view": 2,
    "final_strengths": {"0": 0.12, "1": 0.08, "2": 4.85},
    "winner_ratio": 0.96
  },
  "history": {
    "epochs_logged": [0, 5, 10, ...],
    "pathway_strengths": {
      "0": [0.03, 0.05, 0.08, ...],
      "1": [0.02, 0.04, 0.06, ...],
      "2": [0.04, 0.08, 0.15, ...]
    },
    "accuracy": [0.12, 0.25, 0.45, ...],
    "loss": [2.30, 1.85, 1.42, ...]
  },
  "figure_path": "results/figures/race_dynamics_hard.png"
}
```

### Generated Figure

![Race Dynamics](race_dynamics_hard.png)

The figure shows:
- X-axis: Training epoch
- Y-axis: Pathway strength
- Lines: One per view (M=3)
- Annotation: Winning view labeled

---

## Running the Experiment

### Single Seed

```bash
cd /kdmech/engineer
python experiments/exp_2_3_race_dynamics.py --seed 0 --class 0 --epochs 200
```

### Multiple Seeds Comparison

```bash
python experiments/exp_2_3_race_dynamics.py --multi 5 --class 0 --epochs 200
```

This generates a combined figure showing different winners across seeds.

### Quick Test

```bash
python experiments/exp_2_3_race_dynamics.py --quick
```

Uses 50 epochs (~1 minute)

---

## Multi-Seed Analysis

### Purpose

Running multiple seeds shows that **different seeds → different winners**:

```
Seed 0: Winner = View 2
Seed 1: Winner = View 0
Seed 2: Winner = View 1
Seed 3: Winner = View 2
Seed 4: Winner = View 0
```

This validates that winner selection is determined by initialization, not by any inherent property of the views (which are symmetric in our setup).

### Generated Figure

```
┌────────┬────────┬────────┬────────┬────────┐
│ Seed 0 │ Seed 1 │ Seed 2 │ Seed 3 │ Seed 4 │
│ Win: 2 │ Win: 0 │ Win: 1 │ Win: 2 │ Win: 0 │
│        │        │        │        │        │
│   ╱──  │ ╱──    │  ╱──   │   ╱──  │ ╱──    │
│  ╱     │╱       │ ╱      │  ╱     │╱       │
│ ╱      │        │╱       │ ╱      │        │
└────────┴────────┴────────┴────────┴────────┘
```

---

## Interpretation Guide

### Strong Winner-Take-All

**Expected behavior**:
- One pathway dominates (>80% of total strength)
- Other pathways decay to near zero
- Dynamics show clear separation

### Weak Winner-Take-All

**Potential issues**:
- Winner ratio < 0.7: Training may not have converged
- Multiple pathways survive: Views may not be orthogonal

### Dynamics Shape

**Expected shape**:
1. All pathways start small (~0.03 at init)
2. All grow exponentially in early phase
3. One pulls ahead due to initial advantage
4. Winner saturates; losers decay

**Unexpected shapes**:
- No separation: Check learning rate, epochs
- Oscillation: Check batch size, momentum
- Plateau without separation: Check data orthogonality

---

## Connection to Theory

### What This Validates

1. **Winner-take-all mechanism**: One pathway dominates ✓
2. **Exponential growth phase**: Early dynamics match theory ✓
3. **Saturation**: System reaches equilibrium ✓
4. **Seed diversity**: Different seeds → different winners ✓

### Key Theoretical Link

The race dynamics equation predicts:
- Growth rate ∝ current strength (exponential)
- Competition term enforces one winner
- Initial advantage determines winner

This experiment directly visualizes these predictions.

---

## Troubleshooting

### No Clear Winner

- Increase epochs (try 300-400)
- Check that views are orthogonal
- Verify training is converging (check loss)

### All Pathways Grow Similarly

- This is expected in early phase
- Wait longer for separation
- May indicate views have similar initial advantages

### Oscillating Dynamics

- Reduce learning rate
- Reduce momentum
- Increase batch size

### Figure Not Generated

- Check output directory exists
- Verify matplotlib is installed
- Check for errors in console output
