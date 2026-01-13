# Experiment 2.1: Single-View Convergence

> **Validates**: Theorem 2 — Hard label training leads to $C(f) \approx 1/M$
> **Priority**: P1 (Core validation)
> **Implementation**: `/kdmech/engineer/experiments/exp_2_1_single_view.py`

---

## Theoretical Background

### Theorem 2 Prediction

Under hard label training, winner-take-all dynamics cause each class to converge to a single view:

$$C(f) = \frac{1}{KM} \sum_{y,m} \mathbb{1}[\text{view } (y,m) \text{ detected}] \approx \frac{1}{M}$$

For $M = 3$, we expect:
$$C(f) \approx 0.33$$

### Why This Happens

1. **Self-reinforcement**: Hard label gradients scale with pathway strength
2. **Winner-take-all**: Stronger pathway grows faster
3. **Loser suppression**: Weaker pathways decay to zero
4. **Result**: One view dominates per class

---

## Experimental Protocol

### Setup

```python
def experiment_2_1_single_view_convergence(
    num_seeds: int = 30,
    K: int = 10,
    M: int = 3,
    d_view: int = 50,
    hidden: int = 200,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 128,
    n_samples: int = 10000
):
```

### Procedure

1. **For each seed** (30 total):
   - Create fresh dataset
   - Initialize model with different seed
   - Train with hard labels (cross-entropy loss)
   - Measure view coverage

2. **Compute statistics**:
   - Mean coverage across seeds
   - Standard deviation
   - Compare to expected value $1/M$

---

## Code Walkthrough

### Main Loop

```python
for seed in range(num_seeds):
    # Create fresh dataset and model
    dataset = MultiViewDataset(K=K, M=M, d_view=d_view, n_samples=n_samples, seed=seed)

    torch.manual_seed(seed + 5000)
    model = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
    model.apply(init_weights)

    # Train
    history = train_hard_labels(model, dataset, epochs=epochs, lr=lr, batch_size=batch_size)

    # Measure coverage
    coverage = measure_view_coverage(model, dataset)
    results.append(coverage)
```

### Coverage Measurement

```python
def measure_view_coverage(model, dataset, threshold=0.5):
    detected = 0
    total = dataset.K * dataset.M  # 30 total views

    for y in range(dataset.K):
        for m in range(dataset.M):
            phi_ym = dataset.get_view_feature(y, m)

            with torch.no_grad():
                logits = model(phi_ym.unsqueeze(0)).squeeze(0)
                probs = F.softmax(logits, dim=0)

            pred = logits.argmax().item()
            conf = probs[y].item()

            if pred == y and conf > threshold:
                detected += 1

    return detected / total
```

---

## Expected Results

### Primary Prediction

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| Mean coverage | 0.333 | ± 0.05 |

### Pass Criterion

```python
if abs(mean_cov - expected) < 0.05:
    print("✓ PASSED: Coverage matches 1/M prediction")
else:
    print("✗ FAILED: Coverage differs from 1/M prediction")
```

---

## Output Format

### JSON Results

```json
{
  "experiment": "exp_2_1_single_view_convergence",
  "timestamp": "2024-01-15T10:30:00",
  "config": {
    "K": 10,
    "M": 3,
    "d_view": 50,
    "hidden": 200,
    "num_seeds": 30,
    "epochs": 100,
    "lr": 0.01,
    "batch_size": 128,
    "n_samples": 10000
  },
  "metrics": {
    "mean_coverage": 0.333,
    "std_coverage": 0.045,
    "expected": 0.333,
    "difference": 0.000,
    "passed": true
  },
  "raw_data": [
    {
      "seed": 0,
      "coverage": 0.367,
      "final_accuracy": 0.98,
      "per_class_views": {"0": [2], "1": [0], ...}
    },
    ...
  ]
}
```

### Console Output

```
============================================================
Experiment 2.1: Single-View Convergence
Config: K=10, M=3, d_view=50, epochs=100
Expected coverage: 0.333
============================================================
Seed  0: coverage = 0.367, accuracy = 0.983
Seed  1: coverage = 0.300, accuracy = 0.978
Seed  2: coverage = 0.333, accuracy = 0.985
...
Seed 29: coverage = 0.333, accuracy = 0.981

============================================================
Results Summary
============================================================
Mean coverage: 0.333 ± 0.045
Expected (1/M): 0.333
Difference: 0.000
✓ PASSED: Coverage matches 1/M prediction

Results saved to: results/results_exp_2_1.json
```

---

## Running the Experiment

### Full Run (Recommended)

```bash
cd /kdmech/engineer
python experiments/exp_2_1_single_view.py --seeds 30 --epochs 100
```

**Time estimate**: ~30 minutes on GPU

### Quick Test

```bash
python experiments/exp_2_1_single_view.py --quick
```

Uses 3 seeds, 50 epochs (~2 minutes)

### Custom Parameters

```bash
python experiments/exp_2_1_single_view.py \
    --seeds 50 \
    --epochs 150 \
    --output results/exp_2_1_extended/
```

---

## Interpretation Guide

### Coverage Near 0.33

**This validates Theorem 2**: Networks learn approximately one view per class.

- Coverage slightly above 0.33: Some classes learned 2 views
- Coverage slightly below 0.33: Some views below detection threshold
- High variance: Winner-take-all is noisy but consistent

### Coverage Significantly Different

**Potential issues**:
- Coverage much higher (>0.5): Views not orthogonal enough, or network too wide
- Coverage much lower (<0.25): Detection threshold too high, or training insufficient

### Per-Class Analysis

The `raw_data` field shows which views were learned per class:

```python
per_class_views = {
    "0": [2],      # Class 0 learned only view 2
    "1": [0],      # Class 1 learned only view 0
    "2": [1, 2],   # Class 2 learned views 1 and 2 (rare)
    ...
}
```

---

## Connection to Theory

### This Experiment Validates

1. **Single-view convergence**: Networks learn ~1 view per class ✓
2. **Coverage formula**: $C(f) \approx 1/M$ holds ✓
3. **Self-reinforcing dynamics**: Winner-take-all is real ✓

### This Experiment Does NOT Validate

1. **Which view wins** (tested in Exp 2.4)
2. **Why winner-take-all occurs** (tested in Exp 2.3)
3. **How KD changes this** (tested in Exp 3.1, 3.3)

---

## Troubleshooting

### Coverage Too High

- Check view orthogonality: `verify_dataset(dataset)` should show max dot < 0.01
- Reduce hidden width: Try hidden=100
- Increase epochs: Ensure full convergence

### Coverage Too Low

- Lower detection threshold: Try threshold=0.3
- Check training convergence: Final accuracy should be >95%
- Increase view signal: Try noise_std=0.05

### High Variance

- Increase num_seeds for tighter confidence intervals
- This is expected behavior (winner-take-all is stochastic)
