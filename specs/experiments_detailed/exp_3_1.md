# Experiment 3.1: KD Coverage Transfer

> **Validates**: Theorem 3 — KD enables multi-view learning via external signal
> **Priority**: P1 (Core validation)
> **Implementation**: `/kdmech/engineer/experiments/exp_3_1_kd_coverage.py`

---

## Theoretical Background

### Theorem 3 Prediction

Under KD from a multi-view teacher ensemble:

$$C(S) \approx C(T) \gg C(f_{\text{hard}})$$

Where:
- $C(S)$ = Student coverage under KD
- $C(T)$ = Teacher ensemble coverage
- $C(f_{\text{hard}})$ = Coverage under hard labels ($\approx 1/M$)

### Why KD Works

1. **External signal**: Teacher provides gradient to all its known views
2. **Gradient floor**: Weak pathways still receive signal (unlike hard labels)
3. **Multiple survivors**: Interior equilibrium with multiple non-zero pathways
4. **Coverage inheritance**: Student learns what teacher knows

---

## Experimental Protocol

### Setup

```python
def experiment_3_1_kd_coverage(
    num_teachers: int = 5,
    num_trials: int = 10,
    K: int = 10,
    M: int = 3,
    temperature: float = 4.0,
    epochs: int = 100,
    ...
):
```

### Procedure

For each trial:

1. **Train teacher ensemble**:
   - 5 networks trained independently with hard labels
   - Different seeds → different views learned
   - Ensemble covers most/all views

2. **Train student (hard labels)**:
   - Same architecture as teachers
   - Baseline comparison

3. **Train student (KD)**:
   - Same initialization as hard label student
   - KD loss from teacher ensemble
   - Temperature τ = 4.0

4. **Measure coverage**:
   - Individual teacher coverage
   - Ensemble coverage
   - Student (hard) coverage
   - Student (KD) coverage

---

## Code Walkthrough

### Teacher Ensemble Training

```python
teachers = []
teacher_coverages = []

for i in range(num_teachers):
    torch.manual_seed(trial * 100 + i)
    teacher = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
    teacher.apply(init_weights)

    train_hard_labels(teacher, dataset, epochs=epochs)
    teachers.append(teacher)

    t_cov = measure_view_coverage(teacher, dataset)
    teacher_coverages.append(t_cov)
```

### Student Training with KD

```python
# Same initialization as hard label student!
torch.manual_seed(trial * 1000)
student_kd = MultiViewNet(d=dataset.d, hidden=hidden, K=dataset.K)
student_kd.apply(init_weights)

train_kd(
    student_kd, teachers, dataset,
    epochs=epochs,
    temperature=temperature,
    lr=lr, batch_size=batch_size
)

kd_cov = measure_view_coverage(student_kd, dataset)
```

### KD Training Loss

```python
def train_kd(student, teachers, dataset, temperature=4.0):
    for batch in dataloader:
        x, y = batch

        # Teacher ensemble soft labels
        with torch.no_grad():
            teacher_logits = torch.stack([t(x) for t in teachers]).mean(0)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # Student soft prediction
        student_logits = student(x)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

        # KD loss (scaled)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        loss = loss * (temperature ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Expected Results

### Primary Predictions

| Metric | Expected Value | Tolerance |
|--------|----------------|-----------|
| Hard label coverage | 0.33 | ± 0.05 |
| Ensemble coverage | 0.90+ | (depends on N teachers) |
| KD coverage | 0.85+ | ≈ ensemble |

### Pass Criteria

```python
# KD matches ensemble
if abs(mean_kd - mean_ensemble) < 0.1:
    print("✓ PASSED: KD student coverage ≈ ensemble coverage")

# KD beats hard labels
if mean_kd > mean_hard + 0.1:
    print("✓ PASSED: KD coverage > hard label coverage")
```

---

## Output Format

### JSON Results

```json
{
  "experiment": "exp_3_1_kd_coverage",
  "config": {
    "num_teachers": 5,
    "num_trials": 10,
    "K": 10,
    "M": 3,
    "temperature": 4.0,
    "epochs": 100
  },
  "metrics": {
    "hard_label": {"mean": 0.333, "std": 0.045},
    "kd": {"mean": 0.890, "std": 0.055},
    "ensemble": {"mean": 0.933, "std": 0.047},
    "kd_vs_ensemble_diff": 0.043,
    "kd_vs_hard_diff": 0.557,
    "passed_kd_matches_ensemble": true,
    "passed_kd_better_than_hard": true
  },
  "raw_data": {
    "hard_label": [0.30, 0.33, 0.37, ...],
    "kd": [0.87, 0.90, 0.93, ...],
    "ensemble": [0.90, 0.93, 0.97, ...],
    "individual_teachers": [[0.33, 0.30, ...], ...]
  },
  "figure_path": "results/figures/kd_coverage_comparison.png"
}
```

### Console Output

```
============================================================
Experiment 3.1: KD Coverage Transfer
Config: 5 teachers, 10 trials
K=10, M=3, temperature=4.0
============================================================

=== Trial 1/10 ===
  Teacher 0: coverage = 0.333
  Teacher 1: coverage = 0.333
  Teacher 2: coverage = 0.367
  Teacher 3: coverage = 0.300
  Teacher 4: coverage = 0.333
  Ensemble coverage: 0.933
  Student (hard) coverage: 0.333
  Student (KD) coverage: 0.900

...

============================================================
Results Summary
============================================================
Hard label:  0.333 ± 0.045
KD:          0.890 ± 0.055
Ensemble:    0.933 ± 0.047

Expected (1/M): 0.333
✓ PASSED: KD student coverage ≈ ensemble coverage
✓ PASSED: KD coverage > hard label coverage
```

---

## Generated Figure

The bar chart compares:

```
Coverage
   │
1.0│              ████
   │        ████  ████
0.8│        ████  ████
   │        ████  ████
0.6│        ████  ████
   │        ████  ████
0.4│  ████  ████  ████
   │  ████  ████  ████
0.2│  ████  ████  ████
   │  ████  ████  ████
0.0└──────────────────
     Hard   KD   Ensemble
```

Features:
- Error bars: Standard deviation across trials
- Individual points: Each trial's result
- Dashed line: Expected $1/M$ value

---

## Running the Experiment

### Full Run (Recommended)

```bash
cd /kdmech/engineer
python experiments/exp_3_1_kd_coverage.py \
    --teachers 5 \
    --trials 10 \
    --epochs 100 \
    --temperature 4.0
```

**Time estimate**: ~1 hour on GPU

### Quick Test

```bash
python experiments/exp_3_1_kd_coverage.py --quick
```

Uses 3 teachers, 3 trials, 50 epochs (~10 minutes)

### Parameter Variations

```bash
# More teachers (higher ensemble coverage)
python experiments/exp_3_1_kd_coverage.py --teachers 10

# Different temperature
python experiments/exp_3_1_kd_coverage.py --temperature 8.0
```

---

## Interpretation Guide

### Strong KD Benefit

**Expected behavior**:
- KD coverage 2-3× higher than hard labels
- KD coverage within 0.1 of ensemble
- Consistent across trials

### Weak KD Benefit

**Potential issues**:
- KD only slightly better than hard: Temperature too low
- KD much worse than ensemble: Training insufficient

### Ensemble Coverage

**Expected**:
- 5 teachers → ~90% coverage (some view overlap)
- 10 teachers → ~95%+ coverage
- Single teacher → ~33% (same as hard labels)

---

## Connection to Theory

### What This Validates

1. **Coverage inheritance**: $C(S) \approx C(T)$ ✓
2. **KD enables multi-view**: Single student matches ensemble ✓
3. **External signal effect**: Soft labels break winner-take-all ✓

### The Key Comparison

| Condition | Coverage | Why |
|-----------|----------|-----|
| Hard labels | 0.33 | Self-reinforcing → one view |
| KD | 0.90 | External signal → all teacher views |
| Ensemble | 0.93 | Different seeds → different views |

### Temperature Effect

Temperature τ controls how much view information is in soft labels:
- Low τ (1-2): Approaches hard labels
- Medium τ (4): Good balance
- High τ (8+): Very soft, may lose class info

---

## Troubleshooting

### KD Coverage Not Matching Ensemble

- Increase training epochs
- Check temperature (try τ = 4)
- Ensure teachers are diverse (different seeds)

### Ensemble Coverage Low

- Increase number of teachers
- Verify teachers have different seeds
- Check teacher training converged

### High Variance Across Trials

- Increase num_trials
- Use more teachers per ensemble
- Seed teachers systematically

### KD Not Better Than Hard Labels

- Check temperature (not too low)
- Verify KD loss is being used
- Ensure teachers are multi-view (ensemble)
