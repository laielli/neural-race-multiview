# Synthetic Experiments: Validating the Neural Race × Multi-View Theory

## Overview

This document outlines synthetic experiments to validate Theorems 1-3. The key advantage of synthetic data: **ground truth is exactly known**, enabling precise validation of theoretical predictions.

---

## Part 1: Synthetic Data Design

### 1.1 Multi-View Data Generation

**Design Principle**: Create data where views are explicitly constructed, orthogonal, and identifiable.

**Structure**:
```
Input x ∈ ℝ^d where d = M × d_view

Each view occupies a disjoint "slot" in the input vector:
- Slot 1: dimensions [0, d_view)         → View 1
- Slot 2: dimensions [d_view, 2·d_view)  → View 2
- ...
- Slot M: dimensions [(M-1)·d_view, d)   → View M
```

### 1.2 View Feature Construction

For class $y \in [K]$ and view $m \in [M]$:

**Option A — Random Orthogonal**:
```python
# Generate random orthogonal view features
phi = {}
for y in range(K):
    for m in range(M):
        # Random unit vector in slot m
        v = torch.randn(d_view)
        v = v / v.norm()
        phi[(y, m)] = embed_in_slot(v, slot=m, total_dim=d)
```

**Option B — Structured Patterns** (more interpretable):
```python
# Use distinct patterns per class
patterns = {
    0: [sine_wave, square_wave, sawtooth],
    1: [gaussian_bump, double_peak, flat_top],
    # ... etc
}
```

### 1.3 Sample Generation

```python
def generate_sample(y, view_prob=0.5, noise_std=0.1):
    """Generate sample from class y with random active views."""
    x = torch.zeros(d)
    active_views = []

    for m in range(M):
        if random.random() < view_prob:
            # View m is active
            x += phi[(y, m)]
            active_views.append(m)

    # Ensure at least one view
    if len(active_views) == 0:
        m = random.randint(0, M-1)
        x += phi[(y, m)]
        active_views.append(m)

    # Add noise
    x += noise_std * torch.randn(d)

    return x, y, active_views
```

### 1.4 Default Parameters

| Parameter | Symbol | Default Value | Rationale |
|-----------|--------|---------------|-----------|
| Classes | $K$ | 10 | Enough for meaningful diversity |
| Views per class | $M$ | 3 | Tractable yet non-trivial |
| View dimension | $d_{\text{view}}$ | 50 | Rich enough features |
| Total dimension | $d$ | 150 | $= M \times d_{\text{view}}$ |
| View probability | $p$ | 0.5 | Balanced multi-view structure |
| Noise std | $\sigma$ | 0.1 | Low noise regime |
| Training samples | $N$ | 10,000 | Sufficient statistics |
| Test samples | | 2,000 | Evaluation set |

### 1.5 Data Verification

Before running experiments, verify:

1. **View orthogonality**: $|\langle \phi_{y,m}, \phi_{y',m'} \rangle| < 0.01$ for $(y,m) \neq (y',m')$

2. **View sufficiency**: Linear classifier on single view achieves >95% accuracy

3. **Multi-view benefit**: Oracle with all views > single view performance

---

## Part 2: Network Architecture

### 2.1 Base Architecture

**Two-Layer MLP** (for Theorems 1-2 analysis):
```python
class MultiViewNet(nn.Module):
    def __init__(self, d=150, hidden=200, K=10):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, K)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def get_gating(self, x):
        """Return binary gating pattern."""
        h_pre = self.fc1(x)
        return (h_pre > 0).float()

    def get_pathway_output(self, x, gating):
        """Output through specific gating pattern."""
        h = F.relu(self.fc1(x)) * gating  # Apply gating
        return self.fc2(h)
```

### 2.2 Pathway Strength Measurement

```python
def measure_pathway_strength(model, phi, y, m):
    """Measure strength of pathway (y, m)."""
    # Input with only view (y, m)
    x = phi[(y, m)].unsqueeze(0)

    # Get output (this is R_{y,m})
    with torch.no_grad():
        R_ym = model(x).squeeze()

    return R_ym.norm().item()

def measure_all_pathway_strengths(model, phi, K, M):
    """Return dict of all pathway strengths."""
    strengths = {}
    for y in range(K):
        for m in range(M):
            strengths[(y, m)] = measure_pathway_strength(model, phi, y, m)
    return strengths
```

### 2.3 View Coverage Measurement

```python
def measure_view_coverage(model, phi, K, M, threshold=0.5):
    """Measure fraction of views the network detects."""
    detected = 0

    for y in range(K):
        for m in range(M):
            x = phi[(y, m)].unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)

            # View detected if correct class with confidence > threshold
            if logits.argmax().item() == y and probs[0, y].item() > threshold:
                detected += 1

    return detected / (K * M)
```

---

## Part 3: Experiments for Theorem 1 (View-Pathway Correspondence)

### Experiment 1.1: Output Decomposition Validation

**Goal**: Verify that $f(x) \approx \sum_{m \in \mathcal{S}} R_{y,m}$

**Protocol**:
```python
def experiment_1_1(model, phi, test_data):
    """Validate output decomposition."""
    actual_outputs = []
    decomposed_outputs = []

    for x, y, active_views in test_data:
        # Actual output
        actual = model(x.unsqueeze(0)).squeeze()

        # Decomposed output
        decomposed = torch.zeros_like(actual)
        for m in active_views:
            x_m = phi[(y, m)].unsqueeze(0)
            decomposed += model(x_m).squeeze()

        actual_outputs.append(actual)
        decomposed_outputs.append(decomposed)

    # Compute correlation
    actual_flat = torch.stack(actual_outputs).flatten()
    decomposed_flat = torch.stack(decomposed_outputs).flatten()
    correlation = torch.corrcoef(torch.stack([actual_flat, decomposed_flat]))[0, 1]

    return correlation.item()
```

**Prediction**: Correlation > 0.95

**Visualization**: Scatter plot of actual vs. decomposed outputs

### Experiment 1.2: Gating Pattern Distinctness

**Goal**: Verify different views induce different gating patterns (Assumption A4)

**Protocol**:
```python
def experiment_1_2(model, phi, K, M):
    """Measure gating pattern distinctness."""
    gating_patterns = {}

    for y in range(K):
        for m in range(M):
            x = phi[(y, m)].unsqueeze(0)
            g = model.get_gating(x).squeeze()
            gating_patterns[(y, m)] = g

    # Compute pairwise Hamming distances
    distances = []
    for (y1, m1), g1 in gating_patterns.items():
        for (y2, m2), g2 in gating_patterns.items():
            if (y1, m1) < (y2, m2):
                hamming = (g1 != g2).float().mean().item()
                distances.append(hamming)

    return {
        'mean_hamming': np.mean(distances),
        'min_hamming': np.min(distances),
        'all_distinct': np.min(distances) > 0
    }
```

**Prediction**: All views have distinct gating patterns (min Hamming > 0)

### Experiment 1.3: View Response Spectrum

**Goal**: Characterize which views a trained network responds to

**Protocol**:
```python
def experiment_1_3(model, phi, K, M):
    """Measure view response spectrum."""
    responses = np.zeros((K, M))

    for y in range(K):
        for m in range(M):
            x = phi[(y, m)].unsqueeze(0)
            with torch.no_grad():
                R = model(x).squeeze()
            # Response magnitude for correct class
            responses[y, m] = R[y].item()

    return responses
```

**Visualization**: Heatmap of responses (classes × views)

---

## Part 4: Experiments for Theorem 2 (Race Dynamics)

### Experiment 2.1: Single-View Convergence Under Hard Labels

**Goal**: Verify $C(f) \approx 1/M$ after hard label training

**Protocol**:
```python
def experiment_2_1(phi, K, M, num_seeds=30):
    """Test single-view convergence."""
    coverages = []

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        model = MultiViewNet()
        train_with_hard_labels(model, phi, epochs=100)
        coverage = measure_view_coverage(model, phi, K, M)
        coverages.append(coverage)

    return {
        'mean_coverage': np.mean(coverages),
        'std_coverage': np.std(coverages),
        'expected': 1/M
    }
```

**Prediction**: Mean coverage ≈ 1/M = 0.33 (for M=3)

### Experiment 2.2: View Diversity Across Seeds

**Goal**: Verify different seeds → different winning views

**Protocol**:
```python
def experiment_2_2(phi, K, M, num_seeds=30):
    """Measure view diversity across random seeds."""
    winning_views = {y: [] for y in range(K)}

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        model = MultiViewNet()
        train_with_hard_labels(model, phi, epochs=100)

        for y in range(K):
            # Find winning view for class y
            strengths = [measure_pathway_strength(model, phi, y, m) for m in range(M)]
            winner = np.argmax(strengths)
            winning_views[y].append(winner)

    # Compute diversity for each class
    diversity = {}
    for y in range(K):
        unique_winners = len(set(winning_views[y]))
        diversity[y] = unique_winners / M

    return {
        'per_class_diversity': diversity,
        'mean_diversity': np.mean(list(diversity.values()))
    }
```

**Prediction**: High diversity (>0.6) — different seeds learn different views

### Experiment 2.3: Race Dynamics Visualization

**Goal**: Visualize winner-take-all dynamics during training

**Protocol**:
```python
def experiment_2_3(phi, y_target=0, num_epochs=200, log_interval=5):
    """Track pathway strengths during training."""
    model = MultiViewNet()

    # Track strengths over time
    history = {m: [] for m in range(M)}

    for epoch in range(num_epochs):
        train_one_epoch(model, phi)

        if epoch % log_interval == 0:
            for m in range(M):
                s = measure_pathway_strength(model, phi, y_target, m)
                history[m].append(s)

    return history
```

**Visualization**:
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

**Prediction**: One pathway grows to dominate, others decay

### Experiment 2.4: Winner Prediction from Initialization

**Goal**: Verify that initial advantage predicts winner

**Protocol**:
```python
def experiment_2_4(phi, K, M, num_seeds=50):
    """Test winner prediction accuracy."""
    correct_predictions = 0
    total_predictions = 0

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        model = MultiViewNet()

        for y in range(K):
            # Measure initial advantages
            init_strengths = [measure_pathway_strength(model, phi, y, m) for m in range(M)]
            # For simplicity, use strength as proxy for advantage
            # (assumes equal correlation strength)
            predicted_winner = np.argmax(init_strengths)

            # Train and find actual winner
            train_with_hard_labels(model, phi, epochs=100)
            final_strengths = [measure_pathway_strength(model, phi, y, m) for m in range(M)]
            actual_winner = np.argmax(final_strengths)

            if predicted_winner == actual_winner:
                correct_predictions += 1
            total_predictions += 1

        # Reset for next seed
        torch.manual_seed(seed)
        model = MultiViewNet()

    return correct_predictions / total_predictions
```

**Prediction**: Accuracy > 80%

### Experiment 2.5: Exponential Early Growth

**Goal**: Verify pathway strengths grow exponentially initially

**Protocol**:
```python
def experiment_2_5(phi, y_target=0, early_epochs=30):
    """Fit exponential to early pathway growth."""
    model = MultiViewNet()

    log_strengths = {m: [] for m in range(M)}
    times = []

    for epoch in range(early_epochs):
        train_one_epoch(model, phi)
        times.append(epoch)

        for m in range(M):
            s = measure_pathway_strength(model, phi, y_target, m)
            log_strengths[m].append(np.log(s + 1e-8))

    # Fit linear regression to log-strength vs time
    r_squared = {}
    for m in range(M):
        slope, intercept, r, p, se = scipy.stats.linregress(times, log_strengths[m])
        r_squared[m] = r**2

    return r_squared
```

**Prediction**: $R^2 > 0.9$ for early phase (exponential growth)

### Experiment 2.6: Correlation Strength Effect

**Goal**: Verify stronger correlation → more likely to win

**Protocol**:
```python
def experiment_2_6(K, M, num_seeds=50):
    """Test effect of view strength on winning probability."""
    # Create asymmetric views
    # View 0: norm 1.5 (strong)
    # View 1: norm 1.0 (medium)
    # View 2: norm 0.5 (weak)

    phi_asymmetric = create_asymmetric_views(
        K, M,
        norms=[1.5, 1.0, 0.5]
    )

    win_counts = {m: 0 for m in range(M)}

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        model = MultiViewNet()
        train_with_hard_labels(model, phi_asymmetric, epochs=100)

        for y in range(K):
            strengths = [measure_pathway_strength(model, phi_asymmetric, y, m) for m in range(M)]
            winner = np.argmax(strengths)
            win_counts[winner] += 1

    total = num_seeds * K
    win_rates = {m: win_counts[m] / total for m in range(M)}

    return win_rates
```

**Prediction**: Win rate ordering: View 0 > View 1 > View 2

---

## Part 5: Experiments for Theorem 3 (KD Circumvents Race)

### Experiment 3.1: Multi-View Learning via KD

**Goal**: Verify KD from ensemble → multi-view student

**Protocol**:
```python
def experiment_3_1(phi, K, M, num_teachers=5, num_trials=10):
    """Compare coverage: hard labels vs KD."""
    results = {'hard': [], 'kd': []}

    for trial in range(num_trials):
        # Train ensemble of teachers
        teachers = []
        for i in range(num_teachers):
            torch.manual_seed(trial * 100 + i)
            t = MultiViewNet()
            train_with_hard_labels(t, phi, epochs=100)
            teachers.append(t)

        # Measure ensemble coverage
        ensemble_coverage = measure_ensemble_coverage(teachers, phi, K, M)

        # Train student with hard labels
        torch.manual_seed(trial * 1000)
        student_hard = MultiViewNet()
        train_with_hard_labels(student_hard, phi, epochs=100)
        results['hard'].append(measure_view_coverage(student_hard, phi, K, M))

        # Train student with KD
        torch.manual_seed(trial * 1000)  # Same init
        student_kd = MultiViewNet()
        train_with_kd(student_kd, teachers, phi, epochs=100, temperature=4.0)
        results['kd'].append(measure_view_coverage(student_kd, phi, K, M))

    return {
        'hard_label_coverage': np.mean(results['hard']),
        'kd_coverage': np.mean(results['kd']),
        'expected_hard': 1/M,
        'ensemble_coverage': ensemble_coverage
    }
```

**Prediction**: KD coverage ≈ ensemble coverage >> hard label coverage

### Experiment 3.2: Gradient Distribution Analysis

**Goal**: Verify soft labels distribute gradient across pathways

**Protocol**:
```python
def experiment_3_2(phi, teachers, K, M):
    """Compare gradient distribution under hard vs soft labels."""

    # Student model
    student = MultiViewNet()

    # Compute gradient magnitude for each pathway under hard labels
    grad_hard = compute_pathway_gradients(
        student, phi, K, M,
        loss_type='hard_label'
    )

    # Compute gradient magnitude under KD
    grad_kd = compute_pathway_gradients(
        student, phi, K, M,
        loss_type='kd',
        teachers=teachers,
        temperature=4.0
    )

    # Measure concentration (Gini coefficient or entropy)
    concentration_hard = gini_coefficient(list(grad_hard.values()))
    concentration_kd = gini_coefficient(list(grad_kd.values()))

    return {
        'gradient_concentration_hard': concentration_hard,
        'gradient_concentration_kd': concentration_kd,
        'grad_hard': grad_hard,
        'grad_kd': grad_kd
    }
```

**Prediction**:
- Hard labels: High concentration (gradient to ~1 pathway)
- Soft labels: Low concentration (gradient distributed)

### Experiment 3.3: Pathway Evolution Under KD

**Goal**: Visualize how multiple pathways survive under KD

**Protocol**:
```python
def experiment_3_3(phi, teachers, y_target=0, num_epochs=200):
    """Track pathway strengths during KD training."""
    student = MultiViewNet()

    history_hard = track_training(
        MultiViewNet(), phi,
        loss_type='hard',
        epochs=num_epochs,
        y_target=y_target
    )

    history_kd = track_training(
        MultiViewNet(), phi,
        loss_type='kd',
        teachers=teachers,
        epochs=num_epochs,
        y_target=y_target
    )

    return history_hard, history_kd
```

**Visualization**:
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

**Prediction**: Under KD, multiple pathways reach positive equilibrium

### Experiment 3.4: Coverage Inheritance

**Goal**: Verify $C(S) \approx C(T)$ for various teachers

**Protocol**:
```python
def experiment_3_4(phi, K, M, num_trials=10):
    """Test coverage inheritance with different teachers."""

    # Create teachers with different coverage levels
    teacher_configs = [
        ('single', 1),      # Single network (C ≈ 1/M)
        ('ensemble_3', 3),  # 3-network ensemble
        ('ensemble_5', 5),  # 5-network ensemble
        ('ensemble_10', 10) # 10-network ensemble (C ≈ 1)
    ]

    results = {}

    for name, num_teachers in teacher_configs:
        teacher_coverages = []
        student_coverages = []

        for trial in range(num_trials):
            # Create teacher(s)
            teachers = create_teacher_ensemble(phi, num_teachers, seed=trial)
            t_cov = measure_ensemble_coverage(teachers, phi, K, M)
            teacher_coverages.append(t_cov)

            # Train student via KD
            student = MultiViewNet()
            train_with_kd(student, teachers, phi, epochs=100)
            s_cov = measure_view_coverage(student, phi, K, M)
            student_coverages.append(s_cov)

        results[name] = {
            'teacher_coverage': np.mean(teacher_coverages),
            'student_coverage': np.mean(student_coverages)
        }

    return results
```

**Prediction**: Student coverage tracks teacher coverage

### Experiment 3.5: Temperature Sweep

**Goal**: Verify temperature effect on view transfer

**Protocol**:
```python
def experiment_3_5(phi, teachers, K, M, num_trials=10):
    """Sweep temperature and measure coverage."""
    temperatures = [1, 2, 4, 8, 16, 32]

    results = {tau: [] for tau in temperatures}

    for tau in temperatures:
        for trial in range(num_trials):
            student = MultiViewNet()
            train_with_kd(student, teachers, phi, epochs=100, temperature=tau)
            cov = measure_view_coverage(student, phi, K, M)
            results[tau].append(cov)

    return {tau: np.mean(covs) for tau, covs in results.items()}
```

**Prediction**: Coverage increases with temperature, then plateaus (or slight decrease at very high τ)

### Experiment 3.6: Self-Distillation Generations

**Goal**: Verify self-distillation improves coverage iteratively

**Protocol**:
```python
def experiment_3_6(phi, K, M, num_generations=5, num_trials=10):
    """Track coverage across self-distillation generations."""

    results = {g: [] for g in range(num_generations)}

    for trial in range(num_trials):
        # Generation 0: hard label training
        gen0 = MultiViewNet()
        train_with_hard_labels(gen0, phi, epochs=100)
        results[0].append(measure_view_coverage(gen0, phi, K, M))

        teacher = gen0
        for g in range(1, num_generations):
            student = MultiViewNet()
            train_with_kd(student, [teacher], phi, epochs=100, temperature=4.0)
            results[g].append(measure_view_coverage(student, phi, K, M))
            teacher = student

    return {g: np.mean(covs) for g, covs in results.items()}
```

**Prediction**: Coverage increases: Gen 0 < Gen 1 < Gen 2 ... (eventually saturates)

### Experiment 3.7: Single-View Teacher Baseline

**Goal**: Verify single-view teacher provides no benefit

**Protocol**:
```python
def experiment_3_7(phi, K, M, num_trials=20):
    """Compare KD from single-view teacher vs hard labels."""

    hard_coverages = []
    kd_coverages = []

    for trial in range(num_trials):
        # Single teacher (single-view)
        teacher = MultiViewNet()
        train_with_hard_labels(teacher, phi, epochs=100)

        # Student with hard labels
        student_hard = MultiViewNet()
        train_with_hard_labels(student_hard, phi, epochs=100)
        hard_coverages.append(measure_view_coverage(student_hard, phi, K, M))

        # Student with KD from single teacher
        student_kd = MultiViewNet()
        train_with_kd(student_kd, [teacher], phi, epochs=100)
        kd_coverages.append(measure_view_coverage(student_kd, phi, K, M))

    return {
        'hard_label': np.mean(hard_coverages),
        'kd_single_teacher': np.mean(kd_coverages),
        'difference': np.mean(kd_coverages) - np.mean(hard_coverages)
    }
```

**Prediction**: No significant difference (single-view teacher ≈ hard labels)

---

## Part 6: Implementation Details

### 6.1 Training Functions

```python
def train_with_hard_labels(model, phi, epochs=100, lr=0.01):
    """Standard cross-entropy training."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch in generate_batches(phi, batch_size=128):
            x, y, _ = batch
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

def train_with_kd(model, teachers, phi, epochs=100, temperature=4.0, lr=0.01):
    """Knowledge distillation training."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        for batch in generate_batches(phi, batch_size=128):
            x, y, _ = batch

            # Get teacher ensemble prediction
            with torch.no_grad():
                teacher_logits = torch.stack([t(x) for t in teachers]).mean(0)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

            # Student prediction
            student_logits = model(x)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

            # KD loss
            loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            loss = loss * (temperature ** 2)  # Scale for gradient magnitude

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 6.2 Utility Functions

```python
def measure_ensemble_coverage(teachers, phi, K, M, threshold=0.5):
    """Measure coverage of teacher ensemble."""
    detected = 0

    for y in range(K):
        for m in range(M):
            x = phi[(y, m)].unsqueeze(0)

            # Ensemble prediction
            with torch.no_grad():
                logits = torch.stack([t(x) for t in teachers]).mean(0)
                probs = F.softmax(logits, dim=-1)

            if logits.argmax().item() == y and probs[0, y].item() > threshold:
                detected += 1

    return detected / (K * M)

def gini_coefficient(values):
    """Compute Gini coefficient (measure of concentration)."""
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
```

---

## Part 7: Expected Results Summary

### Theorem 1 Predictions

| Experiment | Metric | Predicted | Validates |
|------------|--------|-----------|-----------|
| 1.1 | Decomposition correlation | > 0.95 | Output = Σ view responses |
| 1.2 | Min Hamming distance | > 0 | Views have distinct gating |
| 1.3 | Response spectrum | Sparse | Few views detected per network |

### Theorem 2 Predictions

| Experiment | Metric | Predicted | Validates |
|------------|--------|-----------|-----------|
| 2.1 | Mean coverage | ≈ 1/M = 0.33 | Single-view convergence |
| 2.2 | View diversity | > 0.6 | Different seeds → different views |
| 2.3 | Dynamics shape | Winner-take-all | Race mechanism |
| 2.4 | Prediction accuracy | > 80% | Init determines winner |
| 2.5 | Early phase R² | > 0.9 | Exponential growth |
| 2.6 | Win rate ordering | Strong > Weak | Correlation effect |

### Theorem 3 Predictions

| Experiment | Metric | Predicted | Validates |
|------------|--------|-----------|-----------|
| 3.1 | KD coverage | ≈ ensemble | Multi-view via KD |
| 3.2 | Gradient Gini | KD < Hard | Distributed gradients |
| 3.3 | Final pathways | Multiple > 0 | Race circumvented |
| 3.4 | C(S) vs C(T) | Correlated | Coverage inheritance |
| 3.5 | Coverage vs τ | Increasing | Temperature effect |
| 3.6 | Gen n+1 vs Gen n | Increasing | Self-distillation |
| 3.7 | KD(single) vs Hard | ≈ Equal | Single teacher fails |

---

## Part 8: Visualization Plan

### Figure 1: Data Illustration
- Show example inputs with different active view combinations
- Illustrate the slot structure

### Figure 2: Theorem 1 Validation
- (a) Scatter: actual vs decomposed output
- (b) Heatmap: gating pattern distances

### Figure 3: Race Dynamics
- (a) Pathway strength vs epoch (hard labels)
- (b) Winner diversity across seeds
- (c) Prediction accuracy histogram

### Figure 4: KD Effect
- (a) Pathway strength vs epoch (KD)
- (b) Side-by-side: hard vs KD dynamics
- (c) Coverage bar chart

### Figure 5: Ablations
- (a) Temperature sweep
- (b) Self-distillation generations
- (c) Teacher coverage vs student coverage

---

## Part 9: Statistical Analysis

### Sample Sizes
- Primary experiments: N = 30 seeds
- Quick validations: N = 10 seeds
- Ablations: N = 20 seeds

### Statistical Tests
- Comparing two conditions: Two-sample t-test
- Comparing multiple conditions: ANOVA + post-hoc Tukey
- Correlation: Pearson r with p-value

### Reporting
- Mean ± std for all metrics
- 95% confidence intervals where appropriate
- Effect sizes (Cohen's d) for key comparisons

---

## Part 10: Compute Requirements

### Per Experiment Estimates

| Experiment | Seeds | Epochs | Est. Time (1 GPU) |
|------------|-------|--------|-------------------|
| 1.x | 10 | 100 | ~10 min |
| 2.1-2.2 | 30 | 100 | ~30 min |
| 2.3 | 1 | 200 | ~5 min |
| 2.4-2.6 | 50 | 100 | ~1 hour |
| 3.1-3.4 | 10 | 100 | ~30 min each |
| 3.5 | 10×6 | 100 | ~1 hour |
| 3.6 | 10×5 | 100 | ~1 hour |

**Total estimated time**: ~6-8 hours on single GPU

### Hardware
- Minimum: 1 GPU (RTX 3080 or equivalent)
- Preferred: Multi-GPU for parallel seeds

---

## Part 11: Potential Issues and Mitigations

### Issue 1: View features not orthogonal enough
**Mitigation**: Use Gram-Schmidt orthogonalization; verify before experiments

### Issue 2: Network too small/large
**Mitigation**: Run width ablation; report sensitivity

### Issue 3: Gating patterns change during training
**Mitigation**: Track gating stability; focus on stable regime

### Issue 4: Coverage measurement threshold sensitivity
**Mitigation**: Report results at multiple thresholds; use ROC-style analysis

### Issue 5: Temperature sensitivity in KD
**Mitigation**: Sweep temperatures; report best and sensitivity

---

## Next Steps

1. **Implement data generator**: Create `MultiViewDataset` class
2. **Implement measurement functions**: Coverage, pathway strength, gradients
3. **Run Theorem 1 experiments**: Validate decomposition
4. **Run Theorem 2 experiments**: Validate race dynamics
5. **Run Theorem 3 experiments**: Validate KD mechanism
6. **Generate visualizations**: All figures
7. **Statistical analysis**: All comparisons
