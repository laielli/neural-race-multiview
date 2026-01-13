# Instructions for ML Engineer: Initial Experiments

**Date**: 2026-01-10
**From**: Advisor
**Priority**: High

---

## Objective

Implement and run the **Tier 1 synthetic experiments** to validate the core theoretical claims before proofs are finalized. Results will:
1. Build confidence that the theory is on the right track
2. Identify any issues early (before investing in proofs)
3. Provide concrete figures for the paper sketch

---

## Source Document

The full experimental design is in `../advisor/initial_plan/Synthetic Experiments - Validation Plan.md`. This document provides implementation-ready specifications. **Read it thoroughly.**

---

## Implementation Priorities

### Phase 1: Infrastructure (Do First)

| Component | Priority | Description |
|-----------|----------|-------------|
| Multi-view data generator | **P0** | Core dataset class |
| Network architecture | **P0** | Two-layer MLP with gating extraction |
| Pathway strength measurement | **P0** | Core metric for all experiments |
| View coverage measurement | **P0** | Core metric for KD experiments |
| Training loops (hard + KD) | **P0** | Basic training infrastructure |

### Phase 2: Theorem 2 Experiments (Validate Race Dynamics)

| Experiment | Priority | Validates |
|------------|----------|-----------|
| 2.1: Single-view convergence | **P1** | C(f) ≈ 1/M after hard label training |
| 2.3: Race dynamics visualization | **P1** | Winner-take-all pathway evolution |
| 2.2: View diversity across seeds | **P2** | Different seeds → different views |
| 2.4: Winner prediction | **P2** | Initial advantage predicts winner |

### Phase 3: Theorem 3 Experiments (Validate KD Mechanism)

| Experiment | Priority | Validates |
|------------|----------|-----------|
| 3.1: Multi-view learning via KD | **P1** | C(student_KD) ≈ C(ensemble) |
| 3.3: Pathway evolution under KD | **P1** | Multiple pathways survive |
| 3.2: Gradient distribution | **P2** | Soft labels distribute gradient |

### Phase 4: Theorem 1 Experiments (Validate Decomposition)

| Experiment | Priority | Validates |
|------------|----------|-----------|
| 1.1: Output decomposition | **P3** | f(x) ≈ Σ R_{y,m} |
| 1.2: Gating pattern distinctness | **P3** | Different views → different gating |

---

## Technical Specifications

### 1. Multi-View Data Generator

```python
class MultiViewDataset:
    """
    Synthetic multi-view dataset where ground truth is exactly known.

    Each class has M views, each view occupies a disjoint "slot" in input space.
    This ensures perfect orthogonality between views.
    """

    def __init__(
        self,
        K: int = 10,           # Number of classes
        M: int = 3,            # Views per class
        d_view: int = 50,      # Dimensions per view slot
        view_prob: float = 0.5, # Probability each view is active
        noise_std: float = 0.1, # Gaussian noise level
        n_samples: int = 10000, # Training samples
        seed: int = 42
    ):
        self.K = K
        self.M = M
        self.d = M * d_view  # Total input dimension = 150
        self.d_view = d_view
        self.view_prob = view_prob
        self.noise_std = noise_std

        # Generate view features: phi[(y, m)] is the feature vector for class y, view m
        self.phi = self._generate_view_features(seed)

        # Generate dataset
        self.data = self._generate_samples(n_samples, seed)

    def _generate_view_features(self, seed):
        """Generate orthogonal view features in disjoint slots."""
        torch.manual_seed(seed)
        phi = {}

        for y in range(self.K):
            for m in range(self.M):
                # Create vector that is non-zero only in slot m
                v = torch.zeros(self.d)
                slot_start = m * self.d_view
                slot_end = (m + 1) * self.d_view

                # Random unit vector in the slot
                v[slot_start:slot_end] = torch.randn(self.d_view)
                v = v / v.norm()

                phi[(y, m)] = v

        return phi

    def _generate_samples(self, n_samples, seed):
        """Generate (x, y, active_views) tuples."""
        torch.manual_seed(seed + 1000)  # Different seed from features
        samples = []

        for _ in range(n_samples):
            y = torch.randint(0, self.K, (1,)).item()

            # Sample active views
            active_views = []
            for m in range(self.M):
                if torch.rand(1).item() < self.view_prob:
                    active_views.append(m)

            # Ensure at least one view
            if len(active_views) == 0:
                active_views = [torch.randint(0, self.M, (1,)).item()]

            # Generate input
            x = torch.zeros(self.d)
            for m in active_views:
                x = x + self.phi[(y, m)]
            x = x + self.noise_std * torch.randn(self.d)

            samples.append((x, y, active_views))

        return samples

    def get_view_feature(self, y, m):
        """Return the pure feature vector for view (y, m)."""
        return self.phi[(y, m)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

**Verification checks** (run before experiments):
```python
def verify_dataset(dataset):
    """Verify dataset properties."""
    # 1. Check orthogonality
    max_dot = 0
    for (y1, m1), v1 in dataset.phi.items():
        for (y2, m2), v2 in dataset.phi.items():
            if (y1, m1) != (y2, m2):
                dot = abs(torch.dot(v1, v2).item())
                max_dot = max(max_dot, dot)
    print(f"Max dot product between views: {max_dot:.6f} (should be ~0)")

    # 2. Check view sufficiency (linear classifier on single view)
    # Train linear probe on single-view samples, should get >95% accuracy

    # 3. Check dimensions
    print(f"Input dimension: {dataset.d} (expected {dataset.M * dataset.d_view})")
    print(f"Classes: {dataset.K}, Views per class: {dataset.M}")
```

---

### 2. Network Architecture

```python
class MultiViewNet(nn.Module):
    """
    Two-layer MLP for multi-view experiments.
    Includes methods to extract gating patterns and measure pathway strengths.
    """

    def __init__(self, d: int = 150, hidden: int = 200, K: int = 10):
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.K = K

        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, K)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def get_gating_pattern(self, x):
        """
        Return binary gating pattern for input x.
        Shape: (batch, hidden) or (hidden,) if single input
        """
        h_pre = self.fc1(x)
        return (h_pre > 0).float()

    def forward_with_gating(self, x, gating):
        """
        Forward pass with externally specified gating pattern.
        Used to compute pathway-specific outputs.
        """
        h_pre = self.fc1(x)
        h = h_pre * gating  # Apply external gating
        return self.fc2(h)

    def get_view_response(self, phi_ym):
        """
        Compute R_{y,m} = network output when only view (y,m) is present.
        This is the "view response" from Theorem 1.

        Args:
            phi_ym: The pure view feature vector, shape (d,)

        Returns:
            R_ym: Output logits, shape (K,)
        """
        with torch.no_grad():
            x = phi_ym.unsqueeze(0)  # Add batch dim
            return self(x).squeeze(0)

    def get_pathway_strength(self, phi_ym):
        """
        Compute pathway strength s_{y,m} = ||R_{y,m}||.

        This is a simplified measure. For full Frobenius norm of pathway matrix,
        see get_pathway_matrix_strength().
        """
        R_ym = self.get_view_response(phi_ym)
        return R_ym.norm().item()

    def get_pathway_matrix_strength(self, phi_ym):
        """
        Compute ||P_{y,m}||_F where P_{y,m} = W2 @ diag(g) @ W1.

        This is the full pathway strength from Theorem 2.
        """
        with torch.no_grad():
            x = phi_ym.unsqueeze(0)
            g = self.get_gating_pattern(x).squeeze(0)  # (hidden,)

            # P = W2 @ diag(g) @ W1
            # ||P||_F can be computed as ||W2 @ diag(g) @ W1||_F
            W1 = self.fc1.weight  # (hidden, d)
            W2 = self.fc2.weight  # (K, hidden)

            # Apply gating to W1 rows
            W1_gated = W1 * g.unsqueeze(1)  # (hidden, d)
            P = W2 @ W1_gated  # (K, d)

            return P.norm().item()
```

---

### 3. Measurement Functions

```python
def measure_all_pathway_strengths(model, dataset):
    """
    Measure pathway strength for all (class, view) pairs.

    Returns:
        dict: {(y, m): strength} for all y in [K], m in [M]
    """
    strengths = {}
    for y in range(dataset.K):
        for m in range(dataset.M):
            phi_ym = dataset.get_view_feature(y, m)
            strengths[(y, m)] = model.get_pathway_strength(phi_ym)
    return strengths


def measure_view_coverage(model, dataset, threshold: float = 0.5):
    """
    Measure fraction of views the network correctly classifies.

    A view (y, m) is "covered" if:
    1. The network classifies phi_{y,m} correctly (argmax = y)
    2. With confidence > threshold

    Returns:
        float: Coverage in [0, 1]
    """
    detected = 0
    total = dataset.K * dataset.M

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


def measure_view_coverage_detailed(model, dataset, threshold: float = 0.5):
    """
    Detailed coverage measurement returning per-class breakdown.

    Returns:
        dict: {
            'total_coverage': float,
            'per_class': {y: [list of detected views]},
            'detection_matrix': np.array of shape (K, M)
        }
    """
    detection_matrix = np.zeros((dataset.K, dataset.M))
    per_class = {y: [] for y in range(dataset.K)}

    for y in range(dataset.K):
        for m in range(dataset.M):
            phi_ym = dataset.get_view_feature(y, m)

            with torch.no_grad():
                logits = model(phi_ym.unsqueeze(0)).squeeze(0)
                probs = F.softmax(logits, dim=0)

            pred = logits.argmax().item()
            conf = probs[y].item()

            if pred == y and conf > threshold:
                detection_matrix[y, m] = 1
                per_class[y].append(m)

    return {
        'total_coverage': detection_matrix.mean(),
        'per_class': per_class,
        'detection_matrix': detection_matrix
    }


def find_winning_view(model, dataset, y):
    """
    Find which view "won" for class y (has highest pathway strength).

    Returns:
        int: winning view index m*
    """
    strengths = []
    for m in range(dataset.M):
        phi_ym = dataset.get_view_feature(y, m)
        strengths.append(model.get_pathway_strength(phi_ym))
    return np.argmax(strengths)
```

---

### 4. Training Functions

```python
def train_hard_labels(
    model,
    dataset,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 128,
    log_interval: int = 10,
    track_pathways: bool = False,
    track_classes: list = None  # Which classes to track pathway strengths for
):
    """
    Train with standard cross-entropy (hard labels).

    Args:
        track_pathways: If True, record pathway strengths during training
        track_classes: List of class indices to track (default: [0])

    Returns:
        dict: Training history including loss, accuracy, and optionally pathway strengths
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    if track_classes is None:
        track_classes = [0]

    history = {
        'loss': [],
        'accuracy': [],
        'pathway_strengths': {y: {m: [] for m in range(dataset.M)} for y in track_classes}
    }

    # Create data loader
    X = torch.stack([s[0] for s in dataset.data])
    Y = torch.tensor([s[1] for s in dataset.data])

    for epoch in range(epochs):
        model.train()

        # Shuffle
        perm = torch.randperm(len(X))
        X_shuf, Y_shuf = X[perm], Y[perm]

        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            x_batch = X_shuf[i:i+batch_size]
            y_batch = Y_shuf[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(X)
                acc = (logits.argmax(dim=1) == Y).float().mean().item()

            history['loss'].append(epoch_loss / (len(X) / batch_size))
            history['accuracy'].append(acc)

            if track_pathways:
                for y in track_classes:
                    for m in range(dataset.M):
                        phi_ym = dataset.get_view_feature(y, m)
                        s = model.get_pathway_strength(phi_ym)
                        history['pathway_strengths'][y][m].append(s)

    return history


def train_kd(
    student,
    teachers,  # List of teacher models
    dataset,
    epochs: int = 100,
    temperature: float = 4.0,
    lr: float = 0.01,
    batch_size: int = 128,
    log_interval: int = 10,
    track_pathways: bool = False,
    track_classes: list = None
):
    """
    Train student via knowledge distillation from teacher ensemble.

    Uses KL divergence between student and ensemble-averaged teacher soft labels.
    """
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)

    if track_classes is None:
        track_classes = [0]

    history = {
        'loss': [],
        'accuracy': [],
        'pathway_strengths': {y: {m: [] for m in range(dataset.M)} for y in track_classes}
    }

    X = torch.stack([s[0] for s in dataset.data])
    Y = torch.tensor([s[1] for s in dataset.data])

    # Set teachers to eval mode
    for t in teachers:
        t.eval()

    for epoch in range(epochs):
        student.train()

        perm = torch.randperm(len(X))
        X_shuf, Y_shuf = X[perm], Y[perm]

        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            x_batch = X_shuf[i:i+batch_size]

            # Get teacher ensemble prediction (soft labels)
            with torch.no_grad():
                teacher_logits = torch.stack([t(x_batch) for t in teachers]).mean(dim=0)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

            # Student prediction
            student_logits = student(x_batch)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

            # KD loss (KL divergence)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            loss = loss * (temperature ** 2)  # Scale gradient magnitude

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            student.eval()
            with torch.no_grad():
                logits = student(X)
                acc = (logits.argmax(dim=1) == Y).float().mean().item()

            history['loss'].append(epoch_loss / (len(X) / batch_size))
            history['accuracy'].append(acc)

            if track_pathways:
                for y in track_classes:
                    for m in range(dataset.M):
                        phi_ym = dataset.get_view_feature(y, m)
                        s = student.get_pathway_strength(phi_ym)
                        history['pathway_strengths'][y][m].append(s)

    return history
```

---

### 5. Experiment Scripts

#### Experiment 2.1: Single-View Convergence

```python
def experiment_2_1_single_view_convergence(num_seeds: int = 30):
    """
    Validate that hard label training leads to C(f) ≈ 1/M.

    Expected result: mean coverage ≈ 0.33 for M=3
    """
    results = []

    for seed in range(num_seeds):
        # Create fresh dataset and model
        dataset = MultiViewDataset(seed=seed)
        model = MultiViewNet(d=dataset.d, K=dataset.K)

        # Initialize with different seed
        torch.manual_seed(seed + 5000)
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        # Train
        train_hard_labels(model, dataset, epochs=100)

        # Measure coverage
        coverage = measure_view_coverage(model, dataset)
        results.append(coverage)

        print(f"Seed {seed}: coverage = {coverage:.3f}")

    print(f"\n=== Results ===")
    print(f"Mean coverage: {np.mean(results):.3f} ± {np.std(results):.3f}")
    print(f"Expected (1/M): {1/3:.3f}")

    return results
```

#### Experiment 2.3: Race Dynamics Visualization

```python
def experiment_2_3_race_visualization(seed: int = 0, y_target: int = 0, epochs: int = 200):
    """
    Visualize pathway strengths over training to show winner-take-all dynamics.

    Expected: One pathway grows to dominate, others decay.
    """
    dataset = MultiViewDataset(seed=seed)
    model = MultiViewNet(d=dataset.d, K=dataset.K)

    torch.manual_seed(seed + 5000)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    # Train with pathway tracking
    history = train_hard_labels(
        model, dataset,
        epochs=epochs,
        log_interval=5,
        track_pathways=True,
        track_classes=[y_target]
    )

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for m in range(dataset.M):
        strengths = history['pathway_strengths'][y_target][m]
        epochs_logged = np.linspace(0, epochs, len(strengths))
        plt.plot(epochs_logged, strengths, label=f'View {m}', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Pathway Strength')
    plt.title(f'Race Dynamics (Class {y_target}, Hard Labels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('race_dynamics_hard.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: race_dynamics_hard.png")

    return history
```

#### Experiment 3.1: KD Coverage Transfer

```python
def experiment_3_1_kd_coverage(num_teachers: int = 5, num_trials: int = 10):
    """
    Compare coverage: hard labels vs KD from ensemble.

    Expected: C(student_KD) ≈ C(ensemble) >> C(student_hard)
    """
    results = {
        'hard_label': [],
        'kd': [],
        'ensemble': []
    }

    for trial in range(num_trials):
        print(f"\n=== Trial {trial} ===")

        # Shared dataset
        dataset = MultiViewDataset(seed=trial)

        # Train teacher ensemble
        teachers = []
        for i in range(num_teachers):
            t = MultiViewNet(d=dataset.d, K=dataset.K)
            torch.manual_seed(trial * 100 + i)
            t.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            train_hard_labels(t, dataset, epochs=100)
            teachers.append(t)

            t_cov = measure_view_coverage(t, dataset)
            print(f"  Teacher {i} coverage: {t_cov:.3f}")

        # Measure ensemble coverage
        ensemble_cov = measure_ensemble_coverage(teachers, dataset)
        results['ensemble'].append(ensemble_cov)
        print(f"  Ensemble coverage: {ensemble_cov:.3f}")

        # Train student with hard labels (same init as KD student for fair comparison)
        student_hard = MultiViewNet(d=dataset.d, K=dataset.K)
        torch.manual_seed(trial * 1000)
        student_hard.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        train_hard_labels(student_hard, dataset, epochs=100)
        hard_cov = measure_view_coverage(student_hard, dataset)
        results['hard_label'].append(hard_cov)
        print(f"  Student (hard) coverage: {hard_cov:.3f}")

        # Train student with KD (same init)
        student_kd = MultiViewNet(d=dataset.d, K=dataset.K)
        torch.manual_seed(trial * 1000)  # Same init!
        student_kd.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        train_kd(student_kd, teachers, dataset, epochs=100, temperature=4.0)
        kd_cov = measure_view_coverage(student_kd, dataset)
        results['kd'].append(kd_cov)
        print(f"  Student (KD) coverage: {kd_cov:.3f}")

    print(f"\n=== Summary ===")
    print(f"Hard label: {np.mean(results['hard_label']):.3f} ± {np.std(results['hard_label']):.3f}")
    print(f"KD:         {np.mean(results['kd']):.3f} ± {np.std(results['kd']):.3f}")
    print(f"Ensemble:   {np.mean(results['ensemble']):.3f} ± {np.std(results['ensemble']):.3f}")

    return results


def measure_ensemble_coverage(teachers, dataset, threshold=0.5):
    """Measure coverage of teacher ensemble."""
    detected = 0
    total = dataset.K * dataset.M

    for y in range(dataset.K):
        for m in range(dataset.M):
            phi_ym = dataset.get_view_feature(y, m)
            x = phi_ym.unsqueeze(0)

            with torch.no_grad():
                logits = torch.stack([t(x) for t in teachers]).mean(dim=0).squeeze(0)
                probs = F.softmax(logits, dim=0)

            pred = logits.argmax().item()
            conf = probs[y].item()

            if pred == y and conf > threshold:
                detected += 1

    return detected / total
```

#### Experiment 3.3: Pathway Evolution Under KD

```python
def experiment_3_3_pathway_evolution_comparison(seed: int = 0, y_target: int = 0, epochs: int = 200):
    """
    Compare pathway evolution: hard labels vs KD.

    Expected:
    - Hard labels: one pathway dominates
    - KD: multiple pathways survive
    """
    dataset = MultiViewDataset(seed=seed)

    # Train teachers
    teachers = []
    for i in range(5):
        t = MultiViewNet(d=dataset.d, K=dataset.K)
        torch.manual_seed(seed * 100 + i)
        t.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        train_hard_labels(t, dataset, epochs=100)
        teachers.append(t)

    # Train with hard labels
    model_hard = MultiViewNet(d=dataset.d, K=dataset.K)
    torch.manual_seed(seed + 5000)
    model_hard.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    history_hard = train_hard_labels(
        model_hard, dataset, epochs=epochs, log_interval=5,
        track_pathways=True, track_classes=[y_target]
    )

    # Train with KD (same init)
    model_kd = MultiViewNet(d=dataset.d, K=dataset.K)
    torch.manual_seed(seed + 5000)  # Same init!
    model_kd.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    history_kd = train_kd(
        model_kd, teachers, dataset, epochs=epochs, log_interval=5,
        track_pathways=True, track_classes=[y_target]
    )

    # Plot side by side
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hard labels
    ax = axes[0]
    for m in range(dataset.M):
        strengths = history_hard['pathway_strengths'][y_target][m]
        epochs_logged = np.linspace(0, epochs, len(strengths))
        ax.plot(epochs_logged, strengths, label=f'View {m}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pathway Strength')
    ax.set_title('Hard Labels (Winner-Take-All)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KD
    ax = axes[1]
    for m in range(dataset.M):
        strengths = history_kd['pathway_strengths'][y_target][m]
        epochs_logged = np.linspace(0, epochs, len(strengths))
        ax.plot(epochs_logged, strengths, label=f'View {m}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pathway Strength')
    ax.set_title('Knowledge Distillation (Multiple Views)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pathway_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: pathway_comparison.png")

    return history_hard, history_kd
```

---

## Expected Results

### Theorem 2 Experiments

| Experiment | Metric | Expected Value | Tolerance |
|------------|--------|----------------|-----------|
| 2.1 | Mean coverage (hard labels) | ~0.33 | ±0.05 |
| 2.3 | Pathway dynamics | Winner-take-all pattern | Visual |
| 2.2 | View diversity across seeds | >0.6 | — |
| 2.4 | Winner prediction accuracy | >80% | — |

### Theorem 3 Experiments

| Experiment | Metric | Expected Value | Tolerance |
|------------|--------|----------------|-----------|
| 3.1 | Coverage (KD) | ~= ensemble | ±0.1 |
| 3.1 | Coverage (hard) | ~0.33 | ±0.05 |
| 3.3 | Final pathway count | >1 under KD | Visual |

---

## Deliverables

### Required Outputs

1. **Code**: Clean, documented Python files
   - `data.py` — MultiViewDataset class
   - `model.py` — MultiViewNet class
   - `metrics.py` — measurement functions
   - `train.py` — training functions
   - `experiments.py` — experiment scripts
   - `run_all.py` — main entry point

2. **Results**: Place in `to_advisor/` directory
   - `results_exp_2_1.json` — single-view convergence
   - `results_exp_3_1.json` — KD coverage comparison
   - `race_dynamics_hard.png` — Figure for paper
   - `pathway_comparison.png` — Figure for paper
   - `experiment_summary.md` — written summary

3. **Summary report** should include:
   - Which experiments passed/failed expectations
   - Any unexpected findings
   - Suggested parameter adjustments
   - Recommendations for next steps

---

## Compute Requirements

- **Hardware**: Single GPU (RTX 3080 or equivalent) sufficient
- **Estimated time**: ~2-4 hours for all P1 experiments
- **Memory**: <8GB GPU memory

---

## Implementation Notes

1. **Reproducibility**: Always set seeds. Record all hyperparameters.

2. **Logging**: Use `wandb` or simple JSON logging for results.

3. **Debugging**: Start with smaller K=3, M=2 to verify code works.

4. **Numerical stability**: Add small epsilon to norms if needed.

5. **Gating patterns**: The gating depends on initialization. Verify gating is consistent for same input.

---

## Questions?

If anything is unclear or you hit unexpected issues:
1. Check the full spec in `../advisor/initial_plan/Synthetic Experiments - Validation Plan.md`
2. Send questions via `to_advisor/`
3. Proceed with best judgment for minor decisions

**Priority**: Get P1 experiments running first. P2/P3 can wait.
