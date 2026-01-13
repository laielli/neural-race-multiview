# Configuration Reference

> **Purpose**: Document all experimental parameters and their rationale
> **Default values**: Chosen for reproducibility and theoretical validity

---

## Data Parameters

### Core Structure

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Classes | $K$ | 10 | Number of classification classes |
| Views per class | $M$ | 3 | Views available for each class |
| View dimension | $d_{\text{view}}$ | 50 | Dimensions per view slot |
| Total dimension | $d$ | 150 | Input dimension ($= M \times d_{\text{view}}$) |

### Sampling

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| View probability | $p$ | 0.5 | Probability each view is active |
| Noise std | $\sigma$ | 0.1 | Gaussian noise level |
| Training samples | $N$ | 10,000 | Number of training examples |

### Rationale

**K = 10**:
- Enough classes for meaningful diversity
- Not so many that training becomes expensive
- Standard benchmark size (like MNIST)

**M = 3**:
- Tractable for analysis (can visualize all views)
- Non-trivial multi-view structure
- $1/M = 0.33$ is clearly distinguishable from random

**d_view = 50**:
- Rich enough for expressive features
- Total $d = 150$ is manageable
- Sufficient capacity for learning distinct patterns

**p = 0.5**:
- Balanced multi-view structure
- Most samples have 1-2 active views
- Some samples have all 3 views

**noise_std = 0.1**:
- Low noise regime (views are clearly distinguishable)
- Non-zero noise prevents perfect memorization
- Realistic training scenario

---

## Network Parameters

### Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| Hidden dimension | 200 | Width of hidden layer |
| Depth | 2 | Number of layers (input → hidden → output) |
| Activation | ReLU | Nonlinearity |
| Initialization | Xavier | Weight initialization scheme |

### Rationale

**Hidden = 200**:
- Large enough for multi-view capacity
- Not so large that training is slow
- $\approx 1.3 \times d$ provides sufficient capacity

**Depth = 2**:
- Matches theoretical analysis (GDLN formalism)
- Sufficient for learning synthetic features
- Allows clear pathway analysis

---

## Training Parameters

### Hard Label Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| Optimizer | SGD | Stochastic gradient descent |
| Learning rate | 0.01 | Step size |
| Momentum | 0.9 | SGD momentum |
| Batch size | 128 | Samples per gradient step |
| Epochs | 100 | Training iterations |

### KD Training

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Temperature | $\tau$ | 4.0 | Soft label temperature |
| Loss | KL Divergence | $\tau^2 \cdot \text{KL}(p_T \| p_S)$ | KD loss function |

### Rationale

**lr = 0.01, momentum = 0.9**:
- Standard SGD hyperparameters
- Fast enough convergence
- Stable training dynamics

**epochs = 100**:
- Sufficient for convergence
- Long enough for winner-take-all to emerge
- Not so long that overfitting occurs

**temperature = 4.0**:
- Sweet spot for view information transfer
- Not so high that class information is lost
- Not so low that it approximates hard labels
- Standard in KD literature (Hinton et al.)

---

## Experiment-Specific Parameters

### Experiment 2.1: Single-View Convergence

```python
{
    'num_seeds': 30,      # Statistical power
    'K': 10,
    'M': 3,
    'd_view': 50,
    'hidden': 200,
    'epochs': 100,
    'lr': 0.01,
    'batch_size': 128,
    'n_samples': 10000
}
```

### Experiment 2.3: Race Dynamics

```python
{
    'seed': 0,            # Reproducibility
    'y_target': 0,        # Class to visualize
    'epochs': 200,        # Longer for dynamics
    'log_interval': 5,    # Track every 5 epochs
    # ... same data/network params
}
```

### Experiment 3.1: KD Coverage

```python
{
    'num_teachers': 5,    # Ensemble size
    'num_trials': 10,     # Statistical replicates
    'temperature': 4.0,   # KD temperature
    # ... same data/network params
}
```

### Experiment 3.3: Pathway Evolution

```python
{
    'seed': 0,
    'y_target': 0,
    'num_teachers': 5,
    'epochs': 200,        # Longer for dynamics
    'temperature': 4.0,
    'log_interval': 5
}
```

---

## Quick Test Parameters

For rapid validation before full runs:

```python
quick_params = {
    'num_seeds': 3,       # Reduced from 30
    'num_trials': 3,      # Reduced from 10
    'num_teachers': 3,    # Reduced from 5
    'epochs': 50,         # Reduced from 100
    'epochs_visual': 100  # Reduced from 200
}
```

Usage:
```bash
python experiments/run_all.py --quick
```

---

## Threshold Parameters

### Coverage Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| Confidence threshold | 0.5 | Min probability for "detected" |

A view $(y, m)$ is detected if:
1. $\arg\max_c p_c = y$ (correct class)
2. $p_y > 0.5$ (sufficient confidence)

### Pathway Survival

| Parameter | Default | Description |
|-----------|---------|-------------|
| Threshold ratio | 0.1 | Fraction of max strength |

A pathway survives if:
$$s_{y,m} > 0.1 \times \max_{m'} s_{y,m'}$$

---

## Parameter Sensitivity

### Critical Parameters

**Most sensitive**:
- `temperature`: Strongly affects view transfer
- `epochs`: Must be long enough for convergence
- `hidden`: Too small → underfitting; too large → slower convergence

**Moderately sensitive**:
- `lr`: Standard range works well
- `num_seeds`: More seeds → tighter confidence intervals

**Least sensitive**:
- `batch_size`: Standard values work
- `noise_std`: Low values all work similarly
- `view_prob`: 0.3-0.7 all give similar results

### Recommended Ablations

1. **Temperature sweep**: $\tau \in \{1, 2, 4, 8, 16\}$
2. **Hidden width**: $h \in \{100, 200, 400\}$
3. **View probability**: $p \in \{0.3, 0.5, 0.7\}$
4. **Number of teachers**: $N_T \in \{1, 3, 5, 10\}$

---

## Command Line Arguments

### Common Arguments

```
--seeds N       Number of random seeds (default: 30)
--epochs N      Training epochs (default: 100)
--quick         Use quick test parameters
--output DIR    Output directory (default: results/)
```

### Experiment-Specific

```
# exp_2_1
--seeds N       Number of seeds to test

# exp_2_3
--seed N        Random seed
--class N       Target class to visualize
--multi N       Run for N seeds with combined figure

# exp_3_1
--teachers N    Number of teachers in ensemble
--trials N      Number of trials
--temperature F KD temperature

# exp_3_3
--seed N        Random seed
--class N       Target class
--teachers N    Number of teachers
--multi-class N Run for multiple classes
```

---

## Reproducibility

### Setting Seeds

```python
# Dataset generation
dataset = MultiViewDataset(
    ...,
    seed=42  # Controls view features and samples
)

# Model initialization
torch.manual_seed(seed + 5000)  # Offset to avoid correlation
model = MultiViewNet(...)
model.apply(init_weights)
```

### Full Reproducibility

For exact reproduction:
1. Fix all random seeds
2. Use deterministic algorithms: `torch.use_deterministic_algorithms(True)`
3. Pin library versions (see requirements.txt)

### Known Sources of Non-Determinism

- Different GPU hardware may give slightly different results
- Multi-threading in data loading
- cuDNN autotuning
