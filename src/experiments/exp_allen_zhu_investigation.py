"""
Investigation: Why Don't Teachers Naturally Develop Diversity?

Allen-Zhu et al. claim different seeds → different views learned.
Our experiments show all teachers converge to identical representations.

This script systematically investigates possible causes:
1. Signal asymmetry: Decaying signal strengths create a clear "best" view
2. Linear networks: GDLNs may have simpler loss landscapes than nonlinear networks
3. Loss function: CE may have different implicit biases than MSE
4. Network capacity: Small networks may not support multiple solutions
5. Initialization scale: Different init scales may affect view selection

For each hypothesis, we run controlled experiments to isolate the effect.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.statistics import format_result_with_stats, mean_ci


# =============================================================================
# Models
# =============================================================================

class LinearNet(nn.Module):
    """2-layer linear network (GDLN-like)."""
    def __init__(self, d_input, hidden, d_output, init_scale=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_input, hidden, bias=False)
        self.W2 = nn.Linear(hidden, d_output, bias=False)
        nn.init.orthogonal_(self.W1.weight, gain=init_scale)
        nn.init.orthogonal_(self.W2.weight, gain=init_scale)

    def forward(self, x):
        return self.W2(self.W1(x))


class NonlinearNet(nn.Module):
    """2-layer ReLU network."""
    def __init__(self, d_input, hidden, d_output, init_scale=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_input, hidden, bias=True)
        self.W2 = nn.Linear(hidden, d_output, bias=True)
        nn.init.kaiming_normal_(self.W1.weight)
        nn.init.kaiming_normal_(self.W2.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.zeros_(self.W2.bias)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


class DeepLinearNet(nn.Module):
    """4-layer linear network."""
    def __init__(self, d_input, hidden, d_output, init_scale=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_input, hidden, bias=False),
            nn.Linear(hidden, hidden, bias=False),
            nn.Linear(hidden, hidden, bias=False),
            nn.Linear(hidden, d_output, bias=False)
        )
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight, gain=init_scale)

    def forward(self, x):
        return self.layers(x)


class DeepNonlinearNet(nn.Module):
    """4-layer ReLU network."""
    def __init__(self, d_input, hidden, d_output, init_scale=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_output)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)


# =============================================================================
# Data Generation
# =============================================================================

class MultiViewDataset:
    """Multi-view dataset with configurable signal strengths."""

    def __init__(self, M=5, K=10, d_view=50, n_samples=5000,
                 signal_strengths=None, noise_scale=0.1, seed=42):
        self.M = M
        self.K = K
        self.d_view = d_view
        self.n_samples = n_samples
        self.noise_scale = noise_scale

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Default: decaying signal strengths
        if signal_strengths is None:
            self.signal_strengths = [0.5 ** m for m in range(M)]
        else:
            self.signal_strengths = signal_strengths

        self._generate_data()

    def _generate_data(self):
        total_dim = self.M * self.d_view

        # Generate class prototypes
        self.prototypes = {}
        for k in range(self.K):
            proto = {}
            for m in range(self.M):
                p = np.random.randn(self.d_view)
                proto[m] = p / np.linalg.norm(p)
            self.prototypes[k] = proto

        # Generate samples
        self.X = np.zeros((self.n_samples, total_dim))
        self.Y = np.zeros(self.n_samples, dtype=np.int64)

        samples_per_class = self.n_samples // self.K
        idx = 0
        for k in range(self.K):
            for _ in range(samples_per_class):
                if idx >= self.n_samples:
                    break
                x = np.zeros(total_dim)
                for m in range(self.M):
                    signal = self.prototypes[k][m] * self.signal_strengths[m]
                    noise = np.random.randn(self.d_view) * self.noise_scale
                    x[m * self.d_view:(m + 1) * self.d_view] = signal + noise
                self.X[idx] = x
                self.Y[idx] = k
                idx += 1

        # Shuffle
        perm = np.random.permutation(self.n_samples)
        self.X = self.X[perm]
        self.Y = self.Y[perm]

    def get_tensors(self):
        return torch.tensor(self.X, dtype=torch.float32), torch.tensor(self.Y, dtype=torch.long)


# =============================================================================
# Training and Metrics
# =============================================================================

def measure_view_contributions(model, M, d_view):
    """Measure per-view weight norms in first layer."""
    with torch.no_grad():
        # Get first layer weights
        if hasattr(model, 'W1'):
            W1 = model.W1.weight.cpu().numpy()
        elif hasattr(model, 'layers'):
            W1 = model.layers[0].weight.cpu().numpy()
        else:
            raise ValueError("Cannot find first layer")

        contributions = []
        for m in range(M):
            W1_view = W1[:, m * d_view:(m + 1) * d_view]
            contributions.append(np.linalg.norm(W1_view, 'fro'))
        total = sum(contributions)
        return [c / total for c in contributions]


def compute_effective_rank(contributions):
    """Effective rank from contribution distribution."""
    p = np.array(contributions)
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    return np.exp(entropy)


def train_model(model, X, Y, epochs, lr, loss_fn='ce', verbose=False):
    """Train a model and return final metrics."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if loss_fn == 'mse':
        num_classes = int(Y.max().item()) + 1
        Y_onehot = F.one_hot(Y, num_classes=num_classes).float()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, leave=False)

    for _ in iterator:
        model.train()
        optimizer.zero_grad()
        output = model(X)

        if loss_fn == 'ce':
            loss = F.cross_entropy(output, Y)
        else:
            loss = F.mse_loss(output, Y_onehot)

        loss.backward()
        optimizer.step()

    # Final accuracy
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        acc = (preds == Y).float().mean().item()

    return acc


def train_multiple_teachers(config, n_teachers=10, verbose=True):
    """Train multiple teachers and measure their view diversity."""
    results = {
        'view0_contributions': [],
        'effective_ranks': [],
        'accuracies': [],
        'all_contributions': []
    }

    dataset = MultiViewDataset(
        M=config['M'],
        K=config['K'],
        d_view=config['d_view'],
        n_samples=config['n_samples'],
        signal_strengths=config.get('signal_strengths'),
        seed=42
    )
    X, Y = dataset.get_tensors()

    for seed in range(n_teachers):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model
        ModelClass = config['model_class']
        model = ModelClass(
            d_input=config['M'] * config['d_view'],
            hidden=config['hidden'],
            d_output=config['K'],
            init_scale=config.get('init_scale', 0.1)
        )

        # Train
        acc = train_model(
            model, X, Y,
            epochs=config['epochs'],
            lr=config['lr'],
            loss_fn=config['loss_fn'],
            verbose=False
        )

        # Measure
        contrib = measure_view_contributions(model, config['M'], config['d_view'])
        eff_rank = compute_effective_rank(contrib)

        results['view0_contributions'].append(contrib[0])
        results['effective_ranks'].append(eff_rank)
        results['accuracies'].append(acc)
        results['all_contributions'].append(contrib)

        if verbose:
            print(f"  Teacher {seed}: acc={acc:.3f}, V0={contrib[0]:.3f}, rank={eff_rank:.2f}")

    # Compute diversity metric: std of View 0 contributions
    v0_std = np.std(results['view0_contributions'])
    v0_mean = np.mean(results['view0_contributions'])

    results['diversity_metric'] = v0_std
    results['v0_mean'] = v0_mean
    results['v0_std'] = v0_std

    return results


# =============================================================================
# Investigation 1: Signal Strength Asymmetry
# =============================================================================

def investigate_signal_asymmetry(n_teachers=10, verbose=True):
    """Test if equal signal strengths create more diversity."""
    print("\n" + "=" * 60)
    print("Investigation 1: Signal Strength Asymmetry")
    print("=" * 60)
    print("Hypothesis: Decaying signal strengths create a clear 'best' view,")
    print("preventing diversity. Equal strengths may allow different views to win.")

    results = {}

    # Baseline: decaying signals (0.5^m)
    print("\n--- Decaying Signal Strengths [1.0, 0.5, 0.25, 0.125, 0.0625] ---")
    config_decay = {
        'M': 5, 'K': 10, 'd_view': 50, 'hidden': 64, 'n_samples': 5000,
        'epochs': 1000, 'lr': 0.1, 'loss_fn': 'ce',
        'model_class': LinearNet,
        'signal_strengths': None  # Default decaying
    }
    results['decaying'] = train_multiple_teachers(config_decay, n_teachers, verbose)

    # Equal signals
    print("\n--- Equal Signal Strengths [1.0, 1.0, 1.0, 1.0, 1.0] ---")
    config_equal = config_decay.copy()
    config_equal['signal_strengths'] = [1.0] * 5
    results['equal'] = train_multiple_teachers(config_equal, n_teachers, verbose)

    # Reverse decay (weak view first)
    print("\n--- Reverse Decay [0.0625, 0.125, 0.25, 0.5, 1.0] ---")
    config_reverse = config_decay.copy()
    config_reverse['signal_strengths'] = [0.5 ** (4-m) for m in range(5)]
    results['reverse'] = train_multiple_teachers(config_reverse, n_teachers, verbose)

    # Summary
    print("\n--- Summary ---")
    for name, r in results.items():
        print(f"  {name}: V0 std = {r['v0_std']:.4f}, V0 mean = {r['v0_mean']:.3f}")

    return results


# =============================================================================
# Investigation 2: Network Architecture
# =============================================================================

def investigate_architecture(n_teachers=10, verbose=True):
    """Test if nonlinear networks develop more diversity."""
    print("\n" + "=" * 60)
    print("Investigation 2: Network Architecture")
    print("=" * 60)
    print("Hypothesis: Linear networks have simpler loss landscapes.")
    print("Nonlinear networks may have multiple basins → different views.")

    results = {}
    base_config = {
        'M': 5, 'K': 10, 'd_view': 50, 'hidden': 64, 'n_samples': 5000,
        'epochs': 1000, 'lr': 0.1, 'loss_fn': 'ce',
        'signal_strengths': [1.0] * 5  # Equal strengths to isolate architecture effect
    }

    architectures = [
        ('2-layer Linear', LinearNet),
        ('2-layer ReLU', NonlinearNet),
        ('4-layer Linear', DeepLinearNet),
        ('4-layer ReLU', DeepNonlinearNet),
    ]

    for name, model_class in architectures:
        print(f"\n--- {name} ---")
        config = base_config.copy()
        config['model_class'] = model_class
        results[name] = train_multiple_teachers(config, n_teachers, verbose)

    # Summary
    print("\n--- Summary ---")
    for name, r in results.items():
        print(f"  {name}: V0 std = {r['v0_std']:.4f}, V0 mean = {r['v0_mean']:.3f}")

    return results


# =============================================================================
# Investigation 3: Loss Function
# =============================================================================

def investigate_loss_function(n_teachers=10, verbose=True):
    """Test if loss function affects teacher diversity."""
    print("\n" + "=" * 60)
    print("Investigation 3: Loss Function")
    print("=" * 60)
    print("Hypothesis: CE loss creates sharper gradients that favor dominant views.")
    print("MSE may distribute gradients more evenly.")

    results = {}
    base_config = {
        'M': 5, 'K': 10, 'd_view': 50, 'hidden': 64, 'n_samples': 5000,
        'epochs': 1000, 'lr': 0.1,
        'model_class': LinearNet,
        'signal_strengths': [1.0] * 5
    }

    loss_functions = ['ce', 'mse']

    for loss_fn in loss_functions:
        print(f"\n--- Loss: {loss_fn.upper()} ---")
        config = base_config.copy()
        config['loss_fn'] = loss_fn
        # Adjust LR for MSE
        if loss_fn == 'mse':
            config['lr'] = 0.5
        results[loss_fn] = train_multiple_teachers(config, n_teachers, verbose)

    # Summary
    print("\n--- Summary ---")
    for name, r in results.items():
        print(f"  {name}: V0 std = {r['v0_std']:.4f}, V0 mean = {r['v0_mean']:.3f}, "
              f"avg rank = {np.mean(r['effective_ranks']):.2f}")

    return results


# =============================================================================
# Investigation 4: Network Capacity
# =============================================================================

def investigate_capacity(n_teachers=10, verbose=True):
    """Test if larger networks develop more diversity."""
    print("\n" + "=" * 60)
    print("Investigation 4: Network Capacity")
    print("=" * 60)
    print("Hypothesis: Small networks may not support multiple solutions.")
    print("Larger capacity may allow different initializations to find different minima.")

    results = {}
    base_config = {
        'M': 5, 'K': 10, 'd_view': 50, 'n_samples': 5000,
        'epochs': 1000, 'lr': 0.1, 'loss_fn': 'ce',
        'model_class': LinearNet,
        'signal_strengths': [1.0] * 5
    }

    hidden_sizes = [16, 64, 256, 512]

    for hidden in hidden_sizes:
        print(f"\n--- Hidden size: {hidden} ---")
        config = base_config.copy()
        config['hidden'] = hidden
        results[f'hidden_{hidden}'] = train_multiple_teachers(config, n_teachers, verbose)

    # Summary
    print("\n--- Summary ---")
    for name, r in results.items():
        print(f"  {name}: V0 std = {r['v0_std']:.4f}, V0 mean = {r['v0_mean']:.3f}")

    return results


# =============================================================================
# Investigation 5: Initialization Scale
# =============================================================================

def investigate_initialization(n_teachers=10, verbose=True):
    """Test if initialization scale affects view selection."""
    print("\n" + "=" * 60)
    print("Investigation 5: Initialization Scale")
    print("=" * 60)
    print("Hypothesis: Different init scales may break symmetry differently,")
    print("allowing different views to win based on initial random structure.")

    results = {}
    base_config = {
        'M': 5, 'K': 10, 'd_view': 50, 'hidden': 64, 'n_samples': 5000,
        'epochs': 1000, 'lr': 0.1, 'loss_fn': 'ce',
        'model_class': LinearNet,
        'signal_strengths': [1.0] * 5
    }

    init_scales = [0.01, 0.1, 1.0, 2.0]

    for scale in init_scales:
        print(f"\n--- Init scale: {scale} ---")
        config = base_config.copy()
        config['init_scale'] = scale
        results[f'scale_{scale}'] = train_multiple_teachers(config, n_teachers, verbose)

    # Summary
    print("\n--- Summary ---")
    for name, r in results.items():
        print(f"  {name}: V0 std = {r['v0_std']:.4f}, V0 mean = {r['v0_mean']:.3f}")

    return results


# =============================================================================
# Investigation 6: Learning Rate
# =============================================================================

def investigate_learning_rate(n_teachers=10, verbose=True):
    """Test if learning rate affects diversity."""
    print("\n" + "=" * 60)
    print("Investigation 6: Learning Rate")
    print("=" * 60)
    print("Hypothesis: Higher LR may create more stochasticity in training,")
    print("allowing different seeds to find different solutions.")

    results = {}
    base_config = {
        'M': 5, 'K': 10, 'd_view': 50, 'hidden': 64, 'n_samples': 5000,
        'epochs': 1000, 'loss_fn': 'ce',
        'model_class': LinearNet,
        'signal_strengths': [1.0] * 5
    }

    learning_rates = [0.01, 0.1, 0.5, 1.0]

    for lr in learning_rates:
        print(f"\n--- Learning rate: {lr} ---")
        config = base_config.copy()
        config['lr'] = lr
        results[f'lr_{lr}'] = train_multiple_teachers(config, n_teachers, verbose)

    # Summary
    print("\n--- Summary ---")
    for name, r in results.items():
        avg_acc = np.mean(r['accuracies'])
        print(f"  {name}: V0 std = {r['v0_std']:.4f}, V0 mean = {r['v0_mean']:.3f}, "
              f"avg acc = {avg_acc:.3f}")

    return results


# =============================================================================
# Main
# =============================================================================

def run_all_investigations(n_teachers=10, output_dir='../log/results/allen_zhu_investigation'):
    """Run all investigations and compile results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'n_teachers': n_teachers
    }

    # Run investigations
    all_results['signal_asymmetry'] = investigate_signal_asymmetry(n_teachers)
    all_results['architecture'] = investigate_architecture(n_teachers)
    all_results['loss_function'] = investigate_loss_function(n_teachers)
    all_results['capacity'] = investigate_capacity(n_teachers)
    all_results['initialization'] = investigate_initialization(n_teachers)
    all_results['learning_rate'] = investigate_learning_rate(n_teachers)

    # Save results
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif callable(obj):
            return str(obj)
        else:
            return obj

    results_file = output_path / 'allen_zhu_investigation.json'
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {results_file}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: What Creates Teacher Diversity?")
    print("=" * 70)

    print("\n1. Signal Asymmetry:")
    for name, r in all_results['signal_asymmetry'].items():
        print(f"   {name}: V0 std = {r['v0_std']:.4f}")

    print("\n2. Architecture:")
    for name, r in all_results['architecture'].items():
        print(f"   {name}: V0 std = {r['v0_std']:.4f}")

    print("\n3. Loss Function:")
    for name, r in all_results['loss_function'].items():
        print(f"   {name}: V0 std = {r['v0_std']:.4f}")

    print("\n4. Capacity:")
    for name, r in all_results['capacity'].items():
        print(f"   {name}: V0 std = {r['v0_std']:.4f}")

    print("\n5. Initialization:")
    for name, r in all_results['initialization'].items():
        print(f"   {name}: V0 std = {r['v0_std']:.4f}")

    print("\n6. Learning Rate:")
    for name, r in all_results['learning_rate'].items():
        print(f"   {name}: V0 std = {r['v0_std']:.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Allen-Zhu Discrepancy Investigation')
    parser.add_argument('--n-teachers', type=int, default=10, help='Teachers per condition')
    parser.add_argument('--output', type=str, default='../log/results/allen_zhu_investigation')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer teachers)')
    args = parser.parse_args()

    if args.quick:
        args.n_teachers = 5
        print("Quick mode: 5 teachers per condition")

    run_all_investigations(n_teachers=args.n_teachers, output_dir=args.output)


if __name__ == '__main__':
    main()
