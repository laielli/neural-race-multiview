"""
Experiment 2: Gradient Singular Value Analysis

Compares gradient SVD structure under hard labels vs knowledge distillation.

Goal: Test if KD gradients have different singular value structure than hard-label gradients.

Predictions:
- κ(G_KD) < κ(G_hard) (KD gradients more balanced)
- Condition number decreases during training (learned structure)
- Layers closer to output have higher κ (more task-specific)

This experiment is from IDEA-014.

Usage:
    cd papers/neural-race-multiview/src
    python experiments/exp_gradient_svd.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# Simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    """Simple CNN with clear linear layers for gradient analysis."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Linear layers - these are what we'll analyze
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 128x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_linear_layers(self):
        """Return linear layers for gradient analysis."""
        return {
            'fc1': self.fc1,
            'fc2': self.fc2,
            'fc3': self.fc3
        }


def compute_gradient_svd_metrics(model, data_loader, criterion, device, max_batches=None):
    """
    Compute gradient SVD metrics for all linear layers.

    Returns dict with per-layer:
    - singular_values: full spectrum
    - condition_number: σ_max / σ_min
    - effective_rank: (Σσ)² / Σσ²
    - spectral_entropy: -Σ(σ_norm * log(σ_norm))
    """
    model.train()

    # Accumulate gradients over batches
    linear_layers = model.get_linear_layers()
    accumulated_grads = {name: None for name in linear_layers}

    num_batches = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if max_batches and batch_idx >= max_batches:
            break

        data, target = data.to(device), target.to(device)

        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Accumulate gradients
        for name, layer in linear_layers.items():
            grad = layer.weight.grad.clone()
            if accumulated_grads[name] is None:
                accumulated_grads[name] = grad
            else:
                accumulated_grads[name] += grad

        num_batches += 1

    # Average gradients
    for name in accumulated_grads:
        accumulated_grads[name] /= num_batches

    # Compute SVD metrics per layer
    metrics = {}
    for name, grad in accumulated_grads.items():
        # SVD
        U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
        S = S.cpu().numpy()

        # Condition number (σ_max / σ_min)
        condition_number = S[0] / (S[-1] + 1e-10)

        # Effective rank: (Σσ)² / Σσ²
        s_sum = S.sum()
        s_sq_sum = (S ** 2).sum()
        effective_rank = (s_sum ** 2) / (s_sq_sum + 1e-10)

        # Spectral entropy: -Σ(σ_norm * log(σ_norm))
        s_norm = S / (s_sum + 1e-10)
        spectral_entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))

        # Top-k fraction (what fraction of total is in top-k singular values)
        top3_fraction = S[:3].sum() / s_sum if len(S) >= 3 else 1.0

        metrics[name] = {
            'singular_values': S.tolist(),
            'condition_number': float(condition_number),
            'effective_rank': float(effective_rank),
            'spectral_entropy': float(spectral_entropy),
            'top3_fraction': float(top3_fraction),
            'grad_norm': float(np.linalg.norm(S)),
        }

    return metrics


def train_epoch(model, train_loader, optimizer, criterion, device, teacher=None, temperature=4.0, alpha=0.9):
    """Train for one epoch, optionally with KD."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        if teacher is not None:
            # Knowledge distillation
            with torch.no_grad():
                teacher_output = teacher(data)

            # Soft targets
            soft_targets = F.softmax(teacher_output / temperature, dim=1)
            soft_loss = F.kl_div(
                F.log_softmax(output / temperature, dim=1),
                soft_targets,
                reduction='batchmean'
            ) * (temperature ** 2)

            # Hard targets
            hard_loss = criterion(output, target)

            # Combined loss
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


def run_experiment(training_type, teacher=None, config=None, device='cpu'):
    """
    Run training and collect gradient SVD metrics at checkpoints.

    Args:
        training_type: 'hard' or 'kd'
        teacher: Teacher model for KD (required if training_type='kd')
        config: Experiment configuration
        device: Device to use

    Returns:
        dict with metrics at each checkpoint
    """
    if config is None:
        config = {
            'epochs': 15,
            'lr': 0.01,
            'batch_size': 128,
            'checkpoints': [1, 5, 10, 15],
            'max_batches_for_grad': 10,
        }

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # Model
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training loop with checkpoints
    results = {
        'training_type': training_type,
        'config': config,
        'checkpoints': {},
        'final_accuracy': None
    }

    if teacher is not None:
        teacher.eval()

    for epoch in tqdm(range(1, config['epochs'] + 1), desc=f"Training ({training_type})"):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            teacher=teacher if training_type == 'kd' else None
        )

        # Checkpoint
        if epoch in config['checkpoints']:
            test_acc = evaluate(model, test_loader, device)

            # Compute gradient SVD metrics
            grad_metrics = compute_gradient_svd_metrics(
                model, train_loader, criterion, device,
                max_batches=config['max_batches_for_grad']
            )

            results['checkpoints'][epoch] = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gradient_svd': grad_metrics
            }

            print(f"\n  Epoch {epoch}: test_acc={test_acc:.3f}")
            for layer_name, metrics in grad_metrics.items():
                print(f"    {layer_name}: κ={metrics['condition_number']:.1f}, "
                      f"eff_rank={metrics['effective_rank']:.1f}, "
                      f"entropy={metrics['spectral_entropy']:.2f}")

    results['final_accuracy'] = evaluate(model, test_loader, device)

    return results, model


def main():
    print("=" * 60)
    print("EXPERIMENT 2: Gradient Singular Value Analysis")
    print("=" * 60)
    print("\nGoal: Compare gradient SVD structure under hard labels vs KD")
    print("Prediction: κ(G_KD) < κ(G_hard) (KD gradients more balanced)\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = {
        'epochs': 15,  # Reduced for faster CPU execution
        'lr': 0.01,
        'batch_size': 128,
        'checkpoints': [1, 5, 10, 15],  # Fewer checkpoints
        'max_batches_for_grad': 10,  # Fewer batches for gradient computation
    }

    all_results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }

    # 1. Train teacher (will be used for KD)
    print("\n" + "=" * 50)
    print("Training TEACHER model")
    print("=" * 50)
    teacher_results, teacher_model = run_experiment('hard', teacher=None, config=config, device=device)
    all_results['teacher'] = teacher_results
    print(f"\nTeacher final accuracy: {teacher_results['final_accuracy']:.3f}")

    # 2. Train with hard labels (student baseline)
    print("\n" + "=" * 50)
    print("Training HARD LABELS student")
    print("=" * 50)
    hard_results, _ = run_experiment('hard', teacher=None, config=config, device=device)
    all_results['hard_labels'] = hard_results
    print(f"\nHard labels final accuracy: {hard_results['final_accuracy']:.3f}")

    # 3. Train with KD
    print("\n" + "=" * 50)
    print("Training KD student")
    print("=" * 50)
    kd_results, _ = run_experiment('kd', teacher=teacher_model, config=config, device=device)
    all_results['kd'] = kd_results
    print(f"\nKD final accuracy: {kd_results['final_accuracy']:.3f}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS: Comparing Gradient Condition Numbers")
    print("=" * 60)

    print("\nCondition Number κ(G) by Layer and Epoch:")
    print("-" * 70)
    print(f"{'Epoch':<8} {'Layer':<8} {'Hard κ':<12} {'KD κ':<12} {'Δ (KD-Hard)':<12}")
    print("-" * 70)

    kd_lower_count = 0
    total_comparisons = 0

    for epoch in config['checkpoints']:
        hard_checkpoint = hard_results['checkpoints'].get(epoch, {})
        kd_checkpoint = kd_results['checkpoints'].get(epoch, {})

        if not hard_checkpoint or not kd_checkpoint:
            continue

        for layer in ['fc1', 'fc2', 'fc3']:
            hard_kappa = hard_checkpoint['gradient_svd'][layer]['condition_number']
            kd_kappa = kd_checkpoint['gradient_svd'][layer]['condition_number']
            delta = kd_kappa - hard_kappa

            marker = "✓" if delta < 0 else "✗"
            print(f"{epoch:<8} {layer:<8} {hard_kappa:<12.1f} {kd_kappa:<12.1f} {delta:<+12.1f} {marker}")

            if delta < 0:
                kd_lower_count += 1
            total_comparisons += 1

    print("-" * 70)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    prediction_supported = kd_lower_count > total_comparisons / 2

    print(f"\nPrediction: κ(G_KD) < κ(G_hard)")
    print(f"Result: KD had lower κ in {kd_lower_count}/{total_comparisons} comparisons ({100*kd_lower_count/total_comparisons:.1f}%)")

    if prediction_supported:
        print("\n✓ PREDICTION SUPPORTED: KD gradients tend to have lower condition numbers")
    else:
        print("\n✗ PREDICTION NOT SUPPORTED: KD gradients do NOT consistently have lower condition numbers")

    # Also compare effective rank and entropy
    print("\n\nEffective Rank (higher = more balanced):")
    for epoch in [1, 15]:  # Just first and last
        if epoch not in hard_results['checkpoints']:
            continue
        print(f"\n  Epoch {epoch}:")
        for layer in ['fc1', 'fc2', 'fc3']:
            hard_er = hard_results['checkpoints'][epoch]['gradient_svd'][layer]['effective_rank']
            kd_er = kd_results['checkpoints'][epoch]['gradient_svd'][layer]['effective_rank']
            print(f"    {layer}: Hard={hard_er:.1f}, KD={kd_er:.1f}")

    # Save results
    out_dir = Path(__file__).parent.parent.parent / 'log' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'exp_gradient_svd_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {out_path}")

    return all_results


if __name__ == '__main__':
    main()
