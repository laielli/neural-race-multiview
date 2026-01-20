"""
Experiment: CIFAR-100 Hierarchical KD - Real Data Validation

This experiment validates Finding 2 (hierarchical structure enables acceleration)
on real data using CIFAR-100's natural class hierarchy.

Setup:
- CIFAR-100: 100 fine classes grouped into 20 superclasses
- Teacher: ResNet-18 trained on 100-class problem
- Student: ResNet-18 trained on 20-class (superclass) problem
- Comparison: Hard labels vs KD with collapsed soft labels

Hypothesis: Soft labels from the fine-grained teacher should accelerate
learning on the coarse (superclass) task by encoding inter-class similarity.

Expected outcome: KD achieves 90% accuracy faster than hard labels,
validating the synthetic experiment findings on real data.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_cifar import (
    get_cifar100_dataloaders,
    collapse_soft_labels,
    analyze_soft_label_structure
)


# =============================================================================
# Model Definitions
# =============================================================================

class BasicBlock(nn.Module):
    """Basic ResNet block."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR (32x32 images)."""

    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_features(self, x):
        """Get features before the classifier."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return out


def ResNet18(num_classes=100):
    """ResNet-18 for CIFAR."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# =============================================================================
# Training Functions
# =============================================================================

def train_teacher(
    model,
    train_loader,
    test_loader,
    epochs: int = 200,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = 'cuda',
    verbose: bool = True
):
    """
    Train teacher on fine-grained (100-class) CIFAR-100.

    Uses standard CIFAR-100 training recipe:
    - SGD with momentum and weight decay
    - Cosine annealing LR schedule
    - Cross-entropy loss
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {
        'epochs': [], 'train_loss': [], 'train_acc': [],
        'test_fine_acc': [], 'test_coarse_acc': []
    }

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        iterator = train_loader
        if verbose:
            iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)

        for images, fine_labels, _ in iterator:
            images, fine_labels = images.to(device), fine_labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, fine_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += fine_labels.size(0)
            correct += predicted.eq(fine_labels).sum().item()

        scheduler.step()

        # Evaluation
        model.eval()
        test_fine_correct = 0
        test_coarse_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, fine_labels, coarse_labels in test_loader:
                images = images.to(device)
                fine_labels = fine_labels.to(device)
                coarse_labels = coarse_labels.to(device)

                outputs = model(images)
                _, fine_pred = outputs.max(1)

                # Coarse prediction: map predicted fine class to its superclass
                coarse_pred = torch.tensor([
                    train_loader.dataset.fine_to_coarse[f.item()]
                    for f in fine_pred
                ], device=device)

                test_fine_correct += fine_pred.eq(fine_labels).sum().item()
                test_coarse_correct += coarse_pred.eq(coarse_labels).sum().item()
                test_total += fine_labels.size(0)

        train_acc = 100. * correct / total
        test_fine_acc = 100. * test_fine_correct / test_total
        test_coarse_acc = 100. * test_coarse_correct / test_total

        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss / total)
        history['train_acc'].append(train_acc)
        history['test_fine_acc'].append(test_fine_acc)
        history['test_coarse_acc'].append(test_coarse_acc)

        if test_fine_acc > best_acc:
            best_acc = test_fine_acc
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Train {train_acc:.1f}%, Test Fine {test_fine_acc:.1f}%, Test Coarse {test_coarse_acc:.1f}%')

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def train_student_hard(
    model,
    train_loader,
    test_loader,
    epochs: int = 200,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = 'cuda',
    log_interval: int = 1,
    verbose: bool = True
):
    """
    Train student on coarse (20-class) problem with hard labels.
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {
        'epochs': [], 'train_loss': [], 'train_acc': [], 'test_acc': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, _, coarse_labels in train_loader:
            images, coarse_labels = images.to(device), coarse_labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coarse_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += coarse_labels.size(0)
            correct += predicted.eq(coarse_labels).sum().item()

        scheduler.step()

        # Logging
        if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for images, _, coarse_labels in test_loader:
                    images, coarse_labels = images.to(device), coarse_labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    test_correct += predicted.eq(coarse_labels).sum().item()
                    test_total += coarse_labels.size(0)

            train_acc = 100. * correct / total
            test_acc = 100. * test_correct / test_total

            history['epochs'].append(epoch)
            history['train_loss'].append(train_loss / total)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(f'  Hard Epoch {epoch+1}: Train {train_acc:.1f}%, Test {test_acc:.1f}%')

    return model, history


def train_student_kd(
    student,
    teacher,
    train_loader,
    test_loader,
    fine_to_coarse: dict,
    epochs: int = 200,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    temperature: float = 4.0,
    alpha: float = 0.7,
    device: str = 'cuda',
    log_interval: int = 1,
    verbose: bool = True
):
    """
    Train student with KD using collapsed soft labels from teacher.

    The teacher outputs 100-class probabilities, which are collapsed
    to 20-class by summing within each superclass.

    Loss = alpha * KD_loss + (1-alpha) * CE_loss
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    optimizer = torch.optim.SGD(
        student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_criterion = nn.CrossEntropyLoss()

    history = {
        'epochs': [], 'train_loss': [], 'train_acc': [], 'test_acc': [],
        'kd_loss': [], 'ce_loss': []
    }

    for epoch in range(epochs):
        # Training
        student.train()
        train_loss = 0
        train_kd_loss = 0
        train_ce_loss = 0
        correct = 0
        total = 0

        for images, _, coarse_labels in train_loader:
            images, coarse_labels = images.to(device), coarse_labels.to(device)

            # Get teacher soft labels
            with torch.no_grad():
                teacher_logits = teacher(images)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                # Collapse to 20-class
                soft_targets = collapse_soft_labels(teacher_probs, fine_to_coarse, n_coarse=20)

            # Student forward
            student_logits = student(images)

            # Hard label loss (CE)
            ce_loss = ce_criterion(student_logits, coarse_labels)

            # Soft label loss (KL divergence)
            student_soft = F.log_softmax(student_logits / temperature, dim=-1)
            kd_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
            kd_loss = kd_loss * (temperature ** 2)

            # Combined loss
            loss = alpha * kd_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_kd_loss += kd_loss.item() * images.size(0)
            train_ce_loss += ce_loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += coarse_labels.size(0)
            correct += predicted.eq(coarse_labels).sum().item()

        scheduler.step()

        # Logging
        if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            student.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for images, _, coarse_labels in test_loader:
                    images, coarse_labels = images.to(device), coarse_labels.to(device)
                    outputs = student(images)
                    _, predicted = outputs.max(1)
                    test_correct += predicted.eq(coarse_labels).sum().item()
                    test_total += coarse_labels.size(0)

            train_acc = 100. * correct / total
            test_acc = 100. * test_correct / test_total

            history['epochs'].append(epoch)
            history['train_loss'].append(train_loss / total)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['kd_loss'].append(train_kd_loss / total)
            history['ce_loss'].append(train_ce_loss / total)

            if verbose and (epoch + 1) % 10 == 0:
                print(f'  KD Epoch {epoch+1}: Train {train_acc:.1f}%, Test {test_acc:.1f}%')

    return student, history


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    teacher_epochs: int = 200,
    student_epochs: int = 200,
    lr: float = 0.1,
    batch_size: int = 128,
    n_teachers: int = 3,
    temperature: float = 4.0,
    alpha: float = 0.7,
    n_seeds: int = 3,
    data_root: str = './data',
    output_dir: str = 'results/cifar',
    device: str = None,
    verbose: bool = True,
    quick: bool = False
):
    """
    Run CIFAR-100 hierarchical KD experiment.

    Compares:
    1. Hard labels: Student trained on 20-class problem with hard labels
    2. KD: Student trained with collapsed soft labels from 100-class teacher

    Expected finding: KD accelerates learning due to hierarchical structure.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    if quick:
        teacher_epochs = 50
        student_epochs = 50
        n_teachers = 1
        n_seeds = 1
        print("Quick mode: reduced epochs and seeds")

    print("=" * 70)
    print("CIFAR-100 Hierarchical KD Experiment")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Teacher epochs: {teacher_epochs}, Student epochs: {student_epochs}")
    print(f"Teachers: {n_teachers}, Seeds: {n_seeds}")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    print("=" * 70)

    # Data loading
    print("\nLoading CIFAR-100...")
    train_loader, test_loader, dataset_info = get_cifar100_dataloaders(
        root=data_root, batch_size=batch_size, num_workers=4
    )
    print(f"  Train: {dataset_info['train_size']}, Test: {dataset_info['test_size']}")
    print(f"  Fine classes: {dataset_info['n_fine']}, Coarse classes: {dataset_info['n_coarse']}")

    fine_to_coarse = dataset_info['fine_to_coarse']

    # =========================================================================
    # Phase 1: Train teacher ensemble on fine-grained (100-class) problem
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 1: Training Teacher Ensemble (100-class)")
    print("=" * 50)

    teachers = []
    teacher_metrics = []

    for i in range(n_teachers):
        print(f"\nTraining teacher {i+1}/{n_teachers}...")
        torch.manual_seed(i)

        teacher = ResNet18(num_classes=100)
        teacher, history = train_teacher(
            teacher, train_loader, test_loader,
            epochs=teacher_epochs, lr=lr, device=device, verbose=verbose
        )
        teachers.append(teacher)

        final_fine_acc = history['test_fine_acc'][-1]
        final_coarse_acc = history['test_coarse_acc'][-1]

        teacher_metrics.append({
            'seed': i,
            'fine_accuracy': final_fine_acc,
            'coarse_accuracy': final_coarse_acc
        })

        print(f"Teacher {i+1}: Fine={final_fine_acc:.1f}%, Coarse={final_coarse_acc:.1f}%")

    # Analyze soft label information
    print("\nAnalyzing soft label information content...")
    soft_label_info = analyze_soft_label_structure(
        teachers[0], test_loader, temperature, fine_to_coarse, device
    )
    print(f"  Avg entropy: {soft_label_info['avg_entropy']:.3f} / {soft_label_info['max_entropy']:.3f} (max)")
    print(f"  Avg true superclass prob: {soft_label_info['avg_true_superclass_prob']:.3f}")
    print(f"  Avg same-superclass confusion: {soft_label_info['avg_same_superclass_confusion']:.3f}")

    # =========================================================================
    # Phase 2: Train students with hard labels vs KD
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 2: Training Students (20-class)")
    print("=" * 50)

    hard_results = []
    kd_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")

        # Hard labels
        print("Training with hard labels...")
        torch.manual_seed(100 + seed)
        student_hard = ResNet18(num_classes=20)
        student_hard, history_hard = train_student_hard(
            student_hard, train_loader, test_loader,
            epochs=student_epochs, lr=lr, device=device, verbose=verbose
        )

        hard_results.append({
            'seed': seed,
            'final_accuracy': history_hard['test_acc'][-1],
            'history': history_hard
        })

        # KD (use first teacher for simplicity, or ensemble average)
        print("Training with KD...")
        torch.manual_seed(100 + seed)
        student_kd = ResNet18(num_classes=20)
        student_kd, history_kd = train_student_kd(
            student_kd, teachers[0], train_loader, test_loader, fine_to_coarse,
            epochs=student_epochs, lr=lr, temperature=temperature, alpha=alpha,
            device=device, verbose=verbose
        )

        kd_results.append({
            'seed': seed,
            'final_accuracy': history_kd['test_acc'][-1],
            'history': history_kd
        })

        print(f"  Hard: {history_hard['test_acc'][-1]:.1f}%, KD: {history_kd['test_acc'][-1]:.1f}%")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    avg_hard_acc = np.mean([r['final_accuracy'] for r in hard_results])
    avg_kd_acc = np.mean([r['final_accuracy'] for r in kd_results])
    std_hard_acc = np.std([r['final_accuracy'] for r in hard_results])
    std_kd_acc = np.std([r['final_accuracy'] for r in kd_results])

    print(f"\n{'Method':<20} | {'Accuracy':<20} | {'Std':<10}")
    print("-" * 55)
    print(f"{'Hard Labels':<20} | {avg_hard_acc:<20.2f} | {std_hard_acc:<10.2f}")
    print(f"{'KD':<20} | {avg_kd_acc:<20.2f} | {std_kd_acc:<10.2f}")

    # Learning speed analysis
    hard_acc_curve = np.mean([r['history']['test_acc'] for r in hard_results], axis=0)
    kd_acc_curve = np.mean([r['history']['test_acc'] for r in kd_results], axis=0)
    epochs_list = hard_results[0]['history']['epochs']

    # Find epochs to reach various thresholds
    thresholds = [70, 80, 90]
    speed_results = {}

    for thresh in thresholds:
        hard_epoch = None
        kd_epoch = None

        for i, (h, k) in enumerate(zip(hard_acc_curve, kd_acc_curve)):
            if hard_epoch is None and h >= thresh:
                hard_epoch = epochs_list[i]
            if kd_epoch is None and k >= thresh:
                kd_epoch = epochs_list[i]

        speed_results[thresh] = {
            'hard_epoch': hard_epoch,
            'kd_epoch': kd_epoch,
            'speedup': hard_epoch / kd_epoch if (hard_epoch and kd_epoch and kd_epoch > 0) else None
        }

    print(f"\nLearning Speed (epochs to reach threshold):")
    print(f"{'Threshold':<12} | {'Hard':<10} | {'KD':<10} | {'Speedup':<10}")
    print("-" * 45)
    for thresh, result in speed_results.items():
        hard_str = str(result['hard_epoch']) if result['hard_epoch'] else 'N/A'
        kd_str = str(result['kd_epoch']) if result['kd_epoch'] else 'N/A'
        speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] else 'N/A'
        print(f"{thresh}%{'':<10} | {hard_str:<10} | {kd_str:<10} | {speedup_str:<10}")

    # Success criteria
    print("\n" + "-" * 50)
    print("Success Criteria:")
    print("-" * 50)

    success = True

    # Criterion 1: KD maintains accuracy
    if avg_kd_acc >= avg_hard_acc - 2.0:
        print(f"[PASS] KD accuracy >= Hard - 2%: {avg_kd_acc:.1f}% >= {avg_hard_acc - 2:.1f}%")
    else:
        print(f"[FAIL] KD accuracy < Hard - 2%: {avg_kd_acc:.1f}% < {avg_hard_acc - 2:.1f}%")
        success = False

    # Criterion 2: KD learns faster
    if speed_results[80]['speedup'] and speed_results[80]['speedup'] > 1.0:
        print(f"[PASS] KD speedup at 80%: {speed_results[80]['speedup']:.2f}x")
    else:
        print(f"[FAIL] KD not faster at 80%")
        success = False

    # Criterion 3: Soft labels contain information
    if soft_label_info['avg_same_superclass_confusion'] > 0.05:
        print(f"[PASS] Soft labels encode class similarity: {soft_label_info['avg_same_superclass_confusion']:.3f}")
    else:
        print(f"[FAIL] Soft labels don't encode similarity: {soft_label_info['avg_same_superclass_confusion']:.3f}")

    print("\n" + "=" * 50)
    if success:
        print("EXPERIMENT PASSED - Hierarchical KD finding validated on CIFAR!")
    else:
        print("EXPERIMENT INCONCLUSIVE - See detailed results")
    print("=" * 50)

    # =========================================================================
    # Plots
    # =========================================================================
    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    ax = axes[0]
    ax.plot(epochs_list, hard_acc_curve, 'b-', linewidth=2, label='Hard Labels')
    ax.plot(epochs_list, kd_acc_curve, 'r-', linewidth=2, label='KD')
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-100 Superclass Learning Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Speedup comparison
    ax = axes[1]
    thresh_labels = [f'{t}%' for t in thresholds]
    hard_epochs = [speed_results[t]['hard_epoch'] or 0 for t in thresholds]
    kd_epochs = [speed_results[t]['kd_epoch'] or 0 for t in thresholds]

    x = np.arange(len(thresh_labels))
    width = 0.35
    ax.bar(x - width/2, hard_epochs, width, label='Hard Labels', color='blue', alpha=0.7)
    ax.bar(x + width/2, kd_epochs, width, label='KD', color='red', alpha=0.7)

    # Add speedup annotations
    for i, t in enumerate(thresholds):
        if speed_results[t]['speedup']:
            ax.annotate(f"{speed_results[t]['speedup']:.1f}x",
                       xy=(i, max(hard_epochs[i], kd_epochs[i]) + 5),
                       ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Accuracy Threshold', fontsize=12)
    ax.set_ylabel('Epochs to Reach', fontsize=12)
    ax.set_title('Learning Speed Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(thresh_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'figures' / 'cifar_hierarchical_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {output_path / 'figures' / 'cifar_hierarchical_results.png'}")

    # =========================================================================
    # Save results
    # =========================================================================
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'teacher_epochs': teacher_epochs,
            'student_epochs': student_epochs,
            'lr': lr,
            'batch_size': batch_size,
            'n_teachers': n_teachers,
            'temperature': temperature,
            'alpha': alpha,
            'n_seeds': n_seeds,
            'device': device
        },
        'dataset': dataset_info,
        'teachers': {
            'avg_fine_accuracy': np.mean([t['fine_accuracy'] for t in teacher_metrics]),
            'avg_coarse_accuracy': np.mean([t['coarse_accuracy'] for t in teacher_metrics]),
            'individual': teacher_metrics
        },
        'soft_label_info': soft_label_info,
        'hard_labels': {
            'avg_accuracy': float(avg_hard_acc),
            'std_accuracy': float(std_hard_acc),
            'per_seed': [{'seed': r['seed'], 'accuracy': r['final_accuracy']} for r in hard_results]
        },
        'kd': {
            'avg_accuracy': float(avg_kd_acc),
            'std_accuracy': float(std_kd_acc),
            'per_seed': [{'seed': r['seed'], 'accuracy': r['final_accuracy']} for r in kd_results]
        },
        'speed_comparison': speed_results,
        'success': success
    }

    with open(output_path / 'cifar_hierarchical_results.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"Saved results: {output_path / 'cifar_hierarchical_results.json'}")

    return save_results, success


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 Hierarchical KD Experiment')
    parser.add_argument('--teacher-epochs', type=int, default=200, help='Teacher training epochs')
    parser.add_argument('--student-epochs', type=int, default=200, help='Student training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--n-teachers', type=int, default=3, help='Number of teachers')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Soft label weight')
    parser.add_argument('--n-seeds', type=int, default=3, help='Number of seeds for student')
    parser.add_argument('--data-root', type=str, default='./data', help='Data directory')
    parser.add_argument('--output', type=str, default='results/cifar', help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    run_experiment(
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_teachers=args.n_teachers,
        temperature=args.temperature,
        alpha=args.alpha,
        n_seeds=args.n_seeds,
        data_root=args.data_root,
        output_dir=args.output,
        device=args.device,
        quick=args.quick
    )


if __name__ == '__main__':
    main()
