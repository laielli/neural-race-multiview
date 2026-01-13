"""
Training functions for KDMech experiments.

Includes hard label training and knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train_hard_labels(
    model,
    dataset,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 128,
    log_interval: int = 10,
    track_pathways: bool = False,
    track_classes: list = None,
    verbose: bool = True
):
    """
    Train with standard cross-entropy (hard labels).

    Args:
        model: MultiViewNet instance
        dataset: MultiViewDataset instance
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        log_interval: How often to log metrics
        track_pathways: If True, record pathway strengths during training
        track_classes: List of class indices to track (default: [0])
        verbose: Whether to print progress

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
        'epochs_logged': [],
        'pathway_strengths': {y: {m: [] for m in range(dataset.M)} for y in track_classes}
    }

    # Get data tensors
    X, Y = dataset.get_tensors()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training (hard labels)")

    for epoch in iterator:
        model.train()

        # Shuffle
        perm = torch.randperm(len(X))
        X_shuf, Y_shuf = X[perm], Y[perm]

        epoch_loss = 0
        num_batches = 0
        for i in range(0, len(X), batch_size):
            x_batch = X_shuf[i:i+batch_size]
            y_batch = Y_shuf[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(X)
                acc = (logits.argmax(dim=1) == Y).float().mean().item()

            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)
            history['accuracy'].append(acc)
            history['epochs_logged'].append(epoch)

            if track_pathways:
                for y in track_classes:
                    for m in range(dataset.M):
                        phi_ym = dataset.get_view_feature(y, m)
                        s = model.get_pathway_strength(phi_ym)
                        history['pathway_strengths'][y][m].append(s)

            if verbose:
                iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    return history


def train_kd(
    student,
    teachers,
    dataset,
    epochs: int = 100,
    temperature: float = 4.0,
    lr: float = 0.01,
    batch_size: int = 128,
    log_interval: int = 10,
    track_pathways: bool = False,
    track_classes: list = None,
    verbose: bool = True
):
    """
    Train student via knowledge distillation from teacher ensemble.

    Uses KL divergence between student and ensemble-averaged teacher soft labels.

    Args:
        student: MultiViewNet student model
        teachers: List of teacher models
        dataset: MultiViewDataset instance
        epochs: Number of training epochs
        temperature: Softmax temperature for distillation
        lr: Learning rate
        batch_size: Batch size
        log_interval: How often to log metrics
        track_pathways: If True, record pathway strengths during training
        track_classes: List of class indices to track (default: [0])
        verbose: Whether to print progress

    Returns:
        dict: Training history including loss, accuracy, and optionally pathway strengths
    """
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)

    if track_classes is None:
        track_classes = [0]

    history = {
        'loss': [],
        'accuracy': [],
        'epochs_logged': [],
        'pathway_strengths': {y: {m: [] for m in range(dataset.M)} for y in track_classes}
    }

    X, Y = dataset.get_tensors()

    # Set teachers to eval mode
    for t in teachers:
        t.eval()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training (KD)")

    for epoch in iterator:
        student.train()

        perm = torch.randperm(len(X))
        X_shuf, Y_shuf = X[perm], Y[perm]

        epoch_loss = 0
        num_batches = 0
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
            num_batches += 1

        # Logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            student.eval()
            with torch.no_grad():
                logits = student(X)
                acc = (logits.argmax(dim=1) == Y).float().mean().item()

            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)
            history['accuracy'].append(acc)
            history['epochs_logged'].append(epoch)

            if track_pathways:
                for y in track_classes:
                    for m in range(dataset.M):
                        phi_ym = dataset.get_view_feature(y, m)
                        s = student.get_pathway_strength(phi_ym)
                        history['pathway_strengths'][y][m].append(s)

            if verbose:
                iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    return history


def train_teachers(
    dataset,
    num_teachers: int = 5,
    base_seed: int = 0,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 128,
    verbose: bool = True
):
    """
    Train an ensemble of teachers with different random seeds.

    Args:
        dataset: MultiViewDataset instance
        num_teachers: Number of teachers to train
        base_seed: Base seed for reproducibility
        epochs: Training epochs per teacher
        lr: Learning rate
        batch_size: Batch size
        verbose: Whether to print progress

    Returns:
        list: List of trained teacher models
    """
    from .model import MultiViewNet, init_weights

    teachers = []

    for i in range(num_teachers):
        if verbose:
            print(f"\nTraining teacher {i+1}/{num_teachers}")

        # Create and initialize model with different seed
        torch.manual_seed(base_seed * 100 + i)
        teacher = MultiViewNet(d=dataset.d, K=dataset.K)
        teacher.apply(init_weights)

        # Train
        train_hard_labels(
            teacher, dataset,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose
        )

        teachers.append(teacher)

    return teachers
