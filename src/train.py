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
    verbose: bool = True,
    loss_type: str = 'ce',
    gradient_flow: bool = False
):
    """
    Train with standard cross-entropy (hard labels) or MSE loss.

    Args:
        model: MultiViewNet or DeepLinearNet instance
        dataset: MultiViewDataset instance
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        log_interval: How often to log metrics
        track_pathways: If True, record pathway strengths during training
        track_classes: List of class indices to track (default: [0])
        verbose: Whether to print progress
        loss_type: 'ce' for cross-entropy, 'mse' for MSE (matches Saxe theory)
        gradient_flow: If True, use vanilla SGD (no momentum) to approximate gradient flow

    Returns:
        dict: Training history including loss, accuracy, and optionally pathway strengths
    """
    # Optimizer: gradient flow = no momentum
    if gradient_flow:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Loss function
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
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

            # MSE needs one-hot targets
            if loss_type == 'mse':
                y_onehot = F.one_hot(y_batch, num_classes=dataset.K).float()
                loss = criterion(logits, y_onehot)
            else:
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
    verbose: bool = True,
    gradient_flow: bool = False
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
        gradient_flow: If True, use vanilla SGD (no momentum) to approximate gradient flow

    Returns:
        dict: Training history including loss, accuracy, and optionally pathway strengths
    """
    # Optimizer: gradient flow = no momentum
    if gradient_flow:
        optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0)
    else:
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


def train_with_competition(
    model,
    dataset,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 128,
    log_interval: int = 10,
    track_pathways: bool = False,
    track_classes: list = None,
    verbose: bool = True,
    competition_weight: float = 0.1
):
    """
    Train with explicit competition loss to encourage winner-take-all.

    Adds a term that penalizes equal pathway strengths, encouraging one to dominate.

    competition_loss = -sum over classes of (max_pathway_strength / sum_pathway_strengths)

    This encourages one pathway to dominate within each class.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.MSELoss()

    if track_classes is None:
        track_classes = [0]

    history = {
        'loss': [],
        'accuracy': [],
        'epochs_logged': [],
        'pathway_strengths': {y: {m: [] for m in range(dataset.M)} for y in track_classes}
    }

    X, Y = dataset.get_tensors()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training (competition)")

    for epoch in iterator:
        model.train()

        perm = torch.randperm(len(X))
        X_shuf, Y_shuf = X[perm], Y[perm]

        epoch_loss = 0
        num_batches = 0
        for i in range(0, len(X), batch_size):
            x_batch = X_shuf[i:i+batch_size]
            y_batch = Y_shuf[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(x_batch)

            # MSE loss
            y_onehot = F.one_hot(y_batch, num_classes=dataset.K).float()
            main_loss = criterion(logits, y_onehot)

            # Competition loss: encourage pathway dominance
            # Sample a few classes and compute their pathway strengths
            comp_loss = torch.tensor(0.0)
            for y in range(min(3, dataset.K)):  # Sample first 3 classes
                strengths = []
                for m in range(dataset.M):
                    phi_ym = dataset.get_view_feature(y, m)
                    R_ym = model(phi_ym.unsqueeze(0)).squeeze(0)
                    strengths.append(R_ym.norm())

                strengths = torch.stack(strengths)
                total = strengths.sum() + 1e-8
                dominance = strengths.max() / total
                comp_loss = comp_loss - dominance  # Negative to maximize dominance

            comp_loss = comp_loss / min(3, dataset.K)

            loss = main_loss + competition_weight * comp_loss
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


def train_gated_dln(
    model,
    X,
    Y,
    epochs: int = 500,
    lr: float = 0.02,
    log_interval: int = 10,
    track_svd: bool = True,
    verbose: bool = True
):
    """
    Train GatedDLN with MSE loss and gradient flow, tracking singular values.

    This matches the Saxe et al. training setup:
    - MSE loss
    - SGD with no momentum (gradient flow)
    - Track singular values to observe race dynamics

    Args:
        model: GatedDLN or GatedMultiViewNet instance
        X: Input data, shape (N, d) or list of (N, d_input) per pathway
        Y: Target data, shape (N, d_output)
        epochs: Number of training epochs
        lr: Learning rate
        log_interval: How often to log metrics
        track_svd: If True, track singular values during training
        verbose: Whether to print progress

    Returns:
        dict: Training history including loss and optionally SVD history
    """
    # Gradient flow: SGD with no momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    history = {
        'loss': [],
        'epochs_logged': [],
        'pathway_strengths': [],
        'dominance': []
    }

    if track_svd:
        history['svd'] = {
            'encoders': [],
            'hidden': [],
            'decoders': []
        }

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training GatedDLN")

    for epoch in iterator:
        model.train()
        optimizer.zero_grad()

        # Forward pass computes gated loss
        loss = model(X, Y)
        loss.backward()
        optimizer.step()

        # Logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            history['loss'].append(loss.item())
            history['epochs_logged'].append(epoch)
            history['pathway_strengths'].append(model.get_all_pathway_strengths())
            history['dominance'].append(model.compute_dominance())

            if track_svd:
                svs = model.get_singular_values()
                history['svd']['encoders'].append(svs['encoders'])
                history['svd']['hidden'].append(svs['hidden'])
                history['svd']['decoders'].append(svs['decoders'])

            if verbose:
                dom = history['dominance'][-1]
                iterator.set_postfix(loss=f"{loss.item():.4f}", dom=f"{dom:.3f}")

    return history


def train_gated_multiview(
    model,
    dataset,
    epochs: int = 500,
    lr: float = 0.02,
    batch_size: int = None,
    log_interval: int = 10,
    track_svd: bool = True,
    verbose: bool = True
):
    """
    Train GatedMultiViewNet on multi-view dataset.

    Args:
        model: GatedMultiViewNet instance
        dataset: MultiViewDataset instance
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: If None, use full batch (required for theory match)
        log_interval: How often to log metrics
        track_svd: If True, track singular values
        verbose: Whether to print progress

    Returns:
        dict: Training history
    """
    # Get data
    X, Y = dataset.get_tensors()

    # Convert labels to one-hot for MSE loss
    Y_onehot = F.one_hot(Y, num_classes=dataset.K).float()

    # Gradient flow: SGD with no momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    history = {
        'loss': [],
        'accuracy': [],
        'epochs_logged': [],
        'pathway_strengths': [],
        'dominance': [],
        'view_coverage': []
    }

    if track_svd:
        history['svd'] = {
            'encoders': [],
            'hidden': [],
            'decoders': []
        }

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training GatedMultiView")

    for epoch in iterator:
        model.train()

        if batch_size is None:
            # Full batch gradient (closer to gradient flow)
            optimizer.zero_grad()

            # Split input into views
            x_views = [X[:, i*model.d_view:(i+1)*model.d_view] for i in range(model.M)]

            # Compute loss through gated pathways
            total_loss = 0.0
            for i, x_v in enumerate(x_views):
                h = model.encoders[i](x_v)
                h = model.hidden_layer(h)
                for j in range(model.M):
                    if model.gate[i, j] > 0:
                        out = model.decoders[j](h)
                        loss = model.loss_fn(out, Y_onehot).mean()
                        total_loss = total_loss + loss

            total_loss.backward()
            optimizer.step()
            epoch_loss = total_loss.item()

        else:
            # Mini-batch training
            perm = torch.randperm(len(X))
            X_shuf, Y_shuf = X[perm], Y_onehot[perm]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(X), batch_size):
                x_batch = X_shuf[i:i+batch_size]
                y_batch = Y_shuf[i:i+batch_size]

                optimizer.zero_grad()
                x_views = [x_batch[:, k*model.d_view:(k+1)*model.d_view] for k in range(model.M)]

                total_loss = 0.0
                for ii, x_v in enumerate(x_views):
                    h = model.encoders[ii](x_v)
                    h = model.hidden_layer(h)
                    for jj in range(model.M):
                        if model.gate[ii, jj] > 0:
                            out = model.decoders[jj](h)
                            loss = model.loss_fn(out, y_batch).mean()
                            total_loss = total_loss + loss

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            epoch_loss /= num_batches

        # Logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            model.eval()

            # Compute accuracy
            with torch.no_grad():
                logits = model.forward_multiview(X)
                acc = (logits.argmax(dim=1) == Y).float().mean().item()

            history['loss'].append(epoch_loss)
            history['accuracy'].append(acc)
            history['epochs_logged'].append(epoch)
            history['pathway_strengths'].append(model.get_all_pathway_strengths())
            history['dominance'].append(model.compute_dominance())

            # Measure view coverage
            cov = model.measure_view_coverage(dataset, threshold=0.1)
            history['view_coverage'].append(cov)

            if track_svd:
                svs = model.get_singular_values()
                history['svd']['encoders'].append(svs['encoders'])
                history['svd']['hidden'].append(svs['hidden'])
                history['svd']['decoders'].append(svs['decoders'])

            if verbose:
                dom = history['dominance'][-1]
                iterator.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{acc:.3f}", dom=f"{dom:.3f}")

    return history


def train_gated_kd(
    student,
    teachers,
    X,
    Y,
    epochs: int = 500,
    temperature: float = 4.0,
    lr: float = 0.02,
    log_interval: int = 10,
    track_svd: bool = True,
    verbose: bool = True,
    use_mse: bool = True  # Use MSE to teacher outputs (better for regression)
):
    """
    Train GatedDLN student via knowledge distillation from teacher ensemble.

    Uses soft targets from teacher ensemble to guide student training.
    The hypothesis (Theorem 3): KD should distribute gradients to preserve
    multiple pathways, preventing winner-take-all dynamics.

    Args:
        student: GatedDLN student model
        teachers: List of GatedDLN teacher models
        X: Input data, shape (N, d) or list of (N, d_input) per pathway
        Y: Target data, shape (N, d_output) - used for teacher targets
        epochs: Number of training epochs
        temperature: Softmax temperature for distillation (ignored if use_mse=True)
        lr: Learning rate
        log_interval: How often to log metrics
        track_svd: If True, track singular values during training
        verbose: Whether to print progress
        use_mse: If True, use MSE to teacher outputs (better for regression)

    Returns:
        dict: Training history including loss and SVD history
    """
    # Gradient flow: SGD with no momentum
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0)
    mse_loss = nn.MSELoss()

    history = {
        'loss': [],
        'epochs_logged': [],
        'pathway_strengths': [],
        'dominance': []
    }

    if track_svd:
        history['svd'] = {
            'encoders': [],
            'hidden': [],
            'decoders': []
        }

    # Set teachers to eval mode
    for t in teachers:
        t.eval()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training GatedDLN (KD)")

    for epoch in iterator:
        student.train()
        optimizer.zero_grad()

        # Get soft targets from teacher ensemble
        with torch.no_grad():
            # Each teacher produces outputs for each gated pathway
            # Average the outputs across teachers
            teacher_outputs = []
            for t in teachers:
                t_outs = t(X, y_target=None)  # Get outputs, not loss
                if isinstance(t_outs, list):
                    # Average across pathways within each teacher
                    t_out = torch.stack(t_outs).mean(dim=0)
                else:
                    t_out = t_outs
                teacher_outputs.append(t_out)

            # Ensemble average - these are the soft targets
            soft_targets = torch.stack(teacher_outputs).mean(dim=0)

        # Compute loss for student by driving each pathway toward teacher ensemble target
        # This mimics how the original training computes loss for each gated pathway
        if use_mse:
            # Compute loss through gated pathways, same as train_gated_dln
            # but using teacher ensemble outputs as targets
            loss = student(X, soft_targets)  # Uses model's internal gated loss computation
        else:
            # Get student outputs for KL divergence
            student_outputs = student(X, y_target=None)
            if isinstance(student_outputs, list):
                student_out = torch.stack(student_outputs).mean(dim=0)
            else:
                student_out = student_outputs

            # KL divergence with temperature (for classification)
            soft_targets_prob = F.softmax(soft_targets / temperature, dim=-1)
            student_log_probs = F.log_softmax(student_out / temperature, dim=-1)
            loss = F.kl_div(student_log_probs, soft_targets_prob, reduction='batchmean')
            loss = loss * (temperature ** 2)

        loss.backward()
        optimizer.step()

        # Logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            history['loss'].append(loss.item())
            history['epochs_logged'].append(epoch)
            history['pathway_strengths'].append(student.get_all_pathway_strengths())
            history['dominance'].append(student.compute_dominance())

            if track_svd:
                svs = student.get_singular_values()
                history['svd']['encoders'].append(svs['encoders'])
                history['svd']['hidden'].append(svs['hidden'])
                history['svd']['decoders'].append(svs['decoders'])

            if verbose:
                dom = history['dominance'][-1]
                iterator.set_postfix(loss=f"{loss.item():.4f}", dom=f"{dom:.3f}")

    return history


def train_gated_multiview_kd(
    student,
    teachers,
    dataset,
    epochs: int = 500,
    temperature: float = 4.0,
    lr: float = 0.02,
    log_interval: int = 10,
    track_svd: bool = True,
    verbose: bool = True
):
    """
    Train GatedMultiViewNet student via KD from teacher ensemble.

    Args:
        student: GatedMultiViewNet student model
        teachers: List of GatedMultiViewNet teacher models
        dataset: MultiViewDataset instance
        epochs: Number of training epochs
        temperature: KD temperature
        lr: Learning rate
        log_interval: How often to log metrics
        track_svd: If True, track singular values
        verbose: Whether to print progress

    Returns:
        dict: Training history
    """
    # Get data
    X, Y = dataset.get_tensors()

    # Gradient flow: SGD with no momentum
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0)

    history = {
        'loss': [],
        'accuracy': [],
        'epochs_logged': [],
        'pathway_strengths': [],
        'dominance': [],
        'view_coverage': []
    }

    if track_svd:
        history['svd'] = {
            'encoders': [],
            'hidden': [],
            'decoders': []
        }

    # Set teachers to eval mode
    for t in teachers:
        t.eval()

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training GatedMultiView (KD)")

    for epoch in iterator:
        student.train()
        optimizer.zero_grad()

        # Get soft targets from teacher ensemble
        with torch.no_grad():
            teacher_logits = []
            for t in teachers:
                t_logits = t.forward_multiview(X)
                teacher_logits.append(t_logits)

            # Ensemble average logits
            ensemble_logits = torch.stack(teacher_logits).mean(dim=0)
            soft_targets = F.softmax(ensemble_logits / temperature, dim=-1)

        # Student forward pass
        student_logits = student.forward_multiview(X)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

        # KD loss (KL divergence)
        loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean')
        loss = loss * (temperature ** 2)

        loss.backward()
        optimizer.step()

        # Logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            student.eval()

            with torch.no_grad():
                logits = student.forward_multiview(X)
                acc = (logits.argmax(dim=1) == Y).float().mean().item()

            history['loss'].append(loss.item())
            history['accuracy'].append(acc)
            history['epochs_logged'].append(epoch)
            history['pathway_strengths'].append(student.get_all_pathway_strengths())
            history['dominance'].append(student.compute_dominance())

            # Measure view coverage
            cov = student.measure_view_coverage(dataset, threshold=0.1)
            history['view_coverage'].append(cov)

            if track_svd:
                svs = student.get_singular_values()
                history['svd']['encoders'].append(svs['encoders'])
                history['svd']['hidden'].append(svs['hidden'])
                history['svd']['decoders'].append(svs['decoders'])

            if verbose:
                dom = history['dominance'][-1]
                iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}", dom=f"{dom:.3f}")

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
