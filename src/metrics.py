"""
Measurement functions for KDMech experiments.

Includes coverage metrics, pathway strength measurement, and analysis utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np


def measure_all_pathway_strengths(model, dataset):
    """
    Measure pathway strength for all (class, view) pairs.

    Args:
        model: MultiViewNet instance
        dataset: MultiViewDataset instance

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

    Args:
        model: MultiViewNet instance
        dataset: MultiViewDataset instance
        threshold: Confidence threshold for detection

    Returns:
        float: Coverage in [0, 1]
    """
    detected = 0
    total = dataset.K * dataset.M

    model.eval()
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

    Args:
        model: MultiViewNet instance
        dataset: MultiViewDataset instance
        threshold: Confidence threshold for detection

    Returns:
        dict: {
            'total_coverage': float,
            'per_class': {y: [list of detected views]},
            'detection_matrix': np.array of shape (K, M)
        }
    """
    detection_matrix = np.zeros((dataset.K, dataset.M))
    per_class = {y: [] for y in range(dataset.K)}

    model.eval()
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

    Args:
        model: MultiViewNet instance
        dataset: MultiViewDataset instance
        y: Class index

    Returns:
        int: winning view index m*
    """
    strengths = []
    for m in range(dataset.M):
        phi_ym = dataset.get_view_feature(y, m)
        strengths.append(model.get_pathway_strength(phi_ym))
    return int(np.argmax(strengths))


def measure_ensemble_coverage(teachers, dataset, threshold=0.5):
    """
    Measure coverage of teacher ensemble.

    Args:
        teachers: List of MultiViewNet teacher models
        dataset: MultiViewDataset instance
        threshold: Confidence threshold

    Returns:
        float: Ensemble coverage in [0, 1]
    """
    detected = 0
    total = dataset.K * dataset.M

    for t in teachers:
        t.eval()

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


def compute_view_diversity(models, dataset):
    """
    Compute diversity of winning views across multiple trained models.

    Measures whether different random seeds lead to different views winning.

    Args:
        models: List of trained MultiViewNet models
        dataset: MultiViewDataset instance

    Returns:
        float: Diversity score in [0, 1], where 1 = all views won equally
    """
    from collections import Counter

    winning_views = {y: [] for y in range(dataset.K)}

    for model in models:
        for y in range(dataset.K):
            winner = find_winning_view(model, dataset, y)
            winning_views[y].append(winner)

    # Compute diversity per class
    diversities = []
    for y in range(dataset.K):
        counts = Counter(winning_views[y])
        # Normalized entropy
        total = len(winning_views[y])
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(dataset.M)  # Maximum possible entropy
        diversities.append(entropy / max_entropy if max_entropy > 0 else 0)

    return np.mean(diversities)


def get_pathway_strength_history(history, y, m):
    """
    Extract pathway strength history for a specific (class, view) pair.

    Args:
        history: Training history dict from train_hard_labels or train_kd
        y: Class index
        m: View index

    Returns:
        list: Pathway strengths over training epochs
    """
    if 'pathway_strengths' not in history:
        raise ValueError("History does not contain pathway_strengths. "
                         "Set track_pathways=True during training.")
    return history['pathway_strengths'][y][m]


def count_surviving_pathways(model, dataset, threshold_ratio=0.1):
    """
    Count how many pathways have non-negligible strength for each class.

    A pathway "survives" if its strength is > threshold_ratio * max_strength for that class.

    Args:
        model: MultiViewNet instance
        dataset: MultiViewDataset instance
        threshold_ratio: Ratio of max strength to count as surviving

    Returns:
        dict: {y: num_surviving_views} for each class
    """
    surviving = {}

    for y in range(dataset.K):
        strengths = []
        for m in range(dataset.M):
            phi_ym = dataset.get_view_feature(y, m)
            strengths.append(model.get_pathway_strength(phi_ym))

        max_s = max(strengths)
        threshold = threshold_ratio * max_s
        surviving[y] = sum(1 for s in strengths if s > threshold)

    return surviving
