# KDMech Experiment Source Code
# See CLAUDE.md for implementation guidelines

from .data import MultiViewDataset, CompetingViewDataset, verify_dataset, verify_competing_dataset
from .model import MultiViewNet, init_weights
from .metrics import (
    measure_all_pathway_strengths,
    measure_view_coverage,
    measure_view_coverage_detailed,
    measure_ensemble_coverage,
    find_winning_view,
    compute_view_diversity,
    count_surviving_pathways
)
from .train import train_hard_labels, train_kd, train_teachers

__all__ = [
    'MultiViewDataset',
    'CompetingViewDataset',
    'verify_dataset',
    'verify_competing_dataset',
    'MultiViewNet',
    'init_weights',
    'measure_all_pathway_strengths',
    'measure_view_coverage',
    'measure_view_coverage_detailed',
    'measure_ensemble_coverage',
    'find_winning_view',
    'compute_view_diversity',
    'count_surviving_pathways',
    'train_hard_labels',
    'train_kd',
    'train_teachers',
]
