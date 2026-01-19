"""
Multi-task multi-view dataset for testing race dynamics.

Unlike single-task multi-view (where all views predict the same class),
this dataset has M independent classification tasks, one per view.
This creates task competition because views share network capacity
but need to solve different objectives.

Key insight: Race dynamics require multi-task structure, not just multi-view.
Single-task multi-view has no inter-view competition (all views serve same goal).
"""

import torch
from torch.utils.data import Dataset


class MultiTaskMultiViewDataset(Dataset):
    """
    Multi-task multi-view dataset with M independent classification tasks.

    Each sample has:
    - M view features (each d_view dimensional, orthogonal slots)
    - M task labels (each 0 to K-1, independent)

    View m's features encode task m's label using class prototypes.

    This creates the conditions for neural race dynamics:
    - Different views/tasks share network capacity
    - Each task wants to maximize its pathway strength
    - Competition emerges naturally

    Args:
        M: Number of views/tasks (default 5)
        K: Number of classes per task (default 10)
        d_view: Dimension per view (default 50)
        n_samples: Number of training samples (default 5000)
        noise_std: Noise added to features (default 0.1)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        M: int = 5,
        K: int = 10,
        d_view: int = 50,
        n_samples: int = 5000,
        noise_std: float = 0.1,
        seed: int = 42
    ):
        super().__init__()

        self.M = M
        self.K = K
        self.d_view = d_view
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.seed = seed

        torch.manual_seed(seed)

        # Generate M independent task labels per sample
        # Each task has its own label in [0, K-1]
        self.labels = torch.randint(0, K, (n_samples, M))

        # Generate orthogonal class prototypes for each task
        # Each task has K prototype vectors of dimension d_view
        self.prototypes = []
        for m in range(M):
            # Random unit vectors as prototypes
            proto = torch.randn(K, d_view)
            proto = proto / proto.norm(dim=1, keepdim=True)
            self.prototypes.append(proto)

        # Generate view features from prototypes + noise
        self.views = []
        for m in range(M):
            # Look up prototype for each sample's task-m label
            features = self.prototypes[m][self.labels[:, m]]  # (n_samples, d_view)
            # Add Gaussian noise
            features = features + noise_std * torch.randn_like(features)
            self.views.append(features)

        # Pre-concatenate for efficiency
        self._data = torch.cat(self.views, dim=1)  # (n_samples, M * d_view)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            x: Concatenated view features, shape (M * d_view,)
            y: Task labels, shape (M,) with values in [0, K-1]
        """
        return self._data[idx], self.labels[idx]

    def get_view_feature(self, task_idx: int, class_idx: int):
        """
        Get prototype feature for (task, class) pair.

        This returns the full input vector with only the specified
        view's slot filled in (other slots are zero).

        Args:
            task_idx: Which task/view (0 to M-1)
            class_idx: Which class (0 to K-1)

        Returns:
            Full input vector, shape (M * d_view,)
        """
        full_feature = torch.zeros(self.M * self.d_view)
        start = task_idx * self.d_view
        end = (task_idx + 1) * self.d_view
        full_feature[start:end] = self.prototypes[task_idx][class_idx]
        return full_feature

    def get_prototype(self, task_idx: int, class_idx: int):
        """
        Get raw prototype vector for (task, class).

        Args:
            task_idx: Which task/view (0 to M-1)
            class_idx: Which class (0 to K-1)

        Returns:
            Prototype vector, shape (d_view,)
        """
        return self.prototypes[task_idx][class_idx]

    def get_task_data(self, task_idx: int):
        """
        Get features and labels for a specific task.

        Args:
            task_idx: Which task (0 to M-1)

        Returns:
            features: View features for this task, shape (n_samples, d_view)
            labels: Class labels for this task, shape (n_samples,)
        """
        return self.views[task_idx], self.labels[:, task_idx]

    @property
    def input_dim(self):
        """Total input dimension."""
        return self.M * self.d_view

    @property
    def total_classes(self):
        """Total number of class-task combinations."""
        return self.M * self.K

    def __repr__(self):
        return (
            f"MultiTaskMultiViewDataset("
            f"M={self.M}, K={self.K}, d_view={self.d_view}, "
            f"n_samples={self.n_samples}, noise_std={self.noise_std})"
        )


def compute_task_accuracies(model, dataset, device='cpu'):
    """
    Compute per-task accuracy for a model on the dataset.

    Args:
        model: GatedDLN or similar with forward_pathway method
        dataset: MultiTaskMultiViewDataset

    Returns:
        dict: {task_idx: accuracy} for each task
    """
    model.eval()
    accuracies = {}

    with torch.no_grad():
        x = dataset._data.to(device)
        y = dataset.labels.to(device)

        for m in range(dataset.M):
            # Get predictions for task m
            # Extract view m's features
            x_view = x[:, m*dataset.d_view:(m+1)*dataset.d_view]
            output = model.forward_pathway(x_view, encoder_idx=m, decoder_idx=m)
            preds = output.argmax(dim=-1)

            # Compare to task m's labels
            correct = (preds == y[:, m]).float().mean().item()
            accuracies[m] = correct

    return accuracies


def compute_coverage(model, dataset, threshold: float = 0.5, device='cpu'):
    """
    Compute fraction of tasks where accuracy exceeds threshold.

    Args:
        model: Model to evaluate
        dataset: MultiTaskMultiViewDataset
        threshold: Accuracy threshold to count as "learned"

    Returns:
        float: Fraction of tasks with accuracy > threshold
    """
    accuracies = compute_task_accuracies(model, dataset, device)
    learned = sum(1 for acc in accuracies.values() if acc > threshold)
    return learned / len(accuracies)


if __name__ == '__main__':
    # Quick test
    print("Testing MultiTaskMultiViewDataset...")

    dataset = MultiTaskMultiViewDataset(M=5, K=10, d_view=50, n_samples=1000)
    print(f"Created: {dataset}")

    x, y = dataset[0]
    print(f"Sample x shape: {x.shape}")
    print(f"Sample y shape: {y.shape}")
    print(f"Sample y values: {y}")

    # Check prototype retrieval
    proto = dataset.get_view_feature(0, 0)
    print(f"View feature shape: {proto.shape}")

    # Check orthogonality of prototypes
    for m in range(dataset.M):
        proto_mat = dataset.prototypes[m]  # (K, d_view)
        gram = proto_mat @ proto_mat.T
        off_diag = gram - torch.eye(dataset.K)
        print(f"Task {m} off-diagonal Gram max: {off_diag.abs().max():.4f}")

    print("\nAll tests passed!")
