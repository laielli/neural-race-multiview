"""
Asymmetric Multi-View Dataset

Creates a dataset where different views have different predictive power.
This is essential for observing race dynamics - the view with highest
signal-to-noise ratio should win the neural race.

Signal strength per view:
- View 0: highest signal (sigma_0 >> others)
- View 1: medium signal
- ...
- View M-1: lowest signal

The theory predicts: view with highest σ_1(Σ_yx) learns fastest and dominates.
"""

import torch
from torch.utils.data import Dataset


class AsymmetricMultiViewDataset(Dataset):
    """
    Multi-view dataset where views have asymmetric predictive power.

    Args:
        M: Number of views
        K: Number of classes
        d_view: Dimensions per view
        signal_strengths: List of signal strengths per view (length M)
                         Higher = more predictive of class label
        noise_std: Background noise level
        n_samples: Number of training samples
        seed: Random seed
    """

    def __init__(
        self,
        M: int = 5,
        K: int = 10,
        d_view: int = 50,
        signal_strengths: list = None,
        noise_std: float = 0.1,
        n_samples: int = 5000,
        seed: int = 42
    ):
        self.M = M
        self.K = K
        self.d_view = d_view
        self.d = M * d_view
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.seed = seed

        # Default: exponentially decreasing signal strength
        if signal_strengths is None:
            # View 0 has signal 1.0, view 1 has 0.5, view 2 has 0.25, etc.
            signal_strengths = [1.0 / (2 ** m) for m in range(M)]
        self.signal_strengths = signal_strengths

        torch.manual_seed(seed)

        # Generate class prototypes for each view
        self.prototypes = self._generate_prototypes()

        # Generate dataset
        self._data, self._labels = self._generate_data()

    def _generate_prototypes(self):
        """Generate class prototypes with view-specific signal strengths."""
        prototypes = {}

        for m in range(self.M):
            for y in range(self.K):
                # Create feature vector in view m's slot
                v = torch.zeros(self.d)
                slot_start = m * self.d_view
                slot_end = (m + 1) * self.d_view

                # Random direction, scaled by signal strength
                v[slot_start:slot_end] = torch.randn(self.d_view)
                v = v / v.norm()  # Normalize
                v = v * self.signal_strengths[m]  # Scale by signal strength

                prototypes[(m, y)] = v

        return prototypes

    def _generate_data(self):
        """Generate samples: x = sum_m(signal_m * prototype_m) + noise."""
        torch.manual_seed(self.seed + 1000)

        X = []
        Y = []

        for _ in range(self.n_samples):
            y = torch.randint(0, self.K, (1,)).item()

            # Combine prototypes from all views
            x = torch.zeros(self.d)
            for m in range(self.M):
                x = x + self.prototypes[(m, y)]

            # Add noise
            x = x + self.noise_std * torch.randn(self.d)

            X.append(x)
            Y.append(y)

        return torch.stack(X), torch.tensor(Y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]

    def get_tensors(self):
        """Return all data as tensors."""
        return self._data, self._labels

    def get_view_feature(self, class_idx, view_idx):
        """Get the prototype feature for (class, view) pair."""
        full_feature = torch.zeros(self.d)
        proto = self.prototypes[(view_idx, class_idx)]
        return proto

    def get_correlation_matrix(self):
        """
        Compute input-output correlation matrix Σ_yx = E[y * x^T].

        For one-hot y, this is the mean of x per class.

        Returns:
            Σ_yx: (K, d) correlation matrix
        """
        Sigma = torch.zeros(self.K, self.d)

        for y in range(self.K):
            mask = self._labels == y
            Sigma[y] = self._data[mask].mean(dim=0)

        return Sigma

    def get_view_correlation_strengths(self):
        """
        Compute correlation strength (top singular value) per view.

        This predicts which view should win the neural race.

        Returns:
            List of σ_1 values per view
        """
        Sigma = self.get_correlation_matrix()

        strengths = []
        for m in range(self.M):
            # Extract view m's portion of correlation matrix
            slot_start = m * self.d_view
            slot_end = (m + 1) * self.d_view
            Sigma_m = Sigma[:, slot_start:slot_end]

            # Compute top singular value
            _, s, _ = torch.linalg.svd(Sigma_m)
            strengths.append(s[0].item())

        return strengths


def create_uniform_dataset(M=5, K=10, d_view=50, n_samples=5000, seed=42):
    """Create dataset where all views have equal signal (no race expected)."""
    signal_strengths = [1.0] * M
    return AsymmetricMultiViewDataset(
        M=M, K=K, d_view=d_view,
        signal_strengths=signal_strengths,
        n_samples=n_samples, seed=seed
    )


def create_dominant_view_dataset(M=5, K=10, d_view=50, n_samples=5000, seed=42, dominant_view=0):
    """Create dataset where one view has much stronger signal."""
    signal_strengths = [0.1] * M
    signal_strengths[dominant_view] = 1.0
    return AsymmetricMultiViewDataset(
        M=M, K=K, d_view=d_view,
        signal_strengths=signal_strengths,
        n_samples=n_samples, seed=seed
    )


def create_hierarchical_dataset(M=5, K=10, d_view=50, n_samples=5000, seed=42):
    """Create dataset with exponentially decreasing signal per view."""
    signal_strengths = [1.0 / (2 ** m) for m in range(M)]
    return AsymmetricMultiViewDataset(
        M=M, K=K, d_view=d_view,
        signal_strengths=signal_strengths,
        n_samples=n_samples, seed=seed
    )


if __name__ == '__main__':
    # Test the dataset
    print("Testing AsymmetricMultiViewDataset...")

    dataset = AsymmetricMultiViewDataset(M=5, K=10, d_view=50)
    print(f"Dataset size: {len(dataset)}")
    print(f"Signal strengths: {dataset.signal_strengths}")

    x, y = dataset[0]
    print(f"Sample x shape: {x.shape}")
    print(f"Sample y: {y}")

    # Check correlation strengths
    strengths = dataset.get_view_correlation_strengths()
    print(f"\nView correlation strengths (σ_1):")
    for m, s in enumerate(strengths):
        print(f"  View {m}: {s:.4f}")
    print(f"Predicted winner: View {strengths.index(max(strengths))}")
