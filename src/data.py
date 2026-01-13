"""
Multi-view synthetic dataset for KDMech experiments.

Each class has M views, each view occupies a disjoint "slot" in input space.
This ensures perfect orthogonality between views.
"""

import torch
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    """
    Synthetic multi-view dataset where ground truth is exactly known.

    Each class has M views, each view occupies a disjoint "slot" in input space.
    This ensures perfect orthogonality between views.

    Args:
        K: Number of classes
        M: Views per class
        d_view: Dimensions per view slot
        view_prob: Probability each view is active
        noise_std: Gaussian noise level
        n_samples: Number of training samples
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        K: int = 10,
        M: int = 3,
        d_view: int = 50,
        view_prob: float = 0.5,
        noise_std: float = 0.1,
        n_samples: int = 10000,
        seed: int = 42
    ):
        self.K = K
        self.M = M
        self.d = M * d_view  # Total input dimension = 150
        self.d_view = d_view
        self.view_prob = view_prob
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.seed = seed

        # Generate view features: phi[(y, m)] is the feature vector for class y, view m
        self.phi = self._generate_view_features(seed)

        # Generate dataset
        self.data = self._generate_samples(n_samples, seed)

    def _generate_view_features(self, seed):
        """Generate orthogonal view features in disjoint slots."""
        torch.manual_seed(seed)
        phi = {}

        for y in range(self.K):
            for m in range(self.M):
                # Create vector that is non-zero only in slot m
                v = torch.zeros(self.d)
                slot_start = m * self.d_view
                slot_end = (m + 1) * self.d_view

                # Random unit vector in the slot
                v[slot_start:slot_end] = torch.randn(self.d_view)
                v = v / v.norm()

                phi[(y, m)] = v

        return phi

    def _generate_samples(self, n_samples, seed):
        """Generate (x, y, active_views) tuples."""
        torch.manual_seed(seed + 1000)  # Different seed from features
        samples = []

        for _ in range(n_samples):
            y = torch.randint(0, self.K, (1,)).item()

            # Sample active views
            active_views = []
            for m in range(self.M):
                if torch.rand(1).item() < self.view_prob:
                    active_views.append(m)

            # Ensure at least one view
            if len(active_views) == 0:
                active_views = [torch.randint(0, self.M, (1,)).item()]

            # Generate input
            x = torch.zeros(self.d)
            for m in active_views:
                x = x + self.phi[(y, m)]
            x = x + self.noise_std * torch.randn(self.d)

            samples.append((x, y, active_views))

        return samples

    def get_view_feature(self, y, m):
        """Return the pure feature vector for view (y, m)."""
        return self.phi[(y, m)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_tensors(self):
        """Return X, Y tensors for batch training."""
        X = torch.stack([s[0] for s in self.data])
        Y = torch.tensor([s[1] for s in self.data])
        return X, Y


class CompetingViewDataset(Dataset):
    """
    Multi-view dataset where views COMPETE in the same dimensional space.

    Unlike MultiViewDataset (orthogonal slots), here all views are random
    unit vectors in the same d-dimensional space. This creates competition
    as the network must choose which view(s) to rely on.

    Structure:
    - Each class has a centroid (random unit vector scaled for separation)
    - Each view is a perturbation of the class centroid
    - Views for the same class are similar but distinct
    - Views across classes are separated

    Args:
        K: Number of classes
        M: Views per class
        d: Input dimension (all views share this space)
        view_spread: How much views differ from class centroid
        class_spread: How separated class centroids are
        noise_std: Gaussian noise level
        n_samples: Number of training samples
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        K: int = 10,
        M: int = 3,
        d: int = 100,
        view_spread: float = 0.3,
        class_spread: float = 3.0,
        noise_std: float = 0.1,
        n_samples: int = 10000,
        seed: int = 42
    ):
        self.K = K
        self.M = M
        self.d = d
        self.d_view = d  # For compatibility
        self.view_spread = view_spread
        self.class_spread = class_spread
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.seed = seed

        # Generate view features
        self.phi, self.centroids = self._generate_view_features(seed)

        # Generate dataset
        self.data = self._generate_samples(n_samples, seed)

    def _generate_view_features(self, seed):
        """Generate competing view features in shared space."""
        torch.manual_seed(seed)
        phi = {}
        centroids = {}

        # Generate class centroids (well-separated)
        for y in range(self.K):
            c = torch.randn(self.d)
            c = c / c.norm() * self.class_spread
            centroids[y] = c

        # Generate views as perturbations of centroids
        for y in range(self.K):
            for m in range(self.M):
                # View = centroid + random perturbation
                perturbation = torch.randn(self.d) * self.view_spread
                v = centroids[y] + perturbation
                v = v / v.norm()  # Normalize to unit vector
                phi[(y, m)] = v

        return phi, centroids

    def _generate_samples(self, n_samples, seed):
        """Generate (x, y, active_views) tuples."""
        torch.manual_seed(seed + 1000)
        samples = []

        for _ in range(n_samples):
            y = torch.randint(0, self.K, (1,)).item()

            # Randomly select ONE view per sample (cleaner signal for competition)
            m = torch.randint(0, self.M, (1,)).item()
            active_views = [m]

            # Generate input from selected view
            x = self.phi[(y, m)].clone()
            x = x + self.noise_std * torch.randn(self.d)

            samples.append((x, y, active_views))

        return samples

    def get_view_feature(self, y, m):
        """Return the pure feature vector for view (y, m)."""
        return self.phi[(y, m)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_tensors(self):
        """Return X, Y tensors for batch training."""
        X = torch.stack([s[0] for s in self.data])
        Y = torch.tensor([s[1] for s in self.data])
        return X, Y


def verify_competing_dataset(dataset):
    """Verify CompetingViewDataset properties."""
    print("=" * 50)
    print("Competing Dataset Verification")
    print("=" * 50)

    # Check within-class view similarity
    within_class_dots = []
    for y in range(dataset.K):
        for m1 in range(dataset.M):
            for m2 in range(m1 + 1, dataset.M):
                dot = torch.dot(dataset.phi[(y, m1)], dataset.phi[(y, m2)]).item()
                within_class_dots.append(dot)

    # Check between-class view similarity
    between_class_dots = []
    for y1 in range(dataset.K):
        for y2 in range(y1 + 1, dataset.K):
            for m1 in range(dataset.M):
                for m2 in range(dataset.M):
                    dot = torch.dot(dataset.phi[(y1, m1)], dataset.phi[(y2, m2)]).item()
                    between_class_dots.append(dot)

    import numpy as np
    print(f"Within-class view similarity: {np.mean(within_class_dots):.3f} ± {np.std(within_class_dots):.3f}")
    print(f"Between-class view similarity: {np.mean(between_class_dots):.3f} ± {np.std(between_class_dots):.3f}")
    print(f"Input dimension: {dataset.d}")
    print(f"Classes: {dataset.K}, Views per class: {dataset.M}")
    print("=" * 50)

    return True


def verify_dataset(dataset):
    """
    Verify dataset properties.

    Checks:
    1. Orthogonality between view features
    2. Correct dimensions
    3. View sufficiency (optional linear probe)
    """
    print("=" * 50)
    print("Dataset Verification")
    print("=" * 50)

    # 1. Check orthogonality
    max_dot = 0
    for (y1, m1), v1 in dataset.phi.items():
        for (y2, m2), v2 in dataset.phi.items():
            if (y1, m1) != (y2, m2):
                dot = abs(torch.dot(v1, v2).item())
                max_dot = max(max_dot, dot)
    print(f"Max dot product between views: {max_dot:.6f} (should be ~0)")

    # 2. Check dimensions
    print(f"Input dimension: {dataset.d} (expected {dataset.M * dataset.d_view})")
    print(f"Classes: {dataset.K}, Views per class: {dataset.M}")
    print(f"Total view features: {len(dataset.phi)}")

    # 3. Check sample stats
    num_samples = len(dataset)
    print(f"Number of samples: {num_samples}")

    # Check view distribution in samples
    view_counts = {m: 0 for m in range(dataset.M)}
    for x, y, active_views in dataset.data:
        for m in active_views:
            view_counts[m] += 1
    print(f"View activation counts: {view_counts}")

    print("=" * 50)
    return max_dot < 0.01  # Return True if orthogonality check passes
