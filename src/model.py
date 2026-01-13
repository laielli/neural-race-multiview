"""
Neural network architecture for KDMech experiments.

Two-layer MLP with methods to extract gating patterns and measure pathway strengths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewNet(nn.Module):
    """
    Two-layer MLP for multi-view experiments.
    Includes methods to extract gating patterns and measure pathway strengths.

    Args:
        d: Input dimension (default 150 = 3 views * 50 dims)
        hidden: Hidden layer width
        K: Number of output classes
    """

    def __init__(self, d: int = 150, hidden: int = 200, K: int = 10):
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.K = K

        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, K)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def get_gating_pattern(self, x):
        """
        Return binary gating pattern for input x.

        Args:
            x: Input tensor, shape (batch, d) or (d,)

        Returns:
            Binary gating pattern, shape (batch, hidden) or (hidden,)
        """
        h_pre = self.fc1(x)
        return (h_pre > 0).float()

    def forward_with_gating(self, x, gating):
        """
        Forward pass with externally specified gating pattern.
        Used to compute pathway-specific outputs.

        Args:
            x: Input tensor
            gating: Binary gating pattern to apply

        Returns:
            Output logits
        """
        h_pre = self.fc1(x)
        h = h_pre * gating  # Apply external gating
        return self.fc2(h)

    def get_view_response(self, phi_ym):
        """
        Compute R_{y,m} = network output when only view (y,m) is present.
        This is the "view response" from Theorem 1.

        Args:
            phi_ym: The pure view feature vector, shape (d,)

        Returns:
            R_ym: Output logits, shape (K,)
        """
        with torch.no_grad():
            x = phi_ym.unsqueeze(0)  # Add batch dim
            return self(x).squeeze(0)

    def get_pathway_strength(self, phi_ym):
        """
        Compute pathway strength s_{y,m} = ||R_{y,m}||.

        This is a simplified measure. For full Frobenius norm of pathway matrix,
        see get_pathway_matrix_strength().

        Args:
            phi_ym: The pure view feature vector, shape (d,)

        Returns:
            float: Pathway strength (L2 norm of view response)
        """
        R_ym = self.get_view_response(phi_ym)
        return R_ym.norm().item()

    def get_pathway_matrix_strength(self, phi_ym):
        """
        Compute ||P_{y,m}||_F where P_{y,m} = W2 @ diag(g) @ W1.

        This is the full pathway strength from Theorem 2.

        Args:
            phi_ym: The pure view feature vector, shape (d,)

        Returns:
            float: Frobenius norm of the pathway matrix
        """
        with torch.no_grad():
            x = phi_ym.unsqueeze(0)
            g = self.get_gating_pattern(x).squeeze(0)  # (hidden,)

            # P = W2 @ diag(g) @ W1
            # ||P||_F can be computed as ||W2 @ diag(g) @ W1||_F
            W1 = self.fc1.weight  # (hidden, d)
            W2 = self.fc2.weight  # (K, hidden)

            # Apply gating to W1 rows
            W1_gated = W1 * g.unsqueeze(1)  # (hidden, d)
            P = W2 @ W1_gated  # (K, d)

            return P.norm().item()

    def get_all_pathway_strengths(self, dataset):
        """
        Measure pathway strength for all (class, view) pairs.

        Args:
            dataset: MultiViewDataset instance

        Returns:
            dict: {(y, m): strength} for all y in [K], m in [M]
        """
        strengths = {}
        for y in range(dataset.K):
            for m in range(dataset.M):
                phi_ym = dataset.get_view_feature(y, m)
                strengths[(y, m)] = self.get_pathway_strength(phi_ym)
        return strengths


def init_weights(module):
    """Standard weight initialization for reproducibility."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
