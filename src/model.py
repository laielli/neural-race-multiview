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


class DeepLinearNet(nn.Module):
    """
    Deep linear network (no ReLU) for matching Saxe et al. theory exactly.

    The race dynamics theory is derived for linear networks with gradient flow.
    This class provides an exact match to the theoretical assumptions.

    Args:
        d: Input dimension
        hidden: Hidden layer width
        K: Number of output classes
    """

    def __init__(self, d: int = 150, hidden: int = 200, K: int = 10):
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.K = K

        self.fc1 = nn.Linear(d, hidden, bias=False)  # No bias for theory match
        self.fc2 = nn.Linear(hidden, K, bias=False)

    def forward(self, x):
        # Linear activations (no ReLU)
        h = self.fc1(x)
        return self.fc2(h)

    def get_pathway_strength(self, phi_ym):
        """Compute pathway strength s_{y,m} = ||R_{y,m}||."""
        with torch.no_grad():
            x = phi_ym.unsqueeze(0)
            R_ym = self(x).squeeze(0)
            return R_ym.norm().item()

    def get_all_pathway_strengths(self, dataset):
        """Measure pathway strength for all (class, view) pairs."""
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


def init_weights_linear(module):
    """Weight initialization for deep linear networks (small scale for gradient flow)."""
    if isinstance(module, nn.Linear):
        # Small initialization to start in the early phase of race dynamics
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class GatedDLN(nn.Module):
    """
    Gated Deep Linear Network following Saxe et al. 2022 architecture.

    This architecture has explicit pathway separation:
    - M separate input encoders (one per pathway/view)
    - Shared hidden layer
    - M separate output decoders (one per pathway/view)
    - Binary gate controlling which input-output pairs are connected

    The race dynamics emerge in the singular values of weight matrices,
    where different modes grow at different rates based on data correlations.

    Args:
        M: Number of pathways (typically = number of views)
        d_input: Input dimension per pathway
        hidden: Hidden layer width
        d_output: Output dimension
        gate_mode: How to construct the gate ('full', 'diagonal', 'k_neighbors')
        k_neighbors: For 'k_neighbors' mode, how many neighbors to connect
        init_scale: Scale for orthogonal initialization
    """

    def __init__(
        self,
        M: int = 3,
        d_input: int = 50,
        hidden: int = 64,
        d_output: int = 10,
        gate_mode: str = 'diagonal',
        k_neighbors: int = 1,
        init_scale: float = 0.2
    ):
        super().__init__()
        self.M = M
        self.d_input = d_input
        self.hidden = hidden
        self.d_output = d_output
        self.gate_mode = gate_mode
        self.k_neighbors = k_neighbors
        self.init_scale = init_scale

        # M separate input encoders (one per pathway)
        self.encoders = nn.ModuleList([
            nn.Linear(d_input, hidden, bias=False)
            for _ in range(M)
        ])

        # Shared hidden layer
        self.hidden_layer = nn.Linear(hidden, hidden, bias=False)

        # M separate output decoders (one per pathway)
        self.decoders = nn.ModuleList([
            nn.Linear(hidden, d_output, bias=False)
            for _ in range(M)
        ])

        # Create gating matrix
        self.register_buffer('gate', self._make_gate())

        # Loss function (MSE for theory match)
        self.loss_fn = nn.MSELoss(reduction='none')

    def _make_gate(self):
        """Create binary gating matrix controlling pathway connectivity."""
        gate = torch.zeros(self.M, self.M)

        if self.gate_mode == 'full':
            # All pathways connected
            gate = torch.ones(self.M, self.M)

        elif self.gate_mode == 'diagonal':
            # Each input only connects to corresponding output
            for i in range(self.M):
                gate[i, i] = 1.0

        elif self.gate_mode == 'k_neighbors':
            # Each input connects to k neighboring outputs (circular)
            for i in range(self.M):
                for j in range(-self.k_neighbors // 2, self.k_neighbors // 2 + 1):
                    gate[i, (i + j) % self.M] = 1.0

        elif self.gate_mode == 'k_plus_mod':
            # Matches gated-dln "k_plus_mod" pattern
            for i in range(self.M):
                for j in range(self.k_neighbors):
                    gate[i, (i + j) % self.M] = 1.0

        elif self.gate_mode == 'k_plus_minus_mod':
            # Matches gated-dln "k_plus_minus_mod" pattern
            for i in range(self.M):
                for j in range(-(self.k_neighbors - 1) // 2, self.k_neighbors // 2 + 1):
                    gate[i, (i + j + self.M) % self.M] = 1.0
        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        return gate

    def init_orthogonal(self, scale=None):
        """Initialize weights with orthogonal initialization at given scale."""
        if scale is None:
            scale = self.init_scale
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=scale)

    def forward_pathway(self, x, encoder_idx, decoder_idx):
        """
        Forward pass through a specific pathway.

        Args:
            x: Input tensor, shape (batch, d_input) or (d_input,)
            encoder_idx: Which encoder to use
            decoder_idx: Which decoder to use

        Returns:
            Output tensor
        """
        h = self.encoders[encoder_idx](x)
        h = self.hidden_layer(h)
        return self.decoders[decoder_idx](h)

    def forward(self, x_list, y_target=None):
        """
        Forward pass computing gated loss.

        Args:
            x_list: List of M input tensors, each shape (batch, d_input)
                    OR single tensor (batch, M * d_input) to be split
            y_target: Target tensor, shape (batch, d_output). If None, returns outputs.

        Returns:
            If y_target is None: List of outputs per active pathway
            If y_target is provided: Total gated loss
        """
        # Handle single concatenated input
        if isinstance(x_list, torch.Tensor) and x_list.dim() == 2:
            if x_list.shape[1] == self.M * self.d_input:
                x_list = [x_list[:, i*self.d_input:(i+1)*self.d_input] for i in range(self.M)]
            elif x_list.shape[1] == self.d_input:
                # Single view input - replicate for all encoders
                x_list = [x_list for _ in range(self.M)]

        # Encode through each pathway
        hidden_states = []
        for i, encoder in enumerate(self.encoders):
            h = encoder(x_list[i] if isinstance(x_list, list) else x_list)
            h = self.hidden_layer(h)
            hidden_states.append(h)

        if y_target is None:
            # Return all gated outputs
            outputs = []
            for i, h in enumerate(hidden_states):
                for j, decoder in enumerate(self.decoders):
                    if self.gate[i, j] > 0:
                        outputs.append(decoder(h))
            return outputs

        # Compute gated loss
        total_loss = 0.0
        num_active = 0
        for i, h in enumerate(hidden_states):
            for j, decoder in enumerate(self.decoders):
                if self.gate[i, j] > 0:
                    output = decoder(h)
                    loss = self.loss_fn(output, y_target).mean()
                    total_loss = total_loss + loss
                    num_active += 1

        return total_loss

    def get_singular_values(self):
        """
        Get singular values of all weight matrices.

        Returns:
            dict: {
                'encoders': list of SVs for each encoder,
                'hidden': SVs for hidden layer,
                'decoders': list of SVs for each decoder
            }
        """
        svs = {
            'encoders': [],
            'hidden': None,
            'decoders': []
        }

        with torch.no_grad():
            for enc in self.encoders:
                _, s, _ = torch.linalg.svd(enc.weight)
                svs['encoders'].append(s.cpu().numpy())

            _, s, _ = torch.linalg.svd(self.hidden_layer.weight)
            svs['hidden'] = s.cpu().numpy()

            for dec in self.decoders:
                _, s, _ = torch.linalg.svd(dec.weight)
                svs['decoders'].append(s.cpu().numpy())

        return svs

    def get_pathway_strength(self, pathway_idx):
        """
        Compute effective strength of a pathway via SVD.

        Pathway strength = product of top singular values across layers.

        Args:
            pathway_idx: Which pathway (encoder/decoder pair) to measure

        Returns:
            float: Pathway strength
        """
        with torch.no_grad():
            _, s_enc, _ = torch.linalg.svd(self.encoders[pathway_idx].weight)
            _, s_hid, _ = torch.linalg.svd(self.hidden_layer.weight)
            _, s_dec, _ = torch.linalg.svd(self.decoders[pathway_idx].weight)

            # Product of top singular values
            return (s_enc[0] * s_hid[0] * s_dec[0]).item()

    def get_all_pathway_strengths(self):
        """Get strengths for all pathways."""
        return [self.get_pathway_strength(i) for i in range(self.M)]

    def compute_dominance(self):
        """
        Compute pathway dominance ratio.

        Dominance = max pathway strength / sum of pathway strengths.
        Winner-take-all: dominance → 1
        Equal pathways: dominance → 1/M

        Returns:
            float: Dominance ratio in [1/M, 1]
        """
        strengths = self.get_all_pathway_strengths()
        total = sum(strengths)
        if total > 0:
            return max(strengths) / total
        return 1.0 / self.M

    def compute_svd_metrics(self):
        """
        Compute SVD-based metrics for all pathways.

        Metrics computed per encoder/decoder:
        - effective_rank: (sum(σ))² / sum(σ²) - higher = more balanced spectrum
        - spectral_entropy: -sum(σ_norm * log(σ_norm)) - higher = more balanced
        - top3_fraction: sum(top 3 σ) / sum(all σ) - lower = more balanced

        Returns:
            dict: Metrics keyed by 'enc_{m}_*' and 'dec_{m}_*'
        """
        metrics = {}

        with torch.no_grad():
            for m, (enc, dec) in enumerate(zip(self.encoders, self.decoders)):
                # Encoder SVD
                _, s_enc, _ = torch.linalg.svd(enc.weight)
                s_sum = s_enc.sum()
                s_sq_sum = (s_enc ** 2).sum()

                # Effective rank
                if s_sq_sum > 0:
                    eff_rank = (s_sum ** 2) / s_sq_sum
                else:
                    eff_rank = 0.0
                metrics[f'enc_{m}_eff_rank'] = eff_rank.item()

                # Spectral entropy
                if s_sum > 0:
                    s_norm = s_enc / s_sum
                    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
                else:
                    entropy = 0.0
                metrics[f'enc_{m}_entropy'] = entropy.item()

                # Top-3 fraction
                if s_sum > 0:
                    top3 = s_enc[:min(3, len(s_enc))].sum() / s_sum
                else:
                    top3 = 1.0
                metrics[f'enc_{m}_top3_frac'] = top3.item()

                # Decoder SVD
                _, s_dec, _ = torch.linalg.svd(dec.weight)
                s_sum = s_dec.sum()
                s_sq_sum = (s_dec ** 2).sum()

                if s_sq_sum > 0:
                    eff_rank = (s_sum ** 2) / s_sq_sum
                else:
                    eff_rank = 0.0
                metrics[f'dec_{m}_eff_rank'] = eff_rank.item()

        return metrics

    def get_all_singular_values(self):
        """
        Get all singular values for detailed analysis.

        Returns:
            dict: {
                'encoders': list of (M,) arrays of singular values,
                'hidden': array of singular values,
                'decoders': list of (M,) arrays of singular values
            }
        """
        result = {
            'encoders': [],
            'hidden': None,
            'decoders': []
        }

        with torch.no_grad():
            for enc in self.encoders:
                _, s, _ = torch.linalg.svd(enc.weight)
                result['encoders'].append(s.cpu().numpy())

            _, s, _ = torch.linalg.svd(self.hidden_layer.weight)
            result['hidden'] = s.cpu().numpy()

            for dec in self.decoders:
                _, s, _ = torch.linalg.svd(dec.weight)
                result['decoders'].append(s.cpu().numpy())

        return result


class GatedMultiViewNet(GatedDLN):
    """
    Gated DLN adapted for multi-view classification experiments.

    Wraps GatedDLN to work with MultiViewDataset:
    - Each view maps to a separate encoder pathway
    - Outputs are class logits
    - Can measure view-specific pathway strengths

    Args:
        dataset: MultiViewDataset to get dimensions from
        hidden: Hidden layer width
        gate_mode: Gating pattern ('diagonal', 'full', 'k_neighbors')
        k_neighbors: For k_neighbors mode
        init_scale: Initialization scale
    """

    def __init__(
        self,
        M: int = 3,
        d_view: int = 50,
        K: int = 10,
        hidden: int = 64,
        gate_mode: str = 'diagonal',
        k_neighbors: int = 1,
        init_scale: float = 0.2
    ):
        super().__init__(
            M=M,
            d_input=d_view,
            hidden=hidden,
            d_output=K,
            gate_mode=gate_mode,
            k_neighbors=k_neighbors,
            init_scale=init_scale
        )
        self.K = K
        self.d_view = d_view

    @classmethod
    def from_dataset(cls, dataset, hidden=64, gate_mode='diagonal', k_neighbors=1, init_scale=0.2):
        """Create from a MultiViewDataset."""
        return cls(
            M=dataset.M,
            d_view=dataset.d_view,
            K=dataset.K,
            hidden=hidden,
            gate_mode=gate_mode,
            k_neighbors=k_neighbors,
            init_scale=init_scale
        )

    def forward_multiview(self, x, view_idx=None):
        """
        Forward pass for multi-view input.

        Args:
            x: Input tensor, shape (batch, d) where d = M * d_view
            view_idx: If provided, only use this view's encoder

        Returns:
            Output logits, shape (batch, K)
        """
        batch_size = x.shape[0] if x.dim() == 2 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Split input into views
        x_views = [x[:, i*self.d_view:(i+1)*self.d_view] for i in range(self.M)]

        if view_idx is not None:
            # Only use specified view
            h = self.encoders[view_idx](x_views[view_idx])
            h = self.hidden_layer(h)
            return self.decoders[view_idx](h)

        # Average outputs from all active pathways
        outputs = []
        for i, x_v in enumerate(x_views):
            h = self.encoders[i](x_v)
            h = self.hidden_layer(h)
            for j in range(self.M):
                if self.gate[i, j] > 0:
                    outputs.append(self.decoders[j](h))

        if outputs:
            return torch.stack(outputs).mean(dim=0)
        return torch.zeros(batch_size, self.K)

    def get_view_pathway_strength(self, dataset, y, m):
        """
        Measure pathway strength for view (y, m) using the view's feature.

        Args:
            dataset: MultiViewDataset
            y: Class index
            m: View index

        Returns:
            float: Pathway strength for this view
        """
        phi_ym = dataset.get_view_feature(y, m)

        # Get the view-specific portion of phi
        view_input = phi_ym[m*self.d_view:(m+1)*self.d_view]

        with torch.no_grad():
            h = self.encoders[m](view_input.unsqueeze(0))
            h = self.hidden_layer(h)
            out = self.decoders[m](h)
            return out.norm().item()

    def measure_view_coverage(self, dataset, threshold=0.5):
        """
        Measure fraction of views correctly classified.

        Args:
            dataset: MultiViewDataset
            threshold: Confidence threshold

        Returns:
            float: Coverage in [0, 1]
        """
        detected = 0
        total = dataset.K * dataset.M

        self.eval()
        for y in range(dataset.K):
            for m in range(dataset.M):
                phi_ym = dataset.get_view_feature(y, m)

                with torch.no_grad():
                    logits = self.forward_multiview(phi_ym.unsqueeze(0))
                    probs = F.softmax(logits, dim=-1)

                pred = logits.argmax(dim=-1).item()
                conf = probs[0, y].item()

                if pred == y and conf > threshold:
                    detected += 1

        return detected / total
