"""
Experiment: Exact Saxe Notebook Replication

Replicates the exact setup from the gated-dln solution verification notebook
where race dynamics were validated against theory.

Key setup differences from our previous experiments:
1. X = identity matrix (orthogonal inputs)
2. Y = structured output with shared (6.0) and unique (3.0) features
3. M=7 pathways, K=4 trained (not diagonal gate)
4. "Decoupled initialization" aligns weights with SVD of Y
5. Track SVD of weight matrices, not "pathway dominance"
6. Theory ODE: b2' = K/M * b3 * b1 * (s - b3*b2*b1)
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

sys.path.insert(0, str(Path(__file__).parent.parent))


class SaxeGatedDLN(nn.Module):
    """
    GatedDLN matching exactly the Saxe notebook implementation.

    Key differences from our GatedDLN:
    - Gate mode "k_plus_minus_mod" creates overlapping connections
    - Multiple hidden layers (though notebook uses 1)
    - Loss computed per-pathway with gating mask
    """

    def __init__(
        self,
        num_pathways: int = 7,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 7,
        gate_mode: str = "4_plus_minus_mod",
        init_scale: float = 0.2
    ):
        super().__init__()
        self.M = num_pathways
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.init_scale = init_scale
        self.gate_mode = gate_mode

        # M separate input encoders
        self.input = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=False)
            for _ in range(num_pathways)
        ])

        # Single shared hidden layer
        self.hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # M separate output decoders
        self.out = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim, bias=False)
            for _ in range(num_pathways)
        ])

        # Create gate matrix
        self.register_buffer('gate', self._make_gate())

        # MSE loss
        self.loss_fn = nn.MSELoss(reduction='none')

    def _make_gate(self):
        """Create gate matching Saxe notebook patterns."""
        gate = torch.zeros(self.M, self.M)

        if self.gate_mode == "fully_connected":
            gate = torch.ones(self.M, self.M)

        elif "_plus_minus_mod" in self.gate_mode:
            # e.g., "4_plus_minus_mod" connects to 4 neighbors centered
            k = int(self.gate_mode.split("_plus_minus_mod")[0])
            for i in range(self.M):
                for j in range(-((k-1)//2), (k//2)+1):
                    gate[i, (i + j + self.M) % self.M] = 1.0

        elif "_plus_mod" in self.gate_mode:
            # e.g., "3_plus_mod" connects to 3 neighbors forward
            k = int(self.gate_mode.split("_plus_mod")[0])
            for i in range(self.M):
                for j in range(k):
                    gate[i, (i + j) % self.M] = 1.0

        elif self.gate_mode == "diagonal":
            for i in range(self.M):
                gate[i, i] = 1.0

        return gate

    def init_orthogonal(self):
        """Standard orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=self.init_scale)

    def init_decoupled(self, Y):
        """
        Decoupled initialization from Saxe notebook.
        Aligns weight matrices with SVD of target Y.
        """
        with torch.no_grad():
            Y_np = Y.T.numpy() if isinstance(Y, torch.Tensor) else Y.T
            u, s, v = np.linalg.svd(Y_np)

            # Get dimensions
            Ninp = self.input_dim
            Nh = self.hidden_dim
            Nout = self.output_dim

            # Get random orthogonal matrices for hidden layer
            r2, _, r1 = np.linalg.svd(self.hidden.weight.detach().numpy())

            # Initialize input encoders
            for m in self.input:
                # Align with right singular vectors of Y
                w = (r1.T)[:, :Ninp] @ (self.init_scale * np.eye(Ninp)) @ v
                m.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32))

            # Initialize hidden layer
            w = r2 @ (self.init_scale * np.eye(Nh)) @ r1
            self.hidden.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32))

            # Initialize output decoders
            for m in self.out:
                # Align with left singular vectors of Y
                w = u @ (self.init_scale * np.eye(Nout)) @ (r2.T)[:Nout, :]
                m.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forward(self, X, Y):
        """Compute gated loss like Saxe notebook."""
        # Encode through each input pathway
        features = [enc(X) for enc in self.input]

        # Pass through shared hidden
        hidden = [self.hidden(f) for f in features]

        # Compute loss for each gated pathway
        loss_tensor = torch.zeros(self.M, self.M)
        for i, h in enumerate(hidden):
            for j, dec in enumerate(self.out):
                if self.gate[i, j] == 1:
                    output = dec(h)
                    loss_tensor[i, j] = self.loss_fn(output, Y).mean()

        return loss_tensor.sum()

    def get_svd(self):
        """Get SVD of all weight matrices."""
        with torch.no_grad():
            # Input encoder 0
            _, s_in, _ = torch.linalg.svd(self.input[0].weight)

            # Hidden
            _, s_hid, _ = torch.linalg.svd(self.hidden.weight)

            # Output decoder 0
            _, s_out, _ = torch.linalg.svd(self.out[0].weight)

        return {
            'input': s_in.numpy(),
            'hidden': s_hid.numpy(),
            'output': s_out.numpy()
        }


def compute_theory_ode(Y, M, K, b0, lr, num_epochs):
    """
    Compute theoretical ODE solution from Saxe paper.

    The race dynamics follow:
    db1/dt = K/M² * b3 * b2 * (s - b3*b2*b1)
    db2/dt = K/M * b3 * b1 * (s - b3*b2*b1)
    db3/dt = K/M² * b2 * b1 * (s - b3*b2*b1)

    where s are the singular values of Y^T @ X (= Y^T for X=I).
    """
    # Get singular values of input-output correlation
    Y_np = Y.T.numpy() if isinstance(Y, torch.Tensor) else Y.T
    _, s, _ = np.linalg.svd(np.dot(Y_np, np.eye(Y_np.shape[1])))

    Nsv = len(s)
    dt = lr * 3.5 / 10  # Scaling from notebook

    # Initialize
    b1 = b0 * np.ones((num_epochs, Nsv))
    b2 = b0 * np.ones((num_epochs, Nsv))
    b3 = b0 * np.ones((num_epochs, Nsv))

    # Integrate ODE
    for t in range(num_epochs - 1):
        shat = b3[t] * b2[t] * b1[t]
        b1[t+1] = b1[t] + K/M**2 * b3[t] * b2[t] * (s - shat) * dt
        b2[t+1] = b2[t] + K/M * b3[t] * b1[t] * (s - shat) * dt
        b3[t+1] = b3[t] + K/M**2 * b2[t] * b1[t] * (s - shat) * dt

    # Theory loss
    loss_theory = K/4 * np.sum((s - b3*b2*b1)**2, axis=1)

    return {
        'b1': b1,
        'b2': b2,
        'b3': b3,
        's': s,
        'loss': loss_theory
    }


def run_experiment(
    M: int = 7,
    K: int = 4,  # Number of trained pathways
    input_dim: int = 4,
    hidden_dim: int = 64,
    output_dim: int = 7,
    init_scale: float = 0.2,
    lr: float = 0.02,
    epochs: int = 500,
    n_reps: int = 10,
    use_decoupled_init: bool = False,
    output_dir: str = 'results',
    verbose: bool = True
):
    """Run exact Saxe notebook experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)

    # Create Saxe-style data
    X = torch.eye(input_dim)

    # Structured output: shared + unique features
    Y = torch.zeros(input_dim, output_dim)
    Y[:, 0] = 6.0  # All inputs map to feature 0 (strongest correlation)
    Y[0:2, 1] = 4.0  # Inputs 0,1 map to feature 1
    Y[2:4, 2] = 4.0  # Inputs 2,3 map to feature 2
    for i in range(input_dim):
        Y[i, 3 + i] = 3.0  # Each input has unique feature

    print("="*70)
    print("Saxe Exact Replication Experiment")
    print("="*70)
    print(f"M={M} pathways, K={K} trained")
    print(f"init_scale={init_scale}, lr={lr}, epochs={epochs}")
    print(f"Decoupled init: {use_decoupled_init}")
    print(f"\nX:\n{X}")
    print(f"\nY:\n{Y}")

    # Get theoretical SVD
    _, s_theory, _ = np.linalg.svd(Y.T.numpy())
    print(f"\nTarget singular values: {s_theory}")
    print("="*70)

    # Compute theory ODE
    theory = compute_theory_ode(Y, M, K, init_scale, lr, epochs * 10)

    # Run multiple replications
    all_losses = []
    all_input_svs = []
    all_hidden_svs = []
    all_output_svs = []

    for rep in range(n_reps):
        if verbose:
            print(f"\nReplication {rep+1}/{n_reps}")

        torch.manual_seed(rep)

        # Create model
        model = SaxeGatedDLN(
            num_pathways=M,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            gate_mode=f"{K}_plus_minus_mod",
            init_scale=init_scale if not use_decoupled_init else init_scale/2.5
        )

        if use_decoupled_init:
            model.init_decoupled(Y)
        else:
            model.init_orthogonal()

        print(f"Gate (K={K}):\n{model.gate}")

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

        # Training
        losses = []
        input_svs = []
        hidden_svs = []
        output_svs = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = model(X, Y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 2 == 0:
                svd = model.get_svd()
                input_svs.append(svd['input'])
                hidden_svs.append(svd['hidden'][:10])
                output_svs.append(svd['output'][:7])

        all_losses.append(np.array(losses))
        all_input_svs.append(np.vstack(input_svs))
        all_hidden_svs.append(np.vstack(hidden_svs))
        all_output_svs.append(np.vstack(output_svs))

        if verbose:
            print(f"  Final loss: {losses[-1]:.4f}")
            print(f"  Final hidden SVs: {hidden_svs[-1][:4]}")

    # Convert to arrays
    all_losses = np.array(all_losses)
    all_hidden_svs = np.array(all_hidden_svs)

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Loss comparison
    ax = axes[0]
    for rep in range(n_reps - 1):
        ax.plot(all_losses[rep], color='red', linewidth=0.5, alpha=0.5)
    ax.plot(all_losses[-1], color='red', linewidth=0.5, label='Simulations')
    ax.plot(theory['loss'], color='green', linestyle='--', linewidth=2, label='Theory ODE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss: Simulation vs Theory')
    ax.legend()
    ax.set_xlim(0, epochs)
    ax.grid(True, alpha=0.3)

    # SVD comparison
    ax = axes[1]
    ep = np.arange(0, epochs, 2)

    # Simulations
    for rep in range(n_reps - 1):
        ax.plot(ep, all_hidden_svs[rep][:, :4], color='red', linewidth=0.5, alpha=0.3)
    ax.plot(ep, all_hidden_svs[-1][:, 0], color='red', linewidth=1, label='Sim (hidden SVs)')
    ax.plot(ep, all_hidden_svs[-1][:, 1:4], color='red', linewidth=1)

    # Theory
    ax.plot(theory['b2'][:, 0], color='blue', linewidth=2, linestyle='--', label='Theory b2 (hidden)')
    ax.plot(theory['b2'][:, 1:4], color='blue', linewidth=2, linestyle='--')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Value Evolution: Simulation vs Theory')
    ax.legend()
    ax.set_xlim(0, epochs)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'figures' / 'saxe_exact_comparison.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_path / 'figures' / 'saxe_exact_comparison.png'}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Final loss (sim): {np.mean(all_losses[:, -1]):.4f} ± {np.std(all_losses[:, -1]):.4f}")
    print(f"Final loss (theory): {theory['loss'][-1]:.4f}")

    # Check if SVs match theory
    final_hidden_svs = np.mean([svs[-1] for svs in all_hidden_svs], axis=0)
    print(f"\nFinal hidden SVs (sim): {final_hidden_svs[:4]}")
    print(f"Final hidden SVs (theory): {theory['b2'][-1, :4]}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'M': M, 'K': K, 'input_dim': input_dim, 'hidden_dim': hidden_dim,
            'output_dim': output_dim, 'init_scale': init_scale, 'lr': lr,
            'epochs': epochs, 'n_reps': n_reps, 'use_decoupled_init': use_decoupled_init
        },
        'target_singular_values': s_theory.tolist(),
        'final_loss_sim_mean': float(np.mean(all_losses[:, -1])),
        'final_loss_sim_std': float(np.std(all_losses[:, -1])),
        'final_loss_theory': float(theory['loss'][-1]),
        'final_hidden_svs_sim': final_hidden_svs[:4].tolist(),
        'final_hidden_svs_theory': theory['b2'][-1, :4].tolist()
    }

    with open(output_path / 'saxe_exact_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path / 'saxe_exact_results.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Saxe Exact Replication')
    parser.add_argument('--M', type=int, default=7, help='Number of pathways')
    parser.add_argument('--K', type=int, default=4, help='Number of trained pathways')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--init-scale', type=float, default=0.2, help='Init scale')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--reps', type=int, default=10, help='Number of replications')
    parser.add_argument('--decoupled', action='store_true', help='Use decoupled init')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 100
        args.reps = 3
        print("Quick mode: epochs=100, reps=3")

    run_experiment(
        M=args.M,
        K=args.K,
        hidden_dim=args.hidden,
        init_scale=args.init_scale,
        lr=args.lr,
        epochs=args.epochs,
        n_reps=args.reps,
        use_decoupled_init=args.decoupled,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
