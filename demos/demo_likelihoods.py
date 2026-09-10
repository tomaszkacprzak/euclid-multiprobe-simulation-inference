"""Train and compare the three registered likelihood families on a toy problem.

The demo learns ``p(y | x)`` for the two-dimensional simulator

    x ~ N(0, [[sigma_1**2, sigma_3**2],
              [sigma_3**2, sigma_2**2]])
    y = 0.5 x + N(0, sigma_n**2 I).

For each registered family (``flow``, ``gmm``, and ``cfm``), eight observations
are drawn and their likelihood is evaluated as a function of ``x`` on
``[-5, 5]^2``.  One 2-by-4 figure is saved per method.  The implementations are
deliberately small, self-contained PyTorch models: an affine conditional flow,
a mixture-density network, and a conditional flow-matching probability-flow
ODE.  They are intended as an executable teaching example rather than as
replacements for MSI's production likelihood wrappers.

Run, for example, with::

    python demos/demo_likelihoods.py --epochs 20 --output-dir likelihood_demo

The requested data size is the default.  ``--n-train`` and ``--grid-size`` are
useful for a quick CPU smoke test.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

# Make the documented ``python demos/demo_likelihoods.py`` invocation work from
# a source checkout as well as from an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msi.likelihood_registry import available_likelihoods

D = 2
N = 100_000
SIGMA_1 = 0.80
SIGMA_2 = 0.65
SIGMA_3 = 0.35
SIGMA_N = 0.20
METHODS = ("flow", "gmm", "cfm")
LOG_2PI = math.log(2.0 * math.pi)


def simulate(n: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Draw ``n`` pairs from the toy simulator using only PyTorch."""
    covariance = torch.tensor([[SIGMA_1**2, SIGMA_3**2], [SIGMA_3**2, SIGMA_2**2]], dtype=torch.float32)
    x = torch.randn(n, D, generator=generator) @ torch.linalg.cholesky(covariance).T
    noise = SIGMA_N * torch.randn(n, D, generator=generator)
    return x, 0.5 * x + noise


def mlp(input_dim: int, output_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output_dim)
    )


class ConditionalLikelihood(nn.Module):
    """Minimal common interface used by the plotting and training code."""

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        return -self.log_prob(y, x).mean()

    def log_prob(self, y: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError


class AffineFlow(ConditionalLikelihood):
    """A learned conditional affine flow with a full 2-D covariance."""

    def __init__(self) -> None:
        super().__init__()
        self.conditioner = mlp(D, D + 3)

    def log_prob(self, y: Tensor, x: Tensor) -> Tensor:
        mean, raw = self.conditioner(x).split((D, 3), dim=-1)
        diagonal = torch.nn.functional.softplus(raw[:, :2]) + 1.0e-3
        lower = torch.zeros(x.shape[0], D, D, device=x.device, dtype=x.dtype)
        lower[:, 0, 0], lower[:, 1, 0], lower[:, 1, 1] = diagonal[:, 0], raw[:, 2], diagonal[:, 1]
        residual = torch.linalg.solve_triangular(lower, (y - mean).unsqueeze(-1), upper=False).squeeze(-1)
        return -0.5 * (residual.square().sum(-1) + D * LOG_2PI) - torch.log(diagonal).sum(-1)


class GaussianMixture(ConditionalLikelihood):
    """A conditional mixture-density network with diagonal components."""

    def __init__(self, components: int = 5) -> None:
        super().__init__()
        self.components = components
        self.conditioner = mlp(D, components * (1 + 2 * D))

    def log_prob(self, y: Tensor, x: Tensor) -> Tensor:
        output = self.conditioner(x).reshape(-1, self.components, 1 + 2 * D)
        logits = output[..., 0]
        means = output[..., 1 : 1 + D]
        scales = torch.nn.functional.softplus(output[..., 1 + D :]) + 1.0e-3
        component_log_prob = -0.5 * (((y[:, None] - means) / scales).square() + LOG_2PI).sum(-1)
        component_log_prob -= torch.log(scales).sum(-1)
        return torch.logsumexp(torch.log_softmax(logits, -1) + component_log_prob, dim=-1)


class ConditionalFlowMatching(ConditionalLikelihood):
    """A compact conditional flow-matching model with Euler ODE likelihoods."""

    def __init__(self, ode_steps: int = 24) -> None:
        super().__init__()
        self.velocity = mlp(2 * D + 1, D)
        self.ode_steps = ode_steps

    def vector_field(self, state: Tensor, time: Tensor, x: Tensor) -> Tensor:
        return self.velocity(torch.cat((state, x, time.expand(state.shape[0], 1)), dim=-1))

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        base = torch.randn_like(y)
        time = torch.rand(y.shape[0], 1, device=y.device)
        path = (1.0 - time) * base + time * y
        return (self.vector_field(path, time, x) - (y - base)).square().mean()

    def log_prob(self, y: Tensor, x: Tensor) -> Tensor:
        # Follow the probability-flow ODE from data (t=1) back to its Gaussian
        # base, accumulating the exact two-dimensional divergence with autograd.
        with torch.enable_grad():
            state = y
            divergence_integral = torch.zeros(y.shape[0], device=y.device)
            dt = 1.0 / self.ode_steps
            for step in range(self.ode_steps, 0, -1):
                state = state.detach().requires_grad_(True)
                time = torch.full((1, 1), step / self.ode_steps, device=y.device)
                velocity = self.vector_field(state, time, x)
                divergence = torch.zeros_like(divergence_integral)
                for coordinate in range(D):
                    gradient = torch.autograd.grad(velocity[:, coordinate].sum(), state, retain_graph=True)[0]
                    divergence = divergence + gradient[:, coordinate]
                divergence_integral = divergence_integral + dt * divergence.detach()
                state = (state - dt * velocity).detach()
        base_log_prob = -0.5 * (state.square().sum(-1) + D * LOG_2PI)
        return base_log_prob - divergence_integral


MODEL_BUILDERS: dict[str, Callable[[], ConditionalLikelihood]] = {
    "flow": AffineFlow,
    "gmm": GaussianMixture,
    "cfm": ConditionalFlowMatching,
}


def train(model: ConditionalLikelihood, x: Tensor, y: Tensor, epochs: int, batch_size: int, device: str) -> None:
    """Fit one estimator by stochastic gradient descent."""
    model.to(device)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    for epoch in range(epochs):
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x_batch, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * x_batch.shape[0]
        print(f"  epoch {epoch + 1:02d}/{epochs}: loss={total / len(loader.dataset):.4f}")


def plot_likelihoods(
    model: ConditionalLikelihood,
    method: str,
    observations: Tensor,
    generating_x: Tensor,
    grid_size: int,
    output_dir: Path,
    device: str,
) -> Path:
    """Plot eight likelihood surfaces ``p(y_observed | x_grid)``."""
    axis = torch.linspace(-5.0, 5.0, grid_size)
    grid_x = torch.cartesian_prod(axis, axis).to(device)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True, sharex=True, sharey=True)
    model.eval()
    for index, ax in enumerate(axes.flat):
        grid_y = observations[index].to(device).expand(grid_x.shape[0], -1)
        log_probability = model.log_prob(grid_y, grid_x).detach().cpu()
        probability = torch.exp(log_probability).reshape(grid_size, grid_size)
        image = ax.contourf(axis, axis, probability.T, levels=30, cmap="viridis")
        ax.plot(*generating_x[index].tolist(), marker="x", color="red", markersize=8, markeredgewidth=2)
        ax.set_title(rf"$y_{{{index + 1}}}=({observations[index, 0]:.2f}, {observations[index, 1]:.2f})$")
        fig.colorbar(image, ax=ax, label=r"$p(y_i\mid x)$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$x_1$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$x_2$")
    fig.suptitle(f"Toy conditional likelihood: {method.upper()} (red x = generating value)")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"likelihood_{method}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=N, help="number of simulator pairs (default: 100000)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--grid-size", type=int, default=80, help="points per grid dimension")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("likelihood_demo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if D != 2 or args.n_train < 1 or args.epochs < 1 or args.grid_size < 2:
        raise ValueError("D must be 2; n-train and epochs must be positive; grid-size must be at least 2")
    registered = available_likelihoods()
    if tuple(registered) != METHODS:
        raise RuntimeError(f"Demo methods {METHODS} no longer match likelihood registry {registered}")

    generator = torch.Generator().manual_seed(args.seed)
    x_train, y_train = simulate(args.n_train, generator)
    generating_x, observations = simulate(8, generator)
    for method in registered:
        print(f"Training {method} likelihood on {args.n_train:,} pairs...")
        model = MODEL_BUILDERS[method]()
        train(model, x_train, y_train, args.epochs, args.batch_size, args.device)
        path = plot_likelihoods(
            model, method, observations, generating_x, args.grid_size, args.output_dir, args.device
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
