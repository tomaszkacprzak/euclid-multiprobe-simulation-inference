"""PyTorch architecture for conditional Gaussian mixture likelihoods."""

import torch
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal


class GaussianMixtureNetwork(nn.Module):
    """Map conditioning variables to a mixture of full-covariance Gaussians."""

    def __init__(
        self,
        n_x,
        n_theta,
        n_gaussians=4,
        n_units=256,
        n_layers=4,
        activation="relu",
        dropout_rate=0.0,
        x_noise_sigma=0.0,
    ):
        super().__init__()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "gelu": nn.GELU}
        if activation not in activations:
            raise ValueError(f"Unsupported activation {activation!r}; choose from {', '.join(activations)}")
        self.n_x = int(n_x)
        self.n_theta = int(n_theta)
        self.n_gaussians = int(n_gaussians)
        self.x_noise_sigma = float(x_noise_sigma)
        blocks = []
        in_features = self.n_theta
        # Preserve the old architecture: one initial and n_layers additional hidden layers.
        for _ in range(n_layers + 1):
            blocks.extend((nn.Linear(in_features, n_units), activations[activation](), nn.Dropout(dropout_rate)))
            in_features = n_units
        triangle_size = self.n_x * (self.n_x + 1) // 2
        output_size = self.n_gaussians * (1 + self.n_x + triangle_size)
        self.network = nn.Sequential(*blocks, nn.Linear(in_features, output_size))

    def forward(self, theta):
        if self.training and self.x_noise_sigma:
            theta = theta + torch.randn_like(theta) * self.x_noise_sigma
        raw = self.network(theta)
        logits, means, triangle = torch.split(
            raw,
            [self.n_gaussians, self.n_gaussians * self.n_x, self.n_gaussians * self.n_x * (self.n_x + 1) // 2],
            dim=-1,
        )
        means = means.reshape(*theta.shape[:-1], self.n_gaussians, self.n_x)
        triangle = triangle.reshape(*theta.shape[:-1], self.n_gaussians, -1)
        scale = raw.new_zeros(*theta.shape[:-1], self.n_gaussians, self.n_x, self.n_x)
        rows, cols = torch.tril_indices(self.n_x, self.n_x, device=raw.device)
        scale[..., rows, cols] = triangle
        diagonal = torch.arange(self.n_x, device=raw.device)
        scale[..., diagonal, diagonal] = torch.nn.functional.softplus(scale[..., diagonal, diagonal]) + 1e-5
        return MixtureSameFamily(Categorical(logits=logits), MultivariateNormal(means, scale_tril=scale))


def get_gmm_layers(n_x, n_theta, **kwargs):
    """Build the conditional PyTorch Gaussian mixture network."""
    return GaussianMixtureNetwork(n_x, n_theta, **kwargs)
