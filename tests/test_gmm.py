import numpy as np
import torch

from msi.gaussian_mixture.architecture import get_gmm_layers


def test_gmm_network_produces_trainable_distribution():
    model = get_gmm_layers(2, 3, n_gaussians=2, n_units=8, n_layers=1)
    context = torch.randn(5, 3)
    target = torch.randn(5, 2)

    distribution = model(context)
    loss = -distribution.log_prob(target).mean()
    loss.backward()

    assert distribution.sample((4,)).shape == (4, 5, 2)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_gmm_network_accepts_multidimensional_batches():
    model = get_gmm_layers(2, 3, n_gaussians=2, n_units=8, n_layers=0)
    distribution = model(torch.randn(4, 5, 3))

    assert distribution.log_prob(torch.randn(4, 5, 2)).shape == (4, 5)
    assert np.isfinite(distribution.sample().detach().numpy()).all()
