"""Application-wrapper tests that do not require the optional ODE package."""

import importlib
import sys
import types

import numpy as np
import pytest
import torch


def _load_wrapper(monkeypatch):
    """Import the wrapper with a deterministic stand-in for torchdiffeq."""
    pytest.importorskip("msfm")

    def odeint(_function, state, _times, **_kwargs):
        if isinstance(state, tuple):
            return tuple(torch.stack((component, component)) for component in state)
        return torch.stack((state, state))

    monkeypatch.setitem(sys.modules, "torchdiffeq", types.SimpleNamespace(odeint=odeint))
    sys.modules.pop("msi.flow_matching.cnf_cfm", None)
    sys.modules.pop("msi.likelihood_cnf", None)
    return importlib.import_module("msi.likelihood_cnf").LikelihoodCFM


def test_numpy_boundary_shapes_and_batching(monkeypatch):
    likelihood_class = _load_wrapper(monkeypatch)
    likelihood = likelihood_class(
        ["p0", "p1"],
        feature_dim=2,
        context_dim=2,
        load_existing=False,
        device="cpu",
    )
    likelihood.model._fitted.fill_(True)

    theta = np.zeros((3, 2), dtype=np.float64)
    samples = likelihood.sample_likelihood(theta, n_samples=1, batch_size=2)
    assert samples.shape == (3, 1, 2)
    assert samples.dtype == np.float32

    observations = np.zeros((2, 1, 2), dtype=np.float64)
    contexts = np.zeros((1, 3, 2), dtype=np.float64)
    log_prob = likelihood.log_likelihood(observations, contexts, return_numpy=True)
    assert log_prob.shape == (2, 3)


def test_checkpoint_restores_initialization_and_state(monkeypatch, tmp_path):
    likelihood_class = _load_wrapper(monkeypatch)
    likelihood = likelihood_class(
        ["p0", "p1"],
        out_dir=tmp_path,
        feature_dim=2,
        context_dim=2,
        dtype=torch.float64,
        random_seed=19,
        load_existing=False,
        device="cpu",
    )
    likelihood.model._fitted.fill_(True)
    likelihood.save()

    restored = likelihood_class.from_checkpoint(checkpoint_file=likelihood.model_file)
    assert restored.feature_dim == restored.context_dim == 2
    assert restored.dtype == torch.float64
    assert restored.random_seed == 19
    assert restored.model._fitted.item()
