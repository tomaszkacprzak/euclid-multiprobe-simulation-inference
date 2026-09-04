"""Application-level context adapter tests."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Unit tests below do not integrate an ODE.  Keep them runnable in the base
# development environment where the optional CFM solver is not installed.
sys.modules.setdefault("torchdiffeq", SimpleNamespace(odeint=None))

from msi.flow_matching.likelihood_cfm import LikelihoodCFM


@pytest.mark.parametrize("theta_dim,feature_dim", [(3, 3), (2, 4), (5, 2)])
def test_adapter_preserves_declared_input_and_produces_cfm_context(theta_dim, feature_dim):
    model = LikelihoodCFM(
        [f"p{i}" for i in range(theta_dim)],
        feature_dim=feature_dim,
        cfm_config={"hidden_features": 4, "num_hidden_layers": 1},
        context_adapter_config={"type": "linear"},
    )
    theta = torch.randn(6, theta_dim)
    summaries = torch.randn(6, feature_dim)

    model.unfreeze()
    loss = model.flow_matching_loss(theta, summaries)
    loss.backward()

    assert model.theta_dim == theta_dim
    assert model.feature_dim == feature_dim
    assert model.context_adapter(theta).shape == (6, feature_dim)
    assert all(parameter.grad is not None for parameter in model.context_adapter.parameters())


def test_adapter_is_used_for_evaluation_and_sampling(monkeypatch):
    model = LikelihoodCFM(
        ["a", "b", "c", "d"],
        feature_dim=2,
        context_adapter_config={"type": "linear"},
    )
    model.cfm._fitted.fill_(True)
    received = {}

    def fake_log_prob(summaries, context, **_):
        received["log_context"] = context
        return summaries.sum(-1)

    def fake_sample(context, **_):
        received["sample_context"] = context
        return context

    monkeypatch.setattr(model.cfm, "log_prob", fake_log_prob)
    monkeypatch.setattr(model.cfm, "sample", fake_sample)
    theta = torch.randn(3, 4)
    expected = model.context_adapter(theta)

    model.log_prob(torch.randn(3, 2), theta)
    model.sample(theta)

    assert torch.equal(received["log_context"], expected)
    assert torch.equal(received["sample_context"], expected)


@pytest.mark.parametrize("conditioning_rows", [1, 3])
@pytest.mark.parametrize("n_samples", [1, 4])
def test_sample_likelihood_supplies_base_noise_with_expected_metadata(monkeypatch, conditioning_rows, n_samples):
    model = LikelihoodCFM(
        ["a", "b", "c"],
        feature_dim=2,
        context_adapter_config={"type": "linear"},
    ).double()
    model.cfm._fitted.fill_(True)
    received = {}

    def fake_sample(context, *, num_samples, base_noise, **_):
        received["context"] = context
        received["num_samples"] = num_samples
        received["base_noise"] = base_noise
        return base_noise

    monkeypatch.setattr(model.cfm, "sample", fake_sample)
    theta = torch.randn(conditioning_rows, 3, dtype=torch.float64)

    result = model.sample_likelihood(theta, n_samples=n_samples, return_numpy=False)

    expected_shape = (conditioning_rows, 2) if n_samples == 1 else (conditioning_rows, n_samples, 2)
    assert received["context"].shape == (conditioning_rows, 2)
    assert received["num_samples"] == n_samples
    assert received["base_noise"].shape == expected_shape
    assert received["base_noise"].dtype == theta.dtype
    assert received["base_noise"].device == theta.device
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "operation,call,pattern",
    [
        ("training", lambda model: model.flow_matching_loss(torch.randn(2, 2), torch.randn(2, 4)), "training: theta"),
        (
            "likelihood evaluation",
            lambda model: model.log_prob(torch.randn(2, 3), torch.randn(2, 3)),
            "likelihood evaluation: summaries",
        ),
        ("sampling", lambda model: model.sample(torch.randn(2, 4)), "sampling: theta"),
    ],
)
def test_dimension_errors_identify_operation(operation, call, pattern):
    del operation
    model = LikelihoodCFM(["a", "b", "c"], feature_dim=4)
    with pytest.raises(ValueError, match=pattern):
        call(model)


def test_checkpoint_metadata_records_dimensions_and_adapter_configuration():
    adapter_config = {
        "type": "mlp",
        "hidden_features": 7,
        "num_hidden_layers": 2,
        "activation": "tanh",
    }
    model = LikelihoodCFM(
        ["a", "b", "c", "d", "e"],
        feature_dim=2,
        context_adapter_config=adapter_config,
    )

    metadata = model.checkpoint_init_kwargs()

    assert metadata["theta_dim"] == 5
    assert metadata["feature_dim"] == 2
    assert metadata["context_adapter_config"] == adapter_config


def test_checkpoint_round_trip_preserves_validation_indices(tmp_path):
    conf = {"analysis": {"prior": "test-prior"}}
    model = LikelihoodCFM(["a", "b"], feature_dim=2, conf=conf, label="example")
    model.vali_indices = torch.tensor([1, 4, 7])
    model.split_metadata = {"method": "grouped", "vali_split": 0.1}

    restored = LikelihoodCFM.from_checkpoint(model.save(str(tmp_path / "likelihood_cfm.pt")))

    assert restored.get_validation_indices().tolist() == [1, 4, 7]
    assert restored.split_metadata == model.split_metadata
    assert restored.params == ["a", "b"]
    assert restored.conf == conf
    assert restored.label == "example"


def test_mcmc_log_posterior_applies_configured_prior(monkeypatch):
    model = LikelihoodCFM(["a", "b"], feature_dim=2, conf={"prior": "configured"})
    captured = {}

    def fake_log_likelihood(x, theta):
        return torch.full((theta.shape[0],), 2.0)

    def fake_log_posterior(theta, likelihood, *, conf, params):
        captured.update(theta=theta, likelihood=likelihood, conf=conf, params=params)
        return likelihood - 1.0

    monkeypatch.setattr(model, "log_likelihood", fake_log_likelihood)
    monkeypatch.setitem(sys.modules, "msfm", SimpleNamespace(utils=SimpleNamespace()))
    monkeypatch.setitem(
        sys.modules, "msfm.utils", SimpleNamespace(prior=SimpleNamespace(log_posterior=fake_log_posterior))
    )

    walkers = np.zeros((3, 2))
    result = model._mcmc_log_posterior(walkers, np.zeros((2, 2)))

    assert np.array_equal(result, np.full(3, 3.0))
    assert captured["theta"] is walkers
    assert np.array_equal(captured["likelihood"], np.full(3, 4.0))
    assert captured["conf"] is model.conf
    assert captured["params"] == model.params


def test_output_directory_uses_likelihood_base_naming_contract(tmp_path):
    model = LikelihoodCFM(["a", "b"], feature_dim=2, out_dir=str(tmp_path), prefix="pre_", suffix="_post", label="run")

    assert model.model_dir == str(tmp_path / "run" / "pre_likelihood_cfm_post")
    assert model.model_file == str(tmp_path / "run" / "pre_likelihood_cfm_post" / "likelihood_cfm.pt")
    assert callable(model.plot_contours)
    assert callable(model.plot_diagnostics)
