"""Application-level context adapter tests."""

import sys
from types import SimpleNamespace

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
