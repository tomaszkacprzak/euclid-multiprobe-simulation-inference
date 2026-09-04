import argparse

import pytest

from msi.apps.run_inference import _config_path, _validate_likelihood_config, configure_parser
from msi.likelihoods import LIKELIHOODS, get_likelihood


def test_likelihood_registry_uses_stable_distinct_model_names():
    assert set(LIKELIHOODS) == {"flow", "cfm", "gmm"}
    assert {entry.model_name for entry in LIKELIHOODS.values()} == {
        "likelihood_flow",
        "likelihood_cfm",
        "likelihood_gmm",
    }


def test_unknown_likelihood_has_actionable_error():
    with pytest.raises(ValueError, match="choose from flow, cfm, gmm"):
        get_likelihood("unknown")


def test_inference_parser_defaults_to_flow_and_accepts_registry_names():
    parser = configure_parser(argparse.ArgumentParser())

    assert parser.parse_args(["--out-dir", "output"]).likelihood_model == "flow"
    for name in LIKELIHOODS:
        args = parser.parse_args(["--out-dir", "output", "--likelihood-model", name])
        assert args.likelihood_model == name


def test_inference_parser_has_model_neutral_config_and_deprecated_alias():
    parser = configure_parser(argparse.ArgumentParser())

    args = parser.parse_args(["--out-dir", "output", "--likelihood-config", "cfm.yaml"])
    assert _config_path(args) == "cfm.yaml"

    args = parser.parse_args(["--out-dir", "output", "--flow-config", "legacy.yaml"])
    with pytest.warns(DeprecationWarning, match="--likelihood-config"):
        assert _config_path(args) == "legacy.yaml"


def test_config_options_are_mutually_exclusive():
    parser = configure_parser(argparse.ArgumentParser())
    args = parser.parse_args(["--out-dir", "output", "--likelihood-config", "new.yaml", "--flow-config", "old.yaml"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config_path(args)


def test_cfm_config_schema_accepts_constructor_and_adapter_settings():
    config = {
        "model": {
            "model_type": "mlp",
            "sigma_min": 0.01,
            "hidden_features": 32,
            "num_hidden_layers": 2,
            "ode_steps": 16,
            "divergence_estimator": "hutchinson",
            "hutchinson_samples": 2,
            "ode_method": "dopri5",
            "ode_rtol": 1e-5,
            "ode_atol": 1e-7,
            "ode_options": {"max_num_steps": 1000},
            "context_adapter": {"type": "linear"},
        },
        "training": {"n_epochs": 1},
        "preprocessing": {},
        "diagnostics": {"n_cosmos": 5},
        "mcmc": {"n_walkers": 8},
    }
    assert _validate_likelihood_config(config, "cfm") is config


def test_cfm_config_rejects_other_likelihood_family_options():
    with pytest.raises(ValueError, match="belong to the legacy flow likelihood"):
        _validate_likelihood_config({"transform": {"n_layers": 4}}, "cfm")


def test_cfm_config_rejects_unknown_nested_option():
    with pytest.raises(ValueError, match=r"model.*not_a_cfm_option"):
        _validate_likelihood_config({"model": {"not_a_cfm_option": True}}, "cfm")


def test_legacy_flow_config_remains_permissive():
    config = {"context_embedding": {"dim": 32}, "transform": {"n_layers": 4}}
    assert _validate_likelihood_config(config, "flow") is config
