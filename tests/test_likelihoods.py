import argparse

import pytest

from msi.apps.run_inference import configure_parser
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
