"""Tests for dependency-free, lazy likelihood discovery."""

from unittest.mock import patch

import pytest

from msi.likelihood_registry import available_likelihoods, get_likelihood_class


def test_available_likelihoods_does_not_import_implementations():
    with patch("msi.likelihood_registry.import_module") as import_module:
        assert available_likelihoods() == ("flow", "gmm", "cfm")
        import_module.assert_not_called()


def test_get_likelihood_class_imports_only_selected_implementation():
    sentinel = object()
    module = type("Module", (), {"LikelihoodCFM": sentinel})

    with patch("msi.likelihood_registry.import_module", return_value=module) as import_module:
        assert get_likelihood_class(" CFM ") is sentinel

    import_module.assert_called_once_with("msi.flow_matching.likelihood_cfm")


def test_unknown_likelihood_reports_available_choices():
    with pytest.raises(ValueError, match="Available likelihoods: flow, gmm, cfm"):
        get_likelihood_class("missing")
