"""Lazy lookup of the likelihood implementations shipped with MSI.

Keeping import paths, rather than imported classes, in the registry prevents an
optional backend from becoming a requirement merely by importing this module.
"""

from importlib import import_module
from typing import Dict, Tuple, Type

_LIKELIHOODS: Dict[str, Tuple[str, str]] = {
    "flow": ("msi.flow_conductor.likelihood_flow", "LikelihoodFlow"),
    "gmm": ("msi.gaussian_mixture.likelihood_gmm", "LikelihoodGMM"),
    "cfm": ("msi.flow_matching.likelihood_cfm", "LikelihoodCFM"),
}


def available_likelihoods() -> Tuple[str, ...]:
    """Return the registered likelihood names without importing any backend."""

    return tuple(_LIKELIHOODS)


def get_likelihood_class(name: str) -> Type:
    """Load and return the likelihood class registered under ``name``.

    Optional dependencies are consequently needed only for the selected
    likelihood. Names are case-insensitive and surrounding whitespace is
    ignored.
    """

    normalized_name = name.strip().lower()
    try:
        module_name, class_name = _LIKELIHOODS[normalized_name]
    except KeyError as exc:
        choices = ", ".join(available_likelihoods())
        raise ValueError(f"Unknown likelihood {name!r}. Available likelihoods: {choices}.") from exc

    module = import_module(module_name)
    return getattr(module, class_name)
