"""Conditional flow-matching models with lazy optional-dependency imports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cnf_cfm import ConditionalFlowMatchingLikelihood

__all__ = ["ConditionalFlowMatchingLikelihood", "LikelihoodCFM"]


def __getattr__(name):
    if name == "ConditionalFlowMatchingLikelihood":
        from .cnf_cfm import ConditionalFlowMatchingLikelihood

        return ConditionalFlowMatchingLikelihood
    if name == "LikelihoodCFM":
        from .likelihood_cfm import LikelihoodCFM

        return LikelihoodCFM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
