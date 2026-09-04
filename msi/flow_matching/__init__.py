"""Conditional flow-matching models with lazy optional-dependency imports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cnf_cfm import ConditionalFlowMatchingLikelihood

__all__ = ["ConditionalFlowMatchingLikelihood"]


def __getattr__(name):
    if name == "ConditionalFlowMatchingLikelihood":
        from .cnf_cfm import ConditionalFlowMatchingLikelihood

        return ConditionalFlowMatchingLikelihood
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
