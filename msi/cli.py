"""Command-line interface for MSI inference workflows."""

import argparse
from collections.abc import Sequence
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level MSI argument parser."""
    parser = argparse.ArgumentParser(
        prog="euclid-deeplss-inference",
        description="Run simulation-based inference and coverage workflows.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Importing these modules is intentionally deferred so importing ``msi.cli``
    # remains lightweight. Their parser configuration functions do not load any
    # models or data.
    from msi.apps import run_inference, run_mcmc_for_coverage_tests

    inference_parser = subparsers.add_parser(
        "inference",
        help="Train or load a flow and run inference.",
        description="Normalizing-flow inference on network summary statistics (maps or Cls).",
    )
    run_inference.configure_parser(inference_parser)
    inference_parser.set_defaults(handler=run_inference.main)

    coverage_parser = subparsers.add_parser(
        "coverage",
        help="Sample posteriors for coverage tests.",
        description="Sample held-out simulations for posterior coverage testing.",
    )
    run_mcmc_for_coverage_tests.configure_parser(coverage_parser)
    coverage_parser.set_defaults(handler=run_mcmc_for_coverage_tests.main)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse ``argv`` and dispatch to the selected workflow."""
    args = build_parser().parse_args(argv)
    handler = args.handler
    del args.handler
    del args.command
    result = handler(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
