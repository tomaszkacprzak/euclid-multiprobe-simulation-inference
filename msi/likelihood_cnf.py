"""Conditional normalizing-flow likelihood entry point."""


def main(config_file: str, input_samples: str):
    """Evaluate the likelihood for a set of samples.

    Parameters
    ----------
    config_file
        Path to the likelihood configuration in YAML format.
    input_samples
        Path to the input samples in HDF4 format.

    Notes
    -----
    The likelihood implementation will be added here.  This entry point is
    currently a stub so callers can integrate with the command-line interface.
    """

    print('config_file', config_file)
    print('input_samples', input_samples)
