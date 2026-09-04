"""Shared posterior-sampling behavior for likelihood implementations."""

import numpy as np
import torch

from msi.utils import mcmc
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def sample_likelihood_posterior(
    likelihood,
    x_obs,
    *,
    n_walkers=1_024,
    n_steps=1_000,
    n_burnin_steps=1_000,
    lambdaCDM=False,
    label=None,
    device=None,
    dont_save=False,
    method="ensemble",
):
    """Sample a prior-constrained posterior through a likelihood wrapper.

    The likelihood supplies ``_mcmc_log_posterior``; this helper owns the
    application-specific emcee setup, optional wCDM-to-lambdaCDM projection,
    output handling, and temporary device movement shared by all backends.
    """
    if method != "ensemble":
        raise ValueError("Only the 'ensemble' posterior sampling method is supported.")

    original_device = likelihood.device
    target_device = original_device if device is None else device
    x_obs = torch.as_tensor(x_obs, dtype=likelihood.floatx, device=target_device)
    x_obs = torch.atleast_2d(x_obs)
    description = "a single observation" if x_obs.shape[0] == 1 else "multiple observations"
    LOGGER.info(f"Sampling the posterior from {description}")

    likelihood.to(target_device)
    if hasattr(likelihood, "eval"):
        likelihood.eval()

    params = likelihood.params
    mcmc_label = label
    if lambdaCDM:
        LOGGER.warning("lambdaCDM")
        i_w = likelihood.params.index("w0")
        params = [parameter for parameter in likelihood.params if parameter != "w0"]
        mcmc_label = (label or "") + "_lambdaCDM"
    else:
        LOGGER.warning("wCDM")

    def log_prob_fn(theta_walkers):
        if lambdaCDM:
            theta_walkers = np.insert(theta_walkers, i_w, -1.0, axis=1)
        return likelihood._mcmc_log_posterior(theta_walkers, x_obs, device=target_device)

    try:
        return mcmc.run_emcee(
            log_prob_fn,
            params,
            conf=likelihood.conf,
            out_dir=likelihood.model_dir if not dont_save else None,
            label=mcmc_label,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burnin_steps=n_burnin_steps,
        )
    finally:
        likelihood.to(original_device)
