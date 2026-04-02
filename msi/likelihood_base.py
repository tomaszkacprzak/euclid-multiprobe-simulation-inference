# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Wrapper around enflows to build a likelihood normalizing flow with training and sampling utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

from msi.utils import plotting, diagnostics
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class LikelihoodBase(ABC):
    @abstractmethod
    def __init__(self, params, conf=None, out_dir=None, label=None, load_existing=False):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def sample_likelihood(self, theta_obs, n_samples, batch_size, return_numpy):
        pass

    @abstractmethod
    def log_likelihood(self, x, theta, return_numpy):
        pass

    @abstractmethod
    def sample_posterior(self, x_obs, n_samples, n_walkers, n_burnin_steps, label, device):
        pass

    @abstractmethod
    def _mcmc_log_posterior(self, theta_walkers, x_obs):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    # plotting ########################################################################################################

    def plot_contours(
        self,
        posterior_samples,
        # cosmetics
        scale_to_prior=True,
        group_params=True,
        density=False,
        # cosmo
        obs_point=None,
        obs_label="synthetic observation",
        with_des_chain=False,
        lambdaCDM=False,
        # output
        label=None,
    ):
        """
        Plot contours of the posterior samples.

        Args:
            samples (array-like): Samples from the posterior distribution.
            scale_to_prior (bool, optional): Whether to scale the plot to the prior distribution. Defaults to True.
            group_params (bool, optional): Whether to group cosmological and astrophysical parameters in the plot.
                Defaults to True.
            plot_fiducial (bool, optional): Whether to include the fiducial point in the plot. Defaults to True.
            fiducial_point (array-like, optional): Fiducial point to plot. Defaults to None.
            with_des_chain (bool, optional): Whether to include the DES chain in the plot. Defaults to False.
            label (str, optional): Additional label for the saved chain, for example to designate different
                observations. Defaults to None.
        """

        if lambdaCDM:
            label += "_lambdaCDM"
            params = [p for p in self.params if p != "w0"]
        else:
            params = self.params

        plotting.plot_chains(
            posterior_samples,
            params,
            self.conf,
            # file
            out_dir=self.model_dir,
            file_label=label,
            # cosmetics
            plot_labels=self.label,
            scale_to_prior=scale_to_prior,
            group_params=group_params,
            density=density,
            # cosmology
            obs_cosmo=obs_point,
            obs_label=obs_label,
            with_des_chain=with_des_chain,
        )

    def plot_diagnostics(
        self,
        grid_preds_true,
        grid_cosmos,
        # sampling
        n_cosmos=None,
        n_samples=100,
        batch_size=10000,
        # flags
        do_hist=False,
        do_dlss=False,
        do_eecp=True,
        do_tarp=True,
        tarp_kwargs={"n_bootstrap": 100, "n_alpha_bins": 20},
        # output
        out_dir=None,
        prefix="",
    ):
        """
        Plot diagnostics of how well the likelihood p(x|theta) has been learned from the (samples of the) true
        distribution.

        Args:
            grid_preds_true (ndarray): Array of shape (n_cosmos, n_examples, n_summary) or (n_cosmos, n_summary) true
                predictions for each cosmology in the grid. These are used as the true baseline to compare to.
            grid_cosmos (ndarray): Array of shape (n_cosmos, n_params) of the cosmologies in the grid. This is used
                to condition the flow and sample from it.
            n_cosmos (int, optional): Number of cosmologies to select randomly from the grid. Defaults to None, then
                all cosmologies are used.
            n_samples (int, optional): Number of samples per cosmology. Defaults to 100.
            batch_size (int, optional): Batch size for sampling. Defaults to 4096.

        Returns:
            ndarray: Array of shape (n_cosmos, n_samples, n_summary) containing samples from the likelihood
            for the whole grid.
        """

        assert grid_preds_true.shape[0] == grid_cosmos.shape[0], "n_cosmos must be the same for both arrays"
        assert grid_cosmos.ndim == 2, "grid_cosmos must have 2 dims containing (n_cosmos, n_params)"

        if out_dir is None:
            out_dir = self.model_dir
        os.makedirs(out_dir, exist_ok=True)

        if grid_preds_true.ndim == 2:
            LOGGER.warning(
                f"grid_preds_true.shape = {grid_preds_true.shape}, for sobol sequence + latin hypercube sampling"
            )
        elif grid_preds_true.ndim == 3:
            LOGGER.warning(f"grid_preds_true.shape = {grid_preds_true.shape}, for sobol sequence")
        else:
            raise ValueError(f"grid_preds_true.ndim = {grid_preds_true.ndim} not supported")

        if n_cosmos is not None:
            LOGGER.info(f"Selecting {n_cosmos} random cosmologies")
            random_indices = np.random.choice(grid_preds_true.shape[0], n_cosmos, replace=False)
            grid_preds_true = grid_preds_true[random_indices]
            grid_cosmos = grid_cosmos[random_indices]

        LOGGER.timer.start("sampling")
        LOGGER.info(f"Drawing samples from the likelihood")
        grid_preds_sample = self.sample_likelihood(
            grid_cosmos, n_samples=n_samples, batch_size=batch_size, return_numpy=True
        )
        LOGGER.info(f"Done drawing samples after {LOGGER.timer.elapsed('sampling')}")

        if do_hist:
            assert (
                grid_preds_true.ndim == 3
            ), "grid_preds_true must have 3 dims containing (n_cosmos, n_samples, n_summaries)"
            diagnostics.plot_histogram_check(
                grid_preds_true, grid_preds_sample, n_random_indices=10, out_dir=out_dir, prefix=prefix
            )
        if do_dlss:
            diagnostics.plot_deeplss_check(grid_preds_true, grid_preds_sample, out_dir=out_dir, prefix=prefix)
        if do_eecp:
            diagnostics.plot_eecp_check(
                grid_preds_true, grid_preds_sample, grid_cosmos, self, out_dir=out_dir, prefix=prefix
            )
        if do_tarp:
            diagnostics.plot_tarp_check(
                grid_preds_true, grid_preds_sample, grid_cosmos, out_dir=out_dir, prefix=prefix, **tarp_kwargs
            )

        # (n_cosmos, n_samples, n_summary)
        return grid_preds_sample, grid_preds_true, grid_cosmos

    def _plot_epochs(self, train_losses, vali_losses):
        """Produce a diagnostics plot of the loss curves after training has finished"""

        all_losses = np.concatenate([train_losses, vali_losses])

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(train_losses, label="training")
        ax.plot(vali_losses, label="validation")
        ax.set(
            xlabel="epoch", ylabel="loss", ylim=(np.nanquantile(all_losses, 0.01), np.nanquantile(all_losses, 0.99))
        )
        ax.grid(True)
        ax.legend()

        if self.model_dir is not None:
            fig.savefig(os.path.join(self.model_dir, "loss_curves.png"))

    # utils ###########################################################################################################

    def _setup_dirs(self, file_type):
        if self.model_dir is not None:
            self.model_file = os.path.join(self.model_dir, self.model_name + file_type)
        elif self.out_dir is not None and self.model_dir is None:
            if self.label is None:
                self.model_dir = os.path.join(self.out_dir, self.prefix + self.model_name + self.suffix)
            else:
                self.model_dir = os.path.join(self.out_dir, self.label, self.prefix + self.model_name + self.suffix)
            os.makedirs(self.model_dir, exist_ok=True)
            LOGGER.info(f"Set up the model directory {self.model_dir}")
            self.model_file = os.path.join(self.model_dir, self.model_name + file_type)
        else:
            self.model_dir = None
            self.model_file = None
