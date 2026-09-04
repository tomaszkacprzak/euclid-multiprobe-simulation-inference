"""Application wrapper for the reusable conditional flow-matching model."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import torch

from msi.flow_matching.cnf_cfm import ConditionalFlowMatchingLikelihood
from msi.likelihood_base import LikelihoodBase
from msi.utils.posterior import sample_likelihood_posterior
from msfm.utils import files, logger, prior

LOGGER = logger.get_logger(__file__)


class LikelihoodCFM(LikelihoodBase):
    """MSI-facing likelihood that composes a reusable CFM estimator."""

    model_name = "likelihood_cfm"

    def __init__(
        self,
        params,
        conf=None,
        out_dir=None,
        model_dir=None,
        prefix="",
        suffix="",
        label=None,
        load_existing=True,
        feature_dim=None,
        context_dim=None,
        model_type="affine",
        sigma_min=0.05,
        hidden_features=128,
        num_hidden_layers=3,
        ode_steps=64,
        divergence_estimator="exact",
        hutchinson_samples=1,
        ode_method="rk4",
        ode_rtol=1.0e-5,
        ode_atol=1.0e-7,
        ode_options: Optional[Mapping[str, Any]] = None,
        device=None,
        floatx=torch.float32,
        torch_seed=7,
    ):
        context_dim = len(params) if context_dim is None else int(context_dim)
        feature_dim = context_dim if feature_dim is None else int(feature_dim)
        if feature_dim != context_dim:
            raise ValueError("ConditionalFlowMatchingLikelihood requires feature_dim == context_dim.")
        if context_dim != len(params):
            raise ValueError("context_dim must equal the number of constrained params.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Keep this dictionary limited to values that torch.save can restore.
        self._init_kwargs = {
            "params": list(params),
            "conf": conf,
            "out_dir": out_dir,
            "model_dir": model_dir,
            "prefix": prefix,
            "suffix": suffix,
            "label": label,
            "feature_dim": feature_dim,
            "context_dim": context_dim,
            "model_type": model_type,
            "sigma_min": sigma_min,
            "hidden_features": hidden_features,
            "num_hidden_layers": num_hidden_layers,
            "ode_steps": ode_steps,
            "divergence_estimator": divergence_estimator,
            "hutchinson_samples": hutchinson_samples,
            "ode_method": ode_method,
            "ode_rtol": ode_rtol,
            "ode_atol": ode_atol,
            "ode_options": dict(ode_options or {}),
            "device": str(device),
            "floatx": floatx,
            "torch_seed": torch_seed,
        }
        self.params = list(params)
        self.conf = files.load_config(conf)
        self.out_dir = out_dir
        self.model_dir = model_dir
        self.prefix = prefix
        self.suffix = suffix
        self.label = label
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.device = str(device)
        self.floatx = floatx
        self.torch_seed = torch_seed
        self._setup_dirs(".pt")

        torch.manual_seed(torch_seed)
        self.model = ConditionalFlowMatchingLikelihood(
            dimension=feature_dim,
            model_type=model_type,
            sigma_min=sigma_min,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            ode_steps=ode_steps,
            divergence_estimator=divergence_estimator,
            hutchinson_samples=hutchinson_samples,
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            ode_options=ode_options,
            device=device,
            dtype=floatx,
        )
        if load_existing:
            try:
                self.load()
            except FileNotFoundError:
                LOGGER.warning(f"Could not load the model from {self.model_file}")

    def to(self, device):
        """Move the composed model without changing the configured home device."""
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.floatx, device=self.model._reference_tensor().device)

    def fit(
        self,
        x,
        theta,
        n_epochs=50,
        batch_size=2048,
        learning_rate=3.0e-4,
        weight_decay=1.0e-6,
        clip_by_global_norm=None,
        save_model=True,
        cosine_decay=True,
        shuffle=True,
        verbose=True,
        **kwargs,
    ):
        """Fit p(x|theta), accepting NumPy arrays at the MSI boundary."""
        if kwargs:
            LOGGER.warning(f"Ignoring unsupported CFM training arguments: {sorted(kwargs)}")
        losses = self.model.fit(
            self._tensor(theta),
            self._tensor(x),
            epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=clip_by_global_norm,
            cosine_decay=cosine_decay,
            shuffle=shuffle,
            verbose=verbose,
        )
        if save_model:
            self.save()
        return {"train_loss": losses.detach().cpu().numpy()}

    def sample_likelihood(self, theta, n_samples=1000, batch_size=None, return_numpy=True):
        """Return samples shaped (conditions, samples, features)."""
        del batch_size  # The composed ODE model evaluates all conditions together.
        conditions = torch.atleast_2d(self._tensor(theta))
        samples = self.model.sample(conditions, num_samples=n_samples)
        if n_samples == 1:
            samples = samples[:, None, :]
        return samples.detach().cpu().numpy() if return_numpy else samples

    def log_likelihood(self, x, theta, return_numpy=False, **kwargs):
        """Evaluate p(x|theta), preserving all leading broadcast dimensions."""
        x_tensor = self._tensor(x)
        theta_tensor = self._tensor(theta)
        if x_tensor.shape[-1] != self.feature_dim or theta_tensor.shape[-1] != self.context_dim:
            raise ValueError("The final dimensions of x and theta do not match the model dimensions.")
        leading_shape = torch.broadcast_shapes(x_tensor.shape[:-1], theta_tensor.shape[:-1])
        x_flat = x_tensor.expand(*leading_shape, self.feature_dim).reshape(-1, self.feature_dim)
        theta_flat = theta_tensor.expand(*leading_shape, self.context_dim).reshape(-1, self.context_dim)
        log_prob = self.model.log_prob(x_flat, theta_flat, **kwargs).reshape(leading_shape)
        return log_prob.detach().cpu().numpy() if return_numpy else log_prob

    def sample_posterior(
        self,
        x_obs,
        n_walkers=1_024,
        n_steps=1_000,
        n_burnin_steps=1_000,
        lambdaCDM=False,
        label=None,
        device=None,
        dont_save=False,
        method="ensemble",
        **_,
    ):
        return sample_likelihood_posterior(
            self,
            x_obs,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burnin_steps=n_burnin_steps,
            lambdaCDM=lambdaCDM,
            label=label,
            device=device,
            dont_save=dont_save,
            method=method,
        )

    def _mcmc_log_posterior(self, theta_walkers, x_obs, device=None):
        device = self.device if device is None else device
        context = torch.as_tensor(theta_walkers, dtype=self.floatx, device=device)
        observation = torch.as_tensor(x_obs, dtype=self.floatx, device=device)
        if observation.ndim != 2:
            raise ValueError("x_obs must have shape (observations, features).")
        log_prob = np.zeros(context.shape[0])
        with torch.no_grad():
            for single_observation in observation:
                values = self.model.log_prob(single_observation[None, :], context)
                log_prob += values.cpu().numpy()
        return prior.log_posterior(theta_walkers, log_prob, conf=self.conf, params=self.params)

    def save(self):
        if self.model_file is None:
            LOGGER.warning("Could not save the model, no output directory specified")
            return
        torch.save({"state_dict": self.model.state_dict(), "init_kwargs": self._init_kwargs}, self.model_file)
        LOGGER.info(f"Saved the model to {self.model_file}")

    def load(self):
        if self.model_file is None:
            return
        checkpoint = torch.load(self.model_file, map_location=self.device, weights_only=False)
        state_dict = (
            checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        )
        self.model.load_state_dict(state_dict)
        LOGGER.info(f"Loaded the model from {self.model_file}")

    @classmethod
    def from_checkpoint(
        cls, checkpoint_file=None, model_dir=None, out_dir=None, prefix="", suffix="", label=None, **kwargs_overrides
    ):
        if checkpoint_file is None:
            if model_dir is None and out_dir is not None:
                base = prefix + cls.model_name + suffix
                model_dir = os.path.join(out_dir, base) if label is None else os.path.join(out_dir, label, base)
            if model_dir is not None:
                checkpoint_file = os.path.join(model_dir, cls.model_name + ".pt")
        if checkpoint_file is None:
            raise ValueError("Insufficient path arguments to determine checkpoint_file.")
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "init_kwargs" not in checkpoint:
            raise ValueError(f"The checkpoint at {checkpoint_file} does not contain 'init_kwargs'.")
        init_kwargs = dict(checkpoint["init_kwargs"])
        init_kwargs.update(kwargs_overrides)
        init_kwargs["model_dir"] = os.path.dirname(checkpoint_file)
        return cls(**init_kwargs, load_existing=True)
