"""Conditional Gaussian-mixture likelihood implemented entirely in PyTorch."""

import copy
import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from msi.gaussian_mixture import architecture
from msi.likelihood_base import LikelihoodBase
from msi.utils import mcmc
from msfm.utils import files, logger, prior

LOGGER = logger.get_logger(__file__)


class LikelihoodGMM(nn.Module, LikelihoodBase):
    """Learn a conditional density ``p(x | theta)`` as a Gaussian mixture."""

    model_name = "likelihood_gmm"

    def __init__(
        self,
        params,
        conf=None,
        layers=None,
        out_dir=None,
        label=None,
        load_existing=True,
        floatx=torch.float32,
        prefix="",
        suffix="",
    ):
        nn.Module.__init__(self)
        self.params = params
        self.conf = files.load_config(conf)
        self.floatx = floatx
        self.out_dir, self.label, self.prefix, self.suffix = out_dir, label, prefix, suffix
        self.model_dir = None
        self._setup_dirs(".pt")
        self.network = layers if layers is not None else architecture.get_gmm_layers(len(params), len(params))
        self.scaler_x = self.scaler_theta = None
        if load_existing and self.model_file and os.path.exists(self.model_file):
            self.load()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, theta):
        return self.network(torch.as_tensor(theta, dtype=self.floatx, device=self.device))

    def fit(
        self,
        x,
        theta,
        n_epochs=1000,
        batch_size=10000,
        vali_split=0.1,
        learning_rate=1e-3,
        weight_decay=0.0,
        clip_by_global_norm=1.0,
        scheduler_kwargs=None,
        n_patience_epochs=10,
        min_delta=1e-3,
        fit_kwargs=None,
        save_model=True,
    ):
        x, theta = np.asarray(x), np.asarray(theta)
        if x.ndim != 2 or theta.ndim != 2 or len(x) != len(theta):
            raise ValueError("x and theta must be two-dimensional arrays with equal lengths")
        self.set_scalers(x, theta)
        x_tensor = torch.as_tensor(self.scale_forward_x(x), dtype=self.floatx)
        theta_tensor = torch.as_tensor(self.scale_forward_theta(theta), dtype=self.floatx)
        dataset = TensorDataset(theta_tensor, x_tensor)
        n_val = int(len(dataset) * vali_split)
        n_train = len(dataset) - n_val
        train_data, val_data = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size) if n_val else None
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = None
        if scheduler_kwargs is not None:
            options = {"min_lr": 1e-6, "factor": 0.75, "patience": 20, "cooldown": 5, "threshold": 1e-4}
            options.update(scheduler_kwargs)
            options.pop("min_delta", None)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **options)
        best, best_state, stale = float("inf"), None, 0
        train_losses, val_losses = [], []
        for _ in range(n_epochs):
            self.train()
            losses = []
            for context, target in train_loader:
                context, target = context.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = -self(context).log_prob(target).mean()
                loss.backward()
                if clip_by_global_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_by_global_norm)
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))
            train_losses.append(train_loss)
            val_loss = train_loss
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = float(
                        np.mean(
                            [-self(c.to(self.device)).log_prob(y.to(self.device)).mean().item() for c, y in val_loader]
                        )
                    )
            val_losses.append(val_loss)
            if scheduler is not None:
                scheduler.step(val_loss)
            if val_loss < best - min_delta:
                best, best_state, stale = val_loss, copy.deepcopy(self.state_dict()), 0
            else:
                stale += 1
                if n_patience_epochs is not None and stale >= n_patience_epochs:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        self._plot_epochs(train_losses, val_losses)
        if save_model:
            self.save()
        return {"loss": train_losses, "val_loss": val_losses}

    def set_scalers(self, x, theta):
        self.scaler_x = MinMaxScaler(feature_range=(1e-5, 1 - 1e-5)).fit(x)
        self.scaler_theta = RobustScaler().fit(theta)

    @staticmethod
    def _scale(inputs, transform):
        is_tensor = torch.is_tensor(inputs)
        array = inputs.detach().cpu().numpy() if is_tensor else np.asarray(inputs)
        if array.ndim < 2:
            raise ValueError("inputs must have at least two dimensions")
        output = transform(array.reshape(-1, array.shape[-1])).reshape(array.shape)
        return torch.as_tensor(output, dtype=inputs.dtype, device=inputs.device) if is_tensor else output

    def scale_forward_x(self, x):
        return self._scale(x, self.scaler_x.transform)

    def scale_inverse_x(self, x):
        return self._scale(x, self.scaler_x.inverse_transform)

    def scale_forward_theta(self, theta):
        return self._scale(theta, self.scaler_theta.transform)

    def scale_inverse_y(self, theta):
        return self._scale(theta, self.scaler_theta.inverse_transform)

    def sample_likelihood(self, theta, n_samples=1000, batch_size=10000, return_numpy=True):
        self.eval()
        batches = []
        with torch.no_grad():
            for batch in np.array_split(theta, max(1, int(np.ceil(len(theta) / batch_size)))):
                context = self.scale_forward_theta(batch)
                samples = self(context).sample((n_samples,)).transpose(0, 1)
                batches.append(self.scale_inverse_x(samples))
        result = torch.cat(batches, dim=0)
        return result.cpu().numpy() if return_numpy else result

    def log_likelihood(self, x, theta, return_numpy=False):
        tensor_input = torch.is_tensor(x) or torch.is_tensor(theta)
        x_scaled, theta_scaled = self.scale_forward_x(x), self.scale_forward_theta(theta)
        x_tensor = torch.as_tensor(x_scaled, dtype=self.floatx, device=self.device)
        theta_tensor = torch.as_tensor(theta_scaled, dtype=self.floatx, device=self.device)
        if x_tensor.shape[:-1] != theta_tensor.shape[:-1]:
            batch_shape = torch.broadcast_shapes(x_tensor.shape[:-1], theta_tensor.shape[:-1])
            x_tensor = x_tensor.expand(*batch_shape, x_tensor.shape[-1])
            theta_tensor = theta_tensor.expand(*batch_shape, theta_tensor.shape[-1])
        shape = x_tensor.shape[:-1]
        with torch.no_grad():
            result = (
                self(theta_tensor.reshape(-1, theta_tensor.shape[-1]))
                .log_prob(x_tensor.reshape(-1, x_tensor.shape[-1]))
                .reshape(shape)
            )
        if return_numpy or not tensor_input:
            return result.cpu().numpy()
        return result

    def sample_posterior(self, x_obs, n_samples=512000, n_walkers=1024, n_burnin_steps=100, label=None, device=None):
        x_obs = np.atleast_2d(np.asarray(x_obs))
        chain = mcmc.run_emcee(
            lambda walkers: self._mcmc_log_posterior(walkers, x_obs),
            self.params,
            conf=self.conf,
            out_dir=self.model_dir,
            label=label,
            n_walkers=n_walkers,
            n_steps=int(np.ceil(n_samples / n_walkers)),
            n_burnin_steps=n_burnin_steps,
        )
        return chain[:n_samples]

    def _mcmc_log_posterior(self, theta_walkers, x_obs):
        log_prob = sum(
            self.log_likelihood(np.repeat(x[None], len(theta_walkers), axis=0), theta_walkers)
            for x in np.atleast_2d(x_obs)
        )
        return prior.log_posterior(theta_walkers, log_prob, params=self.params, conf=self.conf)

    def save(self):
        if self.model_file is None:
            LOGGER.warning("Could not save the model, no model directory specified")
            return
        torch.save(self.state_dict(), self.model_file)
        with open(os.path.join(self.model_dir, "scalers.pkl"), "wb") as handle:
            pickle.dump([self.scaler_x, self.scaler_theta], handle)

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=self.device, weights_only=True))
        with open(os.path.join(self.model_dir, "scalers.pkl"), "rb") as handle:
            self.scaler_x, self.scaler_theta = pickle.load(handle)
