"""Application adapter for the conditional flow-matching likelihood."""

from __future__ import annotations

from copy import deepcopy
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from msi.utils.dataset_split import validation_split_indices

from .cnf_cfm import ConditionalFlowMatchingLikelihood

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class ContextAdapter(nn.Sequential):
    """Trainable map from application parameters to CFM-sized contexts."""

    def __init__(self, input_dim: int, output_dim: int, config: Mapping[str, Any]):
        adapter_type = config.get("type", "mlp")
        if adapter_type not in ("linear", "mlp"):
            raise ValueError("context_adapter.type must be 'linear' or 'mlp'.")

        if adapter_type == "linear":
            layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
        else:
            hidden_features = int(config.get("hidden_features", 64))
            num_hidden_layers = int(config.get("num_hidden_layers", 1))
            if hidden_features < 1:
                raise ValueError("context_adapter.hidden_features must be positive.")
            if num_hidden_layers < 0:
                raise ValueError("context_adapter.num_hidden_layers must be nonnegative.")
            activation_name = str(config.get("activation", "silu")).lower()
            try:
                activation = _ACTIVATIONS[activation_name]
            except KeyError as exc:
                raise ValueError(
                    "context_adapter.activation must be one of: " + ", ".join(_ACTIVATIONS) + "."
                ) from exc
            dimensions = [input_dim] + [hidden_features] * num_hidden_layers + [output_dim]
            layers = []
            for index, (in_features, out_features) in enumerate(zip(dimensions, dimensions[1:])):
                layers.append(nn.Linear(in_features, out_features))
                if index < len(dimensions) - 2:
                    layers.append(activation())
        super().__init__(*layers)


class _Standardizer(nn.Module):
    """A small, state-dict-aware per-coordinate standardizer."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, values: Tensor) -> None:
        with torch.no_grad():
            self.mean.copy_(values.mean(dim=0))
            # Constant columns are left unchanged, as in sklearn's StandardScaler.
            scale = values.std(dim=0, unbiased=False)
            self.scale.copy_(torch.where(scale > 0, scale, torch.ones_like(scale)))
            self.fitted.fill_(True)

    def forward(self, values: Tensor) -> Tensor:
        return (values - self.mean) / self.scale

    def inverse(self, values: Tensor) -> Tensor:
        return values * self.scale + self.mean

    @property
    def log_abs_det(self) -> Tensor:
        return -torch.log(self.scale).sum()


class LikelihoodCFM(nn.Module):
    """Application-facing CFM with a learned cosmology-context adapter.

    The underlying CFM uses ``feature_dim`` for both its state and context.
    This wrapper preserves all ``theta_dim`` cosmological inputs by learning a
    map between the two spaces instead of truncating or padding parameters.
    """

    model_name = ConditionalFlowMatchingLikelihood.model_name

    def __init__(
        self,
        params: Sequence[str],
        *,
        feature_dim: int,
        cfm_config: Optional[Mapping[str, Any]] = None,
        context_adapter_config: Optional[Mapping[str, Any]] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.params = list(params)
        self.theta_dim = len(self.params)
        self.feature_dim = int(feature_dim)
        if self.theta_dim < 1:
            raise ValueError("params must contain at least one cosmological parameter.")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive.")

        self.cfm_config = deepcopy(dict(cfm_config or {}))
        self.context_adapter_config = deepcopy(dict(context_adapter_config or {}))
        self.model_dir = model_dir
        self.context_adapter = ContextAdapter(self.theta_dim, self.feature_dim, self.context_adapter_config)
        self.cfm = ConditionalFlowMatchingLikelihood(dimension=self.feature_dim, **self.cfm_config)
        self.feature_standardizer = _Standardizer(self.feature_dim)
        self.parameter_standardizer = _Standardizer(self.theta_dim)
        self.train_losses: list[float] = []
        self.vali_losses: list[float] = []
        self.freeze()

    def checkpoint_init_kwargs(self) -> dict[str, Any]:
        """Return complete, serializable architecture metadata."""
        return {
            "params": self.params,
            "theta_dim": self.theta_dim,
            "feature_dim": self.feature_dim,
            "cfm_config": deepcopy(self.cfm_config),
            "context_adapter_config": deepcopy(self.context_adapter_config),
            "model_dir": self.model_dir,
        }

    def _plot_epochs(self) -> None:
        """Use the common ``LikelihoodBase`` loss-curve implementation lazily.

        The lazy import keeps the standalone CFM usable without the optional
        application stack imported by :mod:`msi.likelihood_base`.
        """
        from msi.likelihood_base import LikelihoodBase

        LikelihoodBase._plot_epochs(self, self.train_losses, self.vali_losses)

    def freeze(self) -> None:
        self.cfm.freeze()
        for parameter in self.context_adapter.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        self.cfm.unfreeze()
        for parameter in self.context_adapter.parameters():
            parameter.requires_grad_(True)

    def _tensor(self, value: Any) -> Tensor:
        reference = next(self.parameters())
        if isinstance(value, Tensor):
            return value.to(device=reference.device, dtype=reference.dtype)
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _validate_matrix(name: str, value: Tensor, dimension: int, operation: str) -> None:
        if value.ndim != 2 or value.shape[-1] != dimension:
            raise ValueError(
                f"{operation}: {name} must have shape (batch, {dimension}); " f"received {tuple(value.shape)}."
            )
        if value.shape[0] < 1:
            raise ValueError(f"{operation}: {name} must contain at least one batch item.")

    def _adapt(self, theta: Any, operation: str) -> Tensor:
        theta_tensor = self._tensor(theta)
        self._validate_matrix("theta", theta_tensor, self.theta_dim, operation)
        return self.context_adapter(theta_tensor)

    def flow_matching_loss(self, theta: Any, summaries: Any, **kwargs: Any) -> Tensor:
        summaries_tensor = self._tensor(summaries)
        self._validate_matrix("summaries", summaries_tensor, self.feature_dim, "training")
        context = self._adapt(theta, "training")
        if context.shape[0] != summaries_tensor.shape[0]:
            raise ValueError(
                "training: theta and summaries must have the same batch size; "
                f"received {context.shape[0]} and {summaries_tensor.shape[0]}."
            )
        return self.cfm.flow_matching_loss(context, summaries_tensor, **kwargs)

    def fit(
        self,
        theta: Any,
        summaries: Any,
        *,
        epochs: int = 50,
        batch_size: int = 2048,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-6,
        shuffle: bool = True,
        verbose: bool = True,
        vali_split: float = 0.1,
        seed: Optional[int] = None,
        group_ids: Any = None,
        scheduler_type: Optional[str] = None,
        scheduler_kwargs: Optional[Mapping[str, Any]] = None,
        n_patience_epochs: Optional[int] = None,
        min_delta: float = 1.0e-4,
        gradient_clip_norm: Optional[float] = None,
        save_model: bool = False,
    ) -> dict[str, list[float]]:
        theta_tensor, summaries_tensor = self._tensor(theta), self._tensor(summaries)
        self._validate_matrix("theta", theta_tensor, self.theta_dim, "training")
        self._validate_matrix("summaries", summaries_tensor, self.feature_dim, "training")
        if theta_tensor.shape[0] != summaries_tensor.shape[0]:
            raise ValueError("training: theta and summaries must have the same batch size.")
        if epochs < 1 or batch_size < 1:
            raise ValueError("training: epochs and batch_size must be positive.")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("training: learning_rate must be positive and weight_decay nonnegative.")
        if not 0 < vali_split < 1:
            raise ValueError("training: vali_split must be strictly between zero and one.")

        count = theta_tensor.shape[0]
        split_seed = torch.initial_seed() if seed is None else seed
        train_indices, vali_indices, self.split_metadata = validation_split_indices(
            count, vali_split, seed=split_seed, group_ids=group_ids
        )

        train_indices = torch.as_tensor(train_indices, device=theta_tensor.device, dtype=torch.long)
        vali_indices = torch.as_tensor(vali_indices, device=theta_tensor.device, dtype=torch.long)
        self.train_indices = train_indices.detach().cpu()
        self.vali_indices = vali_indices.detach().cpu()
        # Fit both transformations only after splitting, preventing validation leakage.
        self.feature_standardizer.fit(summaries_tensor[train_indices])
        self.parameter_standardizer.fit(theta_tensor[train_indices])
        scaled_theta = self.parameter_standardizer(theta_tensor).detach()
        scaled_summaries = self.feature_standardizer(summaries_tensor).detach()
        dataset = TensorDataset(scaled_theta, scaled_summaries)
        train_dataset = Subset(dataset, train_indices.cpu().tolist())
        vali_dataset = Subset(dataset, vali_indices.cpu().tolist())
        loader_generator = torch.Generator().manual_seed(torch.initial_seed() if seed is None else seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, generator=loader_generator)
        vali_loader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False)

        self.unfreeze()
        self.train(True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler_kwargs = dict(scheduler_kwargs or {})
        if scheduler_type is None:
            scheduler = None
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, **scheduler_kwargs)
        elif scheduler_type == "exp":
            exponential_kwargs = {"gamma": 0.95}
            exponential_kwargs.update(scheduler_kwargs)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **exponential_kwargs)
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler type {scheduler_type!r}.")

        self.train_losses, self.vali_losses = [], []
        best_loss = float("inf")
        best_state = deepcopy(self.state_dict())
        stale_epochs = 0
        for epoch in range(epochs):
            self.train(True)
            total, seen = 0.0, 0
            for theta_batch, summaries_batch in train_loader:
                loss = self.flow_matching_loss(theta_batch, summaries_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_norm)
                optimizer.step()
                total += loss.item() * theta_batch.shape[0]
                seen += theta_batch.shape[0]
            train_loss = total / seen
            self.eval()
            with torch.no_grad():
                total, seen = 0.0, 0
                for theta_batch, summaries_batch in vali_loader:
                    loss = self.flow_matching_loss(theta_batch, summaries_batch)
                    total += loss.item() * theta_batch.shape[0]
                    seen += theta_batch.shape[0]
            vali_loss = total / seen
            self.train_losses.append(train_loss)
            self.vali_losses.append(vali_loss)
            if scheduler_type == "plateau":
                scheduler.step(vali_loss)
            elif scheduler is not None:
                scheduler.step()
            if vali_loss < best_loss - min_delta:
                best_loss, stale_epochs = vali_loss, 0
                best_state = deepcopy(self.state_dict())
            else:
                stale_epochs += 1
            if verbose:
                print(f"epoch {epoch + 1:4d}/{epochs}: train = {train_loss:.7f}, validation = {vali_loss:.7f}")
            if n_patience_epochs is not None and stale_epochs >= n_patience_epochs:
                break
        self.load_state_dict(best_state)
        self.cfm._fitted.fill_(True)
        self.eval()
        self.freeze()
        if self.model_dir is not None:
            self._plot_epochs()
        if save_model:
            self.save()
        return {"train_loss": self.train_losses, "vali_loss": self.vali_losses}

    def log_prob(self, summaries: Any, theta: Any, **kwargs: Any) -> Tensor:
        summaries_tensor = self._tensor(summaries)
        self._validate_matrix("summaries", summaries_tensor, self.feature_dim, "likelihood evaluation")
        theta_tensor = self._tensor(theta)
        self._validate_matrix("theta", theta_tensor, self.theta_dim, "likelihood evaluation")
        scaled_summaries = self.feature_standardizer(summaries_tensor)
        scaled_theta = self.parameter_standardizer(theta_tensor)
        result = self.cfm.log_prob(scaled_summaries, self._adapt(scaled_theta, "likelihood evaluation"), **kwargs)
        return result + self.feature_standardizer.log_abs_det

    def log_likelihood(self, x: Any, theta: Any, return_numpy: bool = False, **kwargs: Any):
        result = self.log_prob(x, theta, **kwargs)
        return result.detach().cpu().numpy() if return_numpy else result

    def sample(self, theta: Any, **kwargs: Any) -> Tensor:
        theta_tensor = self._tensor(theta)
        self._validate_matrix("theta", theta_tensor, self.theta_dim, "sampling")
        scaled_theta = self.parameter_standardizer(theta_tensor)
        scaled = self.cfm.sample(self._adapt(scaled_theta, "sampling"), **kwargs)
        return self.feature_standardizer.inverse(scaled)

    def sample_likelihood(
        self, theta: Any, n_samples: int = 1000, batch_size: Optional[int] = None, return_numpy: bool = True
    ):
        del batch_size  # torchdiffeq operates on the full conditional batch.
        result = self.sample(theta, num_samples=n_samples)
        return result.detach().cpu().numpy() if return_numpy else result

    def _mcmc_log_posterior(self, theta_walkers: Any, x_obs: Any, device: Any = None):
        """Evaluate all observations for an MCMC walker batch in raw units.

        Prior enforcement belongs to the calling sampler; this method supplies
        the (summed) likelihood term and deliberately routes through
        :meth:`log_likelihood`, so both standardizers and the feature Jacobian
        are applied identically to ordinary evaluation.
        """
        del device  # Input conversion follows the module's actual device.
        observations = self._tensor(x_obs)
        self._validate_matrix("x_obs", observations, self.feature_dim, "MCMC")
        theta_tensor = self._tensor(theta_walkers)
        self._validate_matrix("theta", theta_tensor, self.theta_dim, "MCMC")
        total = theta_tensor.new_zeros(theta_tensor.shape[0])
        with torch.no_grad():
            for observation in observations:
                total += self.log_likelihood(observation[None, :], theta_tensor)
        return total.detach().cpu().numpy()

    def sample_posterior(
        self,
        x_obs: Any,
        n_walkers: int = 1024,
        n_steps: int = 1000,
        n_burnin_steps: int = 1000,
        label: Optional[str] = None,
        device: Any = None,
        dont_save: bool = False,
        **_: Any,
    ) -> np.ndarray:
        """Sample the posterior through the application-wide MCMC interface."""
        from msfm.utils import prior

        from msi.utils import mcmc

        def log_posterior(theta_walkers: Any) -> np.ndarray:
            likelihood = self._mcmc_log_posterior(theta_walkers, x_obs, device=device)
            return prior.log_posterior(theta_walkers, likelihood, conf=None, params=self.params)

        return mcmc.run_emcee(
            log_posterior,
            self.params,
            out_dir=None if dont_save else self.model_dir,
            label=label,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burnin_steps=n_burnin_steps,
        )

    def save(self, path: Optional[str] = None) -> str:
        """Save architecture, adapters, scalers, and model weights (never optimizer state)."""
        path = path or (os.path.join(self.model_dir, self.model_name + ".pt") if self.model_dir else None)
        if path is None:
            raise ValueError("A checkpoint path or model_dir is required.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "init_kwargs": self.checkpoint_init_kwargs(),
                "state_dict": self.state_dict(),
                "training_history": {
                    "train_loss": list(self.train_losses),
                    "vali_loss": list(self.vali_losses),
                },
                "split_metadata": getattr(self, "split_metadata", None),
                "validation_indices": getattr(self, "vali_indices", None),
            },
            path,
        )
        return path

    @classmethod
    def from_checkpoint(cls, path: str, *, map_location: Any = "cpu") -> "LikelihoodCFM":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        kwargs = dict(checkpoint["init_kwargs"])
        stored_theta_dim = kwargs.pop("theta_dim")
        model = cls(**kwargs)
        if model.theta_dim != stored_theta_dim:
            raise ValueError("Checkpoint parameter metadata is inconsistent with theta_dim.")
        model.load_state_dict(checkpoint["state_dict"])
        history = checkpoint.get("training_history", {})
        model.train_losses = list(history.get("train_loss", []))
        model.vali_losses = list(history.get("vali_loss", []))
        if checkpoint.get("validation_indices") is not None:
            model.vali_indices = torch.as_tensor(checkpoint["validation_indices"], dtype=torch.long).cpu()
            model.split_metadata = checkpoint.get("split_metadata") or {}
        model.eval()
        model.freeze()
        return model

    def get_validation_indices(self) -> np.ndarray:
        """Return a copy of the held-out row indices stored in the checkpoint."""
        if not hasattr(self, "vali_indices"):
            raise ValueError("This checkpoint does not contain validation indices; retrain it to record the split.")
        return self.vali_indices.detach().cpu().numpy().copy()
