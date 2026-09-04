"""Application adapter for the conditional flow-matching likelihood."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

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
        self.context_adapter = ContextAdapter(self.theta_dim, self.feature_dim, self.context_adapter_config)
        self.cfm = ConditionalFlowMatchingLikelihood(dimension=self.feature_dim, **self.cfm_config)
        self.freeze()

    def checkpoint_init_kwargs(self) -> dict[str, Any]:
        """Return complete, serializable architecture metadata."""
        return {
            "params": self.params,
            "theta_dim": self.theta_dim,
            "feature_dim": self.feature_dim,
            "cfm_config": deepcopy(self.cfm_config),
            "context_adapter_config": deepcopy(self.context_adapter_config),
        }

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
    ) -> Tensor:
        theta_tensor, summaries_tensor = self._tensor(theta), self._tensor(summaries)
        # Validate before constructing the optimizer, including batch agreement.
        self.flow_matching_loss(theta_tensor, summaries_tensor)
        if epochs < 1 or batch_size < 1:
            raise ValueError("training: epochs and batch_size must be positive.")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("training: learning_rate must be positive and weight_decay nonnegative.")

        self.unfreeze()
        self.train(True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        losses: list[Tensor] = []
        count = theta_tensor.shape[0]
        for epoch in range(epochs):
            indices = (
                torch.randperm(count, device=theta_tensor.device)
                if shuffle
                else torch.arange(count, device=theta_tensor.device)
            )
            total = theta_tensor.new_zeros(())
            for start in range(0, count, batch_size):
                batch = indices[start : start + batch_size]
                loss = self.flow_matching_loss(theta_tensor[batch].detach(), summaries_tensor[batch].detach())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.detach() * batch.shape[0]
            epoch_loss = total / count
            losses.append(epoch_loss)
            if verbose:
                print(f"epoch {epoch + 1:4d}/{epochs}: flow-matching loss = {epoch_loss.item():.7f}")
        self.cfm._fitted.fill_(True)
        self.eval()
        self.freeze()
        return torch.stack(losses)

    def log_prob(self, summaries: Any, theta: Any, **kwargs: Any) -> Tensor:
        summaries_tensor = self._tensor(summaries)
        self._validate_matrix("summaries", summaries_tensor, self.feature_dim, "likelihood evaluation")
        return self.cfm.log_prob(summaries_tensor, self._adapt(theta, "likelihood evaluation"), **kwargs)

    def log_likelihood(self, x: Any, theta: Any, return_numpy: bool = False, **kwargs: Any):
        result = self.log_prob(x, theta, **kwargs)
        return result.detach().cpu().numpy() if return_numpy else result

    def sample(self, theta: Any, **kwargs: Any) -> Tensor:
        return self.cfm.sample(self._adapt(theta, "sampling"), **kwargs)

    def sample_likelihood(
        self, theta: Any, n_samples: int = 1000, batch_size: Optional[int] = None, return_numpy: bool = True
    ):
        del batch_size  # torchdiffeq operates on the full conditional batch.
        result = self.sample(theta, num_samples=n_samples)
        return result.detach().cpu().numpy() if return_numpy else result
