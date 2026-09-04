#!/usr/bin/env python3
"""Reusable conditional likelihood estimation with continuous flow matching.

The main class is :class:`ConditionalFlowMatchingLikelihood`.

It learns a normalized conditional density p(y | x), with x and y both in
R^D, using the linear Gaussian conditional-flow-matching path

    s_t = t y + [1 - (1 - sigma_min) t] z,     z ~ N(0, I_D),

whose sample-wise target velocity is

    u_t = y - (1 - sigma_min) z.

A positive ``sigma_min`` is part of the statistical model: the learned
endpoint is the empirical conditional density convolved with
N(0, sigma_min^2 I_D).

Two velocity fields are available:

``model_type="affine"``
    v(t, s; x) = A(t)s + B(t)x + c(t).  This has an exact, inexpensive
    divergence trace(A(t)) and is especially suitable for approximately
    Gaussian conditional densities.  Its parameter count scales as O(D^2).

``model_type="mlp"``
    A generic feed-forward neural network of [time features, s, x].  This is
    more expressive, but likelihood evaluation must differentiate the vector
    field with respect to every state dimension (exact trace) or use a
    Hutchinson trace estimate.

All public data inputs and outputs are PyTorch tensors.  The class is an
``nn.Module``, so standard ``.to(device)``, ``.float()``, ``.double()``,
``state_dict()``, and ``load_state_dict()`` operations work normally.

Minimal use
-----------

    model = ConditionalFlowMatchingLikelihood(
        dimension=D,
        model_type="mlp",       # or "affine"
        sigma_min=0.05,
        ode_steps=64,
    ).to(x_train.device)

    losses = model.fit(x_train, y_train, epochs=50, batch_size=2048)
    log_p = model.log_prob(y_new, x_new)       # shape: (batch,)

For gradient-based inference, make the proposed parameters require gradients:

    x_new = x_new.requires_grad_(True)
    log_p = model.log_prob(y_observed, x_new)  # y_observed may have shape (1,D)
    gradient = torch.autograd.grad(log_p.sum(), x_new)[0]

Forward sampling and backward density evaluation are solved with
``torchdiffeq.odeint``.  The default solver remains fixed-step RK4 so that
``ode_steps`` has the same interpretation as in the original implementation.
Adaptive solvers such as ``dopri5`` can instead be selected at initialization.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from typing import Any, Literal, Optional

import torch
from torch import Tensor, nn

try:
    from torchdiffeq import odeint
except ImportError as exc:  # pragma: no cover - depends on the user environment
    raise ImportError(
        "This module requires torchdiffeq. Install it with "
        "`python -m pip install torchdiffeq`."
    ) from exc


ModelType = Literal["affine", "mlp"]
DivergenceEstimator = Literal["exact", "hutchinson"]

# Methods whose internal grid can be controlled with ``options["step_size"]``.
# The first group is available in torchdiffeq 0.2.x; the additional implicit
# names are accepted by newer releases. Unknown method names are deliberately
# passed through so torchdiffeq can provide its own version-specific error.
_FIXED_GRID_ODE_METHODS = {
    "euler",
    "midpoint",
    "heun2",
    "heun3",
    "rk4",
    "explicit_adams",
    "implicit_adams",
    "fixed_adams",
    "implicit_euler",
    "implicit_midpoint",
    "trapezoid",
    "radauIIA3",
    "radauIIA5",
    "gl4",
    "gl6",
    "sdirk2",
    "trbdf2",
}


def _canonical_time(t: Tensor) -> Tensor:
    """Return time as shape (n_times, 1), without expanding its batch."""
    if not isinstance(t, Tensor):
        raise TypeError("t must be a PyTorch tensor.")
    if not t.is_floating_point():
        raise TypeError("t must have a floating-point dtype.")

    if t.ndim == 0:
        return t.reshape(1, 1)
    if t.ndim == 1:
        return t[:, None]
    if t.ndim == 2 and t.shape[1] == 1:
        return t
    raise ValueError("t must be scalar, shape (batch,), or shape (batch, 1).")


def _expand_time(t: Tensor, batch_size: int) -> Tensor:
    """Return time as shape (batch_size, 1), allowing a shared scalar time."""
    t = _canonical_time(t)
    if t.shape[0] == 1:
        return t.expand(batch_size, 1)
    if t.shape[0] != batch_size:
        raise ValueError(
            f"Batched time has size {t.shape[0]}, but the data batch has "
            f"size {batch_size}."
        )
    return t


def _time_features(t: Tensor) -> Tensor:
    """Six smooth scalar-time features used by both velocity models."""
    t = _canonical_time(t)
    return torch.cat(
        (
            t,
            t.square(),
            torch.sin(math.pi * t),
            torch.cos(math.pi * t),
            torch.sin(2.0 * math.pi * t),
            torch.cos(2.0 * math.pi * t),
        ),
        dim=-1,
    )


def _build_feed_forward_network(
    input_dimension: int,
    output_dimension: int,
    hidden_features: int,
    num_hidden_layers: int,
    *,
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
) -> nn.Sequential:
    if hidden_features < 1:
        raise ValueError("hidden_features must be positive.")
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be at least one.")

    factory_kwargs = {"device": device, "dtype": dtype}
    layers: list[nn.Module] = []
    in_features = input_dimension
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(in_features, hidden_features, **factory_kwargs))
        layers.append(nn.SiLU())
        in_features = hidden_features
    layers.append(nn.Linear(in_features, output_dimension, **factory_kwargs))

    network = nn.Sequential(*layers)
    final_layer = network[-1]
    assert isinstance(final_layer, nn.Linear)

    # A zero terminal layer starts with a numerically benign zero vector field.
    # The terminal layer learns immediately; earlier layers begin receiving
    # gradients after the first optimizer update.
    nn.init.zeros_(final_layer.weight)
    nn.init.zeros_(final_layer.bias)
    return network


class ConditionalAffineVelocity(nn.Module):
    """Time-neural vector field affine in state and conditioning context.

    v(t, s; x) = A(t)s + B(t)x + c(t).
    """

    def __init__(
        self,
        dimension: int,
        hidden_features: int,
        num_hidden_layers: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        n_outputs = 2 * dimension * dimension + dimension
        self.time_network = _build_feed_forward_network(
            input_dimension=6,
            output_dimension=n_outputs,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            device=device,
            dtype=dtype,
        )

    def coefficients(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d = self.dimension
        raw = self.time_network(_time_features(t))
        matrix_size = d * d
        a = raw[:, :matrix_size].reshape(-1, d, d)
        b = raw[:, matrix_size : 2 * matrix_size].reshape(-1, d, d)
        c = raw[:, 2 * matrix_size :]
        return a, b, c

    def forward(self, t: Tensor, state: Tensor, context: Tensor) -> Tensor:
        if state.shape != context.shape:
            raise ValueError("state and context must have identical shapes.")
        if state.ndim != 2 or state.shape[1] != self.dimension:
            raise ValueError(
                f"state and context must have shape (batch, {self.dimension})."
            )

        a, b, c = self.coefficients(t)
        batch_size = state.shape[0]

        # A scalar ODE time is shared by the complete data batch.
        if a.shape[0] == 1:
            return state @ a[0].T + context @ b[0].T + c[0]

        if a.shape[0] != batch_size:
            raise ValueError("A batched time tensor must match the data batch size.")

        a_state = torch.bmm(a, state.unsqueeze(-1)).squeeze(-1)
        b_context = torch.bmm(b, context.unsqueeze(-1)).squeeze(-1)
        return a_state + b_context + c

    def exact_divergence(self, t: Tensor, batch_size: int) -> Tensor:
        """Exact divergence with respect to the state variable."""
        a, _, _ = self.coefficients(t)
        divergence = a.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        if divergence.shape[0] == 1:
            return divergence.expand(batch_size)
        if divergence.shape[0] != batch_size:
            raise ValueError("A batched time tensor must match the data batch size.")
        return divergence


class ConditionalMLPVelocity(nn.Module):
    """Generic feed-forward conditional vector field.

    The network receives [time features, state, context] and returns a vector
    in R^D.  Unlike the affine field, its divergence generally depends on the
    complete state and context.
    """

    def __init__(
        self,
        dimension: int,
        hidden_features: int,
        num_hidden_layers: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.network = _build_feed_forward_network(
            input_dimension=2 * dimension + 6,
            output_dimension=dimension,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            device=device,
            dtype=dtype,
        )

    def forward(self, t: Tensor, state: Tensor, context: Tensor) -> Tensor:
        if state.shape != context.shape:
            raise ValueError("state and context must have identical shapes.")
        if state.ndim != 2 or state.shape[1] != self.dimension:
            raise ValueError(
                f"state and context must have shape (batch, {self.dimension})."
            )

        time = _expand_time(t, state.shape[0])
        features = _time_features(time)
        return self.network(torch.cat((features, state, context), dim=-1))


class ConditionalFlowMatchingLikelihood(nn.Module):
    """Conditional continuous-normalizing-flow likelihood p(y | x).

    Parameters
    ----------
    dimension:
        Common dimension D of x and y.
    model_type:
        ``"affine"`` or ``"mlp"``.
    sigma_min:
        Terminal Gaussian width in the flow-matching path.  The fitted density
        includes this smoothing as part of the likelihood model.
    hidden_features:
        Width of the neural network used by the velocity field.
    num_hidden_layers:
        Number of hidden feed-forward layers.
    ode_steps:
        Default number of steps for fixed-grid torchdiffeq solvers. With the
        default ``ode_method="rk4"``, ``step_size`` is set to ``1/ode_steps``.
        Adaptive solvers ignore this value unless their options use it.
    divergence_estimator:
        For the MLP field, ``"exact"`` computes the full Jacobian trace using
        D reverse-mode derivatives.  ``"hutchinson"`` uses an unbiased trace
        estimator.  The affine field always uses its analytic exact trace.
    hutchinson_samples:
        Number of Rademacher probe vectors per batch item when using the
        Hutchinson estimator.
    ode_method:
        Method passed to ``torchdiffeq.odeint``. The default ``"rk4"`` keeps
        behavior close to the original fixed-step implementation. Typical
        adaptive choices include ``"dopri5"`` and ``"dopri8"``.
    ode_rtol, ode_atol:
        Relative and absolute tolerances passed to ``torchdiffeq.odeint``.
        They primarily control adaptive solvers.
    ode_options:
        Optional mapping forwarded as the solver ``options`` dictionary. For
        fixed-grid methods, ``step_size=1/ode_steps`` is supplied unless the
        mapping already defines ``step_size`` or ``grid_constructor``.
    device, dtype:
        Optional PyTorch construction device and dtype.  Standard ``.to(...)``
        can also be used after initialization.

    Notes
    -----
    ``log_prob`` requires both y and x because this is a conditional density.
    It supports singleton-batch broadcasting, so an observed y with shape
    ``(1, D)`` can be evaluated at proposed parameters x with shape ``(B, D)``.

    When either input requires gradients, ``log_prob`` automatically builds a
    differentiable ODE computation.  Set ``differentiable=False`` explicitly
    for lower-memory density evaluation, or ``True`` explicitly when needed.
    """

    def __init__(
        self,
        dimension: int,
        model_type: ModelType = "affine",
        sigma_min: float = 0.05,
        hidden_features: int = 128,
        num_hidden_layers: int = 3,
        ode_steps: int = 64,
        divergence_estimator: DivergenceEstimator = "exact",
        hutchinson_samples: int = 1,
        *,
        ode_method: str = "rk4",
        ode_rtol: float = 1.0e-5,
        ode_atol: float = 1.0e-7,
        ode_options: Optional[Mapping[str, Any]] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        super().__init__()

        if dimension < 1:
            raise ValueError("dimension must be positive.")
        if not 0.0 <= sigma_min < 1.0:
            raise ValueError("sigma_min must satisfy 0 <= sigma_min < 1.")
        if ode_steps < 1:
            raise ValueError("ode_steps must be positive.")
        if model_type not in ("affine", "mlp"):
            raise ValueError("model_type must be 'affine' or 'mlp'.")
        if divergence_estimator not in ("exact", "hutchinson"):
            raise ValueError(
                "divergence_estimator must be 'exact' or 'hutchinson'."
            )
        if hutchinson_samples < 1:
            raise ValueError("hutchinson_samples must be positive.")
        if not isinstance(ode_method, str) or not ode_method.strip():
            raise ValueError("ode_method must be a nonempty string.")
        if ode_rtol <= 0.0:
            raise ValueError("ode_rtol must be positive.")
        if ode_atol <= 0.0:
            raise ValueError("ode_atol must be positive.")
        if ode_options is not None and not isinstance(ode_options, Mapping):
            raise TypeError("ode_options must be a mapping or None.")

        self.dimension = int(dimension)
        self.model_type: ModelType = model_type
        self.sigma_min = float(sigma_min)
        self.ode_steps = int(ode_steps)
        self.divergence_estimator: DivergenceEstimator = divergence_estimator
        self.hutchinson_samples = int(hutchinson_samples)
        self.ode_method = ode_method.strip()
        self.ode_rtol = float(ode_rtol)
        self.ode_atol = float(ode_atol)
        self.ode_options: dict[str, Any] = dict(ode_options or {})

        if model_type == "affine":
            self.velocity_field: nn.Module = ConditionalAffineVelocity(
                dimension=dimension,
                hidden_features=hidden_features,
                num_hidden_layers=num_hidden_layers,
                device=device,
                dtype=dtype,
            )
        else:
            self.velocity_field = ConditionalMLPVelocity(
                dimension=dimension,
                hidden_features=hidden_features,
                num_hidden_layers=num_hidden_layers,
                device=device,
                dtype=dtype,
            )

        self.register_buffer(
            "_fitted",
            torch.tensor(False, dtype=torch.bool, device=device),
            persistent=True,
        )

        # Instances start in inference-safe mode. ``fit`` unfreezes the field
        # before constructing its optimizer, and freezes it again afterward.
        # This also means a newly constructed object remains frozen after a
        # trained state_dict (including the fitted flag) is loaded into it.
        self.freeze()

    # ------------------------------------------------------------------
    # Tensor validation and batch handling
    # ------------------------------------------------------------------
    def _reference_tensor(self) -> Tensor:
        return next(self.parameters())

    def _validate_tensor(self, name: str, value: Tensor) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a PyTorch tensor.")
        if value.ndim != 2 or value.shape[1] != self.dimension:
            raise ValueError(
                f"{name} must have shape (batch, {self.dimension}); "
                f"received {tuple(value.shape)}."
            )
        if value.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one batch item.")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype.")

        reference = self._reference_tensor()
        if value.device != reference.device:
            raise ValueError(
                f"{name} is on {value.device}, but the model is on "
                f"{reference.device}. Move the model or tensor explicitly."
            )
        if value.dtype != reference.dtype:
            raise ValueError(
                f"{name} has dtype {value.dtype}, but the model has dtype "
                f"{reference.dtype}. Convert one of them explicitly."
            )

    def _validate_training_pair(self, x: Tensor, y: Tensor) -> None:
        self._validate_tensor("x", x)
        self._validate_tensor("y", y)
        if x.shape != y.shape:
            raise ValueError(
                "Training x and y must have identical shape (N, D); "
                f"received {tuple(x.shape)} and {tuple(y.shape)}."
            )

    def _broadcast_evaluation_pair(
        self, y: Tensor, x: Tensor
    ) -> tuple[Tensor, Tensor]:
        self._validate_tensor("y", y)
        self._validate_tensor("x", x)

        y_batch = y.shape[0]
        x_batch = x.shape[0]
        if y_batch == x_batch:
            return y, x
        if y_batch == 1:
            return y.expand(x_batch, -1), x
        if x_batch == 1:
            return y, x.expand(y_batch, -1)
        raise ValueError(
            "The y and x batch sizes must match, or one of them must be one; "
            f"received {y_batch} and {x_batch}."
        )

    def _require_fitted(self) -> None:
        if not bool(self._fitted.item()):
            raise RuntimeError("The likelihood must be fitted before evaluation.")

    def freeze(self) -> None:
        """Freeze velocity parameters while preserving input gradients."""
        for parameter in self.velocity_field.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unfreeze velocity parameters for training or fine-tuning."""
        for parameter in self.velocity_field.parameters():
            parameter.requires_grad_(True)

    # ------------------------------------------------------------------
    # Flow-matching training
    # ------------------------------------------------------------------
    def flow_matching_loss(
        self,
        x: Tensor,
        y: Tensor,
        *,
        time: Optional[Tensor] = None,
        base_noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Return one Monte-Carlo conditional flow-matching loss.

        Parameters ``time`` and ``base_noise`` are optional tensors so callers
        can provide controlled random draws for testing or specialized loops.
        ``time`` may have shape ``()``, ``(1,)``, ``(1,1)``, ``(B,)``, or
        ``(B,1)``.  ``base_noise`` must have shape ``(B,D)``.
        """
        self._validate_training_pair(x, y)
        batch_size = x.shape[0]

        if time is None:
            time = torch.rand(
                batch_size,
                1,
                device=y.device,
                dtype=y.dtype,
            )
        else:
            if time.device != y.device or time.dtype != y.dtype:
                raise ValueError("time must have the same device and dtype as y.")
            time = _expand_time(time, batch_size)

        if base_noise is None:
            base_noise = torch.randn_like(y)
        else:
            self._validate_tensor("base_noise", base_noise)
            if base_noise.shape != y.shape:
                raise ValueError("base_noise must have the same shape as y.")

        beta_t = 1.0 - (1.0 - self.sigma_min) * time
        state_t = time * y + beta_t * base_noise
        target_velocity = y - (1.0 - self.sigma_min) * base_noise
        predicted_velocity = self.velocity_field(time, state_t, x)

        # Squared Euclidean error, averaged over the batch.
        return predicted_velocity.sub(target_velocity).square().sum(dim=-1).mean()

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        epochs: int = 50,
        batch_size: int = 2048,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-6,
        gradient_clip_norm: Optional[float] = None,
        cosine_decay: bool = True,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> Tensor:
        """Train the conditional likelihood and return epoch losses.

        The returned tensor has shape ``(epochs,)`` and remains on the same
        device and with the same floating-point dtype as the model/data.
        """
        self._validate_training_pair(x, y)
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative.")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when provided.")

        self.unfreeze()
        self.train(True)
        optimizer = torch.optim.AdamW(
            self.velocity_field.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR]
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            if cosine_decay
            else None
        )

        n_data = x.shape[0]
        epoch_losses: list[Tensor] = []

        for epoch in range(epochs):
            if shuffle:
                indices = torch.randperm(n_data, device=x.device)
            else:
                indices = torch.arange(n_data, device=x.device)

            accumulated_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            for start in range(0, n_data, batch_size):
                batch_indices = indices[start : start + batch_size]

                # The dataset is treated as fixed; gradients are needed only
                # for the velocity-field parameters.
                x_batch = x[batch_indices].detach()
                y_batch = y[batch_indices].detach()
                loss = self.flow_matching_loss(x_batch, y_batch)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.velocity_field.parameters(), gradient_clip_norm
                    )
                optimizer.step()

                accumulated_loss = (
                    accumulated_loss + loss.detach() * x_batch.shape[0]
                )

            if scheduler is not None:
                scheduler.step()

            epoch_loss = accumulated_loss / n_data
            epoch_losses.append(epoch_loss)
            if verbose:
                print(
                    f"epoch {epoch + 1:4d}/{epochs}: "
                    f"flow-matching loss = {epoch_loss.item():.7f}"
                )

        self._fitted.fill_(True)
        self.eval()
        self.freeze()
        return torch.stack(epoch_losses)

    # ------------------------------------------------------------------
    # Divergence and ODE utilities
    # ------------------------------------------------------------------
    def _make_hutchinson_probes(self, state: Tensor) -> Tensor:
        """Rademacher probes with shape (n_probes, batch, D)."""
        integers = torch.randint(
            0,
            2,
            (self.hutchinson_samples, *state.shape),
            device=state.device,
        )
        return integers.to(dtype=state.dtype).mul_(2.0).sub_(1.0)

    def _mlp_velocity_and_divergence(
        self,
        t: Tensor,
        state: Tensor,
        context: Tensor,
        *,
        differentiable: bool,
        probes: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Evaluate an MLP velocity and its state-Jacobian trace."""
        # Autograd must be enabled even for non-differentiable density
        # evaluation because it is how the MLP divergence is computed.
        with torch.enable_grad():
            if differentiable:
                state_for_grad = state
                if not state_for_grad.requires_grad:
                    state_for_grad = state_for_grad.detach().requires_grad_(True)
                context_for_grad = context
            else:
                state_for_grad = state.detach().requires_grad_(True)
                context_for_grad = context.detach()

            velocity = self.velocity_field(t, state_for_grad, context_for_grad)

            if self.divergence_estimator == "exact":
                diagonal_terms: list[Tensor] = []
                for coordinate in range(self.dimension):
                    retain = differentiable or coordinate < self.dimension - 1
                    gradient = torch.autograd.grad(
                        velocity[:, coordinate].sum(),
                        state_for_grad,
                        create_graph=differentiable,
                        retain_graph=retain,
                        allow_unused=False,
                    )[0]
                    diagonal_terms.append(gradient[:, coordinate])
                divergence = torch.stack(diagonal_terms, dim=0).sum(dim=0)
            else:
                if probes is None:
                    raise RuntimeError("Hutchinson probes were not provided.")
                estimates: list[Tensor] = []
                for probe_index in range(probes.shape[0]):
                    probe = probes[probe_index]
                    retain = differentiable or probe_index < probes.shape[0] - 1
                    vector_jacobian_product = torch.autograd.grad(
                        (velocity * probe).sum(),
                        state_for_grad,
                        create_graph=differentiable,
                        retain_graph=retain,
                        allow_unused=False,
                    )[0]
                    estimates.append(
                        (vector_jacobian_product * probe).sum(dim=-1)
                    )
                divergence = torch.stack(estimates, dim=0).mean(dim=0)

        if not differentiable:
            velocity = velocity.detach()
            divergence = divergence.detach()
        return velocity, divergence

    def _velocity_and_divergence(
        self,
        t: Tensor,
        state: Tensor,
        context: Tensor,
        *,
        differentiable: bool,
        probes: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        # Adaptive torchdiffeq solvers may use float64 for internal time even
        # when the state/model is float32. Neural-network inputs must match the
        # state/model dtype, so cast the scalar time at the RHS boundary.
        model_time = t.to(device=state.device, dtype=state.dtype)

        if self.model_type == "affine":
            affine = self.velocity_field
            assert isinstance(affine, ConditionalAffineVelocity)
            if differentiable:
                velocity = affine(model_time, state, context)
                divergence = affine.exact_divergence(
                    model_time, state.shape[0]
                )
            else:
                with torch.no_grad():
                    velocity = affine(model_time, state, context)
                    divergence = affine.exact_divergence(
                        model_time, state.shape[0]
                    )
            return velocity, divergence

        return self._mlp_velocity_and_divergence(
            model_time,
            state,
            context,
            differentiable=differentiable,
            probes=probes,
        )

    def _odeint_kwargs(self, n_steps: int) -> dict[str, Any]:
        """Build keyword arguments for one ``torchdiffeq.odeint`` solve."""
        options = dict(self.ode_options)
        if (
            self.ode_method in _FIXED_GRID_ODE_METHODS
            and "step_size" not in options
            and "grid_constructor" not in options
        ):
            options["step_size"] = 1.0 / n_steps

        return {
            "rtol": self.ode_rtol,
            "atol": self.ode_atol,
            "method": self.ode_method,
            "options": options or None,
        }

    @staticmethod
    def _model_time(t: Tensor, state: Tensor) -> Tensor:
        """Cast an odeint time scalar to the state/model device and dtype."""
        return t.to(device=state.device, dtype=state.dtype)

    # ------------------------------------------------------------------
    # Conditional density evaluation
    # ------------------------------------------------------------------
    def log_prob(
        self,
        y: Tensor,
        x: Tensor,
        *,
        ode_steps: Optional[int] = None,
        differentiable: Optional[bool] = None,
    ) -> Tensor:
        """Return batched ``log p(y | x)`` as a tensor of shape ``(batch,)``.

        Parameters
        ----------
        y, x:
            Tensors with shape ``(batch,D)``. Their batch sizes may differ
            only when one batch size is one; singleton-batch broadcasting is
            then applied without copying data.
        ode_steps:
            Optional per-call override of the configured fixed-grid step
            count. It sets ``step_size=1/ode_steps`` for fixed-grid methods
            unless ``ode_options`` already supplies a grid or step size.
            Adaptive solvers use ``ode_rtol`` and ``ode_atol`` instead.
        differentiable:
            ``None`` selects differentiable evaluation exactly when PyTorch
            gradient mode is enabled and either input requires gradients.
            ``False`` minimizes graph construction and memory. ``True``
            enables derivatives through ``torchdiffeq.odeint`` and through
            the CNF divergence calculation.
        """
        self._require_fitted()
        y, x = self._broadcast_evaluation_pair(y, x)
        n_steps = self.ode_steps if ode_steps is None else int(ode_steps)
        if n_steps < 1:
            raise ValueError("ode_steps must be positive.")

        if differentiable is None:
            differentiable = bool(
                torch.is_grad_enabled() and (y.requires_grad or x.requires_grad)
            )

        if differentiable:
            initial_state = y.clone()
            context = x
        else:
            initial_state = y.detach().clone()
            context = x.detach()

        initial_logp_change = torch.zeros(
            initial_state.shape[0],
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
        probes = (
            self._make_hutchinson_probes(initial_state)
            if self.model_type == "mlp"
            and self.divergence_estimator == "hutchinson"
            else None
        )

        # Integrate the original CNF dynamics backward from t=1 to t=0.
        # The augmented scalar follows d(log p)/dt = -div v. Starting its
        # change at zero gives +integral_0^1 div(v) at the base endpoint.
        integration_times = initial_state.new_tensor((1.0, 0.0))

        def augmented_dynamics(
            time: Tensor,
            augmented_state: tuple[Tensor, Tensor],
        ) -> tuple[Tensor, Tensor]:
            state_t, _logp_change_t = augmented_state
            velocity, divergence = self._velocity_and_divergence(
                time,
                state_t,
                context,
                differentiable=differentiable,
                probes=probes,
            )
            return velocity, -divergence

        ode_kwargs = self._odeint_kwargs(n_steps)
        if differentiable:
            state_path, logp_change_path = odeint(
                augmented_dynamics,
                (initial_state, initial_logp_change),
                integration_times,
                **ode_kwargs,
            )
        else:
            with torch.no_grad():
                state_path, logp_change_path = odeint(
                    augmented_dynamics,
                    (initial_state, initial_logp_change),
                    integration_times,
                    **ode_kwargs,
                )

        base_state = state_path[-1]
        integrated_divergence = logp_change_path[-1]
        base_log_probability = (
            -0.5 * base_state.square().sum(dim=-1)
            -0.5 * self.dimension * math.log(2.0 * math.pi)
        )
        result = base_log_probability - integrated_divergence
        return result if differentiable else result.detach()

    def log_likelihood(
        self,
        y: Tensor,
        x: Tensor,
        *,
        ode_steps: Optional[int] = None,
        differentiable: Optional[bool] = None,
    ) -> Tensor:
        """Alias for :meth:`log_prob`."""
        return self.log_prob(
            y,
            x,
            ode_steps=ode_steps,
            differentiable=differentiable,
        )

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        """Make ``model(y, x)`` equivalent to ``model.log_prob(y, x)``."""
        return self.log_prob(y, x)

    # ------------------------------------------------------------------
    # Conditional sampling
    # ------------------------------------------------------------------
    def sample(
        self,
        x: Tensor,
        *,
        num_samples: int = 1,
        base_noise: Optional[Tensor] = None,
        ode_steps: Optional[int] = None,
        differentiable: Optional[bool] = None,
    ) -> Tensor:
        """Draw conditional samples from the learned likelihood.

        With ``x.shape == (B,D)``:

        * ``num_samples == 1`` returns shape ``(B,D)``;
        * ``num_samples > 1`` returns shape ``(B,num_samples,D)``.

        Optional ``base_noise`` must have the corresponding output shape and
        provides reparameterized, repeatable sampling.
        """
        self._require_fitted()
        self._validate_tensor("x", x)
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        n_steps = self.ode_steps if ode_steps is None else int(ode_steps)
        if n_steps < 1:
            raise ValueError("ode_steps must be positive.")

        original_batch = x.shape[0]
        if num_samples == 1:
            context = x
            expected_noise_shape = x.shape
        else:
            context = (
                x[:, None, :]
                .expand(original_batch, num_samples, self.dimension)
                .reshape(-1, self.dimension)
            )
            expected_noise_shape = (
                original_batch,
                num_samples,
                self.dimension,
            )

        if base_noise is None:
            state = torch.randn(
                expected_noise_shape,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            if not isinstance(base_noise, Tensor):
                raise TypeError("base_noise must be a PyTorch tensor.")
            if tuple(base_noise.shape) != tuple(expected_noise_shape):
                raise ValueError(
                    f"base_noise must have shape {tuple(expected_noise_shape)}; "
                    f"received {tuple(base_noise.shape)}."
                )
            if base_noise.device != x.device or base_noise.dtype != x.dtype:
                raise ValueError(
                    "base_noise must have the same device and dtype as x."
                )
            state = base_noise

        if num_samples > 1:
            state = state.reshape(-1, self.dimension)

        if differentiable is None:
            differentiable = bool(
                torch.is_grad_enabled()
                and (x.requires_grad or state.requires_grad)
            )
        if not differentiable:
            state = state.detach()
            context = context.detach()

        integration_times = state.new_tensor((0.0, 1.0))

        def forward_dynamics(time: Tensor, state_t: Tensor) -> Tensor:
            model_time = self._model_time(time, state_t)
            return self.velocity_field(model_time, state_t, context)

        ode_kwargs = self._odeint_kwargs(n_steps)
        if differentiable:
            state_path = odeint(
                forward_dynamics,
                state,
                integration_times,
                **ode_kwargs,
            )
        else:
            with torch.no_grad():
                state_path = odeint(
                    forward_dynamics,
                    state,
                    integration_times,
                    **ode_kwargs,
                )
        state = state_path[-1]

        if num_samples > 1:
            state = state.reshape(original_batch, num_samples, self.dimension)
        return state if differentiable else state.detach()


# ----------------------------------------------------------------------
# Optional command-line smoke demo.  This is never run when imported.
# ----------------------------------------------------------------------
def _run_toy_demo(model_type: ModelType, device: torch.device) -> None:
    torch.manual_seed(7)
    dimension = 2
    n_train = 100_000

    covariance = torch.tensor(
        [[0.80**2, 0.45**2], [0.45**2, 0.60**2]],
        device=device,
        dtype=torch.float32,
    )
    cholesky = torch.linalg.cholesky(covariance)
    x_train = torch.randn(n_train, dimension, device=device) @ cholesky.T
    y_train = 0.5 * x_train + 0.25 * torch.randn_like(x_train)

    likelihood = ConditionalFlowMatchingLikelihood(
        dimension=dimension,
        model_type=model_type,
        sigma_min=0.05,
        hidden_features=96,
        num_hidden_layers=3,
        ode_steps=48,
        divergence_estimator="exact",
        device=device,
    )
    losses = likelihood.fit(
        x_train,
        y_train,
        epochs=25,
        batch_size=4096,
        learning_rate=1.0e-3,
        verbose=True,
    )

    x_new = torch.randn(8, dimension, device=device)
    y_new = likelihood.sample(x_new)
    log_p = likelihood.log_prob(y_new, x_new)
    print("final training loss:", losses[-1])
    print("sample shape:", torch.tensor(y_new.shape, device=device))
    print("log p(y | x):", log_p)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditional flow-matching demo using torchdiffeq"
    )
    parser.add_argument("--demo", action="store_true", help="run the toy demo")
    parser.add_argument(
        "--model-type", choices=("affine", "mlp"), default="affine"
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.demo:
        if arguments.device == "auto":
            selected_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            selected_device = torch.device(arguments.device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        _run_toy_demo(arguments.model_type, selected_device)
    else:
        print(
            "Import ConditionalFlowMatchingLikelihood from this module, or "
            "run with --demo."
        )
