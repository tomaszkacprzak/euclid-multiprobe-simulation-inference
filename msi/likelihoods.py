"""Application-level registry and factory for likelihood implementations.

Imports are deliberately lazy: selecting the PyTorch flow must not require the
optional TensorFlow dependencies used by the GMM implementation (and vice
versa).
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LikelihoodImplementation:
    """Everything the inference application needs to build or restore a model."""

    model_name: str
    wrapper_class: Callable[[], type]
    build: Callable[..., object]
    load: Callable[..., object]


def _flow_class():
    from msi.flow_conductor.likelihood_flow import LikelihoodFlow

    return LikelihoodFlow


def _cfm_class():
    from msi.flow_matching.likelihood_cfm import LikelihoodCFM

    return LikelihoodCFM


def _gmm_class():
    from msi.gaussian_mixture.likelihood_gmm import LikelihoodGMM

    return LikelihoodGMM


def _suffix(n_steps):
    return f"_{n_steps}" if n_steps is not None else ""


def _build_flow(*, params, msfm_conf, pred_dir, n_steps, grid_preds, grid_cosmos, config, prefix="", i_signal=None):
    from msi.utils.flow import build_flow_architecture

    cls = _flow_class()
    embedding_net, transform = build_flow_architecture(grid_preds.shape[-1], grid_cosmos.shape[-1], config)
    model = cls(
        params,
        msfm_conf,
        feature_dim=grid_preds.shape[-1],
        embedding_net=embedding_net,
        transform=transform,
        out_dir=pred_dir,
        prefix=prefix,
        suffix=_suffix(n_steps),
        load_existing=False,
    )
    training = config.get("training", {})
    print("Fitting flow...")
    model.fit(
        x=grid_preds,
        theta=grid_cosmos,
        n_epochs=training.get("n_epochs", 100),
        batch_size=training.get("batch_size", 10_000),
        scheduler_type=training.get("scheduler_type", "cosine"),
        save_model=True,
        run_c2st=True,
        group_ids=i_signal,
    )
    return model


def _load_flow(*, pred_dir, n_steps, prefix="", **_):
    return _flow_class().from_checkpoint(out_dir=pred_dir, prefix=prefix, suffix=_suffix(n_steps))


def _build_gmm(*, params, msfm_conf, pred_dir, n_steps, grid_preds, grid_cosmos, config, prefix="", **_):
    from msi.gaussian_mixture import architecture

    cls = _gmm_class()
    layers = architecture.get_gmm_layers(grid_preds.shape[-1], grid_cosmos.shape[-1])
    model = cls(
        params,
        msfm_conf,
        layers=layers,
        out_dir=pred_dir,
        prefix=prefix,
        suffix=_suffix(n_steps),
        load_existing=False,
    )
    training = config.get("training", {})
    print("Fitting GMM...")
    model.fit(
        x=grid_preds,
        theta=grid_cosmos,
        n_epochs=training.get("n_epochs", 100),
        batch_size=training.get("batch_size", 10_000),
        save_model=True,
    )
    return model


def _load_gmm(*, params, msfm_conf, pred_dir, n_steps, grid_preds, grid_cosmos, prefix="", **_):
    from msi.gaussian_mixture import architecture

    return _gmm_class()(
        params,
        msfm_conf,
        layers=architecture.get_gmm_layers(grid_preds.shape[-1], grid_cosmos.shape[-1]),
        out_dir=pred_dir,
        prefix=prefix,
        suffix=_suffix(n_steps),
        load_existing=True,
    )


def _build_cfm(*, params, msfm_conf, pred_dir, n_steps, grid_preds, grid_cosmos, config, prefix="", **_):
    """Build the application CFM and its learned context adapter."""
    import os
    import torch

    model_conf = config.get("model", {})
    adapter_conf = model_conf.get("context_adapter", {})
    cfm_conf = {key: value for key, value in model_conf.items() if key != "context_adapter"}
    model_dir = os.path.join(pred_dir, prefix + _cfm_class().model_name + _suffix(n_steps))
    model = _cfm_class()(
        params,
        feature_dim=grid_preds.shape[-1],
        cfm_config=cfm_conf,
        context_adapter_config=adapter_conf,
        conf=msfm_conf,
        model_dir=model_dir,
    )
    training = config.get("training", {})
    model.fit(
        torch.as_tensor(grid_cosmos, dtype=torch.float32),
        torch.as_tensor(grid_preds, dtype=torch.float32),
        epochs=training.get("n_epochs", training.get("epochs", 50)),
        batch_size=training.get("batch_size", 2048),
        vali_split=training.get("vali_split", 0.1),
        learning_rate=training.get("learning_rate", 3.0e-4),
        weight_decay=training.get("weight_decay", 1.0e-6),
        scheduler_type=training.get("scheduler_type"),
        scheduler_kwargs=training.get("scheduler_kwargs"),
        n_patience_epochs=training.get("n_patience_epochs"),
        min_delta=training.get("min_delta", 1.0e-4),
        gradient_clip_norm=training.get("clip_by_global_norm"),
        seed=training.get("seed"),
        group_ids=_.get("i_signal"),
        save_model=True,
    )
    return model


def _load_cfm(*, params, pred_dir, n_steps, grid_preds, grid_cosmos, prefix="", **_):
    import os
    import torch

    cls = _cfm_class()
    path = os.path.join(pred_dir, prefix + cls.model_name + _suffix(n_steps), cls.model_name + ".pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    init_kwargs = dict(checkpoint["init_kwargs"])
    stored_theta_dim = init_kwargs.pop("theta_dim")
    requested_theta_dim = len(params)
    if requested_theta_dim != stored_theta_dim:
        raise ValueError(
            f"Cannot load CFM checkpoint with theta_dim={stored_theta_dim} for "
            f"an application configured with theta_dim={requested_theta_dim}."
        )
    requested_feature_dim = int(grid_preds.shape[-1])
    if requested_feature_dim != init_kwargs["feature_dim"]:
        raise ValueError(
            f"Cannot load CFM checkpoint with feature_dim={init_kwargs['feature_dim']} "
            f"for summaries with feature_dim={requested_feature_dim}."
        )
    if int(grid_cosmos.shape[-1]) != requested_theta_dim:
        raise ValueError(
            f"Checkpoint loading: grid_cosmos must have theta_dim={requested_theta_dim}; "
            f"received {grid_cosmos.shape[-1]}."
        )
    model = cls.from_checkpoint(path, map_location="cpu")
    if model.theta_dim != stored_theta_dim:
        raise ValueError(
            f"Checkpoint theta_dim is {stored_theta_dim}, but its parameter list has " f"length {model.theta_dim}."
        )
    model.model_dir = os.path.dirname(path)
    return model


LIKELIHOODS = {
    "flow": LikelihoodImplementation("likelihood_flow", _flow_class, _build_flow, _load_flow),
    "cfm": LikelihoodImplementation("likelihood_cfm", _cfm_class, _build_cfm, _load_cfm),
    "gmm": LikelihoodImplementation("likelihood_gmm", _gmm_class, _build_gmm, _load_gmm),
}


def get_likelihood(name):
    """Return the registered implementation for a stable CLI name."""
    try:
        return LIKELIHOODS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown likelihood model {name!r}; choose from {', '.join(LIKELIHOODS)}.") from exc


def build_likelihood(name, **kwargs):
    return get_likelihood(name).build(**kwargs)


def load_likelihood(name, **kwargs):
    return get_likelihood(name).load(**kwargs)


def load_likelihood_checkpoint(name, model_dir, *, map_location="cpu"):
    """Restore a registered wrapper directly from its self-describing checkpoint."""
    import os

    implementation = get_likelihood(name)
    cls = implementation.wrapper_class()
    if not hasattr(cls, "from_checkpoint"):
        raise ValueError(f"The registered {name!r} likelihood does not provide the common checkpoint API.")
    checkpoint_file = os.path.join(model_dir, implementation.model_name + ".pt")
    if name == "flow":
        return cls.from_checkpoint(checkpoint_file=checkpoint_file, device=map_location)
    return cls.from_checkpoint(checkpoint_file, map_location=map_location)
