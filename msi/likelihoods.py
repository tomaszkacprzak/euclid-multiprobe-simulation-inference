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
    from msi.likelihood_cnf import LikelihoodCFM

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
    """Build the application-level CFM likelihood wrapper."""
    model_conf = dict(config.get("model", {}))
    configured_dimension = model_conf.pop("dimension", grid_preds.shape[-1])
    if configured_dimension != grid_preds.shape[-1]:
        raise ValueError("The configured CFM dimension does not match the summary dimension.")
    model = _cfm_class()(
        params,
        msfm_conf,
        feature_dim=grid_preds.shape[-1],
        context_dim=grid_cosmos.shape[-1],
        out_dir=pred_dir,
        prefix=prefix,
        suffix=_suffix(n_steps),
        load_existing=False,
        **model_conf,
    )
    training = config.get("training", {})
    model.fit(
        x=grid_preds,
        theta=grid_cosmos,
        n_epochs=training.get("n_epochs", training.get("epochs", 50)),
        batch_size=training.get("batch_size", 2048),
        save_model=True,
    )
    return model


def _load_cfm(*, pred_dir, n_steps, prefix="", **_):
    return _cfm_class().from_checkpoint(out_dir=pred_dir, prefix=prefix, suffix=_suffix(n_steps))


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
