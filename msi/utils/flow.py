from msfm.utils.input_output import read_yaml
from msi.flow_conductor import architecture
from msi.flow_conductor.likelihood_flow import LikelihoodFlow


def build_flow_architecture(x_dim: int, theta_dim: int, flow_conf: dict):
    """Build embedding net and transform from a config dict.

    Defaults match default.yaml / the original notebook values, so passing an empty
    dict reproduces the standard architecture.

    Two transform families are supported via ``transform.type``:
    - ``"sigmoids"`` (default): ConditionalSVD + MaskedSumOfSigmoids layers.
    - ``"lipschitz"``: Lipschitz-constrained iResBlocks (architecturally independent,
      useful as a cross-check when diagnosing posterior instability).

    Returns:
        tuple: (embedding_net, transform) ready to pass to LikelihoodFlow.
    """
    emb_conf = flow_conf.get("context_embedding", {})
    ctx_emb_dim = emb_conf.get("dim", 32)

    embedding_net = architecture.get_context_embedding_net(
        context_dim=theta_dim,
        context_embedding_dim=ctx_emb_dim,
        hidden_dim=emb_conf.get("hidden_dim", 64),
    )

    tr_conf = flow_conf.get("transform", {})
    transform_type = tr_conf.get("type", "sigmoids")

    if transform_type == "sigmoids":
        sig_conf = tr_conf.get("sigmoids", {})
        transform = architecture.get_sigmoids_transform(
            feature_dim=x_dim,
            context_embedding_dim=ctx_emb_dim,
            n_layers=tr_conf.get("n_layers", 4),
            hidden_dim=tr_conf.get("hidden_dim", 256),
            svd_kwargs={},
            sigmoids_kwargs={
                "n_sigmoids": sig_conf.get("n_sigmoids", 16),
                "num_blocks": sig_conf.get("num_blocks", 3),
                "dropout_probability": sig_conf.get("dropout_probability", 0.0),
            },
        )
    elif transform_type == "lipschitz":
        lip_conf = tr_conf.get("lipschitz", {})
        transform = architecture.get_lipschitz_transform(
            feature_dim=x_dim,
            context_embedding_dim=ctx_emb_dim,
            n_layers=tr_conf.get("n_layers", 8),
            hidden_dim=tr_conf.get("hidden_dim", 128),
            lipschitz_coeff=lip_conf.get("lipschitz_coeff", 0.97),
        )
    else:
        raise ValueError(f"Unknown transform type: {transform_type!r}. Choose 'sigmoids' or 'lipschitz'.")

    return embedding_net, transform


def build_flow(params, msfm_conf, pred_dir, n_steps, grid_preds, grid_cosmos, flow_conf: dict, prefix: str = ""):
    """Build, train, plot diagnostics, and return a LikelihoodFlow.

    Args:
        params: List of cosmological parameter names.
        msfm_conf: Forward-model config dict (passed to LikelihoodFlow).
        pred_dir: Output directory for checkpoints and plots.
        n_steps: Training-step label appended to saved filenames.
        grid_preds: Array of shape (N, x_dim) — network summary statistics.
        grid_cosmos: Array of shape (N, theta_dim) — cosmological parameters.
        flow_conf: Flow config dict (keys: context_embedding, transform, training,
            diagnostics). Use {} or read_yaml(path) to populate.
        prefix: Prepended to the saved model directory name, e.g. ``"larger_"`` →
            ``pred_dir/larger_likelihood_flow_{n_steps}/``. Useful when comparing
            multiple flow configs on the same prediction file.

    Returns:
        LikelihoodFlow: Trained flow with saved checkpoint.
    """
    x_dim = grid_preds.shape[-1]
    theta_dim = grid_cosmos.shape[-1]

    embedding_net, transform = build_flow_architecture(x_dim, theta_dim, flow_conf)

    suffix = f"_{n_steps}" if n_steps is not None else ""
    flow = LikelihoodFlow(
        params,
        msfm_conf,
        feature_dim=x_dim,
        embedding_net=embedding_net,
        transform=transform,
        out_dir=pred_dir,
        prefix=prefix,
        suffix=suffix,
        load_existing=False,
    )

    train_conf = flow_conf.get("training", {})
    print("Fitting flow...")
    flow.fit(
        x=grid_preds,
        theta=grid_cosmos,
        n_epochs=train_conf.get("n_epochs", 100),
        batch_size=train_conf.get("batch_size", 10_000),
        scheduler_type=train_conf.get("scheduler_type", "cosine"),
        save_model=True,
        run_c2st=True,
    )

    diag_conf = flow_conf.get("diagnostics", {})
    print("Plotting diagnostics...")
    try:
        flow.plot_diagnostics(
            grid_preds_true=grid_preds,
            grid_cosmos=grid_cosmos,
            n_cosmos=diag_conf.get("n_cosmos", 1000),
        )
    except Exception as e:
        print(f"WARNING: plot_diagnostics failed ({type(e).__name__}: {e}), skipping.")

    return flow
