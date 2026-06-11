import glob
import os
import re

import h5py
import numpy as np

from msfm.utils.input_output import read_yaml
from msi.flow_conductor import architecture
from msi.flow_conductor.likelihood_flow import LikelihoodFlow
from msi.utils import input_output


def _find_latest_n_steps(pred_dir):
    matches = glob.glob(os.path.join(pred_dir, "preds_*.h5"))
    steps = []
    for f in matches:
        m = re.search(r"preds_(\d+)\.h5$", f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def find_all_n_steps(pred_dir):
    """Return a sorted list of all training-step counts with existing preds_*.h5 files."""
    matches = glob.glob(os.path.join(pred_dir, "preds_*.h5"))
    steps = []
    for f in matches:
        m = re.search(r"preds_(\d+)\.h5$", f)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _fit_pca(x, n_components):
    """Fit PCA via covariance eigendecomposition. Returns (mean, components) where
    components has shape (n_components, n_features), ordered by descending variance."""
    mean = x.mean(axis=0)
    cov = np.cov((x - mean).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx[:n_components]].T  # (n_components, n_features)
    return mean, components


def _load_grid_indices(pred_file):
    """Load and flatten the (i_sobol, i_signal, i_noise) triplet identifying each grid/test row."""
    with h5py.File(pred_file, "r") as f:
        i_sobol = f["grid/i_sobol/test"][:]
        i_signal = f["grid/i_signal/test"][:]
        i_noise = f["grid/i_noise/test"][:]
    if i_sobol.ndim == 2:
        i_sobol, i_signal, i_noise = (np.concatenate(arr, axis=0) for arr in (i_sobol, i_signal, i_noise))
    return i_sobol, i_signal, i_noise


def resolve_pred_file(out_dir, model_name, n_steps=None):
    """Resolve a trained model's prediction file, auto-detecting the latest preds_*.h5 if n_steps is omitted.

    Returns:
        tuple: (pred_dir, pred_file, n_steps)
    """
    pred_dir = os.path.join(out_dir, model_name)
    if n_steps is None:
        n_steps = _find_latest_n_steps(pred_dir)
        if n_steps is not None:
            print(f"Auto-detected n_steps={n_steps}")
        else:
            print("No preds_*.h5 found; falling back to preds.h5")
    pred_file = (
        os.path.join(pred_dir, f"preds_{n_steps}.h5") if n_steps is not None else os.path.join(pred_dir, "preds.h5")
    )
    return pred_dir, pred_file, n_steps


def load_grid_summaries(pred_file, pred_file_2=None):
    """Load network summary statistics for flow training/evaluation, optionally combining two models.

    Loads grid_preds/grid_cosmos/obs_pred_dict/obs_cosmo_dict from pred_file and -- if pred_file_2 is
    given -- row-aligns both grids on (i_sobol, i_signal, i_noise) and concatenates their summaries
    feature-wise, e.g. to combine a maps-level and a Cls-level model. Sharing this between
    run_inference.py (which trains/loads the flow) and run_mcmc_for_coverage_tests.py (which
    reproduces the flow's held-out validation split for coverage testing) guarantees both build
    grid_preds/grid_cosmos identically -- same content, same row order -- which the latter relies on
    to faithfully reproduce the split.

    Also returns i_signal, row-aligned with grid_preds/grid_cosmos, so callers can group rows by
    signal realization -- e.g. to build a deterministic, signal-id-grouped flow train/vali split
    that never places different noise realizations of the same signal in both sets (see
    LikelihoodFlow._prepare_data's group_ids argument).

    Returns:
        tuple: (grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal)
    """
    print(f"Loading predictions from: {pred_file}")
    grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict = input_output.load_network_preds_simple(pred_file)
    _, i_signal, _ = _load_grid_indices(pred_file)

    if pred_file_2:
        print(f"Loading second predictions from: {pred_file_2}")
        grid_preds_2, grid_cosmos_2, obs_pred_dict_2, _ = input_output.load_network_preds_simple(pred_file_2)

        # the two prediction files generally store the same held-out grid examples in different
        # orders, so align them onto a common (i_sobol, i_signal, i_noise) ordering before comparing
        print("Aligning the two grids by (i_sobol, i_signal, i_noise)...")
        i_sobol, i_signal, i_noise = _load_grid_indices(pred_file)
        i_sobol_2, i_signal_2, i_noise_2 = _load_grid_indices(pred_file_2)

        order = np.lexsort((i_noise, i_signal, i_sobol))
        order_2 = np.lexsort((i_noise_2, i_signal_2, i_sobol_2))
        grid_preds, grid_cosmos = grid_preds[order], grid_cosmos[order]
        grid_preds_2, grid_cosmos_2 = grid_preds_2[order_2], grid_cosmos_2[order_2]
        i_signal = i_signal[order]

        aligned = (
            np.array_equal(i_sobol[order], i_sobol_2[order_2])
            and np.array_equal(i_signal, i_signal_2[order_2])
            and np.array_equal(i_noise[order], i_noise_2[order_2])
        )
        if not aligned or not np.allclose(grid_cosmos, grid_cosmos_2):
            raise ValueError(
                "Cannot align the two prediction files' grid examples by (i_sobol, i_signal, i_noise); "
                "cannot concatenate their summaries row-wise. Make sure both models were trained on "
                "the same dataset version and train/eval grid split."
            )

        print("Concatenating summaries from the two models...")
        grid_preds = np.concatenate([grid_preds, grid_preds_2], axis=-1)
        obs_pred_dict = {
            label: np.concatenate([obs_pred_dict[label], obs_pred_dict_2[label]], axis=-1)
            for label in obs_pred_dict.keys() & obs_pred_dict_2.keys()
        }

    return grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal


def load_grid_summaries_multi(pred_files, pca_compress=False):
    """Load and concatenate summary statistics from multiple prediction files feature-wise.

    Row-aligns all files on (i_sobol, i_signal, i_noise) before concatenating, then optionally
    applies PCA to compress the combined data vector back to single-run dimensionality.

    Args:
        pred_files: List of paths to preds_*.h5 files (e.g. for different training-step checkpoints).
        pca_compress: If True, fit PCA on the concatenated grid_preds and project both grid_preds
            and obs_pred_dict down to the same dimensionality as a single file's summaries.

    Returns:
        tuple: (grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal) — same
            signature as load_grid_summaries.
    """
    all_grid_preds, all_grid_cosmos, all_obs_pred_dicts, all_obs_cosmo_dicts, all_indices = [], [], [], [], []

    for pf in pred_files:
        print(f"Loading predictions from: {pf}")
        gp, gc, opd, ocd = input_output.load_network_preds_simple(pf)
        all_grid_preds.append(gp)
        all_grid_cosmos.append(gc)
        all_obs_pred_dicts.append(opd)
        all_obs_cosmo_dicts.append(ocd)
        all_indices.append(_load_grid_indices(pf))

    sort_orders = [np.lexsort((idx[2], idx[1], idx[0])) for idx in all_indices]

    ref_sobol, ref_signal, ref_noise = (arr[sort_orders[0]] for arr in all_indices[0])
    ref_cosmos = all_grid_cosmos[0][sort_orders[0]]

    print("Aligning prediction files by (i_sobol, i_signal, i_noise)...")
    for i in range(1, len(pred_files)):
        s_sobol, s_signal, s_noise = (arr[sort_orders[i]] for arr in all_indices[i])
        aligned = (
            np.array_equal(ref_sobol, s_sobol)
            and np.array_equal(ref_signal, s_signal)
            and np.array_equal(ref_noise, s_noise)
            and np.allclose(ref_cosmos, all_grid_cosmos[i][sort_orders[i]])
        )
        if not aligned:
            raise ValueError(
                f"Prediction file {pred_files[i]} has mismatched grid examples; "
                "make sure all files were evaluated on the same dataset version and grid split."
            )

    sorted_preds = [gp[order] for gp, order in zip(all_grid_preds, sort_orders)]
    print("Concatenating summaries feature-wise...")
    grid_preds = np.concatenate(sorted_preds, axis=-1)
    grid_cosmos = ref_cosmos
    i_signal = ref_signal

    common_keys = set.intersection(*[set(opd.keys()) for opd in all_obs_pred_dicts])
    obs_pred_dict = {
        key: np.concatenate([opd[key] for opd in all_obs_pred_dicts], axis=-1) for key in common_keys
    }
    obs_cosmo_dict = all_obs_cosmo_dicts[0]

    if pca_compress:
        n_components = all_grid_preds[0].shape[-1]
        print(f"Applying PCA compression: {grid_preds.shape[-1]} → {n_components} components")
        mean, components = _fit_pca(grid_preds, n_components)
        grid_preds = (grid_preds - mean) @ components.T
        obs_pred_dict = {key: (val - mean) @ components.T for key, val in obs_pred_dict.items()}

    return grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal


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


def build_flow(
    params, msfm_conf, pred_dir, n_steps, grid_preds, grid_cosmos, flow_conf: dict, prefix: str = "", i_signal=None
):
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
        i_signal: Optional array of shape (N,), row-aligned with grid_preds/grid_cosmos
            (e.g. from load_grid_summaries). When given, the flow's train/vali split is
            made deterministic and grouped by signal realization, so that no signal
            realization (regardless of its noise realizations) appears in both sets --
            see LikelihoodFlow._prepare_data's group_ids argument.

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
        group_ids=i_signal,
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
