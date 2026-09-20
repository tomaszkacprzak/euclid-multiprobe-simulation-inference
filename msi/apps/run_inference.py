import os
import warnings

import yaml

from msi.utils import observations

_CFM_CONFIG_KEYS = {
    "model": {
        "model_type",
        "sigma_min",
        "hidden_features",
        "num_hidden_layers",
        "ode_steps",
        "divergence_estimator",
        "hutchinson_samples",
        "ode_method",
        "ode_rtol",
        "ode_atol",
        "ode_options",
        "context_adapter",
    },
    "training": {
        "epochs",
        "n_epochs",
        "batch_size",
        "vali_split",
        "learning_rate",
        "weight_decay",
        "scheduler_type",
        "scheduler_kwargs",
        "n_patience_epochs",
        "min_delta",
        "clip_by_global_norm",
        "seed",
    },
    # CFM currently always standardizes summaries and parameters.  Keeping this
    # explicit section (and validating that it is empty) gives preprocessing a
    # stable home without advertising switches which the implementation ignores.
    "preprocessing": set(),
    "diagnostics": {"n_cosmos"},
    "mcmc": {"n_walkers", "n_steps", "n_burnin_steps"},
}

_CFM_CONTEXT_ADAPTER_KEYS = {"type", "hidden_features", "num_hidden_layers", "activation"}
_OTHER_LIKELIHOOD_KEYS = {"context_embedding", "transform", "observations"}


def _validate_likelihood_config(config, likelihood_model):
    """Validate the selected likelihood's YAML and return it unchanged.

    Legacy normalizing-flow and GMM dictionaries remain permissive for backward
    compatibility.  CFM is new and uses a deliberately strict, sectioned schema
    so misspelled or flow-family options cannot be silently discarded.
    """
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("Likelihood config must be a YAML mapping at the top level.")
    if likelihood_model != "cfm":
        return config

    unknown_sections = set(config) - set(_CFM_CONFIG_KEYS)
    if unknown_sections:
        family_keys = unknown_sections & _OTHER_LIKELIHOOD_KEYS
        detail = (
            " These options belong to the legacy flow likelihood; select "
            "--likelihood-model flow or use the sectioned CFM schema."
            if family_keys
            else ""
        )
        raise ValueError(
            "Invalid CFM likelihood config section(s): "
            + ", ".join(sorted(unknown_sections))
            + ". Expected sections: "
            + ", ".join(_CFM_CONFIG_KEYS)
            + "."
            + detail
        )

    for section, allowed_keys in _CFM_CONFIG_KEYS.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"CFM config section {section!r} must be a YAML mapping.")
        unknown_keys = set(values) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"Invalid key(s) in CFM {section!r} section: "
                f"{', '.join(sorted(unknown_keys))}. Allowed keys: "
                f"{', '.join(sorted(allowed_keys)) or '(none)'}."
            )

    adapter = (config.get("model") or {}).get("context_adapter", {})
    if adapter is None:
        adapter = {}
    if not isinstance(adapter, dict):
        raise ValueError("CFM model.context_adapter must be a YAML mapping.")
    unknown_adapter_keys = set(adapter) - _CFM_CONTEXT_ADAPTER_KEYS
    if unknown_adapter_keys:
        raise ValueError(
            "Invalid CFM model.context_adapter key(s): "
            + ", ".join(sorted(unknown_adapter_keys))
            + ". Allowed keys: "
            + ", ".join(sorted(_CFM_CONTEXT_ADAPTER_KEYS))
            + "."
        )
    return config



def _load_configs(pred_dir, msfm_config_path, dlss_config_path):
    """Load msfm_conf and dlss_conf from either explicit paths or pred_dir/configs.yaml.

    configs.yaml always ends with [..., dlss_conf, msfm_conf] regardless of whether it
    was written by the map or Cls training script (3-doc or 4-doc format).
    """
    from msfm.utils import files as msfm_files
    from msfm.utils.input_output import read_yaml

    if msfm_config_path and dlss_config_path:
        msfm_conf = msfm_files.load_config(msfm_config_path)
        dlss_conf = read_yaml(dlss_config_path)
    else:
        configs_path = os.path.join(pred_dir, "configs.yaml")
        with open(configs_path) as f:
            docs = list(yaml.load_all(f, Loader=yaml.FullLoader))
        dlss_conf, msfm_conf = docs[-2], docs[-1]
    return dlss_conf, msfm_conf


def configure_parser(parser):
    """Add the inference workflow's documented arguments to ``parser``.

    Both hyphenated and underscored spellings are retained because existing
    launch scripts use both forms.  Paths in this interface refer to artifacts
    produced by the companion network-training workflow; this application does
    not create prediction files itself.
    """
    from msi.likelihoods import LIKELIHOODS

    # Primary prediction source: together these identify OUT_DIR/MODEL_NAME,
    # which contains the prediction HDF5 file and, normally, configs.yaml.
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        required=True,
        metavar="PATH",
        help="Base output directory containing the trained summary-network model directory (required).",
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        dest="model_name",
        default="model",
        metavar="NAME",
        help="Name of the primary model subdirectory inside --out-dir (default: %(default)s).",
    )
    # Select the conditional likelihood implementation trained on (or loaded
    # for) the network summaries.
    parser.add_argument(
        "--likelihood-model",
        choices=tuple(LIKELIHOODS),
        default="flow",
        help="Conditional likelihood implementation to train or load (default: %(default)s).",
    )

    # Optional second prediction source.  When enabled, its rows are aligned
    # with the primary source and its summary features are concatenated.
    parser.add_argument(
        "--out-dir-2",
        "--out_dir_2",
        dest="out_dir_2",
        default=None,
        metavar="PATH",
        help="Optional second model's out_dir; its summary is concatenated feature-wise with the "
        "primary model's summary, e.g. to combine a maps-level and a Cls-level model.",
    )
    parser.add_argument(
        "--model-name-2",
        "--model_name_2",
        dest="model_name_2",
        default="model",
        metavar="NAME",
        help="Name of the second model subdirectory inside --out-dir-2 (default: %(default)s).",
    )
    parser.add_argument(
        "--n-steps-2",
        "--n_steps_2",
        dest="n_steps_2",
        type=int,
        default=None,
        metavar="STEPS",
        help="Training-step count selecting preds_STEPS.h5 for the second model; auto-detects "
        "the largest available count, then falls back to preds.h5, when omitted.",
    )

    # Configuration metadata for the summary network and its dataset.  The two
    # overrides operate as a pair; otherwise both documents come from the
    # primary prediction directory's configs.yaml.
    parser.add_argument(
        "--msfm-config",
        "--msfm_config",
        dest="msfm_config",
        default=None,
        metavar="YAML",
        help="Explicit MSFM model YAML path. Used only together with --dlss-config; otherwise "
        "both configs are read from the primary model directory's configs.yaml.",
    )
    parser.add_argument(
        "--dlss-config",
        "--dlss_config",
        dest="dlss_config",
        default=None,
        metavar="YAML",
        help="Explicit DeepLSS dataset YAML path. Used only together with --msfm-config; otherwise "
        "both configs are read from the primary model directory's configs.yaml.",
    )

    # A single primary prediction checkpoint can be selected explicitly.  With
    # no selection, resolution prefers the greatest numeric preds_*.h5 suffix.
    parser.add_argument(
        "--n-steps",
        "--n_steps",
        dest="n_steps",
        type=int,
        default=None,
        metavar="STEPS",
        help="Prediction file step count; auto-detects the largest preds_*.h5 if omitted.",
    )
    # Multi-checkpoint mode combines features from an explicit set of primary
    # model prediction files.  It is mutually exclusive with --n-steps-all.
    parser.add_argument(
        "--n-steps-multi",
        "--n_steps_multi",
        dest="n_steps_multi",
        nargs="+",
        type=int,
        default=None,
        metavar="STEPS",
        help="Combine predictions from these specific training-step counts (feature-wise concatenation).",
    )
    # This is the discovery-based counterpart to --n-steps-multi.
    parser.add_argument(
        "--n-steps-all",
        "--n_steps_all",
        dest="n_steps_all",
        action="store_true",
        help="Combine predictions from ALL preds_*.h5 files found in the model directory.",
    )
    # PCA is meaningful in multi-checkpoint mode: it projects the concatenated
    # features to the dimensionality of one prediction file's summaries.
    parser.add_argument(
        "--pca-compress",
        "--pca_compress",
        dest="pca_compress",
        action="store_true",
        help="After concatenating multi-step summaries, apply PCA to compress back to single-run dimensionality.",
    )

    # Likelihood configuration is separate from the summary-network metadata.
    # An empty configuration selects implementation-level defaults.
    parser.add_argument(
        "--likelihood-config",
        "--likelihood_config",
        dest="likelihood_config",
        default=None,
        metavar="YAML",
        help="Path to the selected likelihood model's YAML config; uses implementation defaults if omitted.",
    )
    # Loading skips likelihood fitting and diagnostic generation, and restores
    # the implementation-specific checkpoint from the primary model directory.
    parser.add_argument(
        "--load-likelihood",
        "--load_likelihood",
        dest="load_likelihood",
        action="store_true",
        help="Load an existing likelihood checkpoint instead of training a new one.",
    )
    # The label disambiguates likelihood experiments that share predictions by
    # adding a prefix to the implementation's checkpoint directory name.
    parser.add_argument(
        "--likelihood-label",
        "--likelihood_label",
        dest="likelihood_label",
        default="",
        metavar="LABEL",
        help="Prefix added to the selected implementation's likelihood checkpoint directory. "
        "Useful when comparing multiple likelihood configs on the same prediction file.",
    )
    # Observation flags determine which stored summaries receive posterior
    # sampling after the likelihood has been trained or loaded.
    observations.add_obs_args(parser)
    return parser

def get_param_names(dlss_conf):

    # Legacy DES config format
    if 'dset' in dlss_conf:
        params = dlss_conf["dset"]["training"]["params"]
        print(f"params: {params}")


    # New Euclid config format
    else:

        from euclid_multiprobe_deeplss_training.training import TrainingConfig, load_physics_model_class
        from euclid_multiprobe_deeplss_training.utils.config import with_forward_model_config, load_config
        from pathlib import Path

        raw_config = with_forward_model_config(dlss_conf)
        config = TrainingConfig.from_mapping(raw_config)
        physics_model_class = load_physics_model_class(config.physics_model)
        print(f"physics_model_class: {physics_model_class}")
        physics_model = physics_model_class(
            config.forward_model,
            nside=config.forward_model["analysis"]["n_side"],
            **config.physics_model_args,
        )
        params = physics_model.params
        print(f"params: {params}")

    return params


def main(args):
    """Run the inference workload using parsed command-line arguments."""
    from msfm.utils.input_output import read_yaml

    from msi.likelihoods import build_likelihood, load_likelihood
    from msi.utils import flow as flow_utils

    is_multi = args.n_steps_multi is not None or args.n_steps_all
    if args.n_steps_multi is not None and args.n_steps_all:
        raise ValueError("--n_steps_multi and --n_steps_all are mutually exclusive.")

    config_path = args.likelihood_config
    likelihood_conf = read_yaml(config_path) if config_path else {}
    likelihood_conf = _validate_likelihood_config(likelihood_conf, args.likelihood_model)
    prefix = f"{args.likelihood_label}_" if args.likelihood_label else ""

    if is_multi:
        pred_dir = os.path.join(args.out_dir, args.model_name)
        if args.n_steps_all:
            steps_list = flow_utils.find_all_n_steps(pred_dir)
            if not steps_list:
                raise FileNotFoundError(f"No preds_*.h5 found in {pred_dir}")
            print(f"Using all steps: {steps_list}")
        else:
            steps_list = sorted(args.n_steps_multi)
        pred_files = [os.path.join(pred_dir, f"preds_{s}.h5") for s in steps_list]

        grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal = flow_utils.load_grid_summaries_multi(
            pred_files, pca_compress=args.pca_compress
        )

        dlss_conf, msfm_conf = _load_configs(pred_dir, args.msfm_config, args.dlss_config)
        params = get_param_names(dlss_conf)

        steps_str = "_".join(str(s) for s in steps_list)
        n_steps_label = f"multi_{steps_str}" + ("_pca" if args.pca_compress else "")

        if args.load_flow:
            print(f"Loading {args.likelihood_model} likelihood from checkpoint...")
            flow = load_likelihood(
                args.likelihood_model,
                params=params,
                msfm_conf=msfm_conf,
                pred_dir=pred_dir,
                n_steps=n_steps_label,
                prefix=prefix,
                grid_preds=grid_preds,
                grid_cosmos=grid_cosmos,
            )
        else:
            flow = build_likelihood(
                args.likelihood_model,
                params=params,
                msfm_conf=msfm_conf,
                pred_dir=pred_dir,
                n_steps=n_steps_label,
                grid_preds=grid_preds,
                grid_cosmos=grid_cosmos,
                config=likelihood_conf,
                prefix=prefix,
                i_signal=i_signal,
            )
    else:
        pred_dir, pred_file, n_steps = flow_utils.resolve_pred_file(args.out_dir, args.model_name, args.n_steps)
        print(f"pred_dir: {pred_dir}")
        print(f"pred_file: {pred_file}")
        print(f"n_steps: {n_steps}")
        dlss_conf, msfm_conf = _load_configs(pred_dir, args.msfm_config, args.dlss_config)
        params = get_param_names(dlss_conf)

        pred_file_2 = None
        if args.out_dir_2:
            _, pred_file_2, _ = flow_utils.resolve_pred_file(args.out_dir_2, args.model_name_2, args.n_steps_2)

        grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal = flow_utils.load_grid_summaries(
            pred_file, pred_file_2
        )

        if args.load_likelihood:
            print(f"Loading {args.likelihood_model} likelihood from checkpoint...")
            flow = load_likelihood(
                args.likelihood_model,
                params=params,
                msfm_conf=msfm_conf,
                pred_dir=pred_dir,
                n_steps=n_steps,
                prefix=prefix,
                grid_preds=grid_preds,
                grid_cosmos=grid_cosmos,
            )
        else:
            flow = build_likelihood(
                args.likelihood_model,
                params=params,
                msfm_conf=msfm_conf,
                pred_dir=pred_dir,
                n_steps=n_steps,
                grid_preds=grid_preds,
                grid_cosmos=grid_cosmos,
                config=likelihood_conf,
                prefix=prefix,
                i_signal=i_signal,
            )

    if not args.load_likelihood and hasattr(flow, "plot_diagnostics"):
        diagnostics_conf = likelihood_conf.get("diagnostics", {})
        print("Plotting diagnostics...")
        try:
            flow.plot_diagnostics(
                grid_preds_true=grid_preds,
                grid_cosmos=grid_cosmos,
                n_cosmos=diagnostics_conf.get("n_cosmos", 1000),
            )
        except Exception as e:
            print(f"WARNING: plot_diagnostics failed ({type(e).__name__}: {e}), skipping.")

    obs_dict = observations.collect_observations(args, obs_pred_dict, obs_cosmo_dict, params, msfm_conf)

    mcmc_conf = likelihood_conf.get("mcmc", {})
    try:
        observations.run_mcmc(
            flow,
            obs_dict,
            n_walkers=mcmc_conf.get("n_walkers", 1024),
            n_steps=mcmc_conf.get("n_steps", 1000),
            n_burnin_steps=mcmc_conf.get("n_burnin_steps", 1000),
        )
    except Exception as e:
        print(f"ERROR: run_mcmc failed ({type(e).__name__}: {e})")
