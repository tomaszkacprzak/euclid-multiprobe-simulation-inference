import argparse
import os

import yaml

from msfm.utils import files as msfm_files
from msfm.utils.input_output import read_yaml
from msi.flow_conductor.likelihood_flow import LikelihoodFlow
from msi.utils import flow as flow_utils
from msi.utils import observations


def _load_configs(pred_dir, msfm_config_path, dlss_config_path):
    """Load msfm_conf and dlss_conf from either explicit paths or pred_dir/configs.yaml.

    configs.yaml always ends with [..., dlss_conf, msfm_conf] regardless of whether it
    was written by the map or Cls training script (3-doc or 4-doc format).
    """
    if msfm_config_path and dlss_config_path:
        msfm_conf = msfm_files.load_config(msfm_config_path)
        dlss_conf = read_yaml(dlss_config_path)
    else:
        configs_path = os.path.join(pred_dir, "configs.yaml")
        with open(configs_path) as f:
            docs = list(yaml.load_all(f, Loader=yaml.FullLoader))
        dlss_conf, msfm_conf = docs[-2], docs[-1]
    return dlss_conf, msfm_conf


def setup():
    parser = argparse.ArgumentParser(
        description="Normalizing flow inference on network summary statistics (maps or Cls)."
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument(
        "--out_dir_2",
        default=None,
        help="Optional second model's out_dir; its summary is concatenated feature-wise with the "
        "primary model's summary, e.g. to combine a maps-level and a Cls-level model.",
    )
    parser.add_argument("--model_name_2", default="model")
    parser.add_argument("--n_steps_2", type=int, default=None)
    # Optional explicit config overrides (Cls path); falls back to pred_dir/configs.yaml
    parser.add_argument("--msfm_config", default=None)
    parser.add_argument("--dlss_config", default=None)
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="Prediction file step count; auto-detects the largest preds_*.h5 if omitted.",
    )
    parser.add_argument(
        "--n_steps_multi",
        nargs="+",
        type=int,
        default=None,
        help="Combine predictions from these specific training-step counts (feature-wise concatenation).",
    )
    parser.add_argument(
        "--n_steps_all",
        action="store_true",
        help="Combine predictions from ALL preds_*.h5 files found in the model directory.",
    )
    parser.add_argument(
        "--pca_compress",
        action="store_true",
        help="After concatenating multi-step summaries, apply PCA to compress back to single-run dimensionality.",
    )
    parser.add_argument(
        "--flow_config",
        default=None,
        help="Path to flow YAML config; uses hardcoded defaults if omitted.",
    )
    parser.add_argument(
        "--load_flow",
        action="store_true",
        help="Load existing flow checkpoint instead of training a new one.",
    )
    parser.add_argument(
        "--flow_label",
        default="",
        help="Prefix for the flow checkpoint directory, e.g. 'larger' saves to "
        "pred_dir/larger_likelihood_flow_{n_steps}/. Useful when comparing multiple "
        "flow configs on the same prediction file.",
    )
    observations.add_obs_args(parser)
    return parser.parse_args()


def main():
    args = setup()

    is_multi = args.n_steps_multi is not None or args.n_steps_all
    if args.n_steps_multi is not None and args.n_steps_all:
        raise ValueError("--n_steps_multi and --n_steps_all are mutually exclusive.")

    flow_conf = read_yaml(args.flow_config) if args.flow_config else {}
    prefix = f"{args.flow_label}_" if args.flow_label else ""

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
        params = dlss_conf["dset"]["training"]["params"]

        steps_str = "_".join(str(s) for s in steps_list)
        n_steps_label = f"multi_{steps_str}" + ("_pca" if args.pca_compress else "")

        if args.load_flow:
            suffix = f"_{n_steps_label}"
            print("Loading flow from checkpoint...")
            flow = LikelihoodFlow.from_checkpoint(out_dir=pred_dir, prefix=prefix, suffix=suffix)
        else:
            flow = flow_utils.build_flow(
                params, msfm_conf, pred_dir, n_steps_label, grid_preds, grid_cosmos, flow_conf,
                prefix=prefix, i_signal=i_signal,
            )
    else:
        pred_dir, pred_file, n_steps = flow_utils.resolve_pred_file(args.out_dir, args.model_name, args.n_steps)
        dlss_conf, msfm_conf = _load_configs(pred_dir, args.msfm_config, args.dlss_config)
        params = dlss_conf["dset"]["training"]["params"]

        pred_file_2 = None
        if args.out_dir_2:
            _, pred_file_2, _ = flow_utils.resolve_pred_file(args.out_dir_2, args.model_name_2, args.n_steps_2)

        grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict, i_signal = flow_utils.load_grid_summaries(
            pred_file, pred_file_2
        )

        if args.load_flow:
            suffix = f"_{n_steps}" if n_steps is not None else ""
            print("Loading flow from checkpoint...")
            flow = LikelihoodFlow.from_checkpoint(out_dir=pred_dir, prefix=prefix, suffix=suffix)
        else:
            flow = flow_utils.build_flow(
                params, msfm_conf, pred_dir, n_steps, grid_preds, grid_cosmos, flow_conf, prefix=prefix, i_signal=i_signal
            )

    obs_dict = observations.collect_observations(args, obs_pred_dict, obs_cosmo_dict, params, msfm_conf)

    mcmc_conf = flow_conf.get("mcmc", {})
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


if __name__ == "__main__":
    main()
