import argparse
import glob
import os
import re

import yaml

from msfm.utils import files as msfm_files
from msfm.utils.input_output import read_yaml
from msi.flow_conductor.likelihood_flow import LikelihoodFlow
from msi.utils import flow as flow_utils
from msi.utils import input_output, observations


def _find_latest_n_steps(pred_dir):
    matches = glob.glob(os.path.join(pred_dir, "preds_*.h5"))
    steps = []
    for f in matches:
        m = re.search(r"preds_(\d+)\.h5$", f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


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

    pred_dir = os.path.join(args.out_dir, args.model_name)
    dlss_conf, msfm_conf = _load_configs(pred_dir, args.msfm_config, args.dlss_config)
    params = dlss_conf["dset"]["training"]["params"]

    n_steps = args.n_steps
    if n_steps is None:
        n_steps = _find_latest_n_steps(pred_dir)
        if n_steps is not None:
            print(f"Auto-detected n_steps={n_steps}")
        else:
            print("No preds_*.h5 found; falling back to preds.h5")

    pred_file = (
        os.path.join(pred_dir, f"preds_{n_steps}.h5") if n_steps is not None else os.path.join(pred_dir, "preds.h5")
    )

    print(f"Loading predictions from: {pred_file}")
    grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict = input_output.load_network_preds_simple(pred_file)

    flow_conf = read_yaml(args.flow_config) if args.flow_config else {}

    prefix = f"{args.flow_label}_" if args.flow_label else ""

    if args.load_flow:
        suffix = f"_{n_steps}" if n_steps is not None else ""
        print("Loading flow from checkpoint...")
        flow = LikelihoodFlow.from_checkpoint(out_dir=pred_dir, prefix=prefix, suffix=suffix)
    else:
        flow = flow_utils.build_flow(params, msfm_conf, pred_dir, n_steps, grid_preds, grid_cosmos, flow_conf, prefix=prefix)

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
