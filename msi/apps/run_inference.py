import argparse
import glob
import os
import re

import yaml

from msfm.utils.input_output import read_yaml
from msi.flow_conductor import architecture
from msi.flow_conductor.likelihood_flow import LikelihoodFlow
from msi.utils import input_output, observations


def find_latest_n_steps(pred_dir):
    matches = glob.glob(os.path.join(pred_dir, "preds_*.h5"))
    steps = []
    for f in matches:
        m = re.search(r"preds_(\d+)\.h5$", f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def build_flow(pred_dir, n_steps, params, msfm_conf, grid_preds, grid_cosmos, flow_conf):
    x_dim = grid_preds.shape[-1]
    theta_dim = grid_cosmos.shape[-1]

    emb_conf = flow_conf.get("context_embedding", {})
    ctx_emb_dim = emb_conf.get("dim", 32)

    embedding_net = architecture.get_context_embedding_net(
        context_dim=theta_dim,
        context_embedding_dim=ctx_emb_dim,
        hidden_dim=emb_conf.get("hidden_dim", 64),
    )

    tr_conf = flow_conf.get("transform", {})
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

    suffix = f"_{n_steps}" if n_steps is not None else ""
    flow = LikelihoodFlow(
        params,
        msfm_conf,
        feature_dim=x_dim,
        embedding_net=embedding_net,
        transform=transform,
        out_dir=pred_dir,
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
    return flow


def setup():
    parser = argparse.ArgumentParser(description="Map-level normalizing flow inference on network predictions.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="model")
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
    observations.add_obs_args(parser)
    return parser.parse_args()


def main():
    args = setup()

    pred_dir = os.path.join(args.out_dir, args.model_name)

    with open(os.path.join(pred_dir, "configs.yaml"), "r") as f:
        _net_conf, dlss_conf, msfm_conf = list(yaml.load_all(f, Loader=yaml.FullLoader))

    params = dlss_conf["dset"]["training"]["params"]

    n_steps = args.n_steps
    if n_steps is None:
        n_steps = find_latest_n_steps(pred_dir)
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

    if args.load_flow:
        suffix = f"_{n_steps}" if n_steps is not None else ""
        print("Loading flow from checkpoint...")
        flow = LikelihoodFlow.from_checkpoint(out_dir=pred_dir, suffix=suffix)
    else:
        flow = build_flow(pred_dir, n_steps, params, msfm_conf, grid_preds, grid_cosmos, flow_conf)
        diag_conf = flow_conf.get("diagnostics", {})
        print("Plotting diagnostics...")
        flow.plot_diagnostics(
            grid_preds_true=grid_preds,
            grid_cosmos=grid_cosmos,
            n_cosmos=diag_conf.get("n_cosmos", 1000),
        )

    obs_dict = observations.collect_observations(args, obs_pred_dict, obs_cosmo_dict, params, msfm_conf)

    mcmc_conf = flow_conf.get("mcmc", {})
    observations.run_mcmc(
        flow,
        obs_dict,
        n_walkers=mcmc_conf.get("n_walkers", 1024),
        n_steps=mcmc_conf.get("n_steps", 1000),
        n_burnin_steps=mcmc_conf.get("n_burnin_steps", 1000),
    )


if __name__ == "__main__":
    main()
