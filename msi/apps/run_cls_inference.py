import argparse
import os

import yaml

from msfm.utils import files
from msfm.utils.input_output import read_yaml
from msi.flow_conductor.likelihood_flow import LikelihoodFlow
from msi.flow_conductor import architecture
from msi.utils import input_output, observations


def setup():
    parser = argparse.ArgumentParser(description="Inference using LikelihoodFlow on predicted Cls summaries.")
    parser.add_argument("--msfm_config", required=True)
    parser.add_argument("--dlss_config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--n_steps", type=int, default=None, help="Override n_steps; default reads from configs.yaml")

    # Flow training parameters
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10000)

    observations.add_obs_args(parser)
    return parser.parse_args()


def main():
    args = setup()

    msfm_conf = files.load_config(args.msfm_config)
    dlss_conf = read_yaml(args.dlss_config)

    params = dlss_conf["dset"]["training"]["params"]

    pred_dir = os.path.join(args.out_dir, args.model_name)

    if args.n_steps is not None:
        n_steps = args.n_steps
    else:
        training_configs_path = os.path.join(pred_dir, "configs.yaml")
        with open(training_configs_path) as f:
            mlp_conf = next(yaml.safe_load_all(f))
        n_steps = mlp_conf["n_steps"]

    pred_file = os.path.join(pred_dir, f"preds_{n_steps}.h5")

    print(f"Loading predictions from: {pred_file}")
    grid_preds, grid_cosmos, obs_pred_dict, obs_cosmo_dict = input_output.load_network_preds_simple(pred_file)

    x_dim = grid_preds.shape[-1]
    theta_dim = grid_cosmos.shape[-1]

    context_embedding_dim = 32

    embedding_net = architecture.get_context_embedding_net(
        context_dim=theta_dim,
        context_embedding_dim=context_embedding_dim,
        hidden_dim=64,
    )

    transform = architecture.get_sigmoids_transform(
        feature_dim=x_dim,
        context_embedding_dim=context_embedding_dim,
        n_layers=4,
        hidden_dim=256,
        svd_kwargs={},
        sigmoids_kwargs={
            "n_sigmoids": 16,
            "num_blocks": 3,
            "dropout_probability": 0.0,
        },
    )

    flow = LikelihoodFlow(
        params,
        msfm_conf,
        feature_dim=x_dim,
        embedding_net=embedding_net,
        transform=transform,
        out_dir=pred_dir,
        suffix=f"_{n_steps}",
        load_existing=False,
    )

    print("Fitting flow...")
    flow.fit(
        x=grid_preds,
        theta=grid_cosmos,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        scheduler_type="cosine",
        save_model=True,
        run_c2st=True,
    )

    print("Plotting diagnostics...")
    flow.plot_diagnostics(
        grid_preds_true=grid_preds,
        grid_cosmos=grid_cosmos,
        n_cosmos=1000,
    )

    obs_dict = observations.collect_observations(args, obs_pred_dict, obs_cosmo_dict, params, msfm_conf)
    observations.run_mcmc(flow, obs_dict)


if __name__ == "__main__":
    main()
