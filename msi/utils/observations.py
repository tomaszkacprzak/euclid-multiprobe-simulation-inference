from msfm.utils import parameters


def add_obs_args(parser, bench_labels_default=None):
    """Add observation inclusion flags to an argument parser (all default off)."""
    parser.add_argument("--include_grid", action="store_true")
    parser.add_argument("--n_grid_examples", type=int, default=16)
    parser.add_argument("--include_des", action="store_true")
    parser.add_argument("--include_buzzard", action="store_true")
    parser.add_argument("--buzzard_labels", nargs="+", default=["Buzzard_mean"])
    parser.add_argument("--include_bench", action="store_true")
    parser.add_argument("--bench_labels", nargs="+", default=bench_labels_default or ["bench_fidu"])


def get_grid_observations(obs_pred_dict, obs_cosmo_dict, params, msfm_conf, n_examples=16):
    stride = msfm_conf["analysis"]["grid"].get("n_perms_per_cosmo", 1) * msfm_conf["analysis"].get("n_patches", 1)
    obs_dict = {}
    for i in range(n_examples):
        label = f"grid_{i * stride}"
        if label in obs_pred_dict and label in obs_cosmo_dict:
            obs_dict[label] = {
                "pred": obs_pred_dict[label],
                "cosmo": {str(p): v for p, v in zip(params, obs_cosmo_dict[label])},
            }
    return obs_dict


def get_des_observations(obs_pred_dict):
    obs_dict = {}
    for label in ["DESy3", "DESy3_no_sys"]:
        if label in obs_pred_dict:
            obs_dict[label] = {"pred": obs_pred_dict[label], "cosmo": None}
    return obs_dict


def get_buzzard_observations(obs_pred_dict, params, msfm_conf, labels):
    fiducials = parameters.get_fiducials(params, msfm_conf)
    cosmo = {str(p): v for p, v in zip(params, fiducials)}
    obs_dict = {}
    for label in labels:
        if label in obs_pred_dict:
            obs_dict[label] = {"pred": obs_pred_dict[label], "cosmo": cosmo}
        else:
            print(f"Warning: '{label}' not found in predictions, skipping.")
    return obs_dict


def get_benchmark_observations(obs_pred_dict, params, msfm_conf, obs_labels):
    fiducials = parameters.get_fiducials(params, msfm_conf)
    cosmo = {str(p): v for p, v in zip(params, fiducials)}
    obs_dict = {}
    for label in obs_labels:
        full_label = f"{label}_mean"
        if full_label in obs_pred_dict:
            obs_dict[full_label] = {"pred": obs_pred_dict[full_label], "cosmo": cosmo}
        else:
            print(f"Warning: '{full_label}' not found in predictions, skipping.")
    return obs_dict


def collect_observations(args, obs_pred_dict, obs_cosmo_dict, params, msfm_conf):
    """Build obs_dict from CLI args and loaded prediction dictionaries."""
    obs_dict = {}
    if args.include_grid:
        obs_dict.update(get_grid_observations(obs_pred_dict, obs_cosmo_dict, params, msfm_conf, args.n_grid_examples))
    if args.include_des:
        obs_dict.update(get_des_observations(obs_pred_dict))
    if args.include_buzzard:
        obs_dict.update(get_buzzard_observations(obs_pred_dict, params, msfm_conf, args.buzzard_labels))
    if args.include_bench:
        obs_dict.update(get_benchmark_observations(obs_pred_dict, params, msfm_conf, args.bench_labels))
    return obs_dict


def run_mcmc(flow, obs_dict, n_walkers=1024, n_steps=1000, n_burnin_steps=1000):
    for key, obs in obs_dict.items():
        print(f"\nStarting with mock observation {key}")
        posterior_samples = flow.sample_posterior(
            obs["pred"],
            label=key,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burnin_steps=n_burnin_steps,
        )
        if obs["cosmo"] is not None and "des" not in key.lower():
            flow.plot_contours(
                posterior_samples,
                obs_point=obs["cosmo"],
                obs_label=key,
                label=key,
                with_des_chain=False,
                density=True,
            )
        if "des" in key.lower():
            lcdm_label = f"{key}_LambdaCDM"
            print(f"\nStarting LambdaCDM run for {key}")
            flow.sample_posterior(
                obs["pred"],
                label=lcdm_label,
                n_walkers=n_walkers,
                n_steps=n_steps,
                n_burnin_steps=n_burnin_steps,
                lambdaCDM=True,
            )
