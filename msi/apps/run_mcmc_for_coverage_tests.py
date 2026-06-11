# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Sample the MCMC posterior for N random observations from the CosmoGrid for posterior-level coverage testing. This is
meant for CPU nodes.

For a flow trained on combined maps+Cls summaries (run_inference.py --out_dir_2, see
y3-deep-lss/submissions/clariden/combined_inference.sh), also pass --preds_file_2 pointing at the
second model's preds_*.h5, so the held-out validation split can be faithfully reproduced.

example usage:

esub run_mcmc_for_coverage_tests.py \
    --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 \
    --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_test_v4/likelihood_flow \
    --n_sims=1000 --n_jobs=1000 \
    --mode=jobarray --function=all --keep_submit_files \
    --jobname=mcmc_test --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


esub run_mcmc_for_coverage_tests.py \
    --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 \
    --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_test_v4/likelihood_flow \
    --n_sims=1000 --n_jobs=1000 \
    --mode=jobarray --function=rerun_missing --keep_submit_files \
    --jobname=mcmc_test --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

"""

import numpy as np
import torch, os, argparse, warnings, h5py, time

from msfm.utils import logger

from msi.utils import flow as flow_utils
from msi.flow_conductor.likelihood_flow import LikelihoodFlow


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def get_tasks(args):
    args = setup(args)

    return list(range(args.n_sims))


def resources(args):
    args = setup(args)

    if args.cluster == "perlmutter":
        # because of hyperthreading, there's a total of 256 threads per node
        resources = {
            "main_time_per_index": 0.4,
            "main_n_cores": 2,
            "main_memory": 1952,
            "main_scratch": 0,
            "merge_time": 0.5,
            "merge_n_cores": 8,
            "merge_memory": 1952,
            "merge_scratch": 0,
        }
    elif args.cluster == "euler":
        resources = {"main_time": 4, "main_memory": 4096, "main_n_cores": 8, "merge_memory": 4096, "merge_n_cores": 16}

    return resources


def setup(args):
    description = "evaluate the power spectra from the input pipelines"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "--preds_file",
        type=str,
        required=True,
        help="directory containing the predictions of the compression network",
    )
    parser.add_argument(
        "--preds_file_2",
        type=str,
        default=None,
        help="optional second model's predictions file; its summary is concatenated feature-wise "
        "with the primary one's, e.g. to reproduce a flow trained on combined maps+Cls summaries "
        "via run_inference.py's --out_dir_2. Must match what the flow at --flow_dir was trained on.",
    )
    parser.add_argument(
        "--flow_dir",
        type=str,
        required=True,
        help="directory containing the flow network",
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="perlmutter",
        choices=("perlmutter", "euler"),
        help="the cluster to execute on",
    )
    parser.add_argument(
        "--max_sleep",
        type=int,
        default=60,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
    )
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    return args


def main(indices, args):
    args = setup(args)

    n_sims = args.n_sims
    n_walkers = 1024
    n_burnin_steps = 1_000
    n_steps = 1_000
    n_samples_out = 10_000

    if args.debug:
        args.max_sleep = 0
        n_burnin_steps = 10
        n_steps = n_walkers * 100
        LOGGER.warning("!!! debug mode !!!")
    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    model, x_true_all, theta_true_all = _set_up_flow(args)

    # deterministically subselect cosmologies
    n_cosmos = x_true_all.shape[0]
    x_true_sub = x_true_all[:: n_cosmos // n_sims]
    theta_true_sub = theta_true_all[:: n_cosmos // n_sims]

    for index in indices:
        x_true = x_true_sub[index]
        theta_true = theta_true_sub[index]

        theta_sample = model.sample_posterior(
            x_true,
            n_walkers=n_walkers,
            n_burnin_steps=n_burnin_steps,
            n_steps=n_steps,
            dont_save=True,
        )
        # too many samples make the test slow and are not needed
        theta_sample = theta_sample[np.random.choice(theta_sample.shape[0], n_samples_out, replace=False)]

        log_prob_true = model.log_likelihood(
            torch.unsqueeze(x_true, 0), torch.unsqueeze(theta_true, 0), return_numpy=True
        )
        log_prob_sample = model.log_likelihood(
            torch.repeat_interleave(torch.unsqueeze(x_true, 0), repeats=theta_sample.shape[0], dim=0),
            theta_sample,
            return_numpy=True,
        )

        out_file = os.path.join(args.flow_dir, f"mcmc_samples_{index}.h5")
        with h5py.File(out_file, "w") as f:
            f.create_dataset("x_true", data=x_true)
            f.create_dataset("theta_true", data=theta_true)
            f.create_dataset("log_prob_true", data=log_prob_true)

            f.create_dataset("theta_sample", data=theta_sample)
            f.create_dataset("log_prob_sample", data=log_prob_sample)

        yield index


def _set_up_flow(args):
    """Restore the trained flow and reproduce its held-out validation split for coverage testing.

    LikelihoodFlow._prepare_data's train/validation split is made deterministic and grouped by
    signal realization (group_ids=i_signal): the unique signal ids are sorted and partitioned into
    a train and a validation fraction, so no signal realization -- regardless of its noise
    realizations -- appears in both sets. The split therefore depends only on the content and row
    order of (x, theta, i_signal), not on any random seed, so grid_preds/grid_cosmos/i_signal must
    be (re)built exactly as run_inference.py built them at training time -- hence reusing
    flow_utils.load_grid_summaries. This guarantees the mock observations drawn from the
    reconstructed validation set were seen neither by the compression network nor by the flow.
    """
    grid_preds, grid_cosmos, _, _, i_signal = flow_utils.load_grid_summaries(args.preds_file, args.preds_file_2)

    model = LikelihoodFlow.from_checkpoint(model_dir=args.flow_dir, device="cpu")

    # get the correct signal-grouped split of the validation data
    model._prepare_data(
        x=grid_preds,
        theta=grid_cosmos,
        batch_size=10000,
        vali_split=0.1,
        group_ids=i_signal,
    )

    x_vali = model.vali_dset.dataset.tensors[0][model.vali_dset.indices]
    theta_vali = model.vali_dset.dataset.tensors[1][model.vali_dset.indices]
    LOGGER.info(f"Reconstructed {len(x_vali)} held-out validation examples for coverage testing")

    return model, x_vali, theta_vali


def merge(indices, args):
    args = setup(args)
    n_sims = args.n_sims

    out_file = os.path.join(args.flow_dir, f"mcmc_samples.h5")
    with h5py.File(out_file, "w") as f_merged:
        for index in LOGGER.progressbar(indices, desc="merging files", at_level="info"):
            try:
                in_file = os.path.join(args.flow_dir, f"mcmc_samples_{index}.h5")
                with h5py.File(in_file, "r") as f_in:
                    x_true = f_in["x_true"][:]
                    theta_true = f_in["theta_true"][:]
                    log_prob_true = f_in["log_prob_true"][:]

                    theta_sample = f_in["theta_sample"][:]
                    log_prob_sample = f_in["log_prob_sample"][:]

                if index == indices[0]:
                    f_merged.create_dataset("x_true", shape=(n_sims, x_true.shape[0]), dtype=np.float32)
                    f_merged.create_dataset("theta_true", shape=(n_sims, theta_true.shape[0]), dtype=np.float32)
                    f_merged.create_dataset("log_prob_true", shape=(n_sims), dtype=np.float32)

                    # shape as expected by TARP package
                    f_merged.create_dataset(
                        "theta_sample",
                        shape=(theta_sample.shape[0], n_sims, theta_sample.shape[1]),
                        dtype=np.float32,
                    )
                    f_merged.create_dataset("log_prob_sample", shape=(theta_sample.shape[0], n_sims), dtype=np.float32)

                f_merged["x_true"][index] = x_true
                f_merged["theta_true"][index] = theta_true
                f_merged["log_prob_true"][index] = log_prob_true

                # breakpoint()

                f_merged["theta_sample"][:, index] = theta_sample
                f_merged["log_prob_sample"][:, index] = log_prob_sample
            except (FileNotFoundError, TypeError):
                pass
    LOGGER.info(f"Merged all files into {out_file}")

    # only remove the files after the above loop has terminated successfully
    for index in indices:
        in_file = os.path.join(args.flow_dir, f"mcmc_samples_{index}.h5")
        if os.path.exists(in_file):
            os.remove(in_file)
    LOGGER.info(f"Removed temporary files")
