"""
Created June 2023
Author: Arne Thomsen

Utils to load MCMC chains to be plotted.
"""

import os, h5py, yaml, glob
import numpy as np

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def get_abs_dir_repo():
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))

    return repo_dir


def load_msi_config():
    repo_dir = get_abs_dir_repo()
    conf_file = os.path.join(repo_dir, "configs/config.yaml")

    with open(conf_file, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    return conf


def load_network_preds(base_dir, model_dir, n_steps=None, file_label=None, preds_file=None, return_training=False):
    out_dir = os.path.join(base_dir, model_dir)

    # build file name
    if preds_file is None:
        if n_steps is None:
            preds_file = os.path.join(out_dir, f"preds.h5")
        elif file_label is None:
            preds_file = os.path.join(out_dir, f"preds_{n_steps}.h5")
        else:
            preds_file = os.path.join(out_dir, f"preds_{n_steps}_{file_label}.h5")
    else:
        preds_file = os.path.join(out_dir, preds_file)

    h5_keys = [
        "fiducial/vali/pred",
        "fiducial/vali/i_signal",
        "fiducial/vali/i_noise",
        "grid/preds/test",
        "grid/cosmos/test",
        "grid/i_signal/test",
        "grid/i_noise/test",
        "grid/i_sobol/test",
    ]

    if return_training:
        h5_keys.append("fiducial/train/pred")

    LOGGER.info(f"Loading predictions from {preds_file}")
    with h5py.File(preds_file, "r") as f:
        LOGGER.info(f"Array shapes:")

        out_dict = {}
        for h5_key in h5_keys:
            try:
                out_dict[h5_key] = f[h5_key][:]
                LOGGER.info(f"{h5_key:<18} = {out_dict[h5_key].shape}")
            except KeyError:
                LOGGER.warning(f"Could not find {h5_key} in {preds_file}")

        try:
            for h5_key in f["mocks/pred"].keys():
                out_dict[f"mocks/pred/{h5_key}"] = f["mocks/pred"][h5_key][:]
        except KeyError:
            LOGGER.debug(f"Could not find mocks/pred in {preds_file}")

    return out_dict


def load_human_summaries(
    base_dir,
    summary_type,
    file_label=None,
    return_raw_cls=False,
    return_fiducial=True,
    return_grid=True,
    cls_from_maps=False,
):
    LOGGER.timer.start("load_summaries")
    LOGGER.info(f"Loading summaries from {base_dir}")

    assert summary_type in ["peaks", "cls"]

    fidu_file = os.path.join(base_dir, summary_type, f"fiducial_{summary_type}")
    grid_file = os.path.join(base_dir, summary_type, f"grid_{summary_type}")
    if cls_from_maps:
        fidu_file += "_from_maps"
        grid_file += "_from_maps"
    if file_label is not None:
        fidu_file += f"_{file_label}"
        grid_file += f"_{file_label}"
    fidu_file += ".h5"
    grid_file += ".h5"

    out_dict = {}

    # fiducial
    if return_fiducial:
        fidu_keys = ["i_signal", "i_noise"]
        if summary_type == "cls":
            # TODO hacky
            if cls_from_maps:
                fidu_keys += ["cls/binned"]
            else:
                fidu_keys += ["cls/binned", "cls/bin_edges"]
            if return_raw_cls:
                LOGGER.warning(f"Returning the raw Cls, this is potentially slow")
                fidu_keys += ["cls/raw"]
        elif summary_type == "peaks":
            fidu_keys += ["peaks"]

        with h5py.File(fidu_file, "r") as f:
            LOGGER.info(f"Array shapes:")

            for h5_key in fidu_keys:
                dict_key = f"fiducial/{h5_key}"
                out_dict[dict_key] = f[h5_key][:]
                LOGGER.info(f"{dict_key:<18} = {out_dict[dict_key].shape}")

    # grid
    if return_grid:
        grid_keys = ["cosmo", "i_signal", "i_noise", "i_sobol"]
        if summary_type == "cls":
            # TODO hacky
            if cls_from_maps:
                grid_keys += ["cls/binned"]
            else:
                grid_keys += ["cls/binned", "cls/bin_edges"]
            if return_raw_cls:
                grid_keys += ["cls/raw"]
        elif summary_type == "peaks":
            grid_keys += ["peaks"]

        with h5py.File(grid_file, "r") as f:
            for h5_key in grid_keys:
                dict_key = f"grid/{h5_key}"
                out_dict[dict_key] = f[h5_key][:]
                LOGGER.info(f"{dict_key:<18} = {out_dict[dict_key].shape}")

    # TODO hacky
    if cls_from_maps:
        out_dict["grid/cosmo"] = np.repeat(
            out_dict["grid/cosmo"][:, np.newaxis, :], out_dict["grid/cls/binned"].shape[1], axis=1
        )

    LOGGER.info(f"Done loading the summaries after {LOGGER.timer.elapsed('load_summaries')}")
    return out_dict


def load_cl_white_noise(base_dir):
    # TODO
    noise_file = os.path.join(base_dir, "cls/white_noise.h5")
    # noise_file = os.path.join(base_dir, "cls/white_noise_old.h5")
    with h5py.File(noise_file, "r") as f:
        noise_cls = f["cls/binned"][:]

    return noise_cls


def load_network_preds_simple(pred_file):
    LOGGER.info(f"Loading predictions from {pred_file}")

    with h5py.File(pred_file, "r") as f:
        grid_preds = f["grid/preds/test"][:]
        grid_cosmos = f["grid/cosmos/test"][:]

        if grid_preds.ndim == 3:
            grid_preds = np.concatenate(grid_preds, axis=0)
            grid_cosmos = np.concatenate(grid_cosmos, axis=0)

        LOGGER.info(f"grid_preds.shape = {grid_preds.shape}")
        LOGGER.info(f"grid_cosmos.shape = {grid_cosmos.shape}")

        obs_preds = {}
        for key, value in f["obs/preds"].items():
            value = value[:]

            LOGGER.info(f"{key} with shape {value.shape}")
            obs_preds[key] = value

        obs_cosmos = {}
        for key, value in f["obs/cosmos"].items():
            value = value[:]
            obs_cosmos[key] = value

    return grid_preds, grid_cosmos, obs_preds, obs_cosmos
