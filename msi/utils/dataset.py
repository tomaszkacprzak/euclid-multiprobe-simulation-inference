import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from msfm.utils import logger
from msi.utils import preprocessing

LOGGER = logger.get_logger(__file__)


class _PowerSpectrumDataset(Dataset):
    """Apply spectrum augmentation lazily using PyTorch tensors."""

    def __init__(self, signals, labels, noise, transform):
        self.signals = torch.as_tensor(signals)
        self.labels = torch.as_tensor(labels)
        self.noise = None if noise is None else torch.as_tensor(noise)
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        signal, label = self.signals[index].clone(), self.labels[index]
        if self.noise is None:
            return self.transform(signal, label)
        noise = self.noise[torch.randint(len(self.noise), ())]
        return self.transform((signal, label), noise)


def get_binned_power_spectra_dset(
    base_dir,
    # file
    file_label=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    params=None,
    signal_indices=None,
    noise_indices=0.8,
    n_examples_to_plot=10,
    cls_from_maps=False,
    # data loading
    batch_size=2**12,
    shuffle_buffer="full",
    prefetch=3,
    num_parallel_calls=0,
    float_type=np.float32,
    # selection
    probe=None,
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    ggl_only=False,
    with_gaussian_noise=True,
    bin_indices=None,
    # CLs scale cuts
    l_mins=None,
    l_maxs=None,
    theta_fwhms=None,
    white_noise_sigmas=None,
    n_bins=None,
    keep_first_i_bins=None,
    keep_last_i_bins=None,
    # additional preprocessing
    apply_log=True,
    standardize=False,
    ell_weighting=None,  # None | "ell" | "ell_sq" — multiply C_ℓ by ℓ or ℓ² before log
    # scale cut variant: "soft" → Gaussian smoothing; "soft_pruned" → prune noise-dominated bins
    scale_cut=None,
):
    if probe == "lensing":
        with_clustering = False
        with_cross_probe = False
    elif probe == "clustering":
        with_lensing = False
        with_cross_probe = False
    elif probe == "cross":
        with_lensing = False
        with_clustering = False
        with_cross_probe = True
    elif probe == "combined":
        with_lensing = True
        with_clustering = True
        if with_cross_z is None:
            with_cross_z = True
        if with_cross_probe is None:
            with_cross_probe = True

    out_dict = preprocessing.get_binned_power_spectra(
        scale_cut=scale_cut,
        base_dir=base_dir,
        # file
        file_label=file_label,
        # configuration
        msfm_conf=msfm_conf,
        dlss_conf=dlss_conf,
        params=params,
        signal_indices=signal_indices,
        noise_indices=noise_indices,
        n_examples_to_plot=n_examples_to_plot,
        cls_from_maps=cls_from_maps,
        concat_bin_dim=True,
        # selection
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
        with_fiducial=False,
        with_gaussian_noise=with_gaussian_noise,
        bin_indices=bin_indices,
        # Cls scale cuts
        l_mins=l_mins,
        l_maxs=l_maxs,
        theta_fwhms=theta_fwhms,
        white_noise_sigmas=white_noise_sigmas,
        n_bins=n_bins,
        keep_first_i_bins=keep_first_i_bins,
        keep_last_i_bins=keep_last_i_bins,
        # additional preprocessing
        apply_log=apply_log,
        standardize=standardize,
        ell_weighting=ell_weighting,
    )

    for key in out_dict:
        if isinstance(out_dict[key], np.ndarray):
            out_dict[key] = out_dict[key].astype(float_type)

    grid_cls_train = out_dict["grid/cls_raw/train"]
    grid_cls_test = out_dict["grid/cls_raw/test"]
    grid_cosmos_train = out_dict["grid/cosmos/train"]
    grid_cosmos_test = out_dict["grid/cosmos/test"]
    noise_cls = out_dict["noise/cls"]

    ell_weights_torch = torch.as_tensor(out_dict["ell_weights"]) if out_dict.get("ell_weights") is not None else None

    if shuffle_buffer == "full":
        shuffle_buffer = grid_cls_train.shape[0]

    def _augmentations(example, noise):
        signal, label = example

        if with_gaussian_noise:
            signal += noise

        if ell_weights_torch is not None:
            signal = signal * ell_weights_torch

        if apply_log:
            signal = torch.log(torch.abs(signal))

        signal = torch.where(torch.isfinite(signal), signal, torch.zeros_like(signal))

        return signal, label

    train_dataset = _PowerSpectrumDataset(grid_cls_train, grid_cosmos_train, noise_cls, _augmentations)
    test_dataset = _PowerSpectrumDataset(grid_cls_test, grid_cosmos_test, noise_cls, _augmentations)
    workers = 0 if num_parallel_calls is None else int(num_parallel_calls)
    dset_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    dset_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return dset_train, dset_test, out_dict


def get_binned_power_spectra_dset_hard_cut(
    base_dir,
    # file
    file_label=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    params=None,
    signal_indices=None,
    noise_indices=0.8,
    n_examples_to_plot=10,
    cls_from_maps=False,
    # data loading
    batch_size=2**12,
    shuffle_buffer="full",
    prefetch=3,
    num_parallel_calls=0,
    float_type=np.float32,
    # selection
    probe=None,
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    ggl_only=False,
    bin_indices=None,
    # additional preprocessing
    apply_log=True,
    standardize=False,
    ell_weighting=None,  # None | "ell" | "ell_sq"
    n_extra_bins=0,  # 0 → hard cut; 1 → hard_conservative
):
    """Hard scale cut variant of get_binned_power_spectra_dset.

    Instead of Gaussian smoothing + white noise, drops all ℓ bins above
    min(l_max[i], l_max[j]) for each cross-pair.  No noise is added during
    training augmentation.  Returns the same (dset_train, dset_test, out_dict)
    3-tuple so it is interchangeable with get_binned_power_spectra_dset.
    """
    if probe == "lensing":
        with_clustering = False
        with_cross_probe = False
    elif probe == "clustering":
        with_lensing = False
        with_cross_probe = False
    elif probe == "cross":
        with_lensing = False
        with_clustering = False
        with_cross_probe = True
    elif probe == "combined":
        with_lensing = True
        with_clustering = True
        if with_cross_z is None:
            with_cross_z = True
        if with_cross_probe is None:
            with_cross_probe = True

    out_dict = preprocessing.get_binned_power_spectra_hard_cut(
        base_dir=base_dir,
        file_label=file_label,
        msfm_conf=msfm_conf,
        dlss_conf=dlss_conf,
        params=params,
        signal_indices=signal_indices,
        noise_indices=noise_indices,
        n_examples_to_plot=n_examples_to_plot,
        cls_from_maps=cls_from_maps,
        concat_bin_dim=True,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
        with_fiducial=False,
        bin_indices=bin_indices,
        apply_log=apply_log,
        standardize=standardize,
        ell_weighting=ell_weighting,
        n_extra_bins=n_extra_bins,
    )

    for key in out_dict:
        if isinstance(out_dict[key], np.ndarray):
            out_dict[key] = out_dict[key].astype(float_type)

    grid_cls_train = out_dict["grid/cls_raw/train"]
    grid_cls_test = out_dict["grid/cls_raw/test"]
    grid_cosmos_train = out_dict["grid/cosmos/train"]
    grid_cosmos_test = out_dict["grid/cosmos/test"]

    ell_weights_torch = torch.as_tensor(out_dict["ell_weights"]) if out_dict.get("ell_weights") is not None else None

    if shuffle_buffer == "full":
        shuffle_buffer = grid_cls_train.shape[0]

    def _augmentations(signal, label):
        if ell_weights_torch is not None:
            signal = signal * ell_weights_torch

        if apply_log:
            signal = torch.log(torch.abs(signal))

        signal = torch.where(torch.isfinite(signal), signal, torch.zeros_like(signal))

        return signal, label

    train_dataset = _PowerSpectrumDataset(grid_cls_train, grid_cosmos_train, None, _augmentations)
    test_dataset = _PowerSpectrumDataset(grid_cls_test, grid_cosmos_test, None, _augmentations)
    workers = 0 if num_parallel_calls is None else int(num_parallel_calls)
    dset_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    dset_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return dset_train, dset_test, out_dict


def get_binned_power_spectra_dset_for_scale_cut(scale_cut, **kwargs):
    """Unified entry point that dispatches to the appropriate scale-cut variant.

    scale_cut: "hard" | "hard_conservative" | "none" | "soft_pruned" | "soft"
    """
    if scale_cut in ("hard", "none"):
        return get_binned_power_spectra_dset_hard_cut(n_extra_bins=0, **kwargs)
    if scale_cut == "hard_conservative":
        return get_binned_power_spectra_dset_hard_cut(n_extra_bins=1, **kwargs)
    return get_binned_power_spectra_dset(scale_cut=scale_cut, **kwargs)
