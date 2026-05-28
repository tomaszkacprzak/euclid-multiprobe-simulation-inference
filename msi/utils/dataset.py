import tensorflow as tf
import numpy as np

from msfm.utils import logger
from msi.utils import preprocessing, plotting

LOGGER = logger.get_logger(__file__)


def get_binned_power_spectra_dset(
    base_dir,
    # file
    file_label=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    params=None,
    train_test_split=0.8,
    n_examples_to_plot=10,
    cls_from_maps=False,
    # tf.data
    batch_size=2**12,
    shuffle_buffer="full",
    prefetch=3,
    num_parallel_calls=tf.data.AUTOTUNE,
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
        base_dir=base_dir,
        # file
        file_label=file_label,
        # configuration
        msfm_conf=msfm_conf,
        dlss_conf=dlss_conf,
        params=params,
        train_test_split=train_test_split,
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

    ell_weights_tf = (
        tf.constant(out_dict["ell_weights"]) if out_dict.get("ell_weights") is not None else None
    )

    if shuffle_buffer == "full":
        shuffle_buffer = grid_cls_train.shape[0]

    def _augmentations(example, noise):
        signal, label = example

        if with_gaussian_noise:
            signal += noise

        if ell_weights_tf is not None:
            signal = signal * ell_weights_tf

        if apply_log:
            signal = tf.math.log(tf.math.abs(signal))

        signal = tf.where(tf.math.is_finite(signal), signal, tf.zeros_like(signal))

        return signal, label

    # create the datasets
    dset_noise = tf.data.Dataset.from_tensor_slices(noise_cls).cache().shuffle(shuffle_buffer).repeat()

    dset_train = (
        tf.data.Dataset.from_tensor_slices((grid_cls_train, grid_cosmos_train))
        .cache()
        .shuffle(shuffle_buffer)
        .repeat()
    )
    dset_train = (
        tf.data.Dataset.zip((dset_train, dset_noise))
        .batch(batch_size)
        .map(_augmentations, num_parallel_calls=num_parallel_calls, deterministic=False)
        .prefetch(prefetch)
    )

    dset_test = tf.data.Dataset.from_tensor_slices((grid_cls_test, grid_cosmos_test)).cache()
    dset_test = (
        tf.data.Dataset.zip((dset_test, dset_noise))
        .batch(batch_size)
        .map(_augmentations, num_parallel_calls=num_parallel_calls, deterministic=True)
        .prefetch(prefetch)
    )

    return dset_train, dset_test, out_dict


def get_binned_power_spectra_dset_hard_cut(
    base_dir,
    # file
    file_label=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    params=None,
    train_test_split=0.8,
    n_examples_to_plot=10,
    cls_from_maps=False,
    # tf.data
    batch_size=2**12,
    shuffle_buffer="full",
    prefetch=3,
    num_parallel_calls=tf.data.AUTOTUNE,
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
        train_test_split=train_test_split,
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
    )

    for key in out_dict:
        if isinstance(out_dict[key], np.ndarray):
            out_dict[key] = out_dict[key].astype(float_type)

    grid_cls_train = out_dict["grid/cls_raw/train"]
    grid_cls_test = out_dict["grid/cls_raw/test"]
    grid_cosmos_train = out_dict["grid/cosmos/train"]
    grid_cosmos_test = out_dict["grid/cosmos/test"]

    ell_weights_tf = (
        tf.constant(out_dict["ell_weights"]) if out_dict.get("ell_weights") is not None else None
    )

    if shuffle_buffer == "full":
        shuffle_buffer = grid_cls_train.shape[0]

    def _augmentations(signal, label):
        if ell_weights_tf is not None:
            signal = signal * ell_weights_tf

        if apply_log:
            signal = tf.math.log(tf.math.abs(signal))

        signal = tf.where(tf.math.is_finite(signal), signal, tf.zeros_like(signal))

        return signal, label

    dset_train = (
        tf.data.Dataset.from_tensor_slices((grid_cls_train, grid_cosmos_train))
        .cache()
        .shuffle(shuffle_buffer)
        .repeat()
        .batch(batch_size)
        .map(_augmentations, num_parallel_calls=num_parallel_calls, deterministic=False)
        .prefetch(prefetch)
    )

    dset_test = (
        tf.data.Dataset.from_tensor_slices((grid_cls_test, grid_cosmos_test))
        .cache()
        .batch(batch_size)
        .map(_augmentations, num_parallel_calls=num_parallel_calls, deterministic=True)
        .prefetch(prefetch)
    )

    return dset_train, dset_test, out_dict
