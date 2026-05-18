# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2023
Author: Arne Thomsen

Utils to preprocess the raw network predictions and human defined summary statistics for density estimation. This for
example entails concatenating the example and cosmology axes.
"""

import os, re
import numpy as np

from scipy.stats import binned_statistic
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from msfm.utils import logger, cross_statistics, parameters, files, power_spectra, observation, scales
from deep_lss.utils import configuration
from msi.utils.sklearn import GeneralizedSklearnModel
from msi.utils import plotting, input_output

LOGGER = logger.get_logger(__file__)


def get_reshaped_network_preds(
    model_dir,
    base_dir="",
    n_steps=None,
    file_label=None,
    preds_file=None,
    n_params=None,
    n_perms_per_cosmo=None,
    with_fidu=True,
    with_grid=True,
):
    file_dict = input_output.load_network_preds(
        base_dir, model_dir, n_steps=n_steps, file_label=file_label, preds_file=preds_file
    )

    LOGGER.info(f"file_dict.keys = {file_dict.keys()}")

    print("\n")
    LOGGER.info(f"Shapes after concatenation and selection:")

    if with_fidu:
        fidu_preds = file_dict["fiducial/vali/pred"]

        # only relevant for the likelihood loss
        fidu_preds = fidu_preds[..., :n_params]

        LOGGER.info(f"fidu_preds  = {fidu_preds.shape}")
    else:
        fidu_preds = None

    if with_grid:
        grid_preds = file_dict["grid/preds/test"]
        grid_cosmos = file_dict["grid/cosmos/test"]

        # only relevant for the likelihood loss
        grid_preds = grid_preds[..., :n_params]

        # only take a subset of the permutations
        if n_perms_per_cosmo is not None:
            LOGGER.warning(f"Only taking the first {n_perms_per_cosmo} permutations per cosmology")
            LOGGER.warning(f"n_patches and n_noise are hard-coded here!")
            n_patches = 4
            n_noise = 3
            grid_preds = grid_preds[:, : (n_perms_per_cosmo * n_patches * n_noise), :]

        # combine the example and cosmology axes
        if grid_preds.ndim == 3:
            grid_preds = np.concatenate(grid_preds, axis=0)
            grid_cosmos = np.concatenate(grid_cosmos, axis=0)

        LOGGER.info(f"grid_preds  = {grid_preds.shape}")
        LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")
    else:
        grid_preds = None
        grid_cosmos = None

    return fidu_preds, grid_preds, grid_cosmos, file_dict


def get_reshaped_human_summaries(
    base_dir,
    summary_type,
    # file
    file_label=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    params=None,
    concat_example_dim=True,
    concat_bin_dim=True,
    do_plot=True,
    # selection
    with_fiducial=True,
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    ggl_only=False,
    with_grid=True,
    # power spectra specific
    bin_indices=None,
    from_raw_cls=False,
    l_mins=None,
    l_maxs=None,
    theta_fwhms=None,
    white_noise_sigmas=None,
    n_bins=None,
    keep_first_i_bins=None,
    keep_last_i_bins=None,
    fixed_binning=False,
    cls_from_maps=False,
    # peaks specific
    scale_indices=None,
    # additional preprocessing
    apply_log=False,
    standardize=False,
    pca_components=None,
):
    assert summary_type in ["cls", "peaks"], "Only cls and peaks are supported"
    assert with_fiducial or with_grid, "At least one of with_fiducial and with_grid must be True"

    msfm_conf = files.load_config(msfm_conf)

    store_lensing = msfm_conf.get("analysis", {}).get("modelling", {}).get("lensing", {}).get("store", True)
    store_clustering = msfm_conf.get("analysis", {}).get("modelling", {}).get("clustering", {}).get("store", True)
    n_z_lensing_total = len(msfm_conf["survey"]["metacal"]["z_bins"])
    n_z_clustering_total = len(msfm_conf["survey"]["maglim"]["z_bins"])
    n_z_lensing_active = n_z_lensing_total if store_lensing else 0
    n_z_clustering_active = n_z_clustering_total if store_clustering else 0

    if scales_from_conf := dlss_conf is not None:
        dlss_conf = configuration.load_deep_lss_config(dlss_conf)

    noise_cls = None

    if summary_type == "cls":
        if theta_fwhms is None and scales_from_conf:
            theta_fwhms = []
            if store_lensing:
                theta_fwhms.extend(dlss_conf["scale_cuts"]["lensing"]["theta_fwhm"])
            if store_clustering:
                theta_fwhms.extend(dlss_conf["scale_cuts"]["clustering"]["theta_fwhm"])
            LOGGER.info(f"Using theta_fwhm = {theta_fwhms} from the dlss config")
        if white_noise_sigmas is None and scales_from_conf:
            white_noise_sigmas = []
            if store_lensing:
                white_noise_sigmas.extend(dlss_conf["scale_cuts"]["lensing"]["white_noise_sigma"])
            if store_clustering:
                white_noise_sigmas.extend(dlss_conf["scale_cuts"]["clustering"]["white_noise_sigma"])
            LOGGER.info(f"Using white_noise_sigma = {white_noise_sigmas} from the dlss config")
        # this l_max here and the theta_fwhm are fully equivalent. This is not to be confused with the l_max resulting
        # from the white noise
        if l_maxs is None and scales_from_conf:
            theta_fwhms = []
            if store_lensing:
                theta_fwhms.extend(dlss_conf["scale_cuts"]["lensing"]["theta_fwhm"])
            if store_clustering:
                theta_fwhms.extend(dlss_conf["scale_cuts"]["clustering"]["theta_fwhm"])
            l_maxs = scales.angle_to_ell(np.array(theta_fwhms), arcmin=dlss_conf["scale_cuts"]["arcmin"])
            LOGGER.info(f"Using l_maxs = {l_maxs} from the dlss config")
        if l_mins is None and scales_from_conf:
            l_mins = np.zeros_like(l_maxs)
            LOGGER.info(f"Using l_mins = {l_mins} by default (no smoothing)")
        if n_bins is None and scales_from_conf:
            n_bins = msfm_conf["analysis"]["power_spectra"]["n_bins"]
            LOGGER.info(f"Using n_bins = {n_bins} from the msfm config")

        # apply scale cuts to the raw Cls
        if from_raw_cls:
            LOGGER.warning(f"Applying scale cuts to the raw Cls, this is deprecated")

            assert (
                (l_mins is not None) and (l_maxs is not None) and (n_bins is not None)
            ), "The l_mins, l_maxs, and n_bins arguments must be provided"

            with np.printoptions(precision=1, suppress=True, floatmode="fixed"):
                LOGGER.info(f"l_mins = {l_mins}")
                LOGGER.info(f"l_maxs = {l_maxs}")

                bin_names = f"l_mins={np.array(l_mins)},l_maxs={np.array(l_maxs)},n_bins={n_bins},fixed_binning={fixed_binning}"
                bin_names = re.sub(r"\s+", ",", bin_names)

                fidu_file = os.path.join(base_dir, summary_type, f"fiducial_{bin_names}.npy")
                grid_file = os.path.join(base_dir, summary_type, f"grid_{bin_names}.npy")

            try:
                fidu_summs = np.load(fidu_file)
                grid_summs = np.load(grid_file)
                file_dict = input_output.load_human_summaries(
                    base_dir,
                    summary_type,
                    file_label=file_label,
                    return_raw_cls=False,
                    return_fiducial=with_fiducial,
                    return_grid=with_grid,
                )
                LOGGER.info(f"Loaded the binned Cls from {bin_names}")
            except FileNotFoundError:
                file_dict = input_output.load_human_summaries(
                    base_dir,
                    summary_type,
                    file_label=file_label,
                    return_raw_cls=True,
                    return_fiducial=with_fiducial,
                    return_grid=with_grid,
                )
                LOGGER.warning(f"Applying the scale cuts to the raw Cls, this takes a while and consumes a lot of RAM")
                LOGGER.timer.start("scale_cuts")
                fidu_summs, _ = power_spectra.smooth_and_bin_cls(
                    file_dict["fiducial/cls/raw"],
                    l_mins,
                    l_maxs,
                    n_bins,
                    n_side=msfm_conf["analysis"]["n_side"],
                    fixed_binning=fixed_binning,
                )
                grid_summs, _ = power_spectra.smooth_and_bin_cls(
                    file_dict["grid/cls/raw"],
                    l_mins,
                    l_maxs,
                    n_bins,
                    n_side=msfm_conf["analysis"]["n_side"],
                    fixed_binning=fixed_binning,
                )
                LOGGER.info(f"Done after {LOGGER.timer.elapsed('scale_cuts')}")

                np.save(fidu_file, fidu_summs)
                np.save(grid_file, grid_summs)
                LOGGER.info(f"Saved the binned Cls to {bin_names}")

        # load the pre-binned Cls
        else:
            LOGGER.info(f"Loading the pre-binned Cls")
            file_dict = input_output.load_human_summaries(
                base_dir,
                summary_type,
                file_label=file_label,
                cls_from_maps=cls_from_maps,
                return_raw_cls=False,
                return_fiducial=with_fiducial,
                return_grid=with_grid,
            )
            fidu_summs = file_dict[f"fiducial/cls/binned"] if with_fiducial else None
            grid_summs = file_dict[f"grid/cls/binned"]

            LOGGER.info(f"Applying scale cuts to the pre-binned Cls")

            # binning
            ells = np.arange(0, 3 * msfm_conf["analysis"]["n_side"])
            bins = power_spectra.get_cl_bins(
                msfm_conf["analysis"]["power_spectra"]["l_min"],
                msfm_conf["analysis"]["power_spectra"]["l_max"],
                msfm_conf["analysis"]["power_spectra"]["n_bins"],
            )

            # white noise
            try:
                noise_cls = input_output.load_cl_white_noise(base_dir)
                with_noise = True

                n_total_expected = (
                    (n_z_lensing_total + n_z_clustering_total) * (n_z_lensing_total + n_z_clustering_total + 1) // 2
                )
                if (
                    noise_cls.shape[-1] == n_total_expected
                    and n_z_lensing_total + n_z_clustering_total != n_z_lensing_active + n_z_clustering_active
                ):
                    total_indices, _ = cross_statistics.get_cross_bin_indices(
                        n_z_lensing=n_z_lensing_total,
                        n_z_clustering=n_z_clustering_total,
                        with_lensing=store_lensing,
                        with_clustering=store_clustering,
                        with_cross_z=True,
                        with_cross_probe=(store_lensing and store_clustering),
                    )
                    noise_cls = noise_cls[..., total_indices]
                    LOGGER.info(f"Subsampled white noise from {n_total_expected} to {len(total_indices)} bins")
            except FileNotFoundError:
                with_noise = False
                LOGGER.warning(f"No white noise Cls found, continuing without")

            # apply the smoothing to all (cross) bins, the selection only happens later
            n_z = n_z_lensing_active + n_z_clustering_active

            k = 0
            for i in range(n_z):
                for j in range(n_z):
                    if (i == j) or (i < j):
                        if l_mins[i] is not None and l_mins[j] is not None:
                            l_min = max(l_mins[i], l_mins[j])
                        else:
                            raise ValueError("l_mins must be provided")

                        if l_maxs[i] is not None and l_maxs[j] is not None:
                            l_max = min(l_maxs[i], l_maxs[j])
                            theta_fwhms = None
                        elif theta_fwhms[i] is not None and theta_fwhms[j] is not None:
                            l_max = None
                            theta_fwhms = max(theta_fwhms[i], theta_fwhms[j])
                        else:
                            raise ValueError("l_maxs or theta_fwhms must be provided")

                        smoothing_fac = scales.gaussian_high_pass_factor_alm(ells, l_min=l_min)
                        smoothing_fac *= scales.gaussian_low_pass_factor_alm(ells, l_max=l_max, theta_fwhm=theta_fwhms)
                        smoothing_fac = smoothing_fac**2
                        smoothing_fac = binned_statistic(ells, smoothing_fac, statistic="mean", bins=bins)[0]

                        if with_fiducial:
                            fidu_summs[..., k] *= smoothing_fac
                        grid_summs[..., k] *= smoothing_fac
                        if with_noise:
                            noise_cls[..., k] *= white_noise_sigmas[i] * white_noise_sigmas[j]

                        k += 1

    elif summary_type == "peaks":
        LOGGER.info(f"Loading the pre-binned peak statistics")

        LOGGER.warning(
            f"The scale cuts are baked into the peak statistics, ignoring the l_mins, l_maxs, and n_bins arguments"
        )

        file_dict = input_output.load_human_summaries(
            base_dir,
            summary_type,
            file_label=file_label,
            return_raw_cls=False,
            return_fiducial=with_fiducial,
            return_grid=with_grid,
        )
        fidu_summs = file_dict[f"fiducial/{summary_type}"] if with_fiducial else None
        grid_summs = file_dict[f"grid/{summary_type}"]

    else:
        raise ValueError

    grid_cosmos = file_dict["grid/cosmo"]
    grid_i_sobols = file_dict["grid/i_sobol"]

    if bin_indices is None:
        bin_indices, bin_names = cross_statistics.get_cross_bin_indices(
            n_z_lensing=n_z_lensing_active,
            n_z_clustering=n_z_clustering_active,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=with_cross_z,
            with_cross_probe=with_cross_probe,
            ggl_only=ggl_only,
        )
        LOGGER.info(f"Using the bin names {bin_names}")

    # select the right auto and cross bins)
    assert isinstance(bin_indices, (list, np.ndarray)), "bin_indices must be a list or numpy array"
    LOGGER.info(f"Using the bin indices {bin_indices}")
    if with_fiducial:
        fidu_summs = fidu_summs[..., bin_indices]
    grid_summs = grid_summs[..., bin_indices]
    if with_noise:
        noise_cls = noise_cls[..., bin_indices]

    if keep_first_i_bins is not None:
        LOGGER.warning(f"Keeping only the first {keep_first_i_bins} bins")
        if with_fiducial:
            fidu_summs = fidu_summs[..., :keep_first_i_bins, :]
        grid_summs = grid_summs[..., :keep_first_i_bins, :]
        if with_noise:
            noise_cls = noise_cls[..., :keep_first_i_bins, :]
    if keep_last_i_bins is not None:
        LOGGER.warning(f"Keeping only the last {keep_last_i_bins} bins")
        if with_fiducial:
            fidu_summs = fidu_summs[..., -keep_last_i_bins:, :]
        grid_summs = grid_summs[..., -keep_last_i_bins:, :]
        if with_noise:
            noise_cls = noise_cls[..., -keep_last_i_bins:, :]

    # select the right cosmological parameters
    msfm_conf = files.load_config(msfm_conf)

    store_lensing = msfm_conf.get("analysis", {}).get("modelling", {}).get("lensing", {}).get("store", True)
    store_clustering = msfm_conf.get("analysis", {}).get("modelling", {}).get("clustering", {}).get("store", True)
    n_z_lensing_total = len(msfm_conf["survey"]["metacal"]["z_bins"])
    n_z_clustering_total = len(msfm_conf["survey"]["maglim"]["z_bins"])
    n_z_lensing_active = n_z_lensing_total if store_lensing else 0
    n_z_clustering_active = n_z_clustering_total if store_clustering else 0
    all_params = parameters.get_parameters(None, msfm_conf)
    params = parameters.get_parameters(params, msfm_conf)

    param_indices = []
    for i, param in enumerate(all_params):
        if param in params:
            param_indices.append(i)
    grid_cosmos = grid_cosmos[..., param_indices]

    print("\n")
    LOGGER.info(f"Shapes after probe selection")
    if with_fiducial:
        LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
    LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
    LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")
    LOGGER.info(f"grid_i_sobols = {grid_i_sobols.shape}")

    # concatenate the bins along the last axis
    if concat_bin_dim:
        if with_fiducial:
            fidu_summs = np.concatenate([fidu_summs[..., i] for i in range(fidu_summs.shape[-1])], axis=-1)
        grid_summs = np.concatenate([grid_summs[..., i] for i in range(grid_summs.shape[-1])], axis=-1)
        if noise_cls is not None:
            noise_cls = np.concatenate([noise_cls[..., i] for i in range(noise_cls.shape[-1])], axis=-1)

    # TODO implement scale selection
    if summary_type == "peaks":
        if scale_indices is None:
            assert (
                not with_fiducial or fidu_summs.shape[-2] == grid_summs.shape[-2]
            ), "The number of scales must be the same for fiducial and grid"
            scale_indices = range(fidu_summs.shape[-2]) if with_fiducial else range(grid_summs.shape[-2])

        # concatenate the scales along the last axis
        if with_fiducial:
            fidu_summs = np.concatenate([fidu_summs[..., i, :] for i in scale_indices], axis=-1)
        grid_summs = np.concatenate([grid_summs[..., i, :] for i in scale_indices], axis=-1)

        print("\n")
        LOGGER.info("Shapes after scale selection")
        if with_fiducial:
            LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
        LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")

    # concatenate the examples along the first axis
    if concat_example_dim and summary_type:
        # TODO this is due to how it's stored in the .h5 files and not super clean
        if summary_type == "cls":
            grid_cosmos = np.concatenate([grid_cosmos[i, ...] for i in range(grid_cosmos.shape[0])], axis=0)
            grid_i_sobols = np.concatenate([grid_i_sobols[i, ...] for i in range(grid_i_sobols.shape[0])], axis=0)
        elif summary_type == "peaks":
            grid_cosmos = np.repeat(grid_cosmos, repeats=grid_summs.shape[1], axis=0)

        grid_summs = np.concatenate([grid_summs[i, ...] for i in range(grid_summs.shape[0])], axis=0)

        print("\n")
        LOGGER.info("Shapes after concatenation")
        if with_fiducial:
            LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
        LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
        LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")
        LOGGER.info(f"grid_i_sobols = {grid_i_sobols.shape}")

    if do_plot:
        assert concat_example_dim, f"Plotting only works if the examples are concatenated"

        LOGGER.info(f"Plotting the selected raw {summary_type}")
        label = f"lensing={with_lensing},clustering={with_clustering},cross_z={with_cross_z},cross_probe={with_cross_probe}"

        if summary_type == "cls":
            plotting.plot_human_summary(
                fidu_summs,
                grid_summs,
                os.path.join(base_dir, summary_type),
                label=label,
                bin_size=msfm_conf["analysis"]["power_spectra"]["n_bins"] - 1,
                bin_names=bin_names,
            )
        elif summary_type == "peaks":
            plotting.plot_human_summary(
                fidu_summs,
                grid_summs,
                os.path.join(base_dir, summary_type),
                label=label,
                bin_size=msfm_conf["analysis"]["peak_statistics"]["n_bins"],
                bin_names=bin_names,
            )

    grid_summs, scaler, pca = preprocess_human_summaries(
        grid_summs, apply_log, standardize=standardize, pca_components=pca_components
    )
    fidu_summs, _, _ = preprocess_human_summaries(
        fidu_summs, apply_log, standardize=standardize, pca_components=pca_components, scaler=scaler, pca=pca
    )
    if noise_cls is not None:
        noise_cls, _, _ = preprocess_human_summaries(
            noise_cls, apply_log, standardize=standardize, pca_components=pca_components, scaler=scaler, pca=pca
        )

    print("\n")
    LOGGER.info("Shapes after pre-processing")
    if with_fiducial:
        LOGGER.info(f"fidu_{summary_type} = {fidu_summs.shape}")
    LOGGER.info(f"grid_{summary_type} = {grid_summs.shape}")
    LOGGER.info(f"grid_cosmos = {grid_cosmos.shape}")

    return fidu_summs, grid_summs, noise_cls, grid_cosmos, grid_i_sobols, file_dict, scaler, pca


def preprocess_human_summaries(
    summaries, apply_log=False, standardize=False, pca_components=None, scaler=None, pca=None
):
    if apply_log:
        LOGGER.info(f"Taking the logarithm of the absolute values.")
        summaries = np.log(np.abs(summaries))

    if standardize and scaler is None:
        LOGGER.info(f"Fitting the scaler to transform to zero mean and unit variance")
        scaler = GeneralizedSklearnModel(StandardScaler())
        summaries = scaler.fit_transform(summaries)
    elif isinstance(scaler, GeneralizedSklearnModel):
        LOGGER.info(f"Applying the scaler to transform to zero mean and unit variance")
        summaries = scaler.transform(summaries)

    if pca_components is not None and pca is None:
        LOGGER.info(f"Fitting PCA to compress to {pca_components} components")
        pca = GeneralizedSklearnModel(PCA(n_components=pca_components, whiten=False))
        summaries = np.nan_to_num(summaries)
        summaries = pca.fit_transform(summaries)
        LOGGER.info(f"Total explained variance = {np.sum(pca.model.explained_variance_ratio_)}")
    elif isinstance(pca, GeneralizedSklearnModel):
        LOGGER.info(f"Applying PCA to compress to {pca_components} components")
        summaries = np.nan_to_num(summaries)
        summaries = pca.transform(summaries)

    summaries = np.nan_to_num(summaries)

    return summaries, scaler, pca


def get_binned_power_spectra(
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
    concat_bin_dim=True,
    # selection
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    ggl_only=False,
    with_gaussian_noise=True,
    with_fiducial=True,
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
):
    """like msi.utils.dataset.get_binned_power_spectra_dset, but without the TensorFlow dependency and dset"""

    if concat_bin_dim == False:
        LOGGER.warning("concat_example_dim = False should not be used when training networks, it's just for plotting")

    fidu_cls, grid_cls, noise_cls, grid_cosmos, grid_i_sobols, file_dict, scaler, pca = get_reshaped_human_summaries(
        base_dir,
        "cls",
        file_label=file_label,
        # configuration
        msfm_conf=msfm_conf,
        dlss_conf=dlss_conf,
        params=params,
        concat_example_dim=False,
        concat_bin_dim=concat_bin_dim,
        do_plot=False,
        # selection
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
        with_fiducial=with_fiducial,
        bin_indices=bin_indices,
        # power spectra: scales
        from_raw_cls=False,
        l_mins=l_mins,
        l_maxs=l_maxs,
        theta_fwhms=theta_fwhms,
        white_noise_sigmas=white_noise_sigmas,
        n_bins=n_bins,
        cls_from_maps=cls_from_maps,
        keep_first_i_bins=keep_first_i_bins,
        keep_last_i_bins=keep_last_i_bins,
        # unlike the standardization, the logarithm is not linear and has to be applied as log(signal + noise), not
        # log(signal) + log(noise)
        apply_log=False,
        standardize=standardize,
    )

    i_sort = np.argsort(grid_i_sobols, axis=0)
    i_sort = i_sort[:, 0]
    grid_cls = grid_cls[i_sort]
    grid_cosmos = grid_cosmos[i_sort]

    # split along the "examples per cosmo" axis
    i_split = int(train_test_split * grid_cls.shape[1])

    grid_cls_train = grid_cls[:, :i_split, :]
    grid_cls_test = grid_cls[:, i_split:, :]
    grid_cosmos_train = grid_cosmos[:, :i_split, :]
    grid_cosmos_test = grid_cosmos[:, i_split:, :]

    _concat_example_axis = lambda array: np.concatenate([array[i, ...] for i in range(array.shape[0])], axis=0)

    grid_cls_train = _concat_example_axis(grid_cls_train)
    grid_cls_test = _concat_example_axis(grid_cls_test)
    grid_cosmos_train = _concat_example_axis(grid_cosmos_train)
    grid_cosmos_test = _concat_example_axis(grid_cosmos_test)

    rng = np.random.default_rng()

    def _noise_and_log(cls):
        if with_gaussian_noise:
            cls += noise_cls[rng.integers(low=0, high=noise_cls.shape[0], size=cls.shape[0])]

        cls = preprocess_human_summaries(cls, apply_log=apply_log)[0]

        return cls

    out_dict = {
        "grid/cls_raw/train": grid_cls_train.copy(),
        "grid/cls_raw/test": grid_cls_test.copy(),
    }
    if with_fiducial:
        out_dict["fidu/cls_raw"] = fidu_cls.copy()

    if with_fiducial:
        fidu_cls = _noise_and_log(fidu_cls.copy())
    grid_cls_train = _noise_and_log(grid_cls_train.copy())
    grid_cls_test = _noise_and_log(grid_cls_test.copy())

    plotting.plot_human_summary(
        fidu_cls if with_fiducial else grid_cls_train,
        grid_cls_train,
        bin_size=msfm_conf["analysis"]["power_spectra"]["n_bins"] - 1,
        n_random_indices=n_examples_to_plot,
        yscale="linear",
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
    )

    out_dict.update(
        {
            "fidu/cls": fidu_cls if with_fiducial else None,
            "grid/cls/train": grid_cls_train,
            "grid/cls/test": grid_cls_test,
            "grid/cosmos/train": grid_cosmos_train,
            "grid/cosmos/test": grid_cosmos_test,
            "noise/cls": noise_cls,
            "grid/i_sobols": grid_i_sobols,
        }
    )

    return out_dict


def get_preprocessed_cl_observation(
    wl_gamma_map=None,
    gc_count_map=None,
    obs_cl=None,
    # configuration
    msfm_conf=None,
    dlss_conf=None,
    base_dir=None,
    from_raw_cls=False,
    nest_in=False,
    apply_maglim_sys_map=True,
    # selection
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    ggl_only=False,
    # CLs scale cuts
    l_mins=None,
    l_maxs=None,
    n_bins=None,
    only_keep_bins=None,
    bin_indices=None,
    # additional preprocessing
    apply_log=False,
    standardize=False,
    pca_components=None,
    scaler=None,
    pca=None,
    # plotting
    make_plot=True,
    obs_label=None,
):
    """To forward model a mock observation like the Buzzards"""

    assert (
        (obs_cl is not None) or (wl_gamma_map is not None) or (gc_count_map is not None)
    ), "Either obs_cl or wl_gamma_map or gc_count_map must be provided"

    msfm_conf = files.load_config(msfm_conf)

    store_lensing = msfm_conf.get("analysis", {}).get("modelling", {}).get("lensing", {}).get("store", True)
    store_clustering = msfm_conf.get("analysis", {}).get("modelling", {}).get("clustering", {}).get("store", True)
    n_z_lensing_total = len(msfm_conf["survey"]["metacal"]["z_bins"])
    n_z_clustering_total = len(msfm_conf["survey"]["maglim"]["z_bins"])
    n_z_lensing_active = n_z_lensing_total if store_lensing else 0
    n_z_clustering_active = n_z_clustering_total if store_clustering else 0
    dlss_conf = configuration.load_deep_lss_config(dlss_conf)

    if l_maxs is None:
        theta_fwhm = []
        if store_lensing:
            theta_fwhm.extend(dlss_conf["scale_cuts"]["lensing"]["theta_fwhm"])
        if store_clustering:
            theta_fwhm.extend(dlss_conf["scale_cuts"]["clustering"]["theta_fwhm"])
        l_maxs = scales.angle_to_ell(np.array(theta_fwhm), arcmin=dlss_conf["scale_cuts"]["arcmin"])
        LOGGER.info(f"Using l_maxs = {l_maxs} from the dlss config")
    if l_mins is None:
        l_mins = np.zeros_like(l_maxs, dtype=int)
        LOGGER.info(f"Using l_mins = {l_mins} by default (no smoothing)")

    if obs_cl is None:
        _, obs_cl, _ = observation.forward_model_observation_map(
            wl_gamma_map=wl_gamma_map if store_lensing else None,
            gc_count_map=gc_count_map if store_clustering else None,
            conf=msfm_conf,
            apply_norm=False,
            with_padding=True,
            nest_in=nest_in,
            apply_maglim_sys_map=apply_maglim_sys_map,
        )

    # apply the same transformations as in get_reshaped_human_summaries to an observation as put out by
    # msfm.observation.forward_model_observation_map
    with_cross_calc = True
    if not (store_lensing and store_clustering):
        # if one of them is missing, we don't have inter-probe cross-correlation,
        # but power_spectra.smooth_and_bin_cls might still count intra-probe cross-correlations
        pass

    with_cross_calc = True
    if not (store_lensing and store_clustering):
        # if one of them is missing, we don't have inter-probe cross-correlation,
        # but power_spectra.smooth_and_bin_cls might still count intra-probe cross-correlations
        pass

    if from_raw_cls:
        LOGGER.warning(f"Applying scale cuts to the raw Cls, this is deprecated")
        obs_cl, _ = power_spectra.smooth_and_bin_cls(
            obs_cl,
            l_mins_smoothing=l_mins,
            l_maxs_smoothing=l_maxs,
            n_bins=n_bins,
            n_side=msfm_conf["analysis"]["n_side"],
            with_cross=True,
            fixed_binning=False,
        )
    else:
        # Bin without smoothing first, then apply the bin-averaged smoothing factor W²(ℓ) per cross-pair.
        # This matches get_reshaped_human_summaries which multiplies pre-binned Cls by mean(W²) per bin,
        # rather than computing mean(W²·Cℓ) from raw Cls — keeping training and observation consistent.
        n_z_smooth = len(l_mins)
        obs_cl, _ = power_spectra.smooth_and_bin_cls(
            obs_cl,
            l_mins_smoothing=[None] * n_z_smooth,
            l_maxs_smoothing=[None] * n_z_smooth,
            with_cross=True,
            fixed_binning=True,
            n_bins=msfm_conf["analysis"]["power_spectra"]["n_bins"],
            l_min_binning=msfm_conf["analysis"]["power_spectra"]["l_min"],
            l_max_binning=msfm_conf["analysis"]["power_spectra"]["l_max"],
        )
        ells = np.arange(0, 3 * msfm_conf["analysis"]["n_side"])
        bins = power_spectra.get_cl_bins(
            msfm_conf["analysis"]["power_spectra"]["l_min"],
            msfm_conf["analysis"]["power_spectra"]["l_max"],
            msfm_conf["analysis"]["power_spectra"]["n_bins"],
        )
        k = 0
        for i in range(n_z_smooth):
            for j in range(n_z_smooth):
                if (i == j) or (i < j):
                    l_min = max(l_mins[i], l_mins[j])
                    l_max = min(l_maxs[i], l_maxs[j])
                    smoothing_fac = scales.gaussian_high_pass_factor_alm(ells, l_min=l_min)
                    smoothing_fac *= scales.gaussian_low_pass_factor_alm(ells, l_max=l_max)
                    smoothing_fac = smoothing_fac**2
                    smoothing_fac = binned_statistic(ells, smoothing_fac, statistic="mean", bins=bins)[0]
                    obs_cl[..., k] *= smoothing_fac
                    k += 1

    # like in get_reshaped_human_summaries
    if base_dir is not None:
        noise_cl = input_output.load_cl_white_noise(base_dir)[0]

        store_lensing = msfm_conf.get("analysis", {}).get("modelling", {}).get("lensing", {}).get("store", True)
        store_clustering = msfm_conf.get("analysis", {}).get("modelling", {}).get("clustering", {}).get("store", True)

        n_total_expected = (
            (n_z_lensing_total + n_z_clustering_total) * (n_z_lensing_total + n_z_clustering_total + 1) // 2
        )
        if (
            noise_cl.shape[-1] == n_total_expected
            and n_z_lensing_total + n_z_clustering_total != n_z_lensing_active + n_z_clustering_active
        ):
            total_indices, _ = cross_statistics.get_cross_bin_indices(
                n_z_lensing=n_z_lensing_total,
                n_z_clustering=n_z_clustering_total,
                with_lensing=store_lensing,
                with_clustering=store_clustering,
                with_cross_z=True,
                with_cross_probe=(store_lensing and store_clustering),
            )
            noise_cl = noise_cl[..., total_indices]
            LOGGER.info(f"Subsampled white noise from {n_total_expected} to {len(total_indices)} bins")

            if obs_cl.shape[-1] == n_total_expected:
                obs_cl = obs_cl[..., total_indices]
                LOGGER.info(f"Subsampled obs_cl from {n_total_expected} to {len(total_indices)} bins")

        white_noise_sigma = []
        if store_lensing:
            white_noise_sigma.extend(dlss_conf["scale_cuts"]["lensing"]["white_noise_sigma"])
        if store_clustering:
            white_noise_sigma.extend(dlss_conf["scale_cuts"]["clustering"]["white_noise_sigma"])
        n_z = len(white_noise_sigma)
        k = 0
        for i in range(n_z):
            for j in range(n_z):
                if (i == j) or (i < j):
                    noise_cl[:, k] *= white_noise_sigma[i] * white_noise_sigma[j]
                    k += 1

        if obs_cl.shape[-1] == noise_cl.shape[-1]:
            obs_cl += noise_cl
            LOGGER.info(f"Adding white noise to the observation")
        else:
            LOGGER.warning(f"Could not add white noise, shapes {obs_cl.shape} and {noise_cl.shape} do not match.")
    else:
        LOGGER.warning(f"Not adding white noise to the observation!")

    if bin_indices is None:
        n_z_lensing_active = (
            len(msfm_conf["survey"]["metacal"]["z_bins"])
            if msfm_conf.get("analysis", {}).get("modelling", {}).get("lensing", {}).get("store", True)
            else 0
        )
        n_z_clustering_active = (
            len(msfm_conf["survey"]["maglim"]["z_bins"])
            if msfm_conf.get("analysis", {}).get("modelling", {}).get("clustering", {}).get("store", True)
            else 0
        )
        bin_indices, _ = cross_statistics.get_cross_bin_indices(
            n_z_lensing=n_z_lensing_active,
            n_z_clustering=n_z_clustering_active,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=with_cross_z,
            with_cross_probe=with_cross_probe,
            ggl_only=ggl_only,
        )

    assert isinstance(bin_indices, (list, np.ndarray)), "bin_indices must be a list or numpy array"
    LOGGER.info(f"Using the bin indices {bin_indices}")
    obs_cl = obs_cl[..., bin_indices]
    if only_keep_bins is not None:
        obs_cl = obs_cl[..., :only_keep_bins, :]

    # concatenate the bins along the last axis
    obs_cl = np.concatenate([obs_cl[..., i] for i in range(obs_cl.shape[-1])], axis=-1)

    obs_cl, _, _ = preprocess_human_summaries(
        obs_cl[np.newaxis],
        apply_log=apply_log,
        standardize=standardize,
        pca_components=pca_components,
        scaler=scaler,
        pca=pca,
    )

    if make_plot:
        plotting.plot_single_power_spectrum(
            obs_cl,
            bin_size=msfm_conf["analysis"]["power_spectra"]["n_bins"] - 1,
            with_lensing=with_lensing,
            with_clustering=with_clustering,
            with_cross_z=with_cross_z,
            with_cross_probe=with_cross_probe,
        )

    return obs_cl
