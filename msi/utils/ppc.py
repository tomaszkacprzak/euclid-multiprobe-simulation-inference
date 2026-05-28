import os
import numpy as np
import matplotlib.pyplot as plt
from trianglechain import TriangleChain

from msfm.utils import files, logger
from msi.utils import input_output, plotting
from msi.flow_conductor.likelihood_flow import LikelihoodFlow
from msi.flow_conductor import architecture

LOGGER = logger.get_logger(__file__)

_PROBE_ABBREVIATIONS = {
    "lensing": "wl",
    "clustering": "gc",
    "cross": "x",
    "combined": "wl+gc",
}


def _join(*parts):
    """Join non-empty parts with a comma — used to build LaTeX subscripts like 'wl,maps'."""
    return ",".join(p for p in parts if p)


class PosteriorPredictiveChecks:
    """
    Class for running posterior predictive checks (PPC) for LSS probes.

    This class handles loading data, setting up normalizing flows, and running various checks to validate
    the posterior distribution obtained from inference.

    Probes are referred to generically as 'probe1' and 'probe2' (e.g. weak lensing and galaxy clustering).
    """

    def __init__(
        self,
        conf,
        cosmo_params=["Om", "s8", "w0"],
        seed=111,
        # probe names
        probe1_name=None,
        probe2_name=None,
        # data type for each run (free-form, e.g. "maps" or "cls"). Used for plot annotations
        # and to disambiguate runs that share the same probe_name.
        probe1_data=None,
        probe2_data=None,
        # unique labels for each run — auto-derived from probe_name + data; pass explicitly only
        # when overriding the default, must differ between probe1 and probe2.
        probe1_label=None,
        probe2_label=None,
        # data loading
        probe1_pred_file=None,
        probe2_pred_file=None,
        probe1_flow_dir=None,
        probe2_flow_dir=None,
        shared_data=False,
    ):
        """
        Initialize the PosteriorPredictiveChecks object.

        Args:
            conf: Path to the configuration file or dictionary.
            cosmo_params: List of cosmological parameters.
            seed: Random seed for reproducibility.
            probe1_name: Physical probe type for probe 1. One of 'lensing', 'clustering', 'cross',
                'combined'. Determines nuisance parameters and abbreviation.
            probe2_name: Physical probe type for probe 2. Same options as probe1_name.
            probe1_data: Summary-statistic / data-vector type for probe 1 (free-form string,
                typically 'maps' or 'cls'). Optional metadata used to (a) auto-derive a unique
                ``probe1_label`` when both probes share the same ``probe_name``, and (b) enrich
                plot titles, legends, and log lines so the maps-vs-Cls vs lensing-vs-clustering
                distinction is visible end-to-end. Pass ``None`` to fall back to probe-only labels.
            probe2_data: Same as ``probe1_data`` for probe 2.
            probe1_label: Unique identifier for probe 1 used in setup_flow. Defaults to
                ``f"{probe1_name}_{probe1_data}"`` when ``probe1_data`` is set, otherwise to
                ``probe1_name``. Pass explicitly only to override.
            probe2_label: Same as ``probe1_label`` for probe 2.
            probe1_pred_file: Path to the probe 1 predictions file.
            probe2_pred_file: Path to the probe 2 predictions file.
            probe1_flow_dir: Directory for the probe 1 flow model.
            probe2_flow_dir: Directory for the probe 2 flow model.
            shared_data: True when probe1 and probe2 are different summary statistics on the SAME
                physical data (e.g. lensing maps vs lensing Cls). In that case the conditional-
                independence assumption underlying ``independent_cross=True`` does not hold (the
                two summaries share cosmic variance and noise); ``setup_flow`` will refuse
                ``independent_cross=True`` for cross-probe runs.
        """

        self.conf = files.load_config(conf)
        self.cosmo_params = cosmo_params
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.probe1_name = probe1_name
        self.probe2_name = probe2_name
        self.probe1_data = probe1_data
        self.probe2_data = probe2_data
        self.probe1_label = probe1_label or self._default_label(probe1_name, probe1_data)
        self.probe2_label = probe2_label or self._default_label(probe2_name, probe2_data)
        if probe1_name and probe2_name:
            assert self.probe1_label != self.probe2_label, (
                f"probe1_label and probe2_label collide ('{self.probe1_label}'). Set probe*_data "
                "to disambiguate (e.g. data='maps' vs data='cls') or pass probe*_label explicitly."
            )
        self.probe1_abbrv = _PROBE_ABBREVIATIONS[probe1_name] if probe1_name else None
        self.probe2_abbrv = _PROBE_ABBREVIATIONS[probe2_name] if probe2_name else None

        self.probe1_pred_file = probe1_pred_file
        self.probe2_pred_file = probe2_pred_file

        self.probe1_flow_dir = probe1_flow_dir
        self.probe2_flow_dir = probe2_flow_dir
        self.shared_data = shared_data

        if self.probe1_pred_file:
            LOGGER.info(f"Loading {probe1_name} data")
            self.s_probe1_grid, self.theta_probe1_grid, self.probe1_obs_dict, _ = (
                input_output.load_network_preds_simple(self.probe1_pred_file)
            )
            self.probe1_params = self._get_probe_params(probe1_name)
            self.probe1_cosmo_idx = [self.probe1_params.index(p) for p in cosmo_params]

        if self.probe2_pred_file:
            LOGGER.info(f"Loading {probe2_name} data")
            self.s_probe2_grid, self.theta_probe2_grid, self.probe2_obs_dict, _ = (
                input_output.load_network_preds_simple(self.probe2_pred_file)
            )
            self.probe2_params = self._get_probe_params(probe2_name)
            self.probe2_cosmo_idx = [self.probe2_params.index(p) for p in cosmo_params]

        if self.probe1_pred_file and self.probe2_pred_file:
            self._assert_shared_cosmo_grid()

    def _assert_shared_cosmo_grid(self):
        """Verify probe1 and probe2 sims share the same cosmology grid points, row by row.

        Cross-probe ``setup_flow`` uses the joint conditional ``p(s_rep | theta_obs, s_obs)``
        (the Doux et al. 2020 mode), which pairs row ``i`` of ``theta_obs_grid`` /
        ``s_obs_grid`` with row ``i`` of ``s_rep_grid``. For that pairing to be a valid joint,
        the two sim banks must be drawn at the same cosmologies in the same row order.
        ``independent_cross=True`` only needs the cosmo *set* to match, but row-aligned grids
        cover both regimes. Failing this check means the cross-probe results would silently
        train on mismatched (cosmology[i], summary[i]) pairs.
        """
        c1 = np.asarray(self.theta_probe1_grid)[:, self.probe1_cosmo_idx]
        c2 = np.asarray(self.theta_probe2_grid)[:, self.probe2_cosmo_idx]
        assert c1.shape == c2.shape, (
            f"Probe cosmo grids have different shapes ({c1.shape} vs {c2.shape}); the two "
            "pred files must come from the same simulation grid (same train/test split)."
        )
        assert np.allclose(c1, c2), (
            "Probe cosmo grids are not row-aligned: theta_probe1_grid[:, cosmo_idx] does not "
            "match theta_probe2_grid[:, cosmo_idx] row by row. Cross-probe checks pair the two "
            "grids by row index, so the two pred files must share the same train/test split "
            "ordering. Re-export one of the pred files with the same shuffling, or pass "
            "shared_data=False and only use the auto-probe path."
        )

    @staticmethod
    def _default_label(probe_name, data):
        if not probe_name:
            return None
        return f"{probe_name}_{data}" if data else probe_name

    def _setup_descriptor(self):
        """Human-readable descriptor of the current obs→rep setup for plot titles / logs.

        Examples:
            "auto: lensing/maps"
            "cross: lensing/maps → clustering/maps  (joint)"
            "cross: lensing/maps → lensing/cls  (joint, shared_data)"
        """

        def _fmt(probe_name, data):
            return f"{probe_name}/{data}" if data else probe_name

        obs = _fmt(self.obs_probe_name, self.obs_data)
        rep = _fmt(self.rep_probe_name, self.rep_data)
        if not self.is_cross_probe:
            return f"auto: {obs}"
        kind = "indep" if self.independent_cross else "joint"
        if self.shared_data:
            kind += ", shared_data"
        return f"cross: {obs} → {rep}  ({kind})"

    def _summ_subs(self, side):
        """LaTeX subscript for the summary on the obs/rep side, including data type if set."""
        if side == "obs":
            return _join(self.obs_abbrv, self.obs_data)
        return _join(self.rep_abbrv, self.rep_data)

    def _get_probe_params(self, probe_name):
        """Return the full parameter list for a probe: cosmo params + probe-specific nuisances."""
        params = self.cosmo_params.copy()
        if probe_name in ("lensing", "combined", "cross"):
            params += self.conf["analysis"]["params"]["ia"]["nla"]
            if self.conf["analysis"]["modelling"]["lensing"]["extended_nla"]:
                params += self.conf["analysis"]["params"]["ia"]["tatt"]
        if probe_name in ("clustering", "combined", "cross"):
            params += self.conf["analysis"]["params"]["bg"]["linear"]
            if self.conf["analysis"]["modelling"]["clustering"]["quadratic_biasing"]:
                params += self.conf["analysis"]["params"]["bg"]["quadratic"]

        LOGGER.info(f"Probe '{probe_name}' parameters: {params}")
        return params

    def setup_flow(
        self, rep_probe, obs_probe, independent_cross=False, train_flow=False, flow_label="", fit_kwargs={}
    ):
        """
        Set up the normalizing flow for the posterior predictive checks.

        Args:
            rep_probe (str): The probe to be replicated (predicted). Must be one of the names
                passed as probe1_name or probe2_name at construction time.
            obs_probe (str): The probe used for observation (conditioning). Same options.
            independent_cross (bool): Controls the cross-probe flow:

                - ``False`` (default): train the joint conditional ``p(s_rep | theta_obs, s_obs)``.
                  This encodes the cross-probe data-level correlation (cosmic variance, shared
                  noise / footprint) and is the standard Doux et al. 2020 cross-probe consistency
                  formulation — respecting probe correlations is the whole point of the test.
                  Use this for cross-probe consistency checks and for same-data summary-stat
                  comparisons (``shared_data=True``).
                - ``True``: train the marginal ``p(s_rep | theta_cosmo)``, treating the two
                  probes as conditionally independent given cosmology. This is a simplified
                  diagnostic that ignores cross-probe correlations; only appropriate when the
                  two probes really are independent measurements (different sky areas, different
                  physics). Refused when ``shared_data=True`` since shared-data summaries clearly
                  share cosmic variance and noise.

            train_flow (bool): If True, trains the flow from scratch.
            flow_label (str): Label for the flow model.
            fit_kwargs (dict): Additional keyword arguments for fitting the flow.
        """

        assert rep_probe in [
            self.probe1_label,
            self.probe2_label,
        ], f"rep_probe must be one of {[self.probe1_label, self.probe2_label]}, got '{rep_probe}'"
        assert obs_probe in [
            self.probe1_label,
            self.probe2_label,
        ], f"obs_probe must be one of {[self.probe1_label, self.probe2_label]}, got '{obs_probe}'"

        self.rep_probe = "probe1" if rep_probe == self.probe1_label else "probe2"
        self.obs_probe = "probe1" if obs_probe == self.probe1_label else "probe2"

        self.is_cross_probe = self.obs_probe != self.rep_probe
        self.independent_cross = independent_cross

        assert not (self.is_cross_probe and independent_cross and self.shared_data), (
            "independent_cross=True assumes the two probes are conditionally independent given "
            "cosmology, which does not hold when shared_data=True (different summary statistics "
            "on the same physical data share cosmic variance and noise). Use independent_cross=False."
        )

        self.rep_abbrv = self.probe1_abbrv if self.rep_probe == "probe1" else self.probe2_abbrv
        self.obs_abbrv = self.probe1_abbrv if self.obs_probe == "probe1" else self.probe2_abbrv
        self.rep_probe_name = self.probe1_name if self.rep_probe == "probe1" else self.probe2_name
        self.obs_probe_name = self.probe1_name if self.obs_probe == "probe1" else self.probe2_name
        self.rep_data = self.probe1_data if self.rep_probe == "probe1" else self.probe2_data
        self.obs_data = self.probe1_data if self.obs_probe == "probe1" else self.probe2_data

        LOGGER.info(f"Setup: {self._setup_descriptor()}")

        # Bind role → attribute once; all private methods use these names directly.
        self._obs_flow_dir  = getattr(self, f"{self.obs_probe}_flow_dir")
        self._rep_flow_dir  = getattr(self, f"{self.rep_probe}_flow_dir")
        self._s_obs_grid    = getattr(self, f"s_{self.obs_probe}_grid")
        self._s_rep_prior   = getattr(self, f"s_{self.rep_probe}_grid")
        self._theta_obs     = getattr(self, f"theta_{self.obs_probe}_grid")
        self._obs_cosmo_idx = getattr(self, f"{self.obs_probe}_cosmo_idx")
        self._obs_obs_dict  = getattr(self, f"{self.obs_probe}_obs_dict")
        self._rep_obs_dict  = getattr(self, f"{self.rep_probe}_obs_dict") if self.is_cross_probe else None
        self._obs_params    = getattr(self, f"{self.obs_probe}_params")
        self._rep_params    = getattr(self, f"{self.rep_probe}_params")
        self.s_prior        = self._s_rep_prior

        rep_subs = self._summ_subs("rep")
        obs_subs = self._summ_subs("obs")

        if self.is_cross_probe:
            flow_dir = self._obs_flow_dir
            features_grid = self._s_rep_prior
            if independent_cross:
                self.flow_dist = f"p(s_{{{rep_subs}}} | theta_cosmo)"
                # only shared cosmo params: rep probe is insensitive to obs probe nuisance parameters
                context_grid = self._theta_obs[:, self._obs_cosmo_idx]
            else:
                self.flow_dist = f"p(s_{{{rep_subs}}} | theta_{self.obs_abbrv}, s_{{{obs_subs}}})"
                context_grid = np.concatenate([self._theta_obs, self._s_obs_grid], axis=-1)
        else:
            self.flow_dist = f"p(s_{{{rep_subs}}} | theta_{self.rep_abbrv})"
            flow_dir = self._obs_flow_dir
            features_grid = self._s_rep_prior
            context_grid = self._theta_obs

        LOGGER.info(f"flow = {self.flow_dist}")
        self.context_grid = context_grid

        if self.is_cross_probe:
            flow_label += "ppc/cross"
            flow_label += f"_{self.rep_abbrv}_given_{self.obs_abbrv}"
            flow_label += "_independent" if independent_cross else ""
        else:
            flow_label += "ppc/auto_"
            flow_label += self.obs_abbrv

        self.flow = LikelihoodFlow(
            params=[],
            conf=self.conf,
            embedding_net=architecture.get_context_embedding_net(context_grid.shape[-1]),
            base_dist=architecture.get_normal_dist(features_grid.shape[-1]),
            transform=architecture.get_sigmoids_transform(features_grid.shape[-1]),
            out_dir=flow_dir,
            label=flow_label,
            load_existing=not train_flow,
        )
        self.out_dir = self.flow.model_dir

        if train_flow:
            self.flow.fit(
                x=features_grid,
                theta=context_grid,
                batch_size=10_000,
                scheduler_type="cosine",
                save_model=True,
                **fit_kwargs,
            )

    def run_checks(
        self,
        # define observation
        obs_label=None,
        s_obs=None,
        theta_post=None,
        s_obs_rep=None,
        theta_post_rep=None,
        # samples
        n_samples_neural=100_000,
        n_samples_grid=1_000,
        k_highest_grid=None,
        # select checks
        plot_param_posterior=False,
        check_data_marginals=True,
        check_kernel=True,
        check_log_prob=True,
        check_mahalanobis=True,
        check_l2=True,
        check_l1=True,
        check_linf=True,
    ):
        """
        Run the requested posterior predictive checks.

        Args:
            obs_label (str): Label for the observation.
            s_obs (np.ndarray): Observed summary statistics.
            theta_post (np.ndarray): Posterior samples of parameters.
            s_obs_rep (np.ndarray): Observed summary statistics for the replicated probe.
            theta_post_rep (np.ndarray): Posterior samples for the replicated probe.
            n_samples_neural (int): Number of samples to draw from the neural posterior predictive.
            n_samples_grid (int): Number of samples to draw from the grid posterior predictive (importance sampling).
            k_highest_grid (int): Number of highest probability samples to select from the grid.
            plot_param_posterior (bool): Whether to plot the parameter posterior.
            check_data_marginals (bool): Whether to check data marginals.
            check_kernel (bool): Whether to run the kernel similarity outlier test.
            check_log_prob (bool): Whether to run the log-probability posterior predictive check.
            check_mahalanobis (bool): Whether to check the Mahalanobis distance.
            check_l2 (bool): Whether to check the mean L2 distance to the PPD.
            check_l1 (bool): Whether to check the mean L1 distance to the PPD.
            check_linf (bool): Whether to check the max standardised deviation (L∞ norm).
        """

        self._set_observation(obs_label, s_obs, theta_post, s_obs_rep, theta_post_rep)

        if plot_param_posterior:
            self._plot_param_posterior()

        self._sample_neural_posterior_predictive(n_samples=n_samples_neural)
        if not self.is_cross_probe:
            self._sample_grid_posterior_predictive(n_importance_samples=n_samples_grid, k_highest=k_highest_grid)

        if check_data_marginals:
            self._check_data_marginals()

        if check_log_prob:
            self._check_log_prob()

        if check_kernel:
            self._check_one_sample(stat="kernel")

        if check_mahalanobis:
            self._check_one_sample(stat="mahalanobis")

        if check_l2:
            self._check_one_sample(stat="l2")

        if check_l1:
            self._check_one_sample(stat="l1")

        if check_linf:
            self._check_one_sample(stat="linf")

    def _set_observation(self, obs_label=None, s_obs=None, theta_post=None, s_obs_rep=None, theta_post_rep=None):
        """Set up the observation data and configuration for the PPC."""

        self.obs_label = obs_label

        self.post_dist = f"p(theta_{self.obs_abbrv} | s_{{{self._summ_subs('obs')}}})"
        LOGGER.info(f"post = {self.post_dist}")

        obs_flow_dir = self._obs_flow_dir
        obs_dict = self._obs_obs_dict

        if self.is_cross_probe:
            rep_flow_dir = self._rep_flow_dir
            rep_obs_dict = self._rep_obs_dict

        # obs_probe
        if s_obs is None:
            assert obs_label in obs_dict, (
                f"obs_label '{obs_label}' not found in {self.obs_probe_name} observations: "
                f"{sorted(obs_dict.keys())}"
            )
            s_obs = obs_dict[obs_label]
        self.s_obs = s_obs

        if theta_post is None:
            theta_post = np.load(os.path.join(obs_flow_dir, f"chain_{obs_label}.npy"))
        self.theta_post = theta_post

        # rep_probe
        if self.is_cross_probe:
            if s_obs_rep is None:
                assert obs_label in rep_obs_dict, (
                    f"obs_label '{obs_label}' not found in {self.rep_probe_name} observations: "
                    f"{sorted(rep_obs_dict.keys())}"
                )
                s_obs_rep = rep_obs_dict[obs_label]

            if theta_post_rep is None:
                theta_post_rep = np.load(os.path.join(rep_flow_dir, f"chain_{obs_label}.npy"))
        else:
            s_obs_rep = s_obs
            theta_post_rep = theta_post

        self.s_obs_rep = s_obs_rep
        # only for plotting the parameter posterior
        self.theta_post_rep = theta_post_rep

    def _plot_param_posterior(self):
        """Plot the parameter posteriors for the observation and replicated probe."""

        def _chain_label(probe_name, data):
            return f"{probe_name}/{data}" if data else probe_name

        chains = [self.theta_post]
        labels = [_chain_label(self.obs_probe_name, self.obs_data)]
        params = [self._obs_params]

        if self.is_cross_probe:
            chains.append(self.theta_post_rep)
            labels.append(_chain_label(self.rep_probe_name, self.rep_data))
            params.append(self._rep_params)

        plotting.plot_chains(
            chains=chains,
            params=params,
            conf=self.conf,
            plot_labels=labels,
            obs_cosmo=None,
            out_dir=self.out_dir,
            file_label=self.obs_label,
        )

    def _sample_neural_posterior_predictive(self, n_samples=100_000):
        """Sample from the neural posterior predictive distribution."""

        # subsample the posterior
        i_star = self.rng.integers(0, self.theta_post.shape[0], n_samples)
        theta_star = self.theta_post[i_star]

        # sample the flow
        if self.is_cross_probe and not self.independent_cross:
            s_obs_star = np.repeat(np.atleast_2d(self.s_obs), n_samples, axis=0)
            context_star = np.concatenate([theta_star, s_obs_star], axis=-1)
        elif self.is_cross_probe and self.independent_cross:
            # marginalise over probe-specific nuisances by using only the shared cosmo columns
            context_star = theta_star[:, self._obs_cosmo_idx]
        else:
            context_star = theta_star

        LOGGER.info(f"Generating {n_samples} neural samples of {self.flow_dist} flow")
        LOGGER.timer.start("sampling")
        s_rep = self.flow.sample_likelihood(
            context_star,
            n_samples=1,
            batch_size=min(context_star.shape[0], 10_000),
        )
        s_rep = np.squeeze(s_rep)
        LOGGER.info(f"Done sampling after {LOGGER.timer.elapsed('sampling')}")

        self.context_star = context_star
        self.s_rep = s_rep

    def _grid_importance_indices(self, n_samples):
        """Importance-sample indices from the cosmology grid using p(s_obs | theta).

        Computes flow log-likelihoods at ``self.s_obs_rep`` across ``self.context_grid``, turns
        them into normalised weights, draws ``n_samples`` indices with replacement, and returns
        ``(i_picked, ess)``. The number of indices returned is capped at ``int(ESS)`` because
        drawing more than that just yields duplicates.
        """
        log_probs = (
            self.flow.log_likelihood(
                np.repeat(np.atleast_2d(self.s_obs_rep), self.context_grid.shape[0], axis=0),
                self.context_grid,
            )
            .cpu()
            .numpy()
        )
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)

        ess = 1.0 / np.sum(probs**2)
        LOGGER.info(f"Effective Sample Size (ESS) = {ess:.1f} out of {self.context_grid.shape[0]}")

        n_eff = max(1, int(ess))
        if n_samples > n_eff:
            LOGGER.info(f"Capping importance samples at int(ESS)={n_eff} (requested {n_samples}).")
            n_samples = n_eff

        LOGGER.info(f"Drawing {n_samples} samples from the grid with importance weights")
        i_picked = self.rng.choice(self.context_grid.shape[0], size=n_samples, replace=True, p=probs)
        n_unique = np.unique(i_picked).shape[0]
        LOGGER.info(f"Obtained {n_unique} unique samples out of {n_samples} samples")
        return i_picked, ess

    def _sample_grid_posterior_predictive(self, n_importance_samples=None, k_highest=None, ess_floor=500):
        """Sample from the grid posterior predictive using importance sampling or top-k selection.

        When the effective sample size (ESS) falls below ``ess_floor``, the simulation grid is
        too sparse near the posterior mode for the resulting empirical PPD to be meaningful;
        ``self.s_rep_grid`` is set to ``None`` and the caller should skip plotting/using it.
        """
        # TODO for the cross-probe check, this is currently wrong: https://gemini.google.com/share/1e7a829ec98b
        # The weights should be proportional to p(theta|s_obs) ~ p(s_obs|theta) and not p(s_rep|theta, s_obs).
        # For the single probe, it doesn't make a difference as s_rep = s_obs.
        assert not self.is_cross_probe, "Grid PPC not implemented for cross-probe checks yet."

        if (n_importance_samples is not None) and (k_highest is None):
            i_picked, ess = self._grid_importance_indices(n_importance_samples)
            if ess < ess_floor:
                LOGGER.warning(
                    f"Grid PPD ESS={ess:.1f} is below the floor of {ess_floor}; the simulation grid "
                    "is too sparse near the posterior mode for a reliable empirical PPD. Skipping the "
                    "grid-based PPD (s_rep_grid=None)."
                )
                self.s_rep_grid = None
                return
            self.s_rep_grid = self.s_prior[i_picked]

        elif (k_highest is not None) and (n_importance_samples is None):
            log_probs = (
                self.flow.log_likelihood(
                    np.repeat(np.atleast_2d(self.s_obs_rep), self.context_grid.shape[0], axis=0),
                    self.context_grid,
                )
                .cpu()
                .numpy()
            )
            log_probs -= np.max(log_probs)
            LOGGER.info(f"Selecting the {k_highest} highest probability samples from the grid")
            i_sorted = np.argsort(log_probs)[-k_highest:]
            self.s_rep_grid = self.s_prior[i_sorted]

        else:
            raise ValueError("Either n_importance_samples or k_highest must be specified, but not both")

    def _check_data_marginals(self, n_scatter=1_000, outlier_quantile=1e-3):
        """Check and plot the marginal distributions of the data."""

        n_s = self.s_prior.shape[1]

        rep_subs = self._summ_subs("rep")
        obs_subs = self._summ_subs("obs")

        prior_label = r"$p(s_{" + rep_subs + r"})$"
        post_label = r"$p(s_{" + rep_subs + r"}|s_{" + obs_subs + r"}^{obs})$"
        post_label_sim = r"$p(s_{" + rep_subs + r"}|s_{" + obs_subs + r"}^{obs})$ (sims)"
        obs_label_str = r"$s_{" + rep_subs + r"}^{obs}$"

        tri = TriangleChain(
            show_legend=True,
            legend_fontsize=24,
            size=2,
            line_kwargs={"zorder": 0, "linewidths": 2},
            hist_kwargs={"zorder": 0, "lw": 2},
            scatter_kwargs={"s": 1, "marker": "o"},
            params=[str(i) for i in range(n_s)],
            labels=[rf"$s^{{{i}}}_{{{rep_subs}}}$" for i in range(n_s)],
            ranges={
                str(i): (
                    np.quantile(self.s_prior[:, i], outlier_quantile),
                    np.quantile(self.s_prior[:, i], (1 - outlier_quantile)),
                )
                for i in range(n_s)
            },
        )

        def contour_or_scatter(tri, data, color, label):
            if data.shape[0] > n_scatter:
                tri.contour_cl(data, color=color, label=label)
            else:
                tri.scatter(data, color=color, label=label)

        contour_or_scatter(tri, self.s_prior, color="tab:blue", label=prior_label)
        contour_or_scatter(tri, self.s_rep, color="tab:orange", label=post_label)
        if not self.is_cross_probe and self.s_rep_grid is not None:
            contour_or_scatter(tri, self.s_rep_grid, color="tab:green", label=post_label_sim)

        tri.scatter(
            np.atleast_2d(self.s_obs_rep),
            scatter_kwargs={"s": 200, "marker": "*", "zorder": 10},
            color="k",
            scatter_vline_1D=True,
            plot_histograms_1D=False,
            label=obs_label_str,
        )

        # only keep the last legend
        try:
            for legend in tri.fig.legends[:-1]:
                legend.remove()
        except AttributeError:
            pass

        if tri.fig.legends:
            legend = tri.fig.legends[-1]
            legend._loc = 1  # 1 = 'upper right'
            legend.set_bbox_to_anchor((0.80, 0.80))
            for text in legend.get_texts():
                text.set_fontsize(24)
            for marker in legend.legend_handles:
                try:
                    marker.set_sizes([200])
                except AttributeError:
                    pass
                try:
                    marker.set_linewidth(4.0)
                except AttributeError:
                    pass

        # to fix the ugly tick labels
        import matplotlib.ticker as mticker

        _fmt = mticker.FuncFormatter(lambda x, _: f"{x:.4g}")
        for ax in tri.fig.axes:
            try:
                ss = ax.get_subplotspec()
                row = ss.rowspan.start
                col = ss.colspan.start
            except (AttributeError, TypeError):
                continue
            ax.xaxis.set_major_formatter(_fmt)
            ax.yaxis.set_major_formatter(_fmt)
            if row < n_s - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if col > 0:
                ax.tick_params(axis="y", labelleft=False)

        tri.fig.suptitle(f"{self.obs_label} — {self._setup_descriptor()}", fontsize=24, y=0.9)

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_data_marginals.png")
        LOGGER.info(f"Saving data marginals plot to {plot_file}")
        tri.fig.savefig(plot_file, bbox_inches="tight", dpi=100)

    def _check_log_prob(self, n_bootstrap=10_000):
        """Bayesian posterior predictive p-value via paired log-likelihood comparison.

        For each bootstrap draw i, computes
            delta_i = log p(s_rep_i | theta_i) - log p(s_obs | theta_i)
        where theta_i ~ p(theta | s_obs) and s_rep_i ~ p(s | theta_i).
        p-value = fraction of draws where delta_i <= 0 (obs at least as likely as rep).

        Interpretation:
            - p ≈ 0.5 : good fit (obs is a typical draw under the model).
            - p → 0   : tension — the obs is in the lower tail of the predictive log-density,
                        i.e. less likely than rep draws under the same theta.
            - p → 1   : usually NOT a tension signal but a sign of posterior over-dispersion or
                        flow leakage (model assigns the obs higher density than its own samples).

        Note: this p-value is not uniform under the null (a known property of Bayesian PPD
        p-values: the data is used to fit the posterior and to evaluate the test, so p
        concentrates near 0.5 — small p is therefore conservative as a tension signal).

        Cross-probe caveat: when ``independent_cross=False`` the flow models the conditional
        ``p(s_rep | theta, s_obs)``, so the densities here are conditional on s_obs. This is a
        different statistic from the marginal-likelihood test in Doux et al. 2020 and the
        numerical p-values are not directly comparable.

        Args:
            n_bootstrap: Number of paired draws for the p-value estimate.
        """
        s_rep = self.s_rep
        s_obs = np.atleast_2d(self.s_obs_rep)
        i_boot = self.rng.integers(0, s_rep.shape[0], n_bootstrap)

        log_lik = lambda x, ctx: self.flow.log_likelihood(x, ctx, return_numpy=True)  # noqa: E731
        t_diff = log_lik(s_rep[i_boot], self.context_star[i_boot]) - log_lik(
            np.repeat(s_obs, n_bootstrap, axis=0), self.context_star[i_boot]
        )  # positive: rep more likely than obs
        p_val = np.mean(t_diff <= 0)

        rep_subs = self._summ_subs("rep")
        diff_label = (
            r"$\log p(s_{" + rep_subs + r"}^{rep}|\theta_i)" r" - \log p(s_{" + rep_subs + r"}^{obs}|\theta_i)$"
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(t_diff, bins=100, alpha=0.5, label=diff_label)
        ax.axvline(0, color="k", linestyle="--", label=f"p = {p_val:.4f}")
        ax.set(
            xlabel=diff_label,
            ylabel="Count",
            title=f"{self.obs_label}: Log-Prob PPC: p = {p_val:.4f}\n{self._setup_descriptor()}",
        )
        ax.legend()
        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_log_prob_check.png")
        LOGGER.info(f"Saving Log-Prob PPC plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)

    def _check_one_sample(self, stat, n_bootstrap=10_000, n_ref=5_000):
        """Generic one-sample test: is s_obs an outlier relative to the PPD?

        Null distribution: evaluate the same statistic on bootstrap draws from
        the PPD samples s_rep.  A small p-value means s_obs is extreme.

        For kernel/L1/L2/Linf the data are whitened by the ref-pool covariance
        (Cholesky-based; per-dimension standardisation is used as a fallback if
        the covariance is not positive-definite).  Whitening makes the metric
        isotropic in the PPD frame, so deviations along narrow correlated
        directions are not swamped by long axes of the cloud.  s_rep is split
        into non-overlapping halves so the reference cloud (defines the
        statistic) and the bootstrap pool (builds the null) are independent;
        the Mahalanobis covariance is also estimated from the ref pool only.

        Args:
            stat: 'mahalanobis', 'l1', 'l2', 'linf', or 'kernel'.
            n_bootstrap: Number of bootstrap draws for the null.
            n_ref: Reference subsample size for distance-based stats (kernel, L1, L2).
        """
        from scipy.spatial.distance import cdist
        from scipy.linalg import solve_triangular

        s_rep = self.s_rep  # (N, dim)
        s_obs = np.atleast_2d(self.s_obs_rep)  # (1, dim)
        n_rep = s_rep.shape[0]

        # Non-overlapping split: ref pool defines the statistic; boot pool builds the null.
        perm = self.rng.permutation(n_rep)
        i_ref_pool, i_boot_pool = perm[: n_rep // 2], perm[n_rep // 2 :]

        # Whiten using ref-pool stats (fall back to per-dim standardisation if Cholesky fails).
        s_mu = np.mean(s_rep[i_ref_pool], axis=0)
        cov_ref = np.cov(s_rep[i_ref_pool], rowvar=False)
        cov_ref = np.atleast_2d(cov_ref)
        try:
            jitter = 1e-8 * max(1.0, float(np.trace(cov_ref)) / cov_ref.shape[0])
            L = np.linalg.cholesky(cov_ref + jitter * np.eye(cov_ref.shape[0]))

            def _whiten(x):
                return solve_triangular(L, (x - s_mu).T, lower=True).T

            s_rep_n = _whiten(s_rep)
            s_obs_n = _whiten(s_obs)
        except np.linalg.LinAlgError:
            LOGGER.warning(
                "Cholesky decomposition of ref-pool covariance failed; falling back to per-dim standardisation."
            )
            s_std = np.std(s_rep[i_ref_pool], axis=0)
            s_std[s_std == 0] = 1.0
            s_rep_n = (s_rep - s_mu) / s_std
            s_obs_n = (s_obs - s_mu) / s_std

        s_ref_n = s_rep_n[i_ref_pool[: min(n_ref, len(i_ref_pool))]]
        rep_subs = self._summ_subs("rep")

        if stat == "mahalanobis":
            cov_inv = np.linalg.pinv(cov_ref)

            def compute_stat(x):
                diff = x - s_mu
                return np.einsum("...i,ij,...j->...", diff, cov_inv, diff)

            s_obs_eval, s_rep_eval = s_obs, s_rep
            outlier_if_high = True
            xlabel = "Mahalanobis distance²"
            file_tag = "mahalanobis_check"
            title_tag = "Mahalanobis Distance Check"
            stat_label = r"$D_M^2(s_{" + rep_subs + r"}^{obs})$"

        elif stat in ("l1", "l2"):
            metric, norm_ord = ("cityblock", 1) if stat == "l1" else ("euclidean", 2)

            def compute_stat(x):
                return np.mean(cdist(x, s_ref_n, metric=metric), axis=-1)

            s_obs_eval, s_rep_eval = s_obs_n, s_rep_n
            outlier_if_high = True
            xlabel = f"Mean L{norm_ord} distance to PPD"
            file_tag = f"{stat}_check"
            title_tag = f"L{norm_ord} Distance Check"
            stat_label = (
                r"$\bar{d}_{L" + str(norm_ord) + r"}(s_{" + rep_subs + r"}^{obs},\, s_{" + rep_subs + r"}^{rep})$"
            )

        elif stat == "linf":

            def compute_stat(x):
                return np.max(np.abs(x), axis=-1)  # max standardised deviation across dims

            s_obs_eval, s_rep_eval = s_obs_n, s_rep_n
            outlier_if_high = True
            xlabel = r"$\max_j |s_j^{std}|$  (L∞ norm)"
            file_tag = "linf_check"
            title_tag = "L∞ Distance Check"
            stat_label = r"$\|s_{" + rep_subs + r"}^{obs}\|_\infty$"

        elif stat == "kernel":
            n_bw = min(2_000, s_ref_n.shape[0])
            sq_dists_bw = cdist(s_ref_n[:n_bw], s_ref_n[:n_bw], metric="sqeuclidean")
            bw2 = np.median(sq_dists_bw[np.triu_indices(n_bw, k=1)]) or 1.0
            LOGGER.info(f"Kernel bandwidth (squared, normalised): {bw2:.4f}")

            def compute_stat(x):
                return np.mean(np.exp(-cdist(x, s_ref_n, metric="sqeuclidean") / bw2), axis=-1)

            s_obs_eval, s_rep_eval = s_obs_n, s_rep_n
            outlier_if_high = False
            xlabel = "Mean kernel similarity"
            file_tag = "kernel_check"
            title_tag = "Kernel Similarity Check"
            stat_label = r"$\bar{k}(s_{" + rep_subs + r"}^{obs},\, s_{" + rep_subs + r"}^{rep})$"

        else:
            raise ValueError(f"Unknown stat: {stat}")

        i_boot = i_boot_pool[self.rng.integers(0, len(i_boot_pool), n_bootstrap)]
        t_obs = compute_stat(s_obs_eval)[0]
        t_boot = compute_stat(s_rep_eval[i_boot])
        p_val = np.mean(t_boot >= t_obs) if outlier_if_high else np.mean(t_boot <= t_obs)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(t_boot, bins=100, alpha=0.5, label="null (PPD samples)")
        ax.axvline(t_obs, color="k", label=f"{stat_label} = {t_obs:.4f}")
        ax.set(
            xlabel=xlabel,
            ylabel="Count",
            title=f"{self.obs_label}: {title_tag}: p = {p_val:.4f}\n{self._setup_descriptor()}",
        )
        ax.legend()

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_{file_tag}.png")
        LOGGER.info(f"Saving {title_tag} plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)

    _PROBE_NAME_TO_CLS_FLAGS = {
        "lensing": {"with_lensing": True, "with_clustering": False, "with_cross_z": True, "with_cross_probe": False},
        "clustering": {
            "with_lensing": False,
            "with_clustering": True,
            "with_cross_z": True,
            "with_cross_probe": False,
        },
        "cross": {"with_lensing": False, "with_clustering": False, "with_cross_z": True, "with_cross_probe": True},
        "combined": {"with_lensing": True, "with_clustering": True, "with_cross_z": True, "with_cross_probe": True},
    }

    def _build_cls_obs(self, dlss_conf, base_dir, apply_log=False):
        """Build the observed Cls vector for ``self.obs_label`` via the catalog → maps → Cls
        pipeline (matching ``y3-deep-lss/notebooks/2pt_train+eval.ipynb``).

        Only catalog-based labels (currently ``"DESy3"``) are supported. For mock labels that
        live only in the maps obs_dict (e.g. ``"bench_fidu_mean"``, ``"grid_*"``), the caller
        must pass ``cls_obs`` to ``check_cls_marginals`` directly — the underlying maps for
        those mocks are not co-located with the compressed-summary obs_dict.
        """
        # deferred imports: keep PPC usable when the catalog data is not present
        from msfm.utils import catalog
        from msi.utils import preprocessing

        if dlss_conf is None or base_dir is None:
            raise ValueError(
                "_build_cls_obs requires both dlss_conf and base_dir; pass them to "
                "check_cls_marginals or provide cls_obs explicitly."
            )

        if self.obs_label != "DESy3":
            raise ValueError(
                f"Auto-build of cls_obs is only implemented for obs_label='DESy3', "
                f"got '{self.obs_label}'. Pass cls_obs explicitly for mock observations."
            )

        flags = self._PROBE_NAME_TO_CLS_FLAGS[self.obs_probe_name]
        LOGGER.info(f"Auto-building cls_obs for obs_label='{self.obs_label}' with flags {flags}")

        wl_gamma_map, _ = catalog.build_metacal_map_from_cat(self.conf)
        gc_count_map = catalog.build_maglim_map_from_cat(self.conf)

        cls_obs = preprocessing.get_preprocessed_cl_observation(
            wl_gamma_map=wl_gamma_map,
            gc_count_map=gc_count_map,
            msfm_conf=self.conf,
            dlss_conf=dlss_conf,
            base_dir=base_dir,
            nest_in=False,
            apply_log=apply_log,
            standardize=False,
            make_plot=False,
            **flags,
        )
        return np.asarray(cls_obs).reshape(-1)

    def check_cls_marginals(
        self,
        dlss_conf,
        base_dir,
        cls_obs=None,
        apply_log=True,
        train_test_split=0.8,
        n_samples=5_000,
        percentiles=(16, 84),
        file_label="cls",
        x_label="data-vector index",
        log_y=None,
        n_traces=20,
    ):
        """Plot the posterior predictive distribution in Cls space (auto-probe only).

        Fully self-contained: loads the Cls grid from ``base_dir``, applies scale cuts via
        ``preprocessing.get_reshaped_human_summaries``, replicates the sobol-sorted train/test
        split used when generating the predictions file, importance-samples indices, adds an
        independent noise realisation to each picked sample, and optionally applies
        ``log(|Cls|)`` — mirroring the ``get_binned_power_spectra`` training pipeline.

        Args:
            dlss_conf: path or dict for the deep-lss config; supplies scale-cut parameters.
            base_dir: data directory containing ``cls/`` and ``cls/white_noise.h5``.
            cls_obs: (n_cls_dims,) observed Cls vector, already in the same preprocessed space
                (scale cuts + noise + optional log). If None, built automatically via
                ``_build_cls_obs`` (only supports ``self.obs_label='DESy3'``).
            apply_log: apply ``log(|Cls|)`` to both PPD samples and obs after noise addition,
                matching the training pipeline. Default True.
            train_test_split: fraction used as training set when generating the predictions
                file; the complementary fraction is the test set aligned with ``self.s_prior``.
            n_samples: number of importance draws (capped at int(ESS) inside the helper).
            percentiles: low/high percentiles for the shaded band.
            file_label: tag in the output filename.
            x_label: x-axis label.
            log_y: log scale on the y-axis. Defaults to ``not apply_log`` (linear when data
                are already log-transformed).
            n_traces: number of thin individual PPD lines to overlay under the band.
        """
        assert not self.is_cross_probe, "check_cls_marginals is implemented for auto-probe checks only."

        from msi.utils import preprocessing

        if log_y is None:
            log_y = not apply_log

        flags = self._PROBE_NAME_TO_CLS_FLAGS[self.obs_probe_name]

        LOGGER.info(f"Loading and preprocessing Cls grid from {base_dir}")
        _, cls_grid, noise_cls, _, grid_i_sobols, _, _, _ = preprocessing.get_reshaped_human_summaries(
            base_dir,
            "cls",
            msfm_conf=self.conf,
            dlss_conf=dlss_conf,
            concat_example_dim=False,
            concat_bin_dim=True,
            with_fiducial=False,
            do_plot=False,
            apply_log=False,
            standardize=False,
            **flags,
        )
        # cls_grid:      (n_cosmo, n_examples, n_cls_dims) — scale-cut + noise-sigma-scaled
        # noise_cls:     (N_noise, n_cls_dims) or None     — sigma-scaled pool
        # grid_i_sobols: (n_cosmo, n_examples)

        # Replicate sobol-sort + train/test split used when generating the predictions file
        i_sort = np.argsort(grid_i_sobols[:, 0])
        cls_grid = cls_grid[i_sort]
        i_split = int(train_test_split * cls_grid.shape[1])
        cls_grid_test = cls_grid[:, i_split:, :]
        cls_grid_test = np.concatenate([cls_grid_test[i] for i in range(cls_grid_test.shape[0])], axis=0)
        # (n_cosmo * n_test_per_cosmo, n_cls_dims), aligned with self.s_prior

        assert cls_grid_test.shape[0] == self.s_prior.shape[0], (
            f"cls_grid test split has {cls_grid_test.shape[0]} rows but self.s_prior has "
            f"{self.s_prior.shape[0]}; ensure train_test_split={train_test_split} matches "
            "the split used when generating the predictions file."
        )

        from msfm.utils import power_spectra as _ps
        _ps_conf = self.conf["analysis"]["power_spectra"]
        _bin_edges_1d = _ps.get_cl_bins(_ps_conf["l_min"], _ps_conf["l_max"], _ps_conf["n_bins"])
        _ell_centers_1d = (_bin_edges_1d[:-1] + _bin_edges_1d[1:]) / 2    # (n_ell_bins,)

        if cls_obs is None:
            cls_obs_raw = self._build_cls_obs(dlss_conf=dlss_conf, base_dir=base_dir, apply_log=False)
            cls_obs = np.log(np.abs(cls_obs_raw)) if apply_log else cls_obs_raw
        else:
            cls_obs = np.atleast_1d(np.asarray(cls_obs)).reshape(-1)
            cls_obs_raw = cls_obs.copy()
            if apply_log:
                cls_obs = np.log(np.abs(cls_obs))

        assert cls_obs.shape[0] == cls_grid_test.shape[1], (
            f"cls_obs has {cls_obs.shape[0]} elements but cls_grid has {cls_grid_test.shape[1]} "
            "columns; ensure probe/bin selection matches."
        )

        i_picked, _ = self._grid_importance_indices(n_samples)
        cls_picked = cls_grid_test[i_picked].copy()

        if noise_cls is not None:
            i_noise = self.rng.integers(0, noise_cls.shape[0], cls_picked.shape[0])
            cls_picked = cls_picked + noise_cls[i_noise]

        cls_picked_raw = cls_picked.copy()

        if apply_log:
            cls_picked = np.log(np.abs(cls_picked))

        lo_q, hi_q = percentiles
        lo = np.percentile(cls_picked, lo_q, axis=0)
        hi = np.percentile(cls_picked, hi_q, axis=0)
        mid = np.median(cls_picked, axis=0)

        n_dims = cls_picked.shape[1]
        x = np.arange(n_dims)
        assert n_dims % len(_ell_centers_1d) == 0, (
            f"n_cls_dims={n_dims} is not a multiple of n_ell_bins={len(_ell_centers_1d)}; "
            "scale cuts may produce variable-length ell axes across z-bin pairs."
        )
        n_ell_per_pair = len(_ell_centers_1d)
        n_pairs = n_dims // n_ell_per_pair
        ell_flat = np.tile(_ell_centers_1d, n_pairs)                       # (n_cls_dims,)

        cls_picked_lcl = ell_flat * cls_picked_raw
        lo_lcl = np.percentile(cls_picked_lcl, lo_q, axis=0)
        hi_lcl = np.percentile(cls_picked_lcl, hi_q, axis=0)
        mid_lcl = np.median(cls_picked_lcl, axis=0)
        cls_obs_lcl = ell_flat * cls_obs_raw

        mid_raw = np.median(cls_picked_raw, axis=0)
        ppd_std = np.std(cls_picked_raw, axis=0)
        sig_obs = (cls_obs_raw - mid_raw) / np.where(ppd_std > 0, ppd_std, np.nan)

        rel_denom = np.where(np.abs(cls_obs_raw) > 0, np.abs(cls_obs_raw), np.nan)
        cls_rel_samples = (cls_picked_raw - cls_obs_raw) / rel_denom
        lo_rel = np.percentile(cls_rel_samples, lo_q, axis=0)
        hi_rel = np.percentile(cls_rel_samples, hi_q, axis=0)
        mid_rel = np.median(cls_rel_samples, axis=0)

        # --- pair labels ---
        _n_z = int(round((-1 + (1 + 8 * n_pairs) ** 0.5) / 2))
        if _n_z * (_n_z + 1) // 2 == n_pairs:
            if flags.get("with_lensing") and not flags.get("with_clustering"):
                _sym = r"\kappa"
            elif flags.get("with_clustering") and not flags.get("with_lensing"):
                _sym = r"\delta_g"
            else:
                _sym = None
            if _sym is not None:
                _pair_labels = [
                    rf"${_sym}^{{{i + 1}}}\times{_sym}^{{{j + 1}}}$"
                    for i in range(_n_z) for j in range(i, _n_z)
                ]
            else:
                _pair_labels = [f"pair {k}" for k in range(n_pairs)]
        else:
            _pair_labels = [f"pair {k}" for k in range(n_pairs)]

        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True, constrained_layout=True)
        ax = axes[0]

        # --- panel 0: log(|Cl|) or Cl with optional log y-scale ---
        ax.fill_between(x, lo, hi, alpha=0.3, color="tab:orange", label=f"PPD [{lo_q:g}, {hi_q:g}]%")
        ax.plot(x, mid, color="tab:orange", lw=1.5, label="PPD median")
        if n_traces > 0:
            ax.plot(x, cls_picked[: min(n_traces, cls_picked.shape[0])].T, color="tab:orange", alpha=0.2, lw=0.5)
        ax.plot(x, cls_obs, color="k", lw=1.5, label=f"obs ({self.obs_label})")

        if log_y:
            if np.any(cls_picked <= 0) or np.any(cls_obs <= 0):
                LOGGER.warning("Non-positive values in cls_picked / cls_obs; using linear y-scale.")
            else:
                ax.set_yscale("log")

        title = (
            f"{self.obs_label}: Cls PPD — {self._setup_descriptor()}, "
            f"noise={'on' if noise_cls is not None else 'off'}, log={apply_log}"
        )
        ax.set(ylabel=r"$\log C_\ell$" if apply_log else r"$C_\ell$", title=title)
        ax.yaxis.set_ticklabels([])
        ax.legend(fontsize=8)

        # --- panel 1: ℓ·Cl ---
        axes[1].fill_between(x, lo_lcl, hi_lcl, alpha=0.3, color="tab:orange")
        axes[1].plot(x, mid_lcl, color="tab:orange", lw=1.5)
        if n_traces > 0:
            axes[1].plot(
                x, (ell_flat * cls_picked_raw[: min(n_traces, cls_picked_raw.shape[0])]).T,
                color="tab:orange", alpha=0.2, lw=0.5,
            )
        axes[1].plot(x, cls_obs_lcl, color="k", lw=1.5)
        if noise_cls is not None:
            _noise_lcl = ell_flat * np.std(noise_cls, axis=0)
            axes[1].plot(x, _noise_lcl, color="gray", lw=1.0, ls="--", label="noise rms")
            axes[1].legend(fontsize=7)
        axes[1].set(ylabel=r"$\ell \, C_\ell$")
        axes[1].yaxis.set_ticklabels([])

        # --- panel 2: relative PPD/obs difference (zoomed to ±1) ---
        axes[2].fill_between(x, lo_rel, hi_rel, alpha=0.3, color="tab:orange")
        axes[2].plot(x, mid_rel, color="tab:orange", lw=1.5)
        if n_traces > 0:
            axes[2].plot(
                x, cls_rel_samples[: min(n_traces, cls_rel_samples.shape[0])].T,
                color="tab:orange", alpha=0.2, lw=0.5,
            )
        axes[2].axhline(0, color="k", lw=1.5)
        axes[2].set(
            ylabel=r"$(C_\ell^\mathrm{PPD} - C_\ell^\mathrm{obs})\,/\,C_\ell^\mathrm{obs}$",
            ylim=(-1, 1),
        )

        # --- panel 3: significance / pull ---
        axes[3].plot(x, sig_obs, color="tab:orange", lw=1.0)
        axes[3].axhline( 0, color="k",          lw=1.2)
        axes[3].axhline( 2, color="tab:red",    lw=0.8, ls="--")
        axes[3].axhline(-2, color="tab:red",    lw=0.8, ls="--")
        axes[3].axhline( 1, color="tab:orange", lw=0.6, ls=":")
        axes[3].axhline(-1, color="tab:orange", lw=0.6, ls=":")
        axes[3].set(
            ylabel=r"$(C_\ell^\mathrm{obs} - \mathrm{med}_\mathrm{PPD})\,/\,\sigma_\mathrm{PPD}$",
            ylim=(-3, 3),
        )

        # --- segment boundaries ---
        for i_pair in range(1, n_pairs):
            xb = i_pair * n_ell_per_pair
            for a in axes:
                a.axvline(xb, color="gray", lw=0.6, ls="--", alpha=0.5)

        axes[3].set_xlabel(x_label)

        # --- pair labels via secondary top axis ---
        _ax_top = axes[0].twiny()
        _ax_top.set_xlim(0, n_dims)
        _ax_top.set_xticks([(k + 0.5) * n_ell_per_pair for k in range(n_pairs)])
        _ax_top.set_xticklabels(_pair_labels, fontsize=8, color="dimgray")
        _ax_top.tick_params(axis="x", which="both", length=0, pad=2)
        _ax_top.spines["top"].set_visible(False)
        _ax_top.yaxis.set_visible(False)

        plot_file = os.path.join(self.out_dir, f"{self.obs_label}_{file_label}_marginals.png")
        LOGGER.info(f"Saving Cls marginals plot to {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=100)
