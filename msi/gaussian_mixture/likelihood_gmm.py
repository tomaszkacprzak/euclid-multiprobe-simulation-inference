# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2023
Author: Arne Thomsen

Adapted from https://github.com/tomaszkacprzak/deep_lss/blob/main/deep_lss/networks/likemdn.py by Tomasz Kacprzak
"""

import numpy as np
import tensorflow as tf
import os, warnings, pickle

from sklearn.preprocessing import RobustScaler, MinMaxScaler

from msi.utils import mcmc
from msi.likelihood_base import LikelihoodBase
from msi.gaussian_mixture import architecture
from msi.gaussian_mixture.keras import EpochProgressCallback
from msfm.utils import logger, files, prior

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class LikelihoodGMM(tf.keras.Sequential, LikelihoodBase):
    """Conditional Gaussian Mixture Model (GMM) implementing a likelihood function p(x|theta), where x is some summary
    statistic vector and theta a vector of cosmological/astrophysical parameters to be constrained.

    Code adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/MixtureNormal#methods_2
    """

    model_name = "likelihood_gmm"

    def __init__(
        self,
        params,
        conf=None,
        layers=None,
        out_dir=None,
        label=None,
        load_existing=True,
        floatx=tf.float32,
        prefix="",
        suffix="",
    ):
        """
        Initialize the LikelihoodGMM object.

        Args:
            params (list): The cosmological and astrophysical parameters to be constrained. Note that the default
                architecture makes the assumption that the summary statistic has the same dimensionality as the number
                of parameters.
            conf (str, optional): The configuration file path. Defaults to None, then the default is loaded.
            layers (list, optional): List of layers making up the Gaussian mixture model within tf.keras.Sequential.
                Defaults to None, then a standard network architecture is loaded.
            out_dir (str, optional): The output directory path. Defaults to None, then no output is saved.
            label (str, optional): The label used in the saved filenames. Defaults to None.
            load_existing (bool, optional): Whether to load a model from disk if it exists. Defaults to True.
            floatx (tf.dtype, optional): The default float type. Defaults to tf.float32.
        """

        self.params = params
        self.conf = files.load_config(conf)
        self.floatx = floatx

        self.out_dir = out_dir
        self.label = label
        self.prefix = prefix
        self.suffix = suffix
        self._setup_dirs(".tf")

        if layers is None:
            LOGGER.warning(f"Assuming that the feature/summary dimension is equal to the context/parameter dimension")
            LOGGER.info(f"Using the default Gaussian mixture model")
            layers = architecture.get_gmm_layers(len(params), len(params))

        super(LikelihoodGMM, self).__init__(layers=layers)

        if load_existing:
            try:
                self.load()
            except tf.errors.NotFoundError:
                LOGGER.warning(f"Could not load the model from {self.model_file}")
        else:
            LOGGER.info(f"Initializing fresh weights")

    # training ########################################################################################################

    def fit(
        self,
        x,
        theta,
        n_epochs=1000,
        batch_size=10000,
        vali_split=0.1,
        # optimizer
        learning_rate=1e-3,
        weight_decay=0.0,
        clip_by_global_norm=1.0,
        # learning rate scheduler
        scheduler_kwargs=None,
        # early stopping
        n_patience_epochs=10,
        min_delta=1e-3,
        fit_kwargs={},
        save_model=True,
    ):
        """
        Fits the likelihood GMMM model to the given data and saves the resulting model.

        Args:
            x (torch.Tensor): The input features (summary statistics).
            theta (torch.Tensor): The input context (cosmological parameters).
            n_epochs (int, optional): The number of epochs to train for. Defaults to 100.
            batch_size (int, optional): The batch size for training and validation. Defaults to 1024.
            vali_split (float, optional): The validation split ratio. The validation set is used for early
                stopping. Defaults to 0.1.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.0.
            clip_by_global_norm (float, optional): The maximum gradient norm for gradient clipping. Defaults to
                100.0. When None, no clipping is applied.
            learning_rate_min (float, optional): The minimum learning rate for the scheduler. Defaults to 1e-6.
                When None, no scheduler is used.
            n_patience_epochs (int, optional): The number of epochs to wait before early stopping. Defaults to 10.
            min_delta (float, optional): The minimum change in validation loss to consider as improvement for
                early stopping. Defaults to 0.05.
            save_model (bool, optional): Whether to save the model after training. Defaults to True.
        """

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, decay=weight_decay, global_clipnorm=clip_by_global_norm
        )

        # maximize the log likelihood of x ~ p(x|theta)
        self.compile(optimizer=optimizer, loss=lambda x, model: -model.log_prob(x))

        # check dimensions
        assert (
            x.ndim == 2 and theta.ndim == 2
        ), "Something is wrong with the x and theta array dimensions, they should be =2 for training"
        assert x.shape[0] == theta.shape[0], "x and theta should have the same number of points"

        # preprocessing
        self.set_scalers(x, theta)
        x = self.scale_forward_x(x)
        theta = self.scale_forward_theta(theta)

        callbacks = []

        # learning rate scheduler
        if scheduler_kwargs is not None:
            LOGGER.info(f"Using a ReduceLROnPlateau learning rate scheduler")
            scheduler_kwargs.setdefault("min_lr", 1e-6)
            scheduler_kwargs.setdefault("factor", 0.75)
            scheduler_kwargs.setdefault("patience", 20)
            scheduler_kwargs.setdefault("cooldown", 5)
            scheduler_kwargs.setdefault("min_delta", 1e-4)

            callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", verbose=0, **scheduler_kwargs)
            callbacks.append(callback_reduce_lr)

        # early stopping
        if n_patience_epochs is not None:
            callback_early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=min_delta,
                patience=n_patience_epochs,
                verbose=1,
                restore_best_weights=True,
            )
            callbacks.append(callback_early_stop)

        # verbosity
        callback_verbose = EpochProgressCallback(n_epochs)
        callbacks.append(callback_verbose)

        history = super().fit(
            x=theta,
            y=x,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_split=vali_split,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
            **fit_kwargs,
        )

        self._plot_epochs(history.history["loss"], history.history["val_loss"])
        if save_model:
            self.save()

    # preprocessing ###################################################################################################

    def set_scalers(self, x, theta):
        """Fit the scalers to the data. This is very important since there are no normalization layers in the model"""

        eps = 1e-5
        self.scaler_x = MinMaxScaler(feature_range=(eps, 1 - eps))
        self.scaler_theta = RobustScaler()
        self.scaler_x.fit(x)
        self.scaler_theta.fit(theta)

        LOGGER.info(f"Fitted the x and y scalers")

    def _scale(self, inputs, transform):
        """Handle dimensions > 2 and arrays/tensors"""

        # arrays
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 2:
                outputs = transform(inputs)
            elif inputs.ndim > 2:
                n_features = inputs.shape[-1]
                inputs_shape = inputs.shape

                inputs = inputs.reshape(-1, n_features)
                outputs = transform(inputs)
                outputs = outputs.reshape(inputs_shape)
            else:
                raise ValueError

        # tensors
        elif isinstance(inputs, tf.Tensor):
            if len(inputs.shape) == 2:
                outputs = transform(inputs)
            elif len(inputs.shape) > 2:
                n_features = inputs.shape[-1]
                inputs_shape = inputs.shape

                inputs = tf.reshape(inputs, shape=(-1, n_features))
                outputs = transform(inputs)
                outputs = tf.reshape(outputs, shape=inputs_shape)
            else:
                raise ValueError

        else:
            raise ValueError(f"Input must either be a numpy array or a TensorFlow tensor")

        return outputs

    def scale_forward_x(self, x):
        return self._scale(x, self.scaler_x.transform)

    def scale_inverse_x(self, x):
        return self._scale(x, self.scaler_x.inverse_transform)

    def scale_forward_theta(self, theta):
        return self._scale(theta, self.scaler_theta.transform)

    def scale_inverse_y(self, theta):
        return self._scale(theta, self.scaler_theta.inverse_transform)

    # likelihood ######################################################################################################

    def sample_likelihood(self, theta, n_samples=1000, batch_size=10000, return_numpy=True):
        """
        Sample from the likelihood distribution p(x|theta). This can be done directly from the model and doesn't need
        an MCMC sampler.

        Args:
            theta (Union[tf.tensor, np.ndarray]): The theta values to condition on. This array/tensor can have more
                than one dimension.
            n_samples (int, optional): The number of samples to generate for each condition. Defaults to 1000.
            batch_size (int, optional): The batch size for generating samples. Defaults to None.
            return_numpy (bool, optional): Whether to return the samples as a numpy array instead of a tensor.
                Defaults to True.
            out_dir (str, optional): The directory to save the samples. Defaults to None.
            label (str, optional): The label for the saved samples. Defaults to None.

        Returns:
            tf.Tensor or numpy.ndarray: The generated samples of the same shape as theta_obs, except for an
                additional axis of length n_samples.
        """

        assert self.scaler_x is not None, "the GMM has not been fit yet"
        assert self.scaler_theta is not None, "the GMM has not been fit yet"
        assert return_numpy, "returning TensorFlow tensors is not supported"

        n_batches = np.ceil(len(theta) / batch_size)

        x_samples = []
        for theta_batch in LOGGER.progressbar(
            np.array_split(theta, n_batches), desc=f"drawing samples with batch_size={batch_size}", at_level="info"
        ):
            theta_batch = self.scale_forward_theta(theta_batch)
            x_batch = tf.squeeze(self(theta_batch).sample(sample_shape=n_samples))

            # rescale x
            x_batch = self.scale_inverse_x(x_batch)

            x_samples.append(x_batch)

        x_samples = tf.concat(x_samples, axis=1)

        # to be consistent with FlowConductor, shape (n_cosmos, n_samples, n_summaries)
        x_samples = tf.transpose(x_samples, perm=(1, 0, 2))

        if return_numpy:
            x_samples = x_samples.numpy()

        return x_samples

    def log_likelihood(self, x, theta, return_numpy=False):
        """log likelihood p(x|theta) in numpy, meant to be used together with an MCMC.

        Args:
            x (np.ndarray): Conditioning of shape (n_samples, n_conditions).
            y (np.ndarray): Features to be modelled by the Gaussians of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape (n_samples,) containing the log likelihoods.
        """

        x = self.scale_forward_x(x)
        theta = self.scale_forward_theta(theta)

        # arrays
        if isinstance(x, np.ndarray) and isinstance(theta, np.ndarray):
            if x.ndim > 2 or theta.ndim > 2:
                assert x.shape[:-1] == theta.shape[:-1]
                out_shape = x.shape[:-1]

                x_features = x.shape[-1]
                theta_features = theta.shape[-1]

                x = x.reshape(-1, x_features)
                theta = theta.reshape(-1, theta_features)

                log_like = self(theta).log_prob(x).numpy()
                log_like = log_like.reshape(out_shape)

            else:
                log_like = self(theta).log_prob(x).numpy()

        # tensors
        elif isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor):
            if len(x.shape) > 2 and len(theta.shape) > 2:
                assert x.shape[:-1] == theta.shape[:-1]
                out_shape = x.shape[:-1]

                x_features = x.shape[-1]
                theta_features = theta.shape[-1]

                x = tf.reshape(x, shape=(-1, x_features))
                theta = tf.reshape(theta, shape=(-1, theta_features))

                log_like = self(theta).log_prob(x)
                log_like = tf.reshape(log_like, shape=out_shape)

                if return_numpy:
                    log_like = log_like.numpy()

            else:
                raise NotImplementedError

        else:
            raise ValueError(f"Input must either be a numpy array or a TensorFlow tensor")

        return log_like

    # posterior #######################################################################################################

    def sample_posterior(self, x_obs, n_samples=512000, n_walkers=1024, n_burnin_steps=100, label=None):
        """
        Sample from the posterior distribution p(theta|x) using the likelihood learned by the model and the flat
        analysis prior. The sampling is done using the emcee library, which runs on the CPU and in numpy.

        Args:
            x_obs (np.ndarray): The observation to condition the posterior on. It must have shape (n_features,) or
                (1, n_features).
            n_samples (int, optional): The number of samples to generate. Defaults to 512000.
            n_walkers (int, optional): The number of walkers in the MCMC chain. Defaults to 1024.
            n_burnin_steps (int, optional): The number of burn-in steps in the MCMC chain. Defaults to 100.
            label (str, optional): Additional label for the saved chain, for example to designate different
                observations. Defaults to None.

        Returns:
            array-like: The generated samples from the likelihood flow model.
        """

        x_obs = tf.cast(x_obs, dtype=self.floatx)
        if x_obs.ndim == 1:
            x_obs = tf.expand_dims(x_obs, axis=0)
        if x_obs.shape[0] == 1:
            LOGGER.info(f"Sampling the posterior from a single observation")
        else:
            LOGGER.info(f"Sampling the posterior from multiple observations")

        chain = mcmc.run_emcee(
            lambda theta_walkers: self._mcmc_log_posterior(theta_walkers, x_obs),
            self.params,
            conf=self.conf,
            out_dir=self.model_dir,
            label=label,
            n_walkers=n_walkers,
            n_steps=int(np.ceil(n_samples / n_walkers)),
            n_burnin_steps=n_burnin_steps,
        )

        # there can be more samples than requested due the walkers
        chain = chain[:n_samples]

        return chain

    def _single_log_posterior(self, theta_walkers, x_obs):
        """theta_walkers.shape = (n_walkers, theta_dim)"""
        assert x_obs.shape[0] == 1

        # evaluate the mixture model
        log_prob = self.log_likelihood(x_obs, theta_walkers, return_numpy=True)

        # enforce the prior
        log_prob = prior.log_posterior(theta_walkers, log_prob, params=self.params, conf=self.conf)

        return log_prob

    def _mcmc_log_posterior(self, theta_walkers, x_obs):
        assert x_obs.ndim == 2

        if x_obs.shape[0] == 1:
            log_prob = self._single_log_posterior(theta_walkers, x_obs)
        else:
            log_prob = np.zeros((theta_walkers.shape[0]))
            for x in x_obs:
                x = tf.expand_dims(x, axis=0)
                log_prob += self._single_log_posterior(theta_walkers, x)

        return log_prob

    # utils ###########################################################################################################

    def save(self):
        """Save the weights of the model to disk."""

        if self.model_dir is not None:
            self._save_scalers()
            self.save_weights(self.model_file)
            LOGGER.info(f"Saved the model to {self.model_file}")
        else:
            LOGGER.warning(f"Could not save the model, no model directory specified")

    def _save_scalers(self):
        with open(os.path.join(self.model_dir, "scalers.pkl"), "wb") as f:
            pickle.dump([self.scaler_x, self.scaler_theta], f)

    def load(self):
        """Load the weights of the model from disk."""

        if self.model_dir is not None:
            self.load_weights(self.model_file)

            with open(os.path.join(self.model_dir, "scalers.pkl"), "rb") as f:
                self.scaler_x, self.scaler_theta = pickle.load(f)

            LOGGER.info(f"Loaded the model from {self.model_dir}")
