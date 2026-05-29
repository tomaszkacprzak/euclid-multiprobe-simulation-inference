# multiprobe-simulation-inference
[![arXiv](https://img.shields.io/badge/arXiv-2511.04681-b31b1b.svg)](https://arxiv.org/abs/2511.04681)

Collection of inference methods to go from arbitrary summary statistics (neural network, peaks, power spectrum, ...) to posterior parameter constraints. Inference and neural density estimation methods include:

- **Normalizing Flows:** Conditional implementation from [`FlowConductor`](https://github.com/FabricioArendTorres/FlowConductor)
- **Gaussian Mixture Models:** As a simpler baseline neural density estimator.
- **Gaussian Process Approximate Bayesian Computation:** As an alternative to standard SBI methods [[Fluri et al. 2021](https://arxiv.org/abs/2107.09002)]

![](data/figures/example_posterior_small.png)

## Installation

Requires Python >= 3.8, PyTorch (for normalizing flows), and optionally TensorFlow >= 2.0/TensorFlow-Probability (for Gaussian mixture models).

**Main dependencies:**
- [`multiprobe-simulation-forward-model`](https://github.com/des-science/multiprobe-simulation-forward-model) for utilities and data loading
- [`y3-deep-lss`](https://github.com/des-science/y3-deep-lss) for neural network summary statistics preprocessing

**Step 1: Install companion packages from GitHub**
```bash
# Install multiprobe-simulation-forward-model
pip install git+https://github.com/des-science/multiprobe-simulation-forward-model.git

# Install y3-deep-lss
pip install git+https://github.com/des-science/y3-deep-lss.git
```

**Step 2: Install this package**

*On HPC clusters with pre-installed PyTorch* (recommended):
```bash
pip install -e .
```

*On systems without PyTorch*:
```bash
pip install -e .[torch]
```

*To include TensorFlow for Gaussian mixture models*:
```bash
pip install -e .[torch,tf]
```

Use the first option when PyTorch is available via system modules (e.g., `module load pytorch`) to preserve optimized GPU configurations.

## Repository Structure

### `msi`
- `msi/apps` - Inference scripts for normalizing flow training and MCMC sampling
- `msi/flow_conductor` - Normalizing flow implementation using PyTorch and [`enflows`](https://github.com/VincentStimper/normalizing-flows)
- `msi/gaussian_mixture` - Gaussian mixture model implementation using TensorFlow Probability
- `msi/utils` - MCMC sampling, preprocessing, diagnostics, and visualization utilities
- `msi/likelihood_base.py` - Base class for likelihood implementations

### `configs`
Configuration files for inference settings and hyperparameters.

### `data`
Stored chains from DES Y3 analyses and figures.

### `notebooks`
Notebooks for simulation-based inference via neural likelihood estimation and MCMC sampling. 

## Companion Repositories
- Forward modeling: [`euclid-multiprobe-simulation-forward-model`](https://github.com/tomaszkacprzak/euclid-multiprobe-simulation-forward-model/)
- Informative map-level neural summary statistics: [`euclid-deep-lss`](https://github.com/tomaszkacprzak/euclid-deep-lss/)
