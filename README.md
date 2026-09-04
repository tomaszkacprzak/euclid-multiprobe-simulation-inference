# multiprobe-simulation-inference
[![arXiv](https://img.shields.io/badge/arXiv-2511.04681-b31b1b.svg)](https://arxiv.org/abs/2511.04681)

Collection of inference methods to go from arbitrary summary statistics (neural network, peaks, power spectrum, ...) to posterior parameter constraints. Inference and neural density estimation methods include:

- **Normalizing Flows:** Conditional implementation from [`FlowConductor`](https://github.com/FabricioArendTorres/FlowConductor)
- **Conditional Flow Matching (CFM):** PyTorch conditional continuous flows with ODE integration provided by `torchdiffeq`.
- **Gaussian Mixture Models:** As a simpler baseline neural density estimator.
- **Gaussian Process Approximate Bayesian Computation:** As an alternative to standard SBI methods [[Fluri et al. 2021](https://arxiv.org/abs/2107.09002)]

![](data/figures/example_posterior_small.png)

## Installation

Requires Python >= 3.8. Backend dependencies are optional: PyTorch is used by normalizing flows, CFM uses both PyTorch and `torchdiffeq`, and Gaussian mixture models use TensorFlow >= 2.0/TensorFlow Probability.

**Main dependencies:**
- [`euclid-multiprobe-simulation-forward-model`](https://github.com/tomaszkacprzak/euclid-multiprobe-simulation-forward-model/) for utilities and data loading
- [`euclid-deep-lss`](https://github.com/tomaszkacprzak/euclid-deep-lss/) for neural network summary statistics preprocessing

**Step 1: Install companion packages from GitHub**
```bash
# Install euclid-multiprobe-simulation-forward-model
pip install git+https://github.com/tomaszkacprzak/euclid-multiprobe-simulation-forward-model.git

# Install euclid-deep-lss
pip install git+https://github.com/tomaszkacprzak/euclid-deep-lss.git
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

*To use the conditional flow-matching likelihood*:
```bash
pip install -e .[cfm]
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
- `msi/flow_matching` - Conditional flow-matching (CFM) likelihood using PyTorch and `torchdiffeq`
- `msi/gaussian_mixture` - Gaussian mixture model implementation using TensorFlow Probability
- `msi/likelihood_registry.py` - Lazy likelihood selection without importing unselected optional backends
- `msi/utils` - MCMC sampling, preprocessing, diagnostics, and visualization utilities
- `msi/likelihood_base.py` - Base class for likelihood implementations

### `configs`
Configuration files for inference settings and hyperparameters.

### `data`
Stored chains from DES Y3 analyses and figures.

## Selecting a likelihood

Likelihood implementations can be selected by their registry name (`flow`, `gmm`, or `cfm`). The registry imports only the selected implementation, so CFM's `torchdiffeq` dependency is not required when using another likelihood:

```python
from msi.likelihood_registry import get_likelihood_class

Likelihood = get_likelihood_class("cfm")
model = Likelihood(
    params=["parameter_1", "parameter_2"],
    feature_dim=2,
    context_dim=2,
    model_type="mlp",
)
```

The CFM class can also be imported directly after installing the `cfm` extra:

```python
from msi.flow_matching import ConditionalFlowMatchingLikelihood
```

### `notebooks`
Notebooks for simulation-based inference via neural likelihood estimation and MCMC sampling. 

## Apps

Legacy apps can be launched via the script directly:

- `run_inference.py ...` - trains a normalizing flow for likelihood conditional density.

- `run_mcmc_for_coverage_tests.py ...` - run chains for coverage tests.

Euclid apps have a unified command line interface. After installing the package, run:

- `euclid-deeplss-inference inference ...` - trains or loads a likelihood model and runs the inference workflow. Prediction inputs used by this workflow are HDF5 files (`.h5`).

- `euclid-deeplss-inference coverage ...` - samples held-out simulations for posterior coverage tests.

Likelihood evaluation is currently part of these workflows; there is no supported standalone `likelihood` command or standalone input-file schema.



## Companion Repositories
- Forward modeling: [`euclid-multiprobe-simulation-forward-model`](https://github.com/tomaszkacprzak/euclid-multiprobe-simulation-forward-model/)
- Informative map-level neural summary statistics: [`euclid-deep-lss`](https://github.com/tomaszkacprzak/euclid-deep-lss/)
