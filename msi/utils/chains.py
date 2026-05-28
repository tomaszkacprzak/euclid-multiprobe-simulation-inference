"""
Created July 2023
Author: Arne Thomsen

Utils to load and handle MCMC chains.

The official DES chains come from https://des.ncsa.illinois.edu/releases/y3a2/Y3key-products

Example header from chain_3x2pt_wcdm_SR_maglim.txt, which determines how the files are read out. See also
http://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/chains/read_chains_y3a2.py

cosmological_parameters--omega_m
cosmological_parameters--h0
cosmological_parameters--omega_b
cosmological_parameters--n_s
cosmological_parameters--a_s
cosmological_parameters--omnuh2
cosmological_parameters--w
shear_calibration_parameters--m1
shear_calibration_parameters--m2
shear_calibration_parameters--m3
shear_calibration_parameters--m4
wl_photoz_errors--bias_1
wl_photoz_errors--bias_2
wl_photoz_errors--bias_3
wl_photoz_errors--bias_4
lens_photoz_errors--bias_1
lens_photoz_errors--bias_2
lens_photoz_errors--bias_3
lens_photoz_errors--bias_4
lens_photoz_errors--width_1
lens_photoz_errors--width_2
lens_photoz_errors--width_3
lens_photoz_errors--width_4
bias_lens--b1
bias_lens--b2
bias_lens--b3
bias_lens--b4
intrinsic_alignment_parameters--a1
intrinsic_alignment_parameters--a2
intrinsic_alignment_parameters--alpha1
intrinsic_alignment_parameters--alpha2
intrinsic_alignment_parameters--bias_ta
COSMOLOGICAL_PARAMETERS--SIGMA_8
COSMOLOGICAL_PARAMETERS--SIGMA_12
DATA_VECTOR--2PT_CHI2
prior
like
post
weight
"""

import numpy as np
import os

from msi.utils import input_output
from msfm.utils import logger, parameters

LOGGER = logger.get_logger(__file__)

# translation between msfm/msi names and DES key project chains names
param_correspondence = {
    "Om": "omega_m",
    "s8": "SIGMA_8",
    "H0": "h0",
    "Ob": "omega_b",
    "ns": "n_s",
    "w0": "w",
    # only if a2 == 0 and bias_ta == 0
    "Aia": "a1",
    "n_Aia": "alpha1",
    "bg1": "b1",
    "bg2": "b2",
    "bg3": "b3",
    "bg4": "b4",
}


def load_des_y3_key_project_chain(params, probes="3x2pt", cosmo_model="wCDM", ia_model="nla"):
    """Load the parameters of interest in one of the chains from
    https://des.ncsa.illinois.edu/releases/y3a2/Y3key-products or
    https://des.ncsa.illinois.edu/releases/y3a2/Y3key-extensions

    Args:
        params (list): List of strings of the constrained cosmological parameters or list of such lists.
        probes (str, optional): The cosmological probes used, one of
            - "1x2pt": weak lensing only (https://arxiv.org/pdf/2105.13543.pdf)
            - "2x2pt": galaxy clustering and galaxy-galaxy lensing (https://arxiv.org/pdf/2105.13546.pdf)
            - "3x2pt": galaxy clustering and weak lensing (https://arxiv.org/pdf/2105.13549.pdf and
                https://arxiv.org/pdf/2207.05766.pdf)
        cosmo_model (str, optional): "wCDM" or "LambdaCDM. Defaults to "wCDM".
        ia_model (str, optional): "nla" or "tatt", determines the kind of chain to load. NOTE the output will always be
            in nla, since only this is supported by the msfm at the moment. Defaults to "nla".
    """
    assert probes in [
        "1x2pt",
        "2x2pt",
        "3x2pt",
    ], f"The probes argument {probes} has to either of '1x2pt', '2x2pt', '3x2pt'"

    assert cosmo_model in [
        "LambdaCDM",
        "wCDM",
    ], f"The cosmo_model argument {cosmo_model} has to either be 'LambdaCDM' or 'wCDM'"

    assert ia_model in ["nla", "tatt"], f"The ia_model argument {ia_model} has to either be 'nla' or 'tatt'"

    LOGGER.info(f"Loading DESy3 key project chain for probes={probes}, cosmo_model={cosmo_model}, ia_model={ia_model}")

    # set up the file
    conf = input_output.load_msi_config()
    chain_file = os.path.join(input_output.get_abs_dir_repo(), conf["files"]["chains"][cosmo_model][ia_model][probes])

    with open(chain_file, "r") as f:
        chain = []
        weight = []
        for i, line in enumerate(f):
            # the column names are stored in the first line
            if i == 0:
                params_header = line
                params_header = params_header.split()
                params_header = [param.split("--")[1] if "--" in param else param for param in params_header]

                # the parameter's column index within the .txt file
                i_params = [params_header.index(param_correspondence[param]) for param in params]

                # include NLA parameters, but the chain is actually TATT
                transform_tatt_to_nla = any([param in ["Aia", "n_Aia"] for param in params]) and any(
                    header in ["a2", "alpha2", "bias_ta"] for header in params_header
                )
                if transform_tatt_to_nla:
                    LOGGER.warning(f"Returning NLA parameters, even though the chain is TATT")
                    i_a2 = params_header.index("a2")
                    i_bta = params_header.index("bias_ta")

            # skip the file header
            if line.startswith("#"):
                pass
            else:
                columns = line.split()

                # restrict the samples to ones where TATT is close to NLA
                if transform_tatt_to_nla:
                    a2 = float(columns[i_a2])
                    bta = float(columns[i_bta])

                    # TODO are these reasonable values to be hardcoded?
                    if abs(a2) < 0.1 and bta < 0.1:
                        chain.append([float(columns[i_param]) for i_param in i_params])
                        weight.append(float(columns[-1]))

                # use all samples
                else:
                    chain.append([float(columns[i_param]) for i_param in i_params])
                    weight.append(float(columns[-1]))

    chain = np.asarray(chain)
    weight = np.asarray(weight)

    # re-normalize
    weight /= np.sum(weight)

    LOGGER.info(f"Loaded DESy3 key project chain containing {len(chain)} samples")

    return chain, weight


_probe_to_des = {
    "lensing":    ("1x2pt", "tatt"),
    "clustering": ("2x2pt", "tatt"),
    "combined":   ("3x2pt", "nla"),
    "2x2pt":      ("3x2pt", "nla"),
    "cross":      ("3x2pt", "nla"),
}


def load_and_shift_des_chain(test_params, probe, msfm_conf):
    """Load a DES Y3 key project chain and shift it to the fiducial cosmology
    using the same MAP→fiducial blinding as load_shifted_chain in the notebook.

    Args:
        test_params (list): Parameter names to load (subset of param_correspondence keys is used).
        probe (str): Probe name matching a key in _probe_to_des.
        msfm_conf (dict): msfm config (for fiducial values via parameters.get_fiducials).

    Returns:
        chain (np.ndarray or None): Shifted samples, shape (n, len(available)).
        weights (np.ndarray or None): Normalised importance weights.
        foms (dict): {(p1, p2): int} weighted FoM for each parameter pair.
        available (list): Subset of test_params that exist in the DES chain.
    """
    des_probes, des_ia = _probe_to_des[probe]
    available = [p for p in test_params if p in param_correspondence]
    if not available:
        return None, None, {}, []

    chain, weights = load_des_y3_key_project_chain(available, des_probes, "wCDM", des_ia)

    # MAP estimate: weighted mean of top-1% samples (mirrors find_MAP in plotting.py)
    w_threshold = np.percentile(weights, 99)
    high_idx = weights >= w_threshold
    des_map = np.average(chain[high_idx], weights=weights[high_idx], axis=0)

    chain -= des_map
    chain += np.array(parameters.get_fiducials(available, msfm_conf))

    # weighted FoM: inverse sqrt of the 2D covariance determinant (eq. 17 in arXiv:2405.10881)
    foms = {}
    for i, p1 in enumerate(available):
        for j, p2 in enumerate(available):
            if i > j:
                idx = [available.index(p1), available.index(p2)]
                cov = np.cov(chain[:, idx].T, aweights=weights)
                foms[(p1, p2)] = int(np.linalg.det(cov) ** -0.5)
                LOGGER.info(f"DES FoM_({p1},{p2}) = {foms[(p1, p2)]}")

    return chain, weights, foms, available
