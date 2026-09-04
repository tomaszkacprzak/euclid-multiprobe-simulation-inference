"""Shared deterministic dataset splitting helpers."""

import numpy as np


def validation_split_indices(n_examples, vali_split, *, seed=0, group_ids=None):
    """Return reproducible train/validation indices and serializable metadata.

    Grouped splits sort the unique group identifiers before assigning the final
    ``vali_split`` fraction of groups to validation. Ungrouped splits use a
    local PyTorch generator so their result is independent of global RNG state.
    """
    import torch

    if n_examples < 2:
        raise ValueError("A train/validation split requires at least two examples.")
    if not 0 < vali_split < 1:
        raise ValueError("vali_split must be strictly between zero and one.")

    if group_ids is None:
        order = torch.randperm(n_examples, generator=torch.Generator().manual_seed(seed)).numpy()
        split_at = int((1 - vali_split) * n_examples)
        train_indices, validation_indices = order[:split_at], order[split_at:]
        metadata = {"method": "seeded", "seed": int(seed), "vali_split": float(vali_split)}
    else:
        groups = np.asarray(group_ids)
        if groups.ndim != 1 or len(groups) != n_examples:
            raise ValueError("group_ids must be one-dimensional and aligned with the data.")
        unique_groups = np.unique(groups)
        split_at = int((1 - vali_split) * len(unique_groups))
        train_groups = unique_groups[:split_at]
        train_mask = np.isin(groups, train_groups)
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(~train_mask)
        metadata = {
            "method": "grouped",
            "vali_split": float(vali_split),
            "n_groups": int(len(unique_groups)),
        }

    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("vali_split produced an empty training or validation partition.")
    return train_indices, validation_indices, metadata
