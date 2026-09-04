import numpy as np

from msi.utils.dataset_split import validation_split_indices


def test_grouped_validation_split_is_deterministic_and_disjoint():
    groups = np.array([3, 1, 3, 2, 1, 4, 2, 4])

    train, validation, metadata = validation_split_indices(8, 0.5, seed=999, group_ids=groups)

    assert np.array_equal(train, [1, 3, 4, 6])
    assert np.array_equal(validation, [0, 2, 5, 7])
    assert set(groups[train]).isdisjoint(groups[validation])
    assert metadata == {"method": "grouped", "vali_split": 0.5, "n_groups": 4}


def test_seeded_validation_split_is_reproducible():
    first = validation_split_indices(10, 0.2, seed=17)
    second = validation_split_indices(10, 0.2, seed=17)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
