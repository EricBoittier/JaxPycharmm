from pathlib import Path

import pytest
import numpy as np

from physnetjax.data.data import (
    assert_dataset_size,
    prepare_datasets,
    prepare_multiple_datasets,
)
import jax

data_key, train_key = jax.random.split(jax.random.PRNGKey(42), 2)


from physnetjax.directories import MAIN_PATH


def test_prepare_multiple_datasets():
    """Test the prepare_multiple_datasets function with mock datasets."""
    mock_key = data_key
    mock_train_size = 10
    mock_valid_size = 5
    mock_filename = [
        MAIN_PATH / Path("data/basepairs/at_prod.npz"),
        MAIN_PATH / Path("data/basepairs/at_reag.npz"),
        MAIN_PATH / Path("data/basepairs/at_retune.npz"),
        MAIN_PATH / Path("data/basepairs/rattle_neb_at.npz"),
        MAIN_PATH / Path("data/basepairs/rattle_neb_gc.npz"),
    ]
    print(mock_filename)
    # Call function with mocked data
    data, keys, ntrain, nvalid = prepare_multiple_datasets(
        mock_key,
        train_size=mock_train_size,
        valid_size=mock_valid_size,
        filename=mock_filename,
        clip_esp=False,
        natoms=30,
        clean=False,
        verbose=False,
    )

    # Assertions to check the function operates as expected
    assert len(data) == len(keys), "Keys and Data lengths should match!"
    assert mock_valid_size + mock_train_size <= len(
        data[0]
    ), "Dataset should have enough samples."
    assert "R" in keys, "Keys should include 'R' (coordinates)."
    assert "Z" in keys, "Keys should include 'Z' (atomic numbers)."
    assert "E" in keys, "Keys should include 'E' (energies)."


def test_prepare_datasets():
    """Test the prepare_datasets function for consistent dataset preparation."""
    mock_train_size = 20
    mock_valid_size = 10
    mock_filename = [
        MAIN_PATH / Path("data/basepairs/at_prod.npz"),
        MAIN_PATH / Path("data/basepairs/at_reag.npz"),
        MAIN_PATH / Path("data/basepairs/at_retune.npz"),
        MAIN_PATH / Path("data/basepairs/rattle_neb_at.npz"),
        MAIN_PATH / Path("data/basepairs/rattle_neb_gc.npz"),
    ]

    # Call the function (mock actual loading behavior if necessary for larger datasets)
    train_data, valid_data = prepare_datasets(
        data_key,
        train_size=mock_train_size,
        valid_size=mock_valid_size,
        natoms=30,
        files=mock_filename,
        verbose=True,
    )

    # Assertions about train and validation datasets
    for k in ["R", "Z", "E"]:
        assert k in train_data, f"Training data should include key {k}."
        assert k in valid_data, f"Validation data should include key {k}."
        assert len(train_data[k]) == mock_train_size, "Training dataset size mismatch."
        assert (
            len(valid_data[k]) == mock_valid_size
        ), "Validation dataset size mismatch."


def test_assert_dataset_size():
    """Test the assert_dataset_size function for handling size mismatches."""
    data = {
        "train": np.random.rand(50, 10),
        "validation": np.random.rand(20, 10),
    }

    # This should pass since dataset sizes are consistent
    try:
        assert_dataset_size(50 + 20, len(data["train"]), len(data["validation"]))
    except AssertionError:
        pytest.fail("Assertion failed unexpectedly on correct dataset size.")

    # This should fail due to size mismatch
    data["validation"] = np.random.rand(30, 10)
    with pytest.raises(AssertionError):
        assert_dataset_size(50 + 20, len(data["train"]), len(data["validation"]))
