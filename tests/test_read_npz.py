import pytest

import jax
import jax.numpy as jnp
from pathlib import Path

from physnetjax.data.data import prepare_multiple_datasets, prepare_datasets
from physnetjax.data.datasets import process_dataset
from physnetjax.directories import MAIN_PATH
from physnetjax.data.read_npz import process_npz_file

# fixed data for testing
@pytest.fixture
def test_process_dataset():
    data_key = jax.random.PRNGKey(42)
    mock_key = data_key
    mock_train_size = 10
    mock_valid_size = 5
    mock_filename = list(MAIN_PATH.glob("data/basepairs/*.npz"))
    print(mock_filename)

    return mock_filename


def test_process_npz_file(test_process_dataset):

    # out, natoms = process_npz_file(test_process_dataset[0])
    # print(out, natoms)
    _ = process_dataset(test_process_dataset)
    print(_)

