import pytest

import jax
import jax.numpy as jnp
from pathlib import Path

from physnetjax.data.data import prepare_multiple_datasets, prepare_datasets
from physnetjax.directories import MAIN_PATH
from physnetjax.data.read_npz import process_npz_file

# fixed data for testing
@pytest.fixture
def test_process_dataset():
    data_key = jax.random.PRNGKey(42)
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

    return mock_filename


def test_process_npz_file():
    files = test_process_dataset()
    process_npz_file(files[0])

