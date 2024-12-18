import pytest
import numpy as np
from pathlib import Path
from physnetjax.data.datasets import (
    process_dataset,
    process_in_memory,
    prepare_multiple_datasets,
    process_dataset_key,
)


@pytest.fixture
def mock_npz_files(tmp_path):
    # Create Some Mock .npz Files
    file1 = tmp_path / "data1.npz"
    np.savez(
        file1,
        Z=np.array([1, 2, 3]),
        R=np.array([[0.0, 0.0, 0.0]]),
        F=np.zeros((1, 3)),
        E=np.array([0.0]),
    )
    file2 = tmp_path / "data2.npz"
    np.savez(
        file2,
        Z=np.array([6, 8]),
        R=np.array([[0.0, 0.1, 0.2]]),
        F=np.zeros((1, 3)),
        E=np.array([-1.0]),
    )
    return [file1, file2]


@pytest.fixture
def mock_data_dict():
    return [
        {
            "atomic_numbers": np.array([1, 2, 3]),
            "coordinates": np.zeros((3, 3)),
            "energy": -1.0,
        },
        {
            "atomic_numbers": np.array([6, 8]),
            "coordinates": np.zeros((2, 3)),
            "energy": -2.0,
        },
    ]


def test_process_dataset(mock_npz_files):
    result = process_dataset(mock_npz_files)
    assert "atomic_numbers" in result
    assert len(result["atomic_numbers"]) > 0


def test_process_in_memory(mock_data_dict):
    result = process_in_memory(mock_data_dict)
    assert "Z" in result
    assert result["Z"].shape[0] == len(mock_data_dict)


def test_prepare_multiple_datasets(mock_npz_files):
    data, keys = prepare_multiple_datasets(
        "random_key", train_size=2, filename=mock_npz_files
    )
    assert len(data) > 0
    assert "R" in keys
