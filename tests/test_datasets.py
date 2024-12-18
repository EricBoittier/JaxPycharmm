import numpy as np

from physnetjax.data.datasets import (
    MAX_N_ATOMS,
    NUM_ESP_CLIP,
)
from physnetjax.data.datasets import process_dataset, process_in_memory, prepare_multiple_datasets, process_dataset_key, \
    clip_or_default_data
from physnetjax.utils.enums import MolecularData


def test_process_dataset(tmp_path):
    """Test the process_dataset function with mock NPZ files."""
    # Create mock NPZ files
    mock_file1 = tmp_path / "mock1.npz"
    np.savez(
        mock_file1,
        atomic_numbers=np.array([1, 2, 3]),
        coordinates=np.random.rand(3, 3),
        forces=np.random.rand(3, 3),
        energy=np.random.rand(),
    )

    mock_file2 = tmp_path / "mock2.npz"
    np.savez(
        mock_file2,
        atomic_numbers=np.array([6, 8]),
        coordinates=np.random.rand(2, 3),
        forces=np.random.rand(2, 3),
        energy=np.random.rand(),
    )

    # Call the function
    result = process_dataset([mock_file1, mock_file2], MAX_N_ATOMS=MAX_N_ATOMS)

    assert result is not None, "Result should not be None."

    Z = MolecularData.ATOMIC_NUMBERS
    R = MolecularData.COORDINATES
    # Assertions on output structure and content
    assert Z in result.keys(), f"Result should contain 'atomic_numbers'. Got: {result.keys()}"
    assert R in result.keys(), "Result should contain 'coordinates'. Got: {result.keys()}"
    assert result[Z].shape[0] == MAX_N_ATOMS, \
        f"Atomic numbers should be padded to {MAX_N_ATOMS}. Got: {result[Z]}"
    assert result[R].shape[1] == MAX_N_ATOMS, \
        f"Coordinates should be padded to {MAX_N_ATOMS}. Got: {result[R]}"
    # assert "dipole_moment" in result, "Result should contain 'dipole_moment'."
    # assert "charges" in result, "Result should contain 'charges'."


def test_process_in_memory():
    """Test process_in_memory with mock data dictionaries."""
    mock_data = [
        {
            "atomic_numbers": np.array([1, 2, 3]),
            "coordinates": np.random.rand(3, 3),
            "energy": np.random.rand(),
            "forces": np.random.rand(3, 3),
        },
        {
            "atomic_numbers": np.array([6, 8]),
            "coordinates": np.random.rand(2, 3),
            "energy": np.random.rand(),
            "forces": np.random.rand(2, 3),
        },
    ]

    # Call function with mock data
    result = process_in_memory(mock_data, max_atoms=MAX_N_ATOMS)

    # Assertions on keys and structure
    assert "Z" in result, "Processed data should contain key 'Z' for atomic numbers."
    assert "R" in result, "Processed data should contain key 'R' for coordinates."
    assert result["Z"].shape[0] == len(mock_data), \
        "Number of samples should match the input data length."
    assert result["R"].shape[1] == MAX_N_ATOMS, \
        f"Coordinates should be padded to {MAX_N_ATOMS}."


def test_process_dataset_key():
    """Test process_dataset_key with mock datasets."""
    mock_data = [
        {"energy": np.random.rand(10)},
        {"energy": np.random.rand(5)},
    ]
    data_key = "energy"
    not_failed = np.arange(15)  # Mock all rows as valid

    # Call the function
    result = process_dataset_key(
        data_key,
        datasets=mock_data,
        shape=(15,),
        natoms=0,
        not_failed=not_failed
    )

    # Assertions on the output
    assert result.shape == (15,), "The shape of the processed key should match the valid data size."
    assert np.all(result >= 0), "Energy values should be non-negative in mock datasets."


def test_clip_or_default_data():
    """Test clip_or_default_data for functionality with and without clipping."""
    mock_data = [
        {"esp": np.random.rand(150)},
        {"esp": np.random.rand(120)},
    ]
    data_key = "esp"
    not_failed = np.arange(270)  # Mock all rows as valid
    shape = (270,)

    # No clipping case
    result = clip_or_default_data(
        mock_data, data_key, not_failed, shape, clip=False
    )
    assert result.shape == (270,), "Result shape should match when clipping is disabled."

    # Clipping case
    result_clipped = clip_or_default_data(
        mock_data, data_key, not_failed, shape, clip=True
    )
    assert result_clipped.shape == (270, NUM_ESP_CLIP), \
        f"Result should be clipped to {NUM_ESP_CLIP}."


def test_prepare_multiple_datasets():
    """Test prepare_multiple_datasets with random mock data."""
    mock_key = "random_key"
    mock_train_size = 5
    mock_valid_size = 10
    mock_files = ["mock_file1.npz", "mock_file2.npz"]

    def mock_load(file):
        """Mock np.load functionality to generate realistic random data."""
        return {
            "R": np.random.rand(15, MAX_N_ATOMS, 3),
            "Z": np.random.randint(1, 10, size=(15, MAX_N_ATOMS)),
            "F": np.random.rand(15, MAX_N_ATOMS, 3),
            "E": np.random.rand(15, 1),
        }

    # Mock np.load behavior
    np.load = mock_load

    # Call function with mocked inputs
    data, keys = prepare_multiple_datasets(
        key=mock_key,
        train_size=mock_train_size,
        valid_size=mock_valid_size,
        filename=mock_files,
        natoms=MAX_N_ATOMS,
    )


    # Assertions on results
    assert len(data) == len(keys), "Keys and Data lengths should match."
    assert mock_valid_size + mock_train_size <= len(data[0]), "Dataset sizes should match input requirements."
    assert "R" in keys, "Keys should include 'R' for coordinates."
    assert "Z" in keys, "Keys should include 'Z' for atomic numbers."
    assert "E" in keys, "Keys should include 'E' for energy."