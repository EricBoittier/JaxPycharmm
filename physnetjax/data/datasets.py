from pathlib import Path
from typing import Dict, List, Tuple, Union

import ase
import numpy as np
from ase.units import Bohr, Hartree, kcal
from numpy.typing import NDArray
from tqdm import tqdm

from physnetjax.utils.enums import KEY_TRANSLATION, MolecularData

# Constants
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr
MAX_N_ATOMS = 37
MAX_GRID_POINTS = 10000
BOHR_TO_ANGSTROM = 0.529177

from physnetjax.data.data import ATOM_ENERGIES_HARTREE

def process_dataset(
    files: List[Path], batch_index: int = 0
) -> Dict[MolecularData, NDArray]:
    """
    Process a batch of NPZ files and combine their data.

    Args:
        files: List of NPZ files to process
        batch_index: Index of the batch being processed

    Returns:
        Dictionary containing combined and processed data, keyed by MolecularData enum
    """
    # Initialize data collectors
    collected_data = {
        MolecularData.ATOMIC_NUMBERS: [],
        MolecularData.COORDINATES: [],
        MolecularData.QUADRUPOLE: [],
        MolecularData.ESP: [],
        MolecularData.ESP_GRID: [],
        MolecularData.CENTER_OF_MASS: [],
        MolecularData.FORCES: [],
        MolecularData.ENERGY: [],
        MolecularData.DIPOLE: [],
    }

    molecule_ids = []  # Keep track of molecule IDs separately

    for filepath in tqdm(files):
        result, n_atoms = process_npz_file(filepath)
        print(result, n_atoms)
        if result is not None and 3 < n_atoms < MAX_N_ATOMS:
            # Store molecule ID
            molecule_ids.append(str(filepath).split("/")[-2])

            # Add data to collectors
            for data_type in MolecularData:
                if data_type.value in result:
                    collected_data[data_type].append(result[data_type.value])

    # Pad all collected arrays to uniform size
    N = len(collected_data[MolecularData.ATOMIC_NUMBERS])

    # Process ESP grid sizes if present
    if collected_data[MolecularData.ESP]:
        N_grid = [_.shape[0] for _ in collected_data[MolecularData.ESP]]

    # Pad arrays
    processed_data = {}

    # Pad atomic numbers
    Z = [
        np.array([int(_) for _ in collected_data[MolecularData.ATOMIC_NUMBERS][i]])
        for i in range(N)
    ]
    processed_data[MolecularData.ATOMIC_NUMBERS] = np.array(
        [pad_atomic_numbers(Z[i], MAX_N_ATOMS) for i in range(N)]
    )

    # Pad coordinates
    processed_data[MolecularData.COORDINATES] = np.array(
        [
            pad_coordinates(collected_data[MolecularData.COORDINATES][i], MAX_N_ATOMS)
            for i in range(N)
        ]
    )

    # Pad forces
    if collected_data[MolecularData.FORCES]:
        processed_data[MolecularData.FORCES] = np.array(
            [
                pad_forces(
                    collected_data[MolecularData.FORCES][i], len(Z[i]), MAX_N_ATOMS
                )
                for i in range(N)
            ]
        )

    # Process energy
    if collected_data[MolecularData.ENERGY]:
        processed_data[MolecularData.ENERGY] = np.array(
            [[collected_data[MolecularData.ENERGY][i] * Hartree] for i in range(N)]
        )

    # Process other data types that don't need special handling
    for data_type in [MolecularData.DIPOLE, MolecularData.CENTER_OF_MASS]:
        if collected_data[data_type]:
            processed_data[data_type] = np.array(collected_data[data_type])

    # Save processed data
    save_dict = {key.value: processed_data[key] for key in processed_data}
    save_dict["molecule_ids"] = np.array(molecule_ids)

    output_path = f"processed_data_batch_{batch_index}.npz"
    np.savez(output_path, **save_dict)

    return processed_data


def process_in_memory(data: List[Dict], max_atoms=None):
    """
    Process a list of dictionaries containing data.
    """
    if max_atoms is not None:
        MAX_N_ATOMS = max_atoms
    output = {}

    # rename the dataset keys to match the enum:
    Z_KEYS = ["atomic_numbers", "Z"]
    R_KEYS = ["coordinates", "positions", "R"]
    F_KEYS = ["forces", "F"]
    E_KEYS = ["energy", "energies", "E"]
    D_KEYS = ["dipole", "d", "dipoles"]
    Q_KEYS = ["quadrupole", "q"]
    ESP_KEYS = ["esp", "ESP"]
    ESP_GRID_KEYS = ["esp_grid", "ESP_GRID"]
    COM_KEYS = ["com", "center_of_mass"]

    def check_keys(keys, data):
        for key in keys:
            if key in data:
                return key
        return None

    _ = check_keys(Z_KEYS, data[0])
    if _ is not None:
        Z = [np.array([z[_]]) for z in data]
        output[MolecularData.ATOMIC_NUMBERS] = np.array(
            [pad_atomic_numbers(Z[i], MAX_N_ATOMS) for i in range(len(Z))]
        ).squeeze()
        output[MolecularData.NUMBER_OF_ATOMS] = np.array([[_.shape[1]] for _ in Z])
    _ = check_keys(R_KEYS, data[0])
    if _ is not None:
        # print(_.shape)
        output[MolecularData.COORDINATES] = np.array(
            [pad_coordinates(d[_], MAX_N_ATOMS) for d in data]
        )

    _ = check_keys(F_KEYS, data[0])
    if _ is not None:
        output[MolecularData.FORCES] = np.array(
            [
                pad_forces(
                    d[_],
                    len(d[_]),
                    MAX_N_ATOMS,
                )
                for d in data
            ]
        )

    _ = check_keys(E_KEYS, data[0])
    if _ is not None:
        output[MolecularData.ENERGY] = np.array([d[_] for d in data])

    _ = check_keys(D_KEYS, data[0])
    if _ is not None:
        output[MolecularData.DIPOLE] = np.array([d[_] for d in data])

    _ = check_keys(Q_KEYS, data[0])
    if _ is not None:
        output[MolecularData.QUADRUPOLE] = np.array([d[_] for d in data])

    _ = check_keys(ESP_KEYS, data[0])
    if _ is not None:
        output[MolecularData.ESP] = np.array([d[_] for d in data])

    _ = check_keys(ESP_GRID_KEYS, data[0])
    if _ is not None:
        output[MolecularData.ESP_GRID] = np.array([d[_] for d in data])

    _ = check_keys(COM_KEYS, data[0])
    if _ is not None:
        output[MolecularData.CENTER_OF_MASS] = np.array

    keys = list(output.keys())
    for k_old in keys:
        k_old = MolecularData[str(k_old).split(".")[-1]]
        k_new = KEY_TRANSLATION[k_old]
        output[k_new] = output.pop(k_old)

    return output



NUM_ESP_CLIP = 1000  # Introduced constant for ESP clip limit


def process_dataset_key(data_key, datasets, shape, natoms, not_failed, reshape_dims=None):
    """Helper to process a key across datasets and apply reshaping."""
    data_array = np.concatenate([ds[data_key] for ds in datasets])
    if reshape_dims:
        data_array = data_array.reshape(shape[0], *reshape_dims)
    data_array = data_array[not_failed]  # Filter failed rows
    return data_array.squeeze()


def clip_or_default_data(datasets, data_key, not_failed, shape, clip=False):
    """Helper function to process ESP data with optional clipping."""
    if clip:
        return np.concatenate(
            [dataset[data_key][:NUM_ESP_CLIP] for dataset in datasets]
        )[not_failed].reshape(shape[0], NUM_ESP_CLIP)
    else:
        return np.concatenate([dataset[data_key] for dataset in datasets])[not_failed]


def prepare_multiple_datasets(
        key,
        train_size=0,
        valid_size=0,
        filename=None,
        clean=False,
        verbose=False,
        esp_mask=False,
        clip_esp=False,
        natoms=60,
        subtract_atom_energies=False,
        subtract_mean=False,
):
    """
    Prepare multiple datasets for training and validation.

    Args:
        key: Random key for dataset shuffling.
        train_size (int): Number of training samples.
        valid_size (int): Number of validation samples.
        filename (list): List of filenames to load datasets from.
    Returns:
        tuple: A tuple containing the prepared data and keys.
    """
    # Load datasets
    datasets = [np.load(f) for f in filename]

    if verbose:
        for i, dataset in enumerate(datasets):
            data_shape = {k: v.shape for k, v in dataset.items()}
            print_dict_as_table(data_shape, title=Path(filename[i]).name, plot=True)

    # Validate datasets and initialize variables
    data_ids = np.concatenate([ds["id"] for ds in datasets]) if "id" in datasets[0] else None
    shape = (
        np.concatenate([ds["R"] for ds in datasets])
        .reshape(-1, natoms, 3)
        .shape
    )
    not_failed = np.arange(shape[0])  # Default: no failed rows filtered

    # Handle cleaning
    if clean:
        failed_ids = pd.read_csv("/pchem-data/meuwly/boittier/home/jaxeq/data/qm9-fails.csv")["0"].tolist()
        not_failed = np.array([i for i in range(len(data_ids)) if str(data_ids[i]) not in failed_ids])
        print(f"n_failed: {len(data_ids) - len(not_failed)}")

    # Collect processed data
    data = []
    keys = []

    # Handle individual dataset keys
    if "id" in datasets[0]:
        data.append(data_ids[not_failed])
        keys.append("id")

    if "R" in datasets[0]:
        positions = process_dataset_key("R", datasets, shape, natoms, not_failed, reshape_dims=(natoms, 3))
        data.append(positions)
        keys.append("R")

    if "Z" in datasets[0]:
        atomic_numbers = process_dataset_key("Z", datasets, shape, natoms, not_failed, reshape_dims=(natoms,))
        data.append(atomic_numbers)
        keys.append("Z")

    if "F" in datasets[0]:
        forces = process_dataset_key("F", datasets, shape, natoms, not_failed, reshape_dims=(natoms, 3))
        data.append(forces)
        keys.append("F")

    if "E" in datasets[0]:
        energies = np.concatenate([ds["E"] for ds in datasets])[not_failed]
        if subtract_atom_energies:
            tmp_ae = ATOM_ENERGIES_HARTREE[atomic_numbers].sum(axis=1) * 27.2114
            energies -= tmp_ae
        if subtract_mean:
            energies -= np.mean(energies)
        data.append(energies.reshape(-1, 1))
        keys.append("E")

    if "esp" in datasets[0]:
        esp_data = clip_or_default_data(datasets, "esp", not_failed, shape, clip_esp)
        data.append(esp_data)
        keys.append("esp")

    # Additional processing for other keys can follow the same pattern

    return data, keys

