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

# Atomic energies in Hartree
ATOM_ENERGIES_HARTREE = np.array(
    [0, -0.500273, 0, 0, 0, 0, -37.846772, -54.583861, -75.064579]
)


def sort_func(filepath: Path) -> int:
    """
    Extract and return sorting number from filepath.

    Args:
        filepath: Path object containing the file path

    Returns:
        Integer for sorting the file paths
    """
    x = str(filepath)
    spl = x.split("/")
    x = spl[-1]
    spl = x.split("xyz")
    spl = spl[1].split("_")
    spl = [_ for _ in spl if len(_) > 0]
    return abs(int(spl[0]))


def get_input_files(data_path: str) -> List[Path]:
    """
    Get list of NPZ files from the given path.

    Args:
        data_path: Path to directory containing NPZ files

    Returns:
        Sorted list of Path objects
    """
    files = list(Path(data_path).glob("*npz"))
    files.sort(key=sort_func)
    return files


def pad_array(
    arr: NDArray, max_size: int, axis: int = 0, pad_value: float = 0.0
) -> NDArray:
    """
    Pad a numpy array along specified axis to a maximum size.

    Args:
        arr: Input array to pad
        max_size: Maximum size to pad to
        axis: Axis along which to pad (default: 0)
        pad_value: Value to use for padding (default: 0.0)

    Returns:
        Padded array
    """
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, max_size - arr.shape[axis])
    return np.pad(arr, pad_width, mode="constant", constant_values=pad_value)


def pad_coordinates(coords: NDArray, max_atoms: int) -> NDArray:
    """
    Pad coordinates array to maximum number of atoms.

    Args:
        coords: Array of atomic coordinates with shape (n_atoms, 3)
        max_atoms: Maximum number of atoms to pad to

    Returns:
        Padded coordinates array with shape (max_atoms, 3)
    """
    return pad_array(coords, max_atoms)


def pad_forces(forces: NDArray, n_atoms: int, max_atoms: int) -> NDArray:
    """
    Pad and convert forces array from Hartree/Bohr to eV/Angstrom.

    Args:
        forces: Array of forces in Hartree/Bohr with shape (n_atoms, 3)
        n_atoms: Number of atoms
        max_atoms: Maximum number of atoms to pad to

    Returns:
        Padded and converted forces array with shape (max_atoms, 3)
    """
    converted_forces = forces * (-HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM)
    return pad_array(converted_forces, max_atoms)


def pad_atomic_numbers(atomic_numbers: NDArray, max_atoms: int) -> NDArray:
    """
    Pad atomic numbers array to maximum number of atoms.

    Args:
        atomic_numbers: Array of atomic numbers
        max_atoms: Maximum number of atoms to pad to

    Returns:
        Padded atomic numbers array
    """
    return pad_array(atomic_numbers, max_atoms, axis=1, pad_value=0)


def process_npz_file(filepath: Path) -> Tuple[Union[dict, None], int]:
    """
    Process a single NPZ file and extract relevant data.

    Args:
        filepath: Path to NPZ file

    Returns:
        Tuple of (processed data dict or None, number of atoms)
    """
    with np.load(filepath) as load:
        if load is None:
            return None, 0

        keys = load.keys()
        if "R" not in keys or "Z" not in keys:
            raise ValueError("Invalid NPZ file, missing required keys R and Z")

        R = load["R"]
        print("R.shape", R.shape)
        n_atoms = len(np.nonzero(R.sum(axis=1))[0])

        if int(len(R)) != n_atoms:
            return None, n_atoms

        fn = np.linalg.norm(
            np.linalg.norm(
                load["F"] * (-627.509474 / 0.529177) * (kcal / mol),
                axis=(1),
            )
        )

        if not (3 < n_atoms < 1000):
            return None, n_atoms

        R = R[np.nonzero(R.sum(axis=1))]
        Z = load["Z"][np.nonzero(R.sum(axis=1))]
        atom_energies = np.take(ATOM_ENERGIES_HARTREE, Z)
        mol = ase.Atoms(Z, R)

        output = {
            MolecularData.COORDINATES.value: R,
            MolecularData.ATOMIC_NUMBERS.value: Z,
        }
        if MolecularData.FORCES.value in load:
            output[MolecularData.FORCES.value] = load["F"]
        if MolecularData.ENERGY.value in load:
            output[MolecularData.ENERGY.value] = load["E"] - np.sum(atom_energies)
        if MolecularData.DIPOLE.value in load:
            output[MolecularData.DIPOLE.value] = load["dipole"]
        if MolecularData.QUADRUPOLE.value in load:
            output[MolecularData.QUADRUPOLE.value] = load["quadrupole"]
        if MolecularData.ESP.value in load:
            output[MolecularData.ESP.value] = load["esp"]
        if MolecularData.ESP_GRID.value in load:
            output[MolecularData.ESP_GRID.value] = load["esp_grid"]
        if MolecularData.CENTER_OF_MASS.value in load:
            output[MolecularData.CENTER_OF_MASS.value] = mol.get_center_of_mass()
        return output, n_atoms


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

if __name__ == "__main__":
#    files = list(Path("/pchem-data/meuwly/boittier/home/pycharmm_test/data/basepairs").glob("*"))
    files = [Path('/pchem-data/meuwly/boittier/home/pycharmm_test/data/basepairs/at_prod.npz'),
 Path('/pchem-data/meuwly/boittier/home/pycharmm_test/data/basepairs/at_reag.npz'),
 Path('/pchem-data/meuwly/boittier/home/pycharmm_test/data/basepairs/at_retune.npz'),
 Path('/pchem-data/meuwly/boittier/home/pycharmm_test/data/basepairs/rattle_neb_at.npz'),
 Path('/pchem-data/meuwly/boittier/home/pycharmm_test/data/basepairs/rattle_neb_gc.npz')]
    print(files)
    process_dataset(files)


