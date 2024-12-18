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
