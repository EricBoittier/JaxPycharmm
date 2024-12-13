from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class MolecularData(Enum):
    """Types of data that can be present in molecular datasets"""

    COORDINATES = "coordinates"
    ATOMIC_NUMBERS = "atomic_numbers"
    FORCES = "forces"
    ENERGY = "energy"
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"
    ESP = "esp"
    ESP_GRID = "esp_grid"
    CENTER_OF_MASS = "com"
    NUMBER_OF_ATOMS = "N"


KEY_TRANSLATION = {
    MolecularData.COORDINATES: "R",
    MolecularData.ATOMIC_NUMBERS: "Z",
    MolecularData.ENERGY: "E",
    MolecularData.FORCES: "F",
    MolecularData.DIPOLE: "D",
    MolecularData.NUMBER_OF_ATOMS: "N",
}
