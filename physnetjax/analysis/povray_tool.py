import numpy as np
from pathlib import Path
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ase import Atoms
from ase.visualize.plot import plot_atoms
from io import BytesIO
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from ase import io
from ase.io.pov import get_bondpairs
from ase.data import covalent_radii

default_color_dict = {
    'Cl': [102, 227, 115],
    'C': [61, 61, 64],
    'O': [240, 10, 10],
    'N': [10, 10, 240],
    "F": [0,232,0],
    'H': [232, 206, 202],
    'K': [128, 50, 100],
    'X': [200, 200, 200]
}



def render_povray(atoms, pov_name,
                  rotation='0x, 0y, 0z',
                  radius_scale=0.40, color_dict=None):
    print("Rendering POV-Ray image...")
    print("path: ", pov_name)
    if color_dict is None:
        color_dict = default_color_dict

    path = Path(pov_name)
    pov_name = path.name
    base = path.parent

    radius_list = []
    for atomic_number in atoms.get_atomic_numbers():
        radius_list.append(radius_scale * covalent_radii[atomic_number])

    colors = np.array([color_dict[atom.symbol] for atom in atoms]) / 255

    bondpairs = get_bondpairs(atoms, radius=1.1)
    good_bonds = []
    for _ in bondpairs:
        #  remove the Cl-Cl bonds
        if not (atoms[_[0]].symbol == "Cl" and atoms[_[1]].symbol == "Cl"):
            good_bonds.append(_)

    kwargs = {  # For povray files only
        'transparent': True,  # Transparent background
        'canvas_width': 1028,  # Width of canvas in pixels
        'canvas_height': None,  # None,  # Height of canvas in pixels
        'camera_dist': 50.0,  # Distance from camera to front atom,
        'camera_type': 'orthographic angle 0',  # 'perspective angle 20'
        'depth_cueing': False,
        'colors': colors,
        'bondatoms': good_bonds,
        "textures": ["jmol"] * len(atoms),
    }

    generic_projection_settings = {
        'rotation': rotation,
        'radii': radius_list,
    }

    povobj = io.write(
        pov_name,
        atoms,
        **generic_projection_settings,
        povray_settings=kwargs)

    povobj.render(povray_executable='/pchem-data/meuwly/boittier/home/miniforge3/envs/jaxphyscharmm/bin/povray')
    png_name = pov_name.replace(".pov", ".png")
    shutil.move(png_name, base / png_name)
    return png_name