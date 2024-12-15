# Basics
import os

# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ASE
from ase import io

# PyCHARMM
os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm/"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"

# from pycharmm_calculator import PyCharmm_Calculator

import jax
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())


# os.sleep(1)
with open("i_", "w") as f:
    print("...")

# ASE
from ase import io
import ase.units as units

# PyCHARMM
import pycharmm
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.minimize as minimize
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.lingo as stream

import pandas as pd
pdb_file = "/pchem-data/meuwly/boittier/home/pycharmm_test/md/adp.pdb"
atoms = io.read(pdb_file)
pkl_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a/params.pkl"
model_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a/model_kwargs.pkl"

from physnetjax.restart.restart import get_params_model_with_ase

params, model = get_params_model_with_ase(
    pkl_path, model_path, atoms
)

# Read in the topology (rtf) and parameter file (prm) for proteins
# equivalent to the CHARMM scripting command: read rtf card name toppar/top_all36_prot.rtf
read.rtf("/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_prot.rtf")
# equivalent to the CHARMM scripting command: read param card flexible name toppar/par_all36m_prot.prm
read.prm(
    "/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36m_prot.prm", flex=True
)
pycharmm.lingo.charmm_script(
    "stream /pchem-data/meuwly/boittier/home/charmm/toppar/toppar_water_ions.str"
)

settings.set_bomb_level(-2)
settings.set_warn_level(-1)

# Set the coordinates of the atoms
read.pdb(pdb_file)
read.psf_card("/pchem-data/meuwly/boittier/home/pycharmm_test/md/adp.psf")

stats = coor.get_stats()
print(stats)