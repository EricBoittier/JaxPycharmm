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

atoms = io.read("/pchem-data/meuwly/boittier/home/pycharmm_test/md/adp.pdb")
pkl_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a/params.pkl"
model_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a/model_kwargs.pkl"

from physnetjax.restart.restart import get_params_model_with_ase

params, model = get_params_model_with_ase(
    pkl_path, model_path, atoms
)

print(params, model)