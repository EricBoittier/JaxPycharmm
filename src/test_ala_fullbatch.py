import os

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax

# from jax import config
# config.update('jax_enable_x64', True)

# Check JAX configuration
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())

import sys

# Add custom path
sys.path.append("/pchem-data/meuwly/boittier/home/pycharmm_test/src")

import e3x
import jax
import numpy as np
import optax
import orbax

from data import prepare_batches, prepare_datasets
from loss import dipole_calc
from model import EF
from training import train_model  # from model import dipole_calc

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

data_key, train_key = jax.random.split(jax.random.PRNGKey(43), 2)

# files = ["/pchem-data/meuwly/boittier/home/ini.to.dioxi.npz"]
files = ["/pchem-data/meuwly/boittier/home/jaxeq/notebooks/ala-esp-dip-0.npz"]
NATOMS = 37

batch_size = 5 

train_data, valid_data = prepare_datasets(
    data_key, 8000, 1000, files, clip_esp=False, natoms=NATOMS, clean=False
)
print(NATOMS)
ntest = len(valid_data["E"]) // 2
test_data = {k: v[ntest:] for k, v in valid_data.items()}
valid_data = {k: v[:ntest] for k, v in valid_data.items()}
model = EF(
    # attributes
    features=12,
    max_degree=2,
    num_iterations=4,
    num_basis_functions=20,
    cutoff=10.0,
    max_atomic_number=11,
    charges=True,
    natoms=NATOMS,
    total_charge=0,
    n_res=1,
    debug=[],
)

from pathlib import Path
restart_dir_base = Path("/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/")
restart_dir = restart_dir_base / "test-ef20ef7e-f086-428c-867e-c7f631eead87"
from training import get_last, get_files, get_params_model

DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]
params = train_model(
    train_key,
    model,
    train_data,
    valid_data,
    num_epochs=20000,
    learning_rate=0.01,
    # forces_weight=100,
    # charges_weight=1,
    batch_size=batch_size,
    num_atoms=NATOMS,
    data_keys=DEFAULT_DATA_KEYS,
    # restart=restart_dir,
    print_freq=1,
    best=100000,
    # name=name,
)
