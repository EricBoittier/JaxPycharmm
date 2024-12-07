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

from physnetjax.data import prepare_batches, prepare_datasets
from physnetjax.model import EF
from physnetjax.training import train_model  # from model import dipole_calc

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

data_key, train_key = jax.random.split(jax.random.PRNGKey(3), 2)

# files = ["/pchem-data/meuwly/boittier/home/ini.to.dioxi.npz"]
files = ["/pchem-data/meuwly/boittier/home/jaxeq/notebooks/ala-esp-dip-0.npz"]

NATOMS = 37


train_data, valid_data = prepare_datasets(
    data_key, 8, 4, files, clip_esp=False, natoms=NATOMS, clean=False
)

ntest = len(valid_data["E"]) // 2
test_data = {k: v[ntest:] for k, v in valid_data.items()}
valid_data = {k: v[:ntest] for k, v in valid_data.items()}
model = EF(
    # attributes
    features=64,
    max_degree=0,
    num_iterations=2,
    num_basis_functions=20,
    cutoff=10.0,
    max_atomic_number=11,
    charges=True,
    natoms=NATOMS,
    total_charge=0,
    n_res=3,
    debug=["repulsion", "ele"],
)

DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]
params = train_model(
    train_key,
    model,
    train_data,
    valid_data,
    num_epochs=1,
    learning_rate=0.001,
    # forces_weight=1,
    # charges_weight=1,
    batch_size=2,
    num_atoms=NATOMS,
    data_keys=DEFAULT_DATA_KEYS,
    # restart=restart,
    print_freq=1,
    # name=name,
)
