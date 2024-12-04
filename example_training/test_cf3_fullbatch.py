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
from physnetjax.loss import dipole_calc
from physnetjax.model import EF
from physnetjax.training import train_model  # from model import dipole_calc

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

data_key, train_key = jax.random.split(jax.random.PRNGKey(43), 2)

# files = ["/pchem-data/meuwly/boittier/home/ini.to.dioxi.npz"]
files = ["/pchem-data/meuwly/boittier/home/all.npz"]

NATOMS = 8


train_data, valid_data = prepare_datasets(
    data_key, 27030, 2000, files, clip_esp=False, natoms=NATOMS, clean=False
)

ntest = len(valid_data["E"]) // 2
test_data = {k: v[ntest:] for k, v in valid_data.items()}
valid_data = {k: v[:ntest] for k, v in valid_data.items()}

model = EF(
    # attributes
    features=128,
    max_degree=0,
    num_iterations=5,
    num_basis_functions=64,
    cutoff=10.0,
    max_atomic_number=11,
    charges=True,
    natoms=NATOMS,
    total_charge=0,
    n_res=3,
    zbl=True,
    debug=False,
)

DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]

restart = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/test-a3932e32-4923-4b30-b0db-9cb8a3d55bd1/"


params = train_model(
    train_key,
    model,
    train_data,
    valid_data,
    num_epochs=int(1e6),
    learning_rate=0.001,
    forces_weight=100,
    optimizer="adamw",
    batch_size=1,
    num_atoms=NATOMS,
    data_keys=DEFAULT_DATA_KEYS,
    restart=restart,
    print_freq=1,
    objective="valid_loss",
    best=1e6,
)
