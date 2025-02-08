import os

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
sys.path.append("/pchem-data/meuwly/boittier/home/pycharmm_test/physnetjax")

import e3x
import jax
import numpy as np
import optax
import orbax

from physnetjax.data.data import prepare_datasets
from physnetjax.training.loss import dipole_calc
from physnetjax.models.model import EF
from physnetjax.training.training import train_model  # from model import dipole_calc

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

data_key, train_key = jax.random.split(jax.random.PRNGKey(42), 2)

from pathlib import Path

# files = list(Path("/pchem-data/meuwly/boittier/home/pycharmm_test/data/basepairs").glob("*.npz"))
NATOMS = 10
files = [
    Path("/pchem-data/meuwly/boittier/home/dcm_dimers_MP2_20999.npz"),

]

train_data, valid_data = prepare_datasets(
    data_key,
    16800,
     4199,
    files,
    clip_esp=False,
    natoms=NATOMS,
    clean=False,
    subtract_atom_energies=True,
    verbose=True,
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
    max_atomic_number=18,
    charges=True,
    natoms=NATOMS,
    total_charge=0,
    n_res=3,
    zbl=False,
    debug=False,
)

DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]

params = train_model(
    train_key,
    model,
    train_data,
    valid_data,
    num_epochs=10, #int(1e6),
    learning_rate=0.001,
    energy_weight=1,
    # charges_weight=1,
    # forces_weight=100,
    schedule_fn="constant",
    optimizer="amsgrad",
    batch_size=1,
    num_atoms=NATOMS,
    data_keys=DEFAULT_DATA_KEYS,
    # restart=restart,
    name="dichloromethane",
    print_freq=1,
    objective="valid_loss",
    best=1e6,
    batch_method="default",
)



