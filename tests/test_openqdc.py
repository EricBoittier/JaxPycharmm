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
sys.path.append("/pchem-data/meuwly/boittier/home/pycharmm_test/src")

import orbax
from orbax.checkpoint import PyTreeCheckpointer
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

import os
# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import jax
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())
import sys
# Add custom path
sys.path.append("/pchem-data/meuwly/boittier/home/pycharmm_test/src")

import jax

from openqdc.datasets import Spice
from physnetjax.savepad import *

NATOMS = 96

ds = Spice(
    energy_unit="ev",
    distance_unit="ang",
    array_format = "jax"
)
ds.read_preprocess()
datadicts = [dict(ds[_]) for _ in ds.subsample(50)]
output1 = process_in_memory(datadicts, max_atoms=NATOMS)
datadicts = [dict(ds[_]) for _ in ds.subsample(5)]
output2 = process_in_memory(datadicts, max_atoms=NATOMS)

data_key, train_key = jax.random.split(
    jax.random.PRNGKey(42), 2)

from physnetjax.model import EF
from physnetjax.training import train_model
DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]

model = EF(natoms=NATOMS, charges=False, zbl=False, features=8,
           max_degree=0, num_basis_functions=8, num_iterations=1, n_res=1,
           total_charge=0.0, max_atomic_number=18, cutoff=5.0,)

params = train_model(
    train_key,
    model,
    output1,
    output2,
    num_epochs=int(1e6),
    learning_rate=0.001,
    energy_weight=NATOMS,
    #charges_weight=1,
    #forces_weight=100,
    schedule_fn="constant",
    optimizer="amsgrad",
    batch_size=10,
    num_atoms=NATOMS,
    data_keys=DEFAULT_DATA_KEYS,
#    restart=restart,
    print_freq=1,
    objective="valid_loss",
    best=1e6,
    )