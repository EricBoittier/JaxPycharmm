"""
Training script for the EF model on alanine dataset.
Handles model initialization, training, and checkpointing.
"""

import os
import sys
from pathlib import Path

# JAX imports
import jax
import optax
import orbax.checkpoint

# Custom imports
sys.path.append("/pchem-data/meuwly/boittier/home/pycharmm_test/src")
from physnetjax.data.data import prepare_datasets
from physnetjax.models.model import EF
from physnetjax.training.training import train_model

# Configure environment
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Verify JAX configuration
print("JAX devices:", jax.local_devices())
print("JAX backend:", jax.default_backend())

# Configuration
NATOMS = 37
BATCH_SIZE = 8
DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]
DATA_FILES = ["/pchem-data/meuwly/boittier/home/jaxeq/notebooks/ala-esp-dip-0.npz"]
CHECKPOINT_DIR = Path("/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/")

restart = CHECKPOINT_DIR / "test-9392c2e7-af2a-4756-ae2a-35ffdf01951d"
# restart = None

# Initialize random keys
data_key, train_key = jax.random.split(jax.random.PRNGKey(42), 2)

# Prepare datasets
train_data, valid_data = prepare_datasets(
    data_key,
    train_size=8000,
    valid_size=1000,
    files=DATA_FILES,
    natoms=NATOMS,
)

# Split validation data into validation and test sets
ntest = len(valid_data["E"]) // 2
test_data = {k: v[ntest:] for k, v in valid_data.items()}
valid_data = {k: v[:ntest] for k, v in valid_data.items()}

# Initialize model
model = EF(
    features=32,
    max_degree=1,
    num_iterations=2,
    num_basis_functions=16,
    cutoff=10.0,
    max_atomic_number=8,
    charges=True,
    efa=True,
    natoms=NATOMS,
    total_charge=0,
    n_res=1,
    zbl=False,
)


restart = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/test-414ad170-d9f2-4eda-b2ee-5cc17158bdfb"
restart = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/test-878b75bc-025b-4a6e-b676-ea80acac8305"
restart = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/test-2e509d1a-2edb-490c-94b3-d6aec2d8ec87"
# Train model
params = train_model(
    train_key,
    model,
    train_data,
    valid_data,
    num_epochs=int(1e6),
    learning_rate=0.001,
    energy_weight=1,
    # charges_weight=1,
    # forces_weight=100,
    schedule_fn="constant",
    optimizer="amsgrad",
    batch_size=20,
    num_atoms=NATOMS,
    data_keys=DEFAULT_DATA_KEYS,
    #restart=restart,
    print_freq=1,
    objective="valid_loss",
    best=1e6,
)
