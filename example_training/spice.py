import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"


import jax
from openqdc.datasets import SpiceV2 as Spice

from physnetjax.data.datasets import process_in_memory
from physnetjax.models.model import EF
from physnetjax.training.training import train_model

# Configurable Constants
NATOMS = 110
DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]
RANDOM_SEED = 42
BATCH_SIZE = 20

# # Environment configuration
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# JAX Configuration Check
def check_jax_configuration():
    devices = jax.local_devices()
    print("Devices:", devices)
    print("Default Backend:", jax.default_backend())
    print("All Devices:", jax.devices())


check_jax_configuration()


# Dataset preparation
def prepare_spice_dataset(dataset, subsample_size, max_atoms, ignore_indices=None):
    """Prepare the dataset by preprocessing and subsampling."""
    indices = dataset.subsample(subsample_size)
    if ignore_indices is not None:
        indices = [_ for _ in indices if _ not in ignore_indices]
    d = [dict(ds[_]) for _ in indices]
    res = process_in_memory(d, max_atoms=max_atoms, openqdc=True)
    return res, indices


ds = Spice(energy_unit="ev", distance_unit="ang", array_format="jax")
ds.read_preprocess()
training_set, training_set_idxs = prepare_spice_dataset(
    ds, subsample_size=100, max_atoms=NATOMS
)
validation_set, validation_set_idxs = prepare_spice_dataset(
    ds, subsample_size=100, max_atoms=NATOMS
)

# Random key initialization
data_key, train_key = jax.random.split(jax.random.PRNGKey(RANDOM_SEED), 2)

# Model initialization
model = EF(
    features=128,
    max_degree=0,
    num_iterations=5,
    num_basis_functions=16,
    cutoff=5.0,
    max_atomic_number=70,
    charges=False,
    natoms=NATOMS,
    total_charge=0,
    n_res=2,
    zbl=False,
)

batch_kwargs = {
    "batch_shape": int((BATCH_SIZE - 1) * NATOMS),
    "nb_len": int((NATOMS * (NATOMS - 1) * (BATCH_SIZE - 1)) // 1.6),
}

print("Model initialized")
print(batch_kwargs)

# Model training
params = train_model(
    train_key,
    model,
    training_set,
    validation_set,
    num_epochs=int(10**2),
    learning_rate=0.001,
    energy_weight=1,
    schedule_fn="constant",
    optimizer="amsgrad",
    batch_size=BATCH_SIZE,
    num_atoms=NATOMS,
    data_keys=DEFAULT_DATA_KEYS,
    print_freq=1,
    objective="valid_loss",
    best=1e6,
    batch_method="advanced",
    batch_args_dict=batch_kwargs,
)
