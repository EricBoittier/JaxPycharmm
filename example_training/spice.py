import os
# # Environment configuration
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import jax
from openqdc.datasets import SpiceV2 as Spice

from physnetjax.data.datasets import process_in_memory
from physnetjax.models.model import EF
from physnetjax.training.training import train_model

# Constants
NATOMS = 110
# total number of samples, SpiceV2 = 2008628
NTRAIN = 10
NVALID = 10
DEFAULT_DATA_KEYS = ("Z", "R", "D", "E", "F", "N")
RANDOM_SEED = 42
BATCH_SIZE = 1

# JAX Configuration Check
def check_jax_configuration():
    devices = jax.local_devices()
    print("Devices:", devices)
    print("Default Backend:", jax.default_backend())
    print("All Devices:", jax.devices())


check_jax_configuration()


# Dataset preparation
def prepare_spice_dataset(dataset, subsample_size, max_atoms, ignore_indices=None,
                          key=jax.random.PRNGKey(42)):
    """Prepare the dataset by preprocessing and subsampling."""
    key = key[0] if len(key) > 1 else key
    indices = dataset.subsample(subsample_size, seed=key)
    if ignore_indices is not None:
        indices = [_ for _ in indices if _ not in ignore_indices]
    d = [dict(ds[_]) for _ in indices]
    res = process_in_memory(d, max_atoms=max_atoms, openqdc=True)
    return res, indices


ds = Spice(energy_unit="ev", distance_unit="ang", array_format="jax")
ds.read_preprocess()


# Random key initialization
data_key, train_key = jax.random.split(jax.random.PRNGKey(RANDOM_SEED), 2)
# load the training set
training_set, training_set_idxs = prepare_spice_dataset(
    ds, subsample_size=NTRAIN, max_atoms=NATOMS, key=data_key
)
# get a new data key
data_key, _ = jax.random.split(data_key, 2)
# load the validation set
validation_set, validation_set_idxs = prepare_spice_dataset(
    ds, subsample_size=NVALID, max_atoms=NATOMS,
    ignore_indices=training_set_idxs, key=data_key
)

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

del ds

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
