import jax
from openqdc.datasets import SpiceV2 as Spice

from physnetjax.data.datasets import process_in_memory
from physnetjax.models.model import EF
from physnetjax.training.training import train_model

# Configurable Constants
NATOMS = 110
DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]
RANDOM_SEED = 42
BATCH_SIZE = 3

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
def prepare_spice_dataset(dataset, subsample_size, max_atoms):
    """Prepare the dataset by preprocessing and subsampling."""
    d = dataset.subsample(subsample_size)
    d = [dict(ds[_]) for _ in d]
    return process_in_memory(d, max_atoms=max_atoms)


ds = Spice(energy_unit="ev", distance_unit="ang", array_format="jax")
ds.read_preprocess()
output1 = prepare_spice_dataset(ds, subsample_size=10000, max_atoms=NATOMS)
output2 = prepare_spice_dataset(ds, subsample_size=1000, max_atoms=NATOMS)

# Random key initialization
data_key, train_key = jax.random.split(jax.random.PRNGKey(RANDOM_SEED), 2)

# Model initialization
model = EF(
    features=128,
    max_degree=0,
    num_iterations=5,
    num_basis_functions=20,
    cutoff=10.0,
    max_atomic_number=53,
    charges=False,
    natoms=NATOMS,
    total_charge=0,
    n_res=5,
    zbl=False,
)

batch_kwargs = {
    "batch_shape" : int((BATCH_SIZE - 1) * NATOMS),
    "nb_len" : int((NATOMS * (NATOMS - 1) * (BATCH_SIZE - 1)) // 1.6)
}

# Model training
params = train_model(
    train_key,
    model,
    output1,
    output2,
    num_epochs=int(1e6),
    learning_rate=0.001,
    energy_weight=NATOMS,
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