from physnetjax.data.batches import prepare_batches, prepare_batches_advanced_minibatching
import numpy as np
import jax

def test_prepare_batches_advanced_minibatching_dummy():
    NSAMPLES = 10
    NATOMS = 4
    BATCH_SIZE = 3
    BATCH_SHAPE = int(BATCH_SIZE * NATOMS // 1.5)
    NB_LEN = 20

    R_shape1 = (NSAMPLES, NATOMS, 3)
    D_shape1 = (NSAMPLES, 1, 3)
    Z_shape1 = (NSAMPLES, NATOMS,)

    dummy_data = {
        "R": np.random.rand(*R_shape1),
        "F": np.random.rand(*R_shape1),
        "E": np.random.rand(NSAMPLES, 1),
        "D": np.random.rand(*D_shape1),
        "Z": np.random.rand(*Z_shape1),
        "N": np.random.randint(1, NATOMS, size=(NSAMPLES,)),
    }
    key = jax.random.PRNGKey(42)

    output = prepare_batches_advanced_minibatching(
        key,
        dummy_data,
        batch_size=BATCH_SIZE,
        batch_shape=BATCH_SHAPE,
        batch_nbl_len=NB_LEN,
        data_keys=("R", "Z", "N", "D", "F"),
    )

    assert len(output) == NSAMPLES // BATCH_SIZE
    for k in output[0]:
        assert len(output[0][k]) == len(output[1][k]), f"Lengths of {k} should match."

    assert len(output[0]["dst_idx"]) > 0
    assert len(output[0]["dst_idx"]) == len(output[0]["src_idx"])
    assert len(output[1]["dst_idx"]) == len(output[0]["src_idx"])
    assert len(output[0]["dst_idx"]) == NB_LEN
    assert len(output[1]["R"]) == BATCH_SHAPE
    assert len(output[1]["Z"]) == BATCH_SHAPE
    assert len(output[1]["N"]) == BATCH_SIZE
    assert len(output[1]["D"]) == BATCH_SIZE

def test_prepare_batches_advanced_minibatching_basepairs():
    from physnetjax.directories import MAIN_PATH
    from pathlib import Path

    mock_filename = [
        MAIN_PATH / Path("data/basepairs/at_prod.npz"),
        MAIN_PATH / Path("data/basepairs/at_reag.npz"),
        MAIN_PATH / Path("data/basepairs/at_retune.npz"),
        MAIN_PATH / Path("data/basepairs/rattle_neb_at.npz"),
        MAIN_PATH / Path("data/basepairs/rattle_neb_gc.npz"),
    ]

    NATOMS = 30
    BATCH_SIZE = 3
    BATCH_SHAPE = int(BATCH_SIZE * NATOMS // 1.5)
    NB_LEN = 870 * 3

    from physnetjax.data.data import prepare_datasets
    print(mock_filename)
    key = jax.random.PRNGKey(42)

    # Call function with mocked data
    train_data, valid_data = prepare_datasets(
        key,
        10,
        10,
        mock_filename,
        natoms=NATOMS,
    )

    output = prepare_batches_advanced_minibatching(
        key,
        train_data,
        batch_size=BATCH_SIZE,
        batch_shape=BATCH_SHAPE,
        batch_nbl_len=NB_LEN,
        data_keys=("R", "Z", "N", "F"),
    )

    assert len(output) == int(len(train_data["R"])) // BATCH_SIZE
    for k in output[0]:
        assert len(output[0][k]) == len(output[1][k]), f"Lengths of {k} should match."

    assert len(output[0]["dst_idx"]) > 0
    assert len(output[0]["dst_idx"]) == len(output[0]["src_idx"])
    assert len(output[1]["dst_idx"]) == len(output[0]["src_idx"])
    assert len(output[0]["dst_idx"]) == NB_LEN
    assert len(output[1]["R"]) == BATCH_SHAPE
    assert len(output[1]["Z"]) == BATCH_SHAPE
    assert len(output[1]["N"]) == BATCH_SIZE


def test_prepare_batches_advanced_minibatching():
    import jax
    from openqdc.datasets import SpiceV2 as Spice

    from physnetjax.data.datasets import process_in_memory
    from physnetjax.models.model import EF
    from physnetjax.training.training import train_model

    # Configurable Constants
    NATOMS = 110
    DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]
    RANDOM_SEED = 42
    key = jax.random.PRNGKey(RANDOM_SEED)

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
    output1 = prepare_spice_dataset(ds, subsample_size=100, max_atoms=NATOMS)
    output2 = prepare_spice_dataset(ds, subsample_size=1000, max_atoms=NATOMS)
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches_advanced_minibatching(
        shuffle_key,
        output2,
        20,
        num_atoms=110,
        data_keys=("Z", "R", "E", "F", "N"),
    )
    print(valid_batches)
    assert len(valid_batches) < 1000

