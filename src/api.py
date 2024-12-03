import os

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import jax

# from jax import config
# config.update('jax_enable_x64', True)

# Check JAX configuration
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())


import argparse

import e3x
import jax
import numpy as np
import optax
from dcmnet.analysis import create_model_and_params
from dcmnet.data import prepare_batches, prepare_datasets

from model import EF
from training import train_model

"""
bash python api.py --data  --ntrain 5000 --nvalid 500 --features 32 --max_degree 3 --num_iterations 2 --num_basis_functions 16 --cutoff 6.0 --max_atomic_number 118 --n_res 3
"""

DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--restart", type=str, default=False, help="Restart training from a checkpoint"
    )
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--train_key", type=str, default=None)
    parser.add_argument("--nepochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nvalid", type=int, default=500)
    parser.add_argument("--name", type=str, default="diox-q3.2")
    parser.add_argument("--natoms", type=int, default=8)
    parser.add_argument("--charge_w", type=float, default=52.91772105638412)
    parser.add_argument("--forces_w", type=float, default=27.211386024367243)
    # model parameters
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--max_degree", type=int, default=3)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--num_basis_functions", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max_atomic_number", type=int, default=9)
    parser.add_argument("--n_res", type=int, default=3)
    parser.add_argument("--debug", type=bool, default=False)
    print("parsing arguments:")
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    batch_size = args.batch_size
    restart = args.restart
    data = args.data
    model = args.model
    train_key = args.train_key
    num_epochs = args.nepochs
    learning_rate = args.lr
    clip = args.clip
    logdir = args.logdir
    ntrain = args.ntrain
    nvalid = args.nvalid
    name = args.name
    features = args.features
    max_degree = args.max_degree
    num_iterations = args.num_iterations
    num_basis_functions = args.num_basis_functions
    cutoff = args.cutoff
    max_atomic_number = args.max_atomic_number
    n_res = args.n_res
    params = False
    debug = args.debug
    NATOMS = args.natoms
    forces_weight = args.forces_w
    charges_weight = args.charge_w
    data_key, train_key = jax.random.split(jax.random.PRNGKey(43), 2)
    NATOMS = args.natoms
    # load data
    files = [data]
    train_data, valid_data = prepare_datasets(
        data_key, ntrain, nvalid, files, clip_esp=False, natoms=NATOMS, clean=False
    )
    ntest = len(valid_data["E"]) // 2
    test_data = {k: v[ntest:] for k, v in valid_data.items()}
    valid_data = {k: v[:ntest] for k, v in valid_data.items()}

    print("train data", len(train_data["E"]))
    print("valid data", len(valid_data["E"]))
    print("test data", len(test_data["E"]))
    model = EF(  # attributes
        features=features,
        max_degree=max_degree,
        num_iterations=num_iterations,
        num_basis_functions=num_basis_functions,
        cutoff=cutoff,
        max_atomic_number=max_atomic_number,
        charges=True,
        natoms=NATOMS,
        total_charge=0,
        n_res=n_res,
        debug=debug,
    )
    print(model)
    print("train_model:")
    params = train_model(
        train_key,
        model,
        train_data,
        valid_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        forces_weight=forces_weight,
        charges_weight=charges_weight,
        batch_size=batch_size,
        num_atoms=NATOMS,
        data_keys=DEFAULT_DATA_KEYS,
        restart=restart,
        print_freq=1,
        name=name,
    )
