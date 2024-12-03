import os
import sys
import uuid
from pathlib import Path

from flax.training import orbax_utils

# Add custom path
sys.path.append("/home/boittier/jaxeq/dcmnet")

import functools
from datetime import datetime

import ase
import dcmnet
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax
from dcmnet.utils import apply_model, reshape_dipole, safe_mkdir
from flax.training import checkpoints, train_state
from jax.random import randint
from optax import contrib
from optax import tree_utils as otu
from tqdm import tqdm

from data import prepare_batches, prepare_datasets
from evalstep import eval_step
from model import EF
from optimizer import optimizer, transform
from trainstep import train_step

DTYPE = jnp.float32

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def get_files(path):
    dirs = list(Path(path).glob("*/"))
    dirs.sort(key=lambda x: int(str(x).split("/")[-1].split("-")[-1]))
    dirs = [_ for _ in dirs if "tmp" not in str(_)]
    return dirs


def get_last(path):
    dirs = get_files(path)
    if "tmp" in str(dirs[-1]):
        dirs.pop()
    return dirs[-1]


def get_params_model(restart, natoms=None):
    restored = orbax_checkpointer.restore(restart)
    print("Restoring from", restart)
    # Get the modification time of the file
    modification_time = os.path.getmtime(restart)
    # Convert the timestamp to a human-readable format
    modification_date = datetime.fromtimestamp(modification_time)
    print(f"The file was last modified on: {modification_date}")
    print("Restored keys:", restored.keys())
    params = restored["params"]
    restored["ema_params"]
    # transform_state = transform.init(restored["transform_state"])
    # print("transform_state", transform_state)
    restored["epoch"] + 1
    restored["best_loss"]
    print("scale:", restored["transform_state"]["scale"])
    if "model_attributes" not in restored.keys():
        return params, None
    kwargs = restored["model_attributes"]
    # print(kwargs)
    # print(kwargs)
    kwargs["features"] = int(kwargs["features"])
    kwargs["max_degree"] = int(kwargs["max_degree"])
    kwargs["num_iterations"] = int(kwargs["num_iterations"])
    kwargs["num_basis_functions"] = int(kwargs["num_basis_functions"])
    kwargs["cutoff"] = float(kwargs["cutoff"])
    kwargs["natoms"] = int(kwargs["natoms"])
    kwargs["total_charge"] = float(kwargs["total_charge"])
    kwargs["n_res"] = int(kwargs["n_res"])
    kwargs["max_atomic_number"] = int(kwargs["max_atomic_number"])
    kwargs["charges"] = bool(kwargs["charges"])
    kwargs["debug"] = []
    if natoms is not None:
        kwargs["natoms"] = natoms

    model = EF(**kwargs)
    print(model)
    return params, model


conversion = {
    "energy": 1 / (ase.units.kcal / ase.units.mol),
    "forces": 1 / (ase.units.kcal / ase.units.mol),
}


def train_model(
    key,
    model,
    train_data,
    valid_data,
    num_epochs=1,
    learning_rate=0.001,
    forces_weight=52.91772105638412,
    charges_weight=27.211386024367243,
    batch_size=1,
    num_atoms=60,
    restart=False,
    conversion=conversion,
    print_freq=1,
    name="test",
    best=False,
    objective="valid_forces_mae",
    data_keys=["R", "Z", "F", "E", "D", "dst_idx", "src_idx", "batch_segments"],
):
    best_loss = 10000
    doCharges = model.charges
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)

    print("Train data keys:", train_data.keys())
    print("Valid data keys:", valid_data.keys())

    uuid_ = str(uuid.uuid4())
    CKPT_DIR = f"/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/{name}-{uuid_}/"
    schedule_fn = optax.schedules.warmup_exponential_decay_schedule(
        init_value=learning_rate,
        peak_value=learning_rate * 1.05,
        warmup_steps=10,
        transition_steps=10,
        decay_rate=0.999,
    )
    # Batches for the validation set need to be prepared only once.
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(
        shuffle_key,
        valid_data,
        batch_size,
        num_atoms=num_atoms,
        data_keys=data_keys,
    )

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    params = model.init(
        init_key,
        atomic_numbers=train_data["Z"][0],
        positions=train_data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    # load from restart
    if restart:
        restart = get_last(restart)
        _, _model = get_params_model(restart, num_atoms)
        print(_, _model)
        if _model is not None:
            model = _model
        restored = orbax_checkpointer.restore(restart)
        print("Restoring from", restart)
        print("Restored keys:", restored.keys())
        params = restored["params"]
        ema_params = restored["ema_params"]
        # transform_state = transform.init(restored["transform_state"])
        # print("transform_state", transform_state)
        step = restored["epoch"] + 1
        best_loss = restored["best_loss"]
        CKPT_DIR = Path(restart).parent  # optimizer = restored["optimizer"]
    # initialize
    else:
        ema_params = params
        best_loss = 10000
        step = 1

    if best:
        best_loss = best

    opt_state = optimizer.init(params)
    transform_state = transform.init(params)

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    # print("Trainable params:", state.num_params)
    print("model", model)
    # Train for 'num_epochs' epochs.
    for epoch in range(step, num_epochs + 1):

        # adjust weights for the loss function
        # forces_weight = np.tanh(1.1**(0.05 * (epoch - 100))) * 100 + 1
        # energy_weight = np.tanh(1.1**(0.05 * (-epoch + 500))) * 10 + 10
        if epoch < 500:
            energy_weight = 1
            forces_weight = 1000
        elif epoch < 1000:
            energy_weight = 1000
            forces_weight = 1
        else:
            forces_weight = 50
            energy_weight = 1

        print("Wf, We =", forces_weight, energy_weight)

        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(
            shuffle_key,
            train_data,
            batch_size,
            num_atoms=num_atoms,
            data_keys=data_keys,
        )
        # Loop over train batches.
        train_loss = 0.0
        train_energy_mae = 0.0
        train_forces_mae = 0.0
        train_dipoles_mae = 0.0
        for i, batch in enumerate(train_batches):
            (
                params,
                ema_params,
                opt_state,
                transform_state,
                loss,
                energy_mae,
                forces_mae,
                dipole_mae,
            ) = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                transform_state=transform_state,
                batch=batch,
                batch_size=batch_size,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                charges_weight=charges_weight,
                opt_state=opt_state,
                doCharges=doCharges,
                params=params,
                ema_params=ema_params,
                debug=bool("grad" in model.debug),
            )
            train_loss += (loss - train_loss) / (i + 1)
            train_energy_mae += (energy_mae - train_energy_mae) / (i + 1)
            train_forces_mae += (forces_mae - train_forces_mae) / (i + 1)
            train_dipoles_mae += (dipole_mae - train_dipoles_mae) / (i + 1)

        # Evaluate on validation set.
        valid_loss = 0.0
        valid_energy_mae = 0.0
        valid_forces_mae = 0.0
        valid_dipoles_mae = 0.0
        for i, batch in enumerate(valid_batches):
            loss, energy_mae, forces_mae, dipole_mae = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                charges_weight=charges_weight,
                charges=doCharges,
                params=ema_params,
            )
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_energy_mae += (energy_mae - valid_energy_mae) / (i + 1)
            valid_forces_mae += (forces_mae - valid_forces_mae) / (i + 1)
            valid_dipoles_mae += (dipole_mae - valid_dipoles_mae) / (i + 1)

        _, transform_state = transform.update(
            updates=params, state=transform_state, value=valid_loss
        )

        # convert statistics to kcal/mol for printing
        valid_energy_mae *= conversion["energy"]
        valid_forces_mae *= conversion["forces"]
        train_energy_mae *= conversion["energy"]
        train_forces_mae *= conversion["forces"]

        obj_res = {
            "valid_energy_mae": valid_energy_mae,
            "valid_forces_mae": valid_forces_mae,
            "train_energy_mae": train_energy_mae,
            "train_forces_mae": train_forces_mae,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }

        lr_eff = transform_state.scale * schedule_fn(epoch)
        best_ = False
        if obj_res[objective] < best_loss:

            model_attributes = {
                _.split(" = ")[0].strip(): _.split(" = ")[-1]
                for _ in str(model).split("\n")[2:-1]
            }

            # checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=epoch)
            ckpt = {
                "model": state,
                "model_attributes": model_attributes,
                "transform_state": transform_state,
                "ema_params": ema_params,
                "params": params,
                "epoch": epoch,
                "opt_state": opt_state,
                "best_loss": best_loss,
                "lr_eff": lr_eff,
                "objectives": obj_res,
            }

            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(
                Path(CKPT_DIR) / f"epoch-{epoch}", ckpt, save_args=save_args
            )
            print(Path(CKPT_DIR) / f"epoch-{epoch}")
            # update best loss
            best_loss = obj_res[objective]
            print("best!")
            best_ = True

        if best_ or (epoch % print_freq == 0):
            # Print progress.
            print(f"epoch: {epoch: 3d}\t\t\t\t train:   valid:")
            print(
                f"    loss\t\t[a.u.]     \t{train_loss : 8.3f} {valid_loss : 8.3f} {best_loss:8.3f}"
            )
            print(
                f"    energy mae\t\t[kcal/mol]\t{train_energy_mae: 8.3f} {valid_energy_mae: 8.3f}"
            )
            print(
                f"    forces mae\t\t[kcal/mol/Å]\t{train_forces_mae: 8.3f} {valid_forces_mae: 8.3f}"
            )
            if doCharges:
                print(
                    f"    dipoles mae\t\t[e Å]     \t{train_dipoles_mae: 8.3f} {valid_dipoles_mae: 8.3f}"
                )
            print(
                "scale:",
                f"{transform_state.scale:.6f} {schedule_fn(epoch):.6f} LR={lr_eff:.9f}",
            )

    # Return final model parameters.
    return ema_params
