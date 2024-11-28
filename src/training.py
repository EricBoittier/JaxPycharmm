import ctypes
import os
import pickle
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from flax.training import orbax_utils

# Add custom path
sys.path.append("/home/boittier/jaxeq/dcmnet")

import functools

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
from dcmnet.analysis import create_model_and_params
from dcmnet.data import prepare_batches, prepare_datasets
from dcmnet.electrostatics import batched_electrostatic_potential, calc_esp
from dcmnet.modules import NATOMS, MessagePassingModel
from dcmnet.plotting import evaluate_dc, plot_esp, plot_model
from dcmnet.training import train_model
from dcmnet.training_dipole import train_model_dipo
from dcmnet.utils import apply_model, reshape_dipole, safe_mkdir
from flax.training import checkpoints, train_state
from jax.random import randint
from optax import contrib
from optax import tree_utils as otu
from tqdm import tqdm

# from jax import config
# config.update('jax_enable_x64', True)
from loss import (
    dipole_calc,
    mean_absolute_error,
    mean_squared_loss,
    mean_squared_loss_D,
    mean_squared_loss_QD,
)

DTYPE = jnp.float32

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


@functools.partial(
    jax.jit,
    static_argnames=("model_apply", "optimizer_update", "batch_size", "doCharges"),
)
def train_step(
    model_apply,
    optimizer_update,
    transform_state,
    batch,
    batch_size,
    doCharges,
    forces_weight,
    charges_weight,
    opt_state,
    params,
    ema_params,
):
    if doCharges:

        def loss_fn(params):
            output = model_apply(
                params,
                atomic_numbers=batch["Z"],
                positions=batch["R"],
                dst_idx=batch["dst_idx"],
                src_idx=batch["src_idx"],
                batch_segments=batch["batch_segments"],
                batch_size=batch_size,
            )
            # jax.debug.print("{x}", x=output)
            # jax.debug.print("{x}", x=batch)
            dipole = dipole_calc(
                batch["R"],
                batch["Z"],
                output["charges"],
                batch["batch_segments"],
                batch_size,
            )
            sum_charges = jax.ops.segment_sum(
                output["charges"],
                segment_ids=batch["batch_segments"],
                num_segments=batch_size,
            )
            loss = mean_squared_loss_QD(
                energy_prediction=output["energy"],
                energy_target=batch["E"],
                forces_prediction=output["forces"],
                forces_target=batch["F"],
                forces_weight=forces_weight,
                dipole_prediction=dipole,
                dipole_target=batch["D"],
                dipole_weight=charges_weight,
                total_charges_prediction=sum_charges,
                total_charge_target=jnp.zeros_like(sum_charges),
                total_charge_weight=14.399645351950548,
            )
            return loss, (output["energy"], output["forces"], output["charges"], dipole)

    else:

        def loss_fn(params):
            output = model_apply(
                params,
                atomic_numbers=batch["Z"],
                positions=batch["R"],
                dst_idx=batch["dst_idx"],
                src_idx=batch["src_idx"],
                batch_segments=batch["batch_segments"],
                batch_size=batch_size,
            )
            loss = mean_squared_loss(
                energy_prediction=output["energy"],
                energy_target=batch["E"],
                forces_prediction=output["forces"],
                forces_target=batch["F"],
                forces_weight=forces_weight,
            )
            return loss, (energy, forces)

    if doCharges:
        (loss, (energy, forces, charges, dipole)), grad = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
    else:
        (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )

    updates, opt_state = optimizer_update(grad, opt_state, params)
    # update the reduce on plateau
    updates = otu.tree_scalar_mul(transform_state.scale, updates)
    params = optax.apply_updates(params, updates)

    energy_mae = mean_absolute_error(energy, batch["E"])
    forces_mae = mean_absolute_error(forces, batch["F"])
    if doCharges:
        dipole_mae = mean_absolute_error(dipole, batch["D"])
    else:
        dipole_mae = 0

    # Update EMA weights
    ema_decay = 0.999
    ema_params = jax.tree_map(
        lambda ema, new: ema_decay * ema + (1 - ema_decay) * new, ema_params, params
    )

    return (
        params,
        ema_params,
        opt_state,
        transform_state,
        loss,
        energy_mae,
        forces_mae,
        dipole_mae,
    )


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size", "charges"))
def eval_step(
    model_apply, batch, batch_size, charges, forces_weight, charges_weight, params
):
    if charges:
        output = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
        )

        dipole = dipole_calc(
            batch["R"],
            batch["Z"],
            output["charges"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
        )

        loss = mean_squared_loss_D(
            energy_prediction=output["energy"],
            energy_target=batch["E"],
            forces_prediction=output["forces"],
            forces_target=batch["F"],
            forces_weight=forces_weight,
            dipole_prediction=dipole,
            dipole_target=batch["D"],
            dipole_weight=charges_weight,
        )
        energy_mae = mean_absolute_error(output["energy"], batch["E"])
        forces_mae = mean_absolute_error(output["forces"], batch["F"])
        dipole_mae = mean_absolute_error(dipole, batch["D"])
        return loss, energy_mae, forces_mae, dipole_mae
    else:
        output = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
        )
        loss = mean_squared_loss(
            energy_prediction=output["energy"],
            energy_target=batch["E"],
            forces_prediction=output["forces"],
            forces_target=batch["F"],
            forces_weight=forces_weight,
        )
        energy_mae = mean_absolute_error(output["energy"], batch["E"])
        forces_mae = mean_absolute_error(output["forces"], batch["F"])
        return loss, energy_mae, forces_mae, 0


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
    data_keys=["R", "Z", "F", "E", "dst_idx", "src_idx", "batch_segments"],
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
        peak_value=learning_rate * 1.001,
        warmup_steps=1000,
        transition_steps=1000,
        decay_rate=0.99999,
    )
    optimizer = optax.chain(
        # optax.adaptive_grad_clip(1.0),
        optax.clip_by_global_norm(1000.0),
        optax.amsgrad(learning_rate=schedule_fn, b1=0.9, b2=0.99, eps=1e-3),
        # optax.adam(learning_rate=schedule_fn, b1=0.9, b2=0.99, eps=1e-3, eps_root=1e-8),
        # optax.adam(learning_rate=learning_rate),
        # optax.ema(decay=0.999, debias=False),
    )
    # optimizer = optax.adam(learning_rate=learning_rate)
    transform = optax.contrib.reduce_on_plateau(
        patience=10,
        cooldown=100,
        factor=0.95,
        rtol=1e-4,
        accumulation_size=5,
        min_scale=0.1,
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
    if restart:
        opt_state = optimizer.init(restart)
        params = restart
    else:
        opt_state = optimizer.init(params)

    ema_params = params

    # Creates initial state for `contrib.reduce_on_plateau` transformation.
    transform_state = transform.init(params)

    # Train for 'num_epochs' epochs.
    for epoch in range(1, num_epochs + 1):
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
                forces_weight=forces_weight,
                charges_weight=charges_weight,
                opt_state=opt_state,
                doCharges=doCharges,
                params=params,
                ema_params=ema_params,
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
        lr_eff = transform_state.scale * schedule_fn(epoch)
        best_ = False
        if valid_forces_mae < best_loss:
            state = train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=optimizer
            )
            # checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=epoch)
            ckpt = {
                "model": state,
                "transform_state": transform_state,
                "epoch": epoch,
                "best_loss": best_loss,
            }

            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(
                Path(CKPT_DIR) / f"epoch-{epoch}", ckpt, save_args=save_args
            )
            # update best loss
            best_loss = valid_forces_mae
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
