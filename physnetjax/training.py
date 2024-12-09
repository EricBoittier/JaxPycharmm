import uuid
from pathlib import Path
from typing import Tuple

import ase
import e3x
import jax
import orbax
import orbax.checkpoint
import tensorflow as tf
from flax.training import orbax_utils, train_state
from jax import random

import physnetjax
from physnetjax.data import prepare_batches
from physnetjax.pretty_printer import epoch_printer
from physnetjax.evalstep import eval_step
from physnetjax.optimizer import (
    base_optimizer,
    base_schedule_fn,
    base_transform,
    get_optimizer,
)
from physnetjax.tensorboard_logging import write_tb_log
from physnetjax.trainstep import train_step
from physnetjax.utils import get_last, get_params_model, pretty_print
from physnetjax.pretty_printer import init_table, epoch_printer, training_printer
from rich.console import Console
console = Console()

schedule_fn = base_schedule_fn
transform = base_transform
optimizer = base_optimizer

BASE_CKPT_DIR = Path("/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts")

# Energy/force unit conversions
CONVERSION = {
    "energy": 1 / (ase.units.kcal / ase.units.mol),
    "forces": 1 / (ase.units.kcal / ase.units.mol),
}

# Initialize checkpointer
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def create_checkpoint_dir(name: str, base: Path) -> Path:
    """Create a unique checkpoint directory path.

    Args:
        name: Base name for the checkpoint directory

    Returns:
        Path object for the checkpoint directory
    """
    uuid_ = str(uuid.uuid4())
    return base / f"/{name}-{uuid_}/"


def get_epoch_weights(epoch: int) -> Tuple[float, float]:
    """Calculate energy and forces weights based on epoch number.

    Args:
        epoch: Current training epoch

    Returns:
        Tuple of (energy_weight, forces_weight)
    """
    if epoch < 500:
        return 1.0, 1000.0
    elif epoch < 1000:
        return 1000.0, 1.0
    else:
        return 1.0, 50.0





def train_model(
    key,
    model,
    train_data,
    valid_data,
    num_epochs=1,
    learning_rate=0.001,
    energy_weight=1.0,
    forces_weight=52.91772105638412,
    dipole_weight=27.211386024367243,
    charges_weight=14.399645351950548,
    batch_size=1,
    num_atoms=60,
    restart=False,
    conversion=CONVERSION,
    print_freq=1,
    name="test",
    best=False,
    optimizer=None,
    transform=None,
    schedule_fn=None,
    objective="valid_forces_mae",
    ckpt_dir=BASE_CKPT_DIR,
    log_tb=False,
    data_keys=["R", "Z", "F", "E", "D", "dst_idx", "src_idx", "batch_segments"],
):
    """Train a model."""

    print("Training Routine")

    best_loss = 10000
    doCharges = model.charges
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)
    optimizer, transform, schedule_fn = get_optimizer(
        learning_rate=learning_rate,
        schedule_fn=schedule_fn,
        optimizer=optimizer,
        transform=transform,
    )
    pretty_print(optimizer, transform, schedule_fn)

    table = training_printer(learning_rate, energy_weight, forces_weight, dipole_weight, charges_weight, batch_size, num_atoms,
                         restart, conversion, print_freq, name, best, objective, data_keys, ckpt_dir, train_data,
                         valid_data)
    console.print(table)

    table = init_table()

    uuid_ = str(uuid.uuid4())
    CKPT_DIR = ckpt_dir / f"{name}-{uuid_}"
    writer = None

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
        # print(_, _model)
        if _model is not None:
            model = _model
        restored = orbax_checkpointer.restore(restart)
        print("Restoring from", restart)
        print("Restored keys:", restored.keys())
        params = restored["params"]
        ema_params = restored["ema_params"]
        transform_state = transform.init(restored["transform_state"])
        print("transform_state", transform_state)
        step = restored["epoch"] + 1
        best_loss = restored["best_loss"]
        CKPT_DIR = Path(restart).parent  # optimizer = restored["optimizer"]
    # initialize
    else:
        ema_params = params
        best_loss = 10000
        step = 1
        opt_state = optimizer.init(params)
        transform_state = transform.init(params)

    if best:
        best_loss = best

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    # print("Trainable params:", state.num_params)
    print("model", model)

    runInDebug = True if model.debug else False

    # Train for 'num_epochs' epochs.
    for epoch in range(step, num_epochs + 1):

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
                dipole_weight=dipole_weight,
                charges_weight=charges_weight,
                opt_state=opt_state,
                doCharges=doCharges,
                params=params,
                ema_params=ema_params,
                debug=runInDebug,
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
                dipole_weight=dipole_weight,
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

        obj_res = {
            "valid_energy_mae": valid_energy_mae,
            "valid_forces_mae": valid_forces_mae,
            "train_energy_mae": train_energy_mae,
            "train_forces_mae": train_forces_mae,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "lr": lr_eff,
            "batch_size": batch_size,
            "energy_w": energy_weight,
            "charges_w": charges_weight,
            "dipole_w": dipole_weight,
            "forces_w": forces_weight,
        }

        if log_tb:
            writer = tf.summary.create_file_writer(str(CKPT_DIR / "tfevents"))
            # Correct usage within the context manager
            # Use the writer for logging
            writer.set_as_default()
            # Log to TensorBoard
            write_tb_log(writer, obj_res, epoch)  # Call your logging function here

        best_ = False
        if obj_res[objective] < best_loss:

            model_attributes = model.return_attributes()

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
            print(CKPT_DIR / f"epoch-{epoch}")
            orbax_checkpointer.save(
                CKPT_DIR / f"epoch-{epoch}", ckpt, save_args=save_args
            )
            # update best loss
            best_loss = obj_res[objective]
            print("best!")
            best_ = True

    if best_ or (epoch % print_freq == 0):

        table = epoch_printer(table, epoch, train_loss, valid_loss, best_loss, train_energy_mae, valid_energy_mae,
                      train_forces_mae, valid_forces_mae, doCharges, train_dipoles_mae, valid_dipoles_mae,
                      transform_state.state, schedule_fn(epoch), lr_eff)
        console.print(table)

    # Return final model parameters.
    return ema_params
