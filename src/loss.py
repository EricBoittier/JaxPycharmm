import sys

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
from dcmnet.analysis import create_model_and_params
from dcmnet.data import prepare_batches, prepare_datasets
from dcmnet.electrostatics import batched_electrostatic_potential, calc_esp
from dcmnet.modules import NATOMS, MessagePassingModel
from dcmnet.plotting import evaluate_dc, plot_esp, plot_model
from dcmnet.training import train_model
from dcmnet.training_dipole import train_model_dipo
from dcmnet.utils import apply_model, reshape_dipole, safe_mkdir
from jax.random import randint
from optax import contrib
from optax import tree_utils as otu
from tqdm import tqdm

# from jax import config
# config.update('jax_enable_x64', True)


DTYPE = jnp.float32


def mean_squared_loss(
    energy_prediction, energy_target, forces_prediction, forces_target, forces_weight
):
    energy_loss = jnp.mean(optax.l2_loss(energy_prediction, energy_target.reshape(-1)))
    forces_loss = jnp.mean(optax.l2_loss(forces_prediction, forces_target.squeeze()))
    return energy_loss + forces_weight * forces_loss


def mean_squared_loss_D(
    energy_prediction,
    energy_target,
    forces_prediction,
    forces_target,
    forces_weight,
    dipole_prediction,
    dipole_target,
    dipole_weight,
):
    energy_loss = jnp.mean(
        optax.l2_loss(energy_prediction.squeeze(), energy_target.squeeze())
    )
    forces_loss = jnp.mean(
        optax.l2_loss(forces_prediction.squeeze(), forces_target.squeeze())
    )
    dipole_loss = jnp.mean(
        optax.l2_loss(dipole_prediction.squeeze(), dipole_target.squeeze())
    )
    return energy_loss + forces_weight * forces_loss + dipole_weight * dipole_loss


def mean_squared_loss_QD(
    energy_prediction,
    energy_target,
    energy_weight,
    forces_prediction,
    forces_target,
    forces_weight,
    dipole_prediction,
    dipole_target,
    dipole_weight,
    total_charges_prediction,
    total_charge_target,
    total_charge_weight,
    atomic_mask,
):
    forces_prediction = forces_prediction * atomic_mask[..., None]
    forces_target = forces_target * atomic_mask[..., None]

    energy_loss = (
        jnp.sum(optax.l2_loss(energy_prediction.flatten(), energy_target.flatten()))
        / atomic_mask.sum()
    )
    forces_loss = (
        jnp.sum(optax.l2_loss(forces_prediction.flatten(), forces_target.flatten()))
        / atomic_mask.sum()
        * 3
    )
    dipole_loss = (
        jnp.sum(optax.l2_loss(dipole_prediction.flatten(), dipole_target.flatten())) / 3
    )
    charges_loss = (
        jnp.sum(
            optax.l2_loss(
                total_charges_prediction.flatten(), total_charge_target.flatten()
            )
        )
        / atomic_mask.sum()
    )
    # jax.debug.print("loss {x}",x=charges_loss)
    # jax.debug.print("pred {x}",x=total_charges_prediction.squeeze())
    # jax.debug.print("ref {x}",x=total_charge_target.squeeze())
    return (
        energy_weight * energy_loss
        + forces_weight * forces_loss
        + dipole_weight * dipole_loss
        + total_charge_weight * charges_loss
    )


def mean_absolute_error(prediction, target, nsamples):
    return jnp.sum(jnp.abs(prediction.squeeze() - target.squeeze())) / nsamples


@functools.partial(jax.jit, static_argnames=("batch_size"))
def dipole_calc(positions, atomic_numbers, charges, batch_segments, batch_size):
    """"""
    charges = charges.squeeze()
    positions = positions.squeeze()
    atomic_numbers = atomic_numbers.squeeze()
    masses = jnp.take(ase.data.atomic_masses, atomic_numbers)
    # nonzero = jnp.where(atomic_numbers != 0.0)
    bs_masses = jax.ops.segment_sum(
        masses, segment_ids=batch_segments, num_segments=batch_size
    )
    masses_per_atom = jnp.take(bs_masses, batch_segments)
    dis_com = positions * masses[..., None] / masses_per_atom[..., None]
    # jax.debug.print("dis_com {dis_com}", dis_com=dis_com)
    com = jnp.sum(dis_com, axis=1)
    # jax.debug.print("com {com}", com=com)
    pos_com = positions - com[..., None]
    # jax.debug.print("pos_com {pos_com}", pos_com=pos_com)
    # jax.debug.print("{com} {masses_per_atom}", com=com, masses_per_atom=masses_per_atom)
    dipoles = jax.ops.segment_sum(
        pos_com * charges[..., None],
        segment_ids=batch_segments,
        num_segments=batch_size,
    )

    # jax.debug.print("dipoles {dipoles}", dipoles=dipoles)
    return dipoles  # * 0.2081943 # to debye
