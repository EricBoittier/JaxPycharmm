import ctypes
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


class EF(nn.Module):
    features: int = 32
    max_degree: int = 3
    num_iterations: int = 2
    num_basis_functions: int = 16
    cutoff: float = 6.0
    max_atomic_number: int = 118  # This is overkill for most applications.
    charges: bool = False
    natoms: int = 60
    total_charge: float = 0
    n_res: int = 3
    debug: bool = False

    def energy(
        self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
    ):
        # 1. Calculate displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Shape (num_pairs, 3).
        # 2. Expand displacement vectors in basis functions.
        basis = e3x.nn.basis(  # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            # radial_fn=e3x.nn.reciprocal_bernstein,
            radial_fn=functools.partial(
                e3x.nn.exponential_gaussian,
                cuspless=False,
                use_exponential_weighting=True,
            ),
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )
        if self.debug:
            jax.debug.print("basis {x}", x=basis)
        # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.features,
            dtype=DTYPE,
        )(atomic_numbers)
        if self.debug:
            jax.debug.print("x {x}", x=x)
        x_hold = e3x.nn.Dense(self.features, use_bias=False)(x)
        x_hold = e3x.nn.hard_tanh(x_hold)

        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                # Since we will only use scalar features after the final message-pass, we do not want to produce non-scalar
                # features for efficiency reasons.
                x = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(
                    x, basis, dst_idx=dst_idx, src_idx=src_idx
                )
                # After the final message pass, we can safely throw away all non-scalar features.
                x = e3x.nn.change_max_degree_or_type(
                    x, max_degree=0, include_pseudotensors=False
                )
            else:
                # In intermediate iterations, the message-pass should consider all possible coupling paths.
                x = e3x.nn.MessagePass(include_pseudotensors=False)(
                    x, basis, dst_idx=dst_idx, src_idx=src_idx
                )

            for _ in range(self.n_res):
                y = e3x.nn.silu(x)
                # Atom-wise refinement MLP.
                y = e3x.nn.Dense(self.features)(y)
                y = e3x.nn.silu(y)
                y = e3x.nn.Dense(self.features)(y)
                # Residual connection.
                x = e3x.nn.add(x, y)

            y = e3x.nn.shifted_softplus(y)
            y = e3x.nn.Dense(self.features)(y)
            # Residual connection.
            x = e3x.nn.add(x_hold, y)

        if self.charges:
            charge_bias = self.param(
                "charge_bias",
                lambda rng, shape: jnp.zeros(shape),
                (self.max_atomic_number + 1),
            )
            atomic_charges = nn.Dense(
                1, use_bias=False, kernel_init=jax.nn.initializers.zeros, dtype=DTYPE
            )(x)
            atomic_charges += charge_bias[atomic_numbers][..., None, None, None]
            atomic_charges = atomic_charges.reshape(batch_size * self.natoms, 1)
            # constrain the total charge to the charge of the molecule
        #     sum_charges = jax.ops.segment_sum(
        #     atomic_charges,
        #         segment_ids=batch_segments,
        #         num_segments=batch_size
        # ) - self.total_charge
        #     sum_charges = sum_charges.reshape(batch_size, 1)
        #     atomic_charges_mean = jnp.take(sum_charges/self.natoms, batch_segments)
        #     atomic_charges = atomic_charges - atomic_charges_mean.reshape(atomic_charges.shape)

        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.max_atomic_number + 1),
        )
        atomic_energies = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.zeros, dtype=DTYPE
        )(x)
        atomic_energies = e3x.nn.silu(atomic_energies)
        atomic_energies = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.zeros, dtype=DTYPE
        )(atomic_energies)
        atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))
        atomic_energies += element_bias[atomic_numbers]

        energy = jax.ops.segment_sum(
            atomic_energies, segment_ids=batch_segments, num_segments=batch_size
        )

        if self.charges:
            # add on the electrostatic energy
            # displacements, positions_dst, positions_src, atomic_charges
            distances = (displacements**2).sum(axis=1) ** 0.5
            sqrt_sqr_distances_plus_1 = (distances**2 + 1) ** 0.5
            switch_dist = e3x.nn.smooth_switch(2 * distances, 0, 10)
            one_minus_switch_dist = 1 - switch_dist
            q1 = jnp.take(atomic_charges, dst_idx)
            q2 = jnp.take(atomic_charges, src_idx)
            q1_batches = jnp.take(batch_segments, dst_idx)
            Relectrostatics1 = switch_dist / sqrt_sqr_distances_plus_1
            Relectrostatics2 = one_minus_switch_dist / distances
            electrostatics = (
                7.199822675975274 * q1 * q2 * (Relectrostatics1 + Relectrostatics2)
            )
            electrostatics = jax.ops.segment_sum(
                electrostatics,
                segment_ids=q1_batches,
                num_segments=batch_size,  # *self.natoms
            )
            if self.debug:
                jax.debug.print("{x}", x=atomic_charges)
                jax.debug.print("{x}", x=atomic_energies)
                jax.debug.print("{x}", x=electrostatics)
                jax.debug.print("{x}", x=energy)
            return (
                -1 * (jnp.sum(energy + electrostatics)),
                (energy + electrostatics, atomic_charges, electrostatics),
            )  # Forces are the negative gradient, hence the minus sign.
        else:
            return (
                -jnp.sum(energy),
                energy,
            )  # Forces are the negative gradient, hence the minus sign.

    @nn.compact
    def __call__(
        self,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments=None,
        batch_size=None,
    ):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        # Since we want to also predict forces, i.e. the gradient of the energy w.r.t. positions (argument 1), we use
        # jax.value_and_grad to create a function for predicting both energy and forces for us.
        energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)

        if self.charges:
            (_, (energy, charges, electrostatics)), forces = energy_and_forces(
                atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
            )
            output = {
                "energy": energy,
                "forces": forces,
                "charges": charges,
                "electrostatics": electrostatics,
            }
            return output
        else:
            charges = False
            electrostatics = False
            (_, energy), forces = energy_and_forces(
                atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
            )
            output = {
                "energy": energy,
                "forces": forces,
                "charges": charges,
                "electrostatics": electrostatics,
            }
            return output
