import sys
from typing import List

from ase import atom

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
    debug: bool | List[str] = False

    def energy(
        self,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments,
        batch_size,
        batch_mask,
        atom_mask,
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
            radial_fn=e3x.nn.reciprocal_bernstein,
            # radial_fn=functools.partial(
            #    e3x.nn.exponential_gaussian,
            #    cuspless=False,
            #    use_exponential_weighting=True,
            # ),
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )
        # if self.debug:
        #     jax.debug.print("basis {x}", x=basis)
        # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.features,
            dtype=DTYPE,
        )(atomic_numbers)

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
                y = e3x.nn.relu(y)
                y = e3x.nn.Dense(self.features)(y)
                # Residual connection.
                x = e3x.nn.add(x, y)

            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            # Residual connection.
            x = e3x.nn.add(x, y)
        # x = e3x.nn.silu(x)
        # jax.debug.print("nans in x? {x}", x=jnp.isnan(x).any())

        if self.charges:
            charge_bias = self.param(
                "charge_bias",
                lambda rng, shape: jnp.zeros(shape),
                (self.max_atomic_number + 1),
            )
            atomic_charges = nn.Dense(
                1, use_bias=False, kernel_init=jax.nn.initializers.zeros, dtype=DTYPE
            )(x)
            # atomic_charges = e3x.nn.hard_tanh(atomic_charges)
            atomic_charges += charge_bias[atomic_numbers][..., None, None, None]
            atomic_charges *= atom_mask[..., None, None, None]
            # atomic_charges = e3x.nn.hard_tanh(atomic_charges)
            # atomic_charges = atomic_charges.reshape(batch_size * self.natoms, 1)
            # atomic_charges = jnp.nan_to_num(atomic_charges, nan=0.0)
            # constrain the total charge to the charge of the molecule
            #     sum_charges = jax.ops.segment_sum(
            #     atomic_charges,
            #         segment_ids=batch_segments,
            #         num_segments=batch_size
            # ) - self.total_charge
            #     sum_charges = sum_charges.reshape(batch_size, 1)
            #     atomic_charges_mean = jnp.take(sum_charges/self.natoms, batch_segments)
            #     atomic_charges = atomic_charges - atomic_charges_mean.reshape(atomic_charges.shape)
            # add on the electrostatic energy
            # displacements, positions_dst, positions_src, atomic_charges
            # valid indices
            # valid_idx = jnp.where(atomic_numbers != 0)
            # jax.debug.print("valid_idx {x}", x=valid_idx)
            # jax.debug.print("batch_mask {x}", x=batch_mask)
            # jax.debug.print("displacements {x}", x=displacements)
            displacements = displacements + (1 - batch_mask)[..., None]
            distances = (displacements**2).sum(axis=1) ** 0.5
            if self.debug and "dist" in self.debug:
                jax.debug.print("atomic_numbers {x}", x=atomic_numbers)
                jax.debug.print("atom mask {x}", x=atom_mask.shape)
                jax.debug.print("atom mask {x}", x=atom_mask)
                jax.debug.print("batch mask {x}", x=batch_mask.shape)
                jax.debug.print(
                    "distances has 0 {c}", c=jnp.isnan(jnp.log(distances)).any()
                )
                jax.debug.print("distances {c}", c=distances)
                jax.debug.print("batch_mask {x}", x=batch_mask)
                jax.debug.print(
                    "distances has 0 {c}", c=jnp.isnan(jnp.log(distances)).any()
                )
                jax.debug.print("distances {c}", c=distances)
            # distances = jnp.nan_to_num(distances, nan=10000)
            sqrt_sqr_distances_plus_1 = (distances**2 + 1) ** 0.5
            switch_dist = e3x.nn.smooth_switch(2 * distances, 0, 10)
            one_minus_switch_dist = 1 - switch_dist

            q1 = jnp.take(atomic_charges, dst_idx, fill_value=0.0)
            q2 = jnp.take(atomic_charges, src_idx, fill_value=0.0)
            if self.debug and "batch" in self.debug:
                jax.debug.print("batch segments {x}", x=batch_segments)
            q1_batches = jnp.take(
                batch_segments,
                dst_idx,  # fill_value=batch_size * self.natoms + 1
            )

            R1 = switch_dist / sqrt_sqr_distances_plus_1
            R2 = one_minus_switch_dist / distances
            R = R1 + R2
            # R = jnp.nan_to_num(R, nan=0.0)
            # jax.debug.print("batch_mask {x}", x=batch_mask)
            # R = R * batch_mask
            electrostatics = 7.199822675975274 * q1 * q2 * R
            electrostatics = electrostatics * batch_mask
            # electrostatics = jnp.nan_to_num(electrostatics, nan=0.0)
            if self.debug and "ele" in self.debug:
                jax.debug.print("q1 {x}", x=q1)
                jax.debug.print("nan in q1? {x}", x=jnp.isnan(q1).any())
                jax.debug.print("q2 {x}", x=q2)
                jax.debug.print("nan in q2? {x}", x=jnp.isnan(q2).any())
                jax.debug.print("ele1 {x}", x=electrostatics)
                jax.debug.print("nan in ele1? {x}", x=jnp.isnan(electrostatics).any())
                jax.debug.print("R {x}", x=R)
                jax.debug.print("nan in R? {x}", x=jnp.isnan(R).any())
                jax.debug.print("q1_batches {x}", x=q1_batches.shape)
                jax.debug.print("nan in q1_batches? {x}", x=jnp.isnan(q1_batches).any())

            # do the sum over all atoms
            atomic_electrostatics = jax.ops.segment_sum(
                electrostatics,
                segment_ids=dst_idx,
                num_segments=batch_size * self.natoms,
            )
            atomic_electrostatics = atomic_electrostatics * atom_mask
            batch_electrostatics = jax.ops.segment_sum(
                atomic_electrostatics,
                segment_ids=batch_segments,
                num_segments=batch_size,
            )

            # electrostatics = electrostatics  # * (atomic_numbers != 0)
            if self.debug and "ele" in self.debug:
                jax.debug.print("batch_electrostatics {x}", x=batch_electrostatics)
                jax.debug.print("electrostatics! {x}", x=atomic_electrostatics)
                # jax.debug.print("q {x}", x=atomic_charges)
                jax.debug.print("nan in q? {x}", x=jnp.isnan(atomic_charges).any())

                jax.debug.print("ele {x}", x=electrostatics)
                jax.debug.print("nan in ele2? {x}", x=jnp.isnan(electrostatics).any())

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
        atomic_energies = atomic_energies * atom_mask
        # atomic_energies *= atomic_numbers != 0
        if self.debug and "e" in self.debug:
            jax.debug.print("atomic_energies {x}", x=atomic_energies.shape)
            jax.debug.print("atomic_energies {x}", x=atomic_energies)
        if self.charges:
            energy = jax.ops.segment_sum(
                atomic_energies + atomic_electrostatics,
                segment_ids=batch_segments,
                num_segments=batch_size,
            )
            return (
                -1 * jnp.sum(energy),
                (energy, atomic_charges, batch_electrostatics),
            )  # Forces are the negative gradient, hence the minus sign.

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
        batch_mask=None,
        atom_mask=None,
    ):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1
            batch_mask = jnp.ones_like(dst_idx)
            atom_mask = jnp.ones_like(atomic_numbers)

        # print("atom_mask", atom_mask)
        # Since we want to also predict forces, i.e. the gradient of the energy w.r.t. positions (argument 1), we use
        # jax.value_and_grad to create a function for predicting both energy and forces for us.
        energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)

        if "idx" in self.debug:
            jax.debug.print("dst_idx {x}", x=dst_idx.shape)
            jax.debug.print("src_idx {x}", x=src_idx.shape)
            jax.debug.print("batch_segments {x}", x=batch_segments.shape)
            jax.debug.print("batch_size {x}", x=batch_size)
            jax.debug.print("batch_mask {x}", x=batch_mask.shape)
            jax.debug.print("atom_mask {x}", x=atom_mask.shape)

        if self.charges:
            (_, (energy, charges, electrostatics)), forces = energy_and_forces(
                atomic_numbers,
                positions,
                dst_idx,
                src_idx,
                batch_segments,
                batch_size,
                batch_mask,
                atom_mask,
            )
            # charges = jnp.nan_to_num(charges, nan=0.0)
            # forces = jnp.nan_to_num(forces, nan=0.0)
            forces = forces * atom_mask[..., None]

            output = {
                "energy": energy,
                "forces": forces,
                "charges": charges,
                "electrostatics": electrostatics,
            }
            if "forces" in self.debug:
                for k in output.keys():
                    hasnans = jnp.isnan(output[k]).any()
                    jax.debug.print("{k} {nank} {blah}", k=k, nank=hasnans, blah="")
                jax.debug.print("forces {x}", x=forces)
            return output
        else:
            charges = False
            electrostatics = False
            (_, energy), forces = energy_and_forces(
                atomic_numbers,
                positions,
                dst_idx,
                src_idx,
                batch_segments,
                batch_size,
                batch_mask,
                atom_mask,
            )

            output = {
                "energy": energy,
                "forces": forces,
                "charges": charges,
                "electrostatics": electrostatics,
            }
            return output
