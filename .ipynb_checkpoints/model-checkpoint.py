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

import jax
#from jax import config
#config.update('jax_enable_x64', True)

import dcmnet
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import optax
from optax import tree_utils as otu
from optax import contrib

from dcmnet.analysis import create_model_and_params
from dcmnet.data import prepare_batches, prepare_datasets
from dcmnet.modules import MessagePassingModel
from dcmnet.plotting import evaluate_dc, plot_esp, plot_model
from dcmnet.training import train_model
from dcmnet.training_dipole import train_model_dipo
from dcmnet.utils import apply_model, safe_mkdir
from tqdm import tqdm
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dcmnet.electrostatics import batched_electrostatic_potential, calc_esp
from dcmnet.modules import NATOMS
from dcmnet.utils import reshape_dipole
from jax.random import randint

import ase

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
            radial_fn=functools.partial(e3x.nn.exponential_gaussian, 
                                        cuspless = False, use_exponential_weighting = True),
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )
        # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1, features=self.features, dtype=DTYPE
        )(atomic_numbers)
        x_hold = e3x.nn.Dense(self.features, use_bias=False)(x)
        x_hold = e3x.nn.hard_tanh(x_hold)
        
        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
          # Message-pass.
          if i == self.num_iterations-1:  # Final iteration.
            # Since we will only use scalar features after the final message-pass, we do not want to produce non-scalar
            # features for efficiency reasons.
            x = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, 
                                                                              dst_idx=dst_idx, src_idx=src_idx)
            # After the final message pass, we can safely throw away all non-scalar features.
            x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
          else:
            # In intermediate iterations, the message-pass should consider all possible coupling paths.
            x = e3x.nn.MessagePass(include_pseudotensors=False)(x, basis, 
                                                                dst_idx=dst_idx, src_idx=src_idx)

          
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
            atomic_charges = nn.Dense(1, use_bias=False,
                                      kernel_init=jax.nn.initializers.zeros,dtype=DTYPE)(x) 
            atomic_charges += charge_bias[atomic_numbers][..., None, None, None]
            atomic_charges = atomic_charges.reshape(batch_size*self.natoms, 1)
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
            1, use_bias=False, kernel_init=jax.nn.initializers.zeros,dtype=DTYPE
        )(
            x
        )  
        atomic_energies = e3x.nn.silu(atomic_energies)
        atomic_energies = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.zeros,dtype=DTYPE
        )(
            atomic_energies
        )    
        atomic_energies = jnp.squeeze(
            atomic_energies, axis=(-1, -2, -3)
        )  
        atomic_energies += element_bias[atomic_numbers]

        energy = jax.ops.segment_sum(
            atomic_energies, segment_ids=batch_segments, num_segments=batch_size
        )

        if self.charges:
            # add on the electrostatic energy
            # displacements, positions_dst, positions_src, atomic_charges
            distances = (displacements**2).sum(axis=1)**0.5
            sqrt_sqr_distances_plus_1 = (distances**2 + 1)**0.5
            switch_dist = e3x.nn.smooth_switch(2*distances, 0, 10)
            one_minus_switch_dist = 1 - switch_dist
            q1 = jnp.take(atomic_charges, dst_idx)
            q2 = jnp.take(atomic_charges, src_idx)
            q1_batches = jnp.take(batch_segments, dst_idx)
            Relectrostatics1 = switch_dist / sqrt_sqr_distances_plus_1
            Relectrostatics2 = one_minus_switch_dist / distances
            electrostatics =  7.199822675975274 * q1*q2*(Relectrostatics1+Relectrostatics2)
            electrostatics = jax.ops.segment_sum(
            electrostatics, 
                segment_ids=q1_batches, 
                num_segments=batch_size #*self.natoms
        )
            return (
                -1 * (jnp.sum(energy + electrostatics)),
                (energy+electrostatics, atomic_charges, electrostatics)
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
            output = {"energy": energy, "forces": forces, "charges": charges, "electrostatics": electrostatics}
            return output
        else:
            charges = False
            electrostatics = False
            (_, energy), forces = energy_and_forces(
                atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
            )
            output = {"energy":energy, "forces": forces, "charges": charges, "electrostatics": electrostatics}
            return output





def mean_squared_loss(
    energy_prediction, energy_target, forces_prediction, forces_target, forces_weight
):
    energy_loss = jnp.mean(optax.l2_loss(energy_prediction, energy_target.reshape(-1)))
    forces_loss = jnp.mean(optax.l2_loss(forces_prediction, forces_target.squeeze()))
    return energy_loss + forces_weight * forces_loss

def mean_squared_loss_D(
    energy_prediction, energy_target, 
    forces_prediction, forces_target, forces_weight, 
    dipole_prediction, dipole_target, dipole_weight, 
):
    energy_loss = jnp.mean(optax.l2_loss(energy_prediction.squeeze(), 
                                         energy_target.squeeze()))
    forces_loss = jnp.mean(optax.l2_loss(forces_prediction.squeeze(), 
                                         forces_target.squeeze()))
    dipole_loss = jnp.mean(optax.l2_loss(dipole_prediction.squeeze(), 
                                         dipole_target.squeeze()))
    return energy_loss + forces_weight * forces_loss + dipole_weight * dipole_loss

def mean_squared_loss_QD(
    energy_prediction, energy_target, 
    forces_prediction, forces_target, forces_weight, 
    dipole_prediction, dipole_target, dipole_weight, 
    total_charges_prediction, total_charge_target, total_charge_weight
):
    energy_loss = jnp.mean(optax.l2_loss(energy_prediction.flatten(), 
                                         energy_target.flatten()))
    forces_loss = jnp.mean(optax.l2_loss(forces_prediction.flatten(), 
                                         forces_target.flatten()))
    dipole_loss = jnp.mean(optax.l2_loss(dipole_prediction.flatten(), 
                                         dipole_target.flatten()))
    charges_loss = jnp.mean(optax.l2_loss(total_charges_prediction.flatten(), 
                                         total_charge_target.flatten()))
    # jax.debug.print("loss {x}",x=charges_loss)
    # jax.debug.print("pred {x}",x=total_charges_prediction.squeeze())
    # jax.debug.print("ref {x}",x=total_charge_target.squeeze())
    return energy_loss + forces_weight * forces_loss + dipole_weight * dipole_loss + total_charge_weight * charges_loss 

def mean_absolute_error(prediction, target):
    return jnp.mean(jnp.abs(prediction.squeeze() - target.squeeze()))

@functools.partial(
    jax.jit, static_argnames=("model_apply", "optimizer_update", "batch_size", "doCharges")
)
def train_step(
    model_apply, optimizer_update, transform_state, batch, batch_size, doCharges, forces_weight, charges_weight, opt_state, params, ema_params
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
            dipole = dipole_calc(batch["R"], batch["Z"], output["charges"], batch["batch_segments"], batch_size)
            sum_charges = jax.ops.segment_sum(
            output["charges"], 
                segment_ids=batch["batch_segments"], 
                num_segments=batch_size
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
                total_charges_prediction = sum_charges,
                total_charge_target = jnp.zeros_like(sum_charges),
                total_charge_weight = 14.399645351950548,
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
        (loss, (energy, forces,charges, dipole)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    else:
        (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
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
        lambda ema, new: ema_decay * ema + (1 - ema_decay) * new,
        ema_params, params
    )
        
    return params, ema_params, opt_state, transform_state, loss, energy_mae, forces_mae, dipole_mae

@functools.partial(jax.jit, static_argnames=("batch_size"))
def dipole_calc(positions, atomic_numbers, charges, batch_segments, batch_size):
    """"""
    charges = charges.squeeze()
    positions = positions.squeeze()
    atomic_numbers = atomic_numbers.squeeze()
    masses = jnp.take(ase.data.atomic_masses, atomic_numbers)
    bs_masses = jax.ops.segment_sum(
            masses, segment_ids=batch_segments, num_segments=batch_size
        )
    masses_per_atom = jnp.take(bs_masses, batch_segments)
    com = jnp.sum(positions*masses[...,None]/ masses_per_atom[...,None], axis=1)
    pos_com = positions - com[...,None]
    # jax.debug.print("{com} {masses_per_atom}", com=com, masses_per_atom=masses_per_atom)
    dipoles = jax.ops.segment_sum(pos_com*charges[...,None], segment_ids=batch_segments, num_segments=batch_size)
    # jax.debug.print("{dipoles}",dipoles=dipoles)
    return dipoles #* 0.2081943 # to debye


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size", "charges"))
def eval_step(model_apply, batch, batch_size, charges, forces_weight, charges_weight, params):
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
        
        dipole = dipole_calc(batch["R"], batch["Z"], output["charges"], batch_segments=batch["batch_segments"], batch_size=batch_size)
        
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

conversion = {"energy": 1/(ase.units.kcal/ase.units.mol), "forces": 1/(ase.units.kcal/ase.units.mol)}

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
    print_freq = 1,
    name = "test",
    data_keys = ["R", "Z", "F", "E", "dst_idx", "src_idx", "batch_segments"]
):
    best_loss = 10000
    doCharges = model.charges
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)

    
    schedule_fn = optax.schedules.warmup_exponential_decay_schedule(
        init_value=learning_rate,
        peak_value=learning_rate*1.001,
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
    # optax.adam(learning_rate=learning_rate)
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

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data["Z"][0]))
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
            params, ema_params, opt_state, transform_state, loss, energy_mae, forces_mae, dipole_mae = train_step(
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
        valid_energy_mae*=conversion["energy"]
        valid_forces_mae*=conversion["forces"]
        train_energy_mae*=conversion["energy"]
        train_forces_mae*=conversion["forces"]

        lr_eff = transform_state.scale*schedule_fn(epoch)
        best_ = False
        if valid_forces_mae < best_loss:
            with open(f"checkpoints/{name}.pkl", "wb") as file:
                pickle.dump(ema_params, file)
            best_loss = valid_forces_mae
            print("best!")
            best_ = True
        
        if best_ or (epoch % print_freq == 0):
            # Print progress.
            print(f"epoch: {epoch: 3d}\t\t\t\t train:   valid:")
            print(f"    loss\t\t[a.u.]     \t{train_loss : 8.3f} {valid_loss : 8.3f} {best_loss:8.3f}")
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
            print("scale:", f"{transform_state.scale:.6f} {schedule_fn(epoch):.6f} LR={lr_eff:.9f}")

    # Return final model parameters.
    return ema_params


