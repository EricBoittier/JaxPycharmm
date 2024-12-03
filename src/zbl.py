"""
JAX/Flax implementation of Ziegler-Biersack-Littmark nuclear repulsion model.

This module provides a neural network model for calculating nuclear repulsion
using the ZBL potential with smooth cutoffs.
"""

from typing import Any, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn import standardize

# Constants
BOHR_TO_ANGSTROM = 0.529177249  # Conversion factor from Bohr to Angstrom
HARTREE_TO_EV = 27.211386245988  # Conversion factor from Hartree to eV


class ZBLRepulsion(nn.Module):
    """Ziegler-Biersack-Littmark nuclear repulsion model.

    Attributes:
        cutoff: Upper cutoff distance
        cuton: Lower cutoff distance starting switch-off function
        trainable: If True, repulsion parameters are trainable
        dtype: Data type for computations
    """

    cutoff: float
    cuton: Optional[float] = None
    trainable: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        """Initialize model parameters."""
        # Default ZBL parameters
        a_coefficient = 0.8854  # Bohr
        a_exponent = 0.23
        phi_coefficients = [0.18175, 0.50986, 0.28022, 0.02817]
        phi_exponents = [3.19980, 0.94229, 0.40290, 0.20162]

        # Setup cutoffs
        self.cutoff_dist = jnp.array([self.cutoff], dtype=self.dtype)

        if self.cuton is not None and self.cuton < self.cutoff:
            self.cuton_dist = jnp.array([self.cuton], dtype=self.dtype)
            self.switchoff_range = jnp.array(
                [self.cutoff - self.cuton], dtype=self.dtype
            )
            self.use_switch = True
        else:
            self.cuton_dist = jnp.array([0.0], dtype=self.dtype)
            self.switchoff_range = jnp.array([self.cutoff], dtype=self.dtype)
            self.use_switch = True if self.cuton is None else False

        # Initialize parameters
        def make_param(name, value):
            if self.trainable:
                return self.param(name, lambda key: jnp.array(value, dtype=self.dtype))
            return jnp.array(value, dtype=self.dtype)

        self.a_coefficient = make_param("a_coefficient", a_coefficient)
        self.a_exponent = make_param("a_exponent", a_exponent)
        self.phi_coefficients = make_param("phi_coefficients", phi_coefficients)
        self.phi_exponents = make_param("phi_exponents", phi_exponents)

    def switch_fn(self, distances: jnp.ndarray) -> jnp.ndarray:
        """Compute smooth switch factors from 1 to 0.

        Args:
            distances: Array of interatomic distances

        Returns:
            Array of switch factors
        """
        x = (self.cutoff_dist - distances) / self.switchoff_range

        switch = jnp.where(
            distances < self.cuton_dist,
            jnp.ones_like(x),
            jnp.where(
                distances >= self.cutoff_dist,
                jnp.zeros_like(x),
                ((6.0 * x - 15.0) * x + 10.0) * x**3,
            ),
        )
        return switch

    def __call__(
        self,
        atomic_numbers: jnp.ndarray,
        displacements: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_size: int,
    ) -> jnp.ndarray:
        """Calculate ZBL nuclear repulsion energies.

        Args:
            atomic_numbers: Array of atomic numbers
            distances: Array of interatomic distances
            idx_i: Array of indices for first atoms in pairs
            idx_j: Array of indices for second atoms in pairs

        Returns:
            Array of repulsion energies per atom
        """
        # Compute distances with numerical stability
        distances = jnp.maximum(jnp.linalg.norm(displacements, axis=-1), 1e-10)

        # Compute switch-off function
        if self.use_switch:
            switch_off = self.switch_fn(distances)
        else:
            switch_off = jnp.where(
                distances < self.cutoff_dist,
                jnp.ones_like(distances),
                jnp.zeros_like(distances),
            )

        # Compute atomic number dependent screening length with safe operations
        za = jnp.power(atomic_numbers, jnp.abs(self.a_exponent))
        denominator = jnp.maximum(za[idx_i] + za[idx_j], 1e-10)
        a_ij = jnp.abs(self.a_coefficient) / denominator

        # Compute screening function phi with numerical stability
        arguments = distances / a_ij
        # Normalize coefficients using softmax for better numerical stability
        coefficients = jax.nn.softmax(jnp.abs(self.phi_coefficients))
        exponents = jnp.abs(self.phi_exponents)

        # Use log-space operations for numerical stability
        log_terms = -exponents[None, ...] * arguments[..., None]
        max_log = jnp.max(log_terms, axis=1, keepdims=True)
        exp_terms = jnp.exp(log_terms - max_log)
        phi = jnp.sum(coefficients[None, ...] * exp_terms, axis=1)

        # Compute nuclear repulsion potential with numerical stability
        # Factor 1.0 represents e^2/(4πε₀) in atomic units
        # Use log-space operations for better numerical stability
        log_repulsion = (
            jnp.log(0.5)
            + jnp.log(atomic_numbers[idx_i])
            + jnp.log(atomic_numbers[idx_j])
            - jnp.log(distances)
            + jnp.log(jnp.maximum(phi, 1e-30))
            + jnp.log(jnp.maximum(switch_off, 1e-30))
        )

        repulsion = jnp.exp(log_repulsion)

        # Apply batch segmentation
        repulsion = jnp.multiply(repulsion, batch_mask)

        # Sum contributions for each atom using safe operations
        Erep = jax.ops.segment_sum(
            repulsion, segment_ids=idx_i, num_segments=atomic_numbers.shape[0]
        )

        # Apply atom mask
        Erep = jnp.multiply(Erep, atom_mask)

        # print everything for temporary debugging
        jax.debug.print("erep {x} {y}", x=Erep, y=Erep.shape)
        jax.debug.print("dist {x} {y}", x=distances, y=distances.shape)
        jax.debug.print("switch {x} {y}", x=switch_off, y=switch_off.shape)
        jax.debug.print("phi {x} {y}", x=phi, y=phi.shape)
        jax.debug.print("rep {x} {y}", x=repulsion, y=repulsion.shape)
        jax.debug.print("a {x} {y}", x=a_ij, y=a_ij.shape)
        jax.debug.print("denom {x} {y}", x=denominator, y=denominator.shape)
        jax.debug.print("za {x} {y}", x=za, y=za.shape)
        jax.debug.print("dist {x} {y}", x=distances, y=distances.shape)
        jax.debug.print("idxi {x} {y}", x=idx_i, y=idx_i.shape)
        jax.debug.print("idxj {x} {y}", x=idx_j, y=idx_j.shape)
        jax.debug.print("atom {x} {y}", x=atomic_numbers, y=atomic_numbers.shape)
        jax.debug.print("rep {x} {y}", x=repulsion, y=repulsion.shape)

        return Erep[..., None, None, None]
