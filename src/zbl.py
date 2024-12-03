
"""
JAX/Flax implementation of Ziegler-Biersack-Littmark nuclear repulsion model.

This module provides a neural network model for calculating nuclear repulsion
using the ZBL potential with smooth cutoffs.
"""

from typing import Dict, Optional, Any
import jax
import jax.numpy as jnp
import flax.linen as nn

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
            self.switchoff_range = jnp.array([self.cutoff - self.cuton], dtype=self.dtype)
            self.use_switch = True
        else:
            self.cuton_dist = jnp.array([0.0], dtype=self.dtype)
            self.switchoff_range = jnp.array([self.cutoff], dtype=self.dtype)
            self.use_switch = True if self.cuton is None else False

        # Initialize parameters
        param_init = lambda x: self.param(
            'param',
            lambda _: jnp.array(x, dtype=self.dtype),
            x
        ) if self.trainable else jnp.array(x, dtype=self.dtype)

        self.a_coefficient = param_init(a_coefficient)
        self.a_exponent = param_init(a_exponent)
        self.phi_coefficients = param_init(phi_coefficients)
        self.phi_exponents = param_init(phi_exponents)

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
                ((6.0 * x - 15.0) * x + 10.0) * x ** 3
            )
        )
        return switch

    def __call__(
            self,
            atomic_numbers: jnp.ndarray,
            distances: jnp.ndarray,
            idx_i: jnp.ndarray,
            idx_j: jnp.ndarray,
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
        # Compute switch-off function
        if self.use_switch:
            switch_off = self.switch_fn(distances)
        else:
            switch_off = jnp.where(
                distances < self.cutoff_dist,
                jnp.ones_like(distances),
                jnp.zeros_like(distances)
            )

        # Compute atomic number dependent screening length
        za = atomic_numbers ** jnp.abs(self.a_exponent)
        a_ij = (
                jnp.abs(self.a_coefficient)
                / (za[idx_i] + za[idx_j])
        )

        # Compute screening function phi
        arguments = distances / a_ij
        coefficients = jax.nn.normalize(
            jnp.abs(self.phi_coefficients), axis=0, ord=1
        )
        exponents = jnp.abs(self.phi_exponents)
        phi = jnp.sum(
            coefficients[None, ...] * jnp.exp(
                -exponents[None, ...] * arguments[..., None]
            ),
            axis=1
        )

        # Compute nuclear repulsion potential
        # Factor 1.0 represents e^2/(4πε₀) in atomic units
        repulsion = (
                0.5 * 1.0
                * atomic_numbers[idx_i] * atomic_numbers[idx_j] / distances
                * phi
                * switch_off
        )

        # Sum contributions for each atom
        Erep = jax.ops.segment_sum(
            repulsion,
            segment_ids=idx_i,
            num_segments=atomic_numbers.shape[0]
        )

        return Erep
