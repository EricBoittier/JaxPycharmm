from typing import Dict, List, Optional

import e3x.ops
import jax
import jax.numpy as jnp
from ase.units import Bohr, Hartree

# Constants
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr
MAX_N_ATOMS = 37
MAX_GRID_POINTS = 10000
BOHR_TO_ANGSTROM = 0.529177


def prepare_batches_one(
    key,
    data,
    batch_size,
    include_id=False,
    data_keys=None,
    num_atoms=60,
    dst_idx=None,
    src_idx=None,
) -> list:
    """
    Prepare batches for training.

    Args:
        key: Random key for shuffling.
        data (dict): Dictionary containing the dataset.
        batch_size (int): Size of each batch.
        include_id (bool): Whether to include ID in the output.
        data_keys (list): List of keys to include in the output.

    Returns:
        list: A list of dictionaries, each representing a batch.
    """
    # Determine the number of training steps per epoch.
    # print(batch_size)
    data_size = len(data["R"])
    # print("data_size", data_size)
    steps_per_epoch = data_size // batch_size

    # Draw random permutations for fetching batches from the train data.
    perms = jax.random.permutation(key, data_size)
    perms = perms[
        : steps_per_epoch * batch_size
    ]  # Skip the last batch (if incomplete).
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Prepare entries that are identical for each batch.
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = dst_idx + offsets[:, None]  # .reshape(-1)  # * good_indices
    src_idx = src_idx + offsets[:, None]  # .reshape(-1)  # * good_indices

    output = []
    for perm in perms:
        # print(perm)
        dict_ = dict()
        for k, v in data.items():
            if k in data_keys:
                # print(k, v.shape)
                if k == "R":
                    dict_[k] = v[perm].reshape(batch_size * num_atoms, 3)
                    # print(dict_[k].
                elif k == "F":
                    dict_[k] = v[perm].reshape(batch_size * num_atoms, 3)
                elif k == "E":
                    dict_[k] = v[perm].reshape(batch_size, 1)
                elif k == "Z":
                    dict_[k] = v[perm].reshape(batch_size * num_atoms)
                elif k == "mono":
                    dict_[k] = v[perm].reshape(batch_size * num_atoms)
                else:
                    dict_[k] = v[perm]

        good_indices = []
        for i, nat in enumerate(dict_["N"]):
            # print("nat", nat)
            cond = (dst_idx[i] < (nat + i * num_atoms)) * (
                src_idx[i] < (nat + i * num_atoms)
            )
            good_indices.append(
                jnp.where(
                    cond,
                    1,
                    0,
                )
            )
        good_indices = jnp.concatenate(good_indices).flatten()
        dict_["dst_idx"] = dst_idx.flatten()
        dict_["src_idx"] = src_idx.flatten()
        dict_["batch_mask"] = good_indices  # .reshape(-1)
        dict_["batch_segments"] = batch_segments.reshape(-1)
        dict_["atom_mask"] = jnp.where(dict_["Z"] > 0, 1, 0).reshape(-1)
        output.append(dict_)
    # print(output)
    return output


def prepare_batches_jit(
    key,
    data: Dict[str, jnp.ndarray],
    batch_size: int,
    data_keys: Optional[List[str]] = None,
    num_atoms: int = 60,
    dst_idx: Optional[jnp.ndarray] = None,
    src_idx: Optional[jnp.ndarray] = None,
    include_id: bool = False,
    debug_mode: bool = False,
) -> List[Dict[str, jnp.ndarray]]:
    """
    Efficiently prepare batches for training.

    Args:
        key: JAX random key for shuffling.
        data (dict): Dictionary containing the dataset.
            Expected keys: 'R', 'N', 'Z', 'F', 'E', and optionally others.
        batch_size (int): Size of each batch.
        data_keys (list, optional): List of keys to include in the output.
            If None, all keys in `data` are included.
        num_atoms (int, optional): Number of atoms per example. Default is 60.
        dst_idx (jax.numpy.ndarray, optional): Precomputed destination indices for atom pairs.
        src_idx (jax.numpy.ndarray, optional): Precomputed source indices for atom pairs.
        include_id (bool, optional): Whether to include 'id' key if present in data.
        debug_mode (bool, optional): If True, run assertions and extra checks.

    Returns:
        list: A list of dictionaries, each representing a batch.
    """

    # -------------------------------------------------------------------------
    # Validation and Setup
    # -------------------------------------------------------------------------

    # Check for mandatory keys
    required_keys = ["R", "N", "Z", "F", "E"]
    for req_key in required_keys:
        if req_key not in data:
            raise ValueError(f"Data dictionary must contain '{req_key}' key.")

    # Default to all keys in data if none provided
    if data_keys is None:
        data_keys = list(data.keys())

    # Verify data sizes
    data_size = len(data["R"])
    steps_per_epoch = data_size // batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            "Batch size is larger than the dataset size or no full batch available."
        )

    # -------------------------------------------------------------------------
    # Compute Random Permutation for Batches
    # -------------------------------------------------------------------------
    perms = jax.random.permutation(key, data_size)
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    # -------------------------------------------------------------------------
    # Precompute Batch Segments and Indices
    # -------------------------------------------------------------------------
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms

    # Compute pairwise indices only if not provided
    # E3x: e3x.ops.sparse_pairwise_indices(num_atoms) -> returns (dst_idx, src_idx)
    if dst_idx is None or src_idx is None:
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

    # Adjust indices for batching
    dst_idx = dst_idx + offsets[:, None]
    src_idx = src_idx + offsets[:, None]

    # Centralize reshape logic
    # For keys not listed here, we default to their original shape after indexing.
    reshape_rules = {
        "R": (batch_size * num_atoms, 3),
        "F": (batch_size * num_atoms, 3),
        "E": (batch_size, 1),
        "Z": (batch_size * num_atoms,),
        "N": (batch_size,),
        "mono": (batch_size * num_atoms,),
    }

    output = []

    # -------------------------------------------------------------------------
    # Batch Preparation Loop
    # -------------------------------------------------------------------------
    for perm in perms:
        # Build the batch dictionary
        batch = {}
        for k in data_keys:
            if k not in data:
                continue
            v = data[k][perm]
            new_shape = reshape_rules.get(k, None)
            if new_shape is not None:
                batch[k] = v.reshape(new_shape)
            else:
                # Default to just attaching the permuted data without reshape
                batch[k] = v

        # Optionally include 'id' if requested and present
        if include_id and "id" in data and "id" in data_keys:
            batch["id"] = data["id"][perm]

        # Compute good_indices (mask for valid atom pairs)
        # Vectorized approach: We know N is shape (batch_size,)
        # Expand N to compare with dst_idx/src_idx
        # dst_idx[i], src_idx[i] range over atom pairs within the ith example
        # Condition: (dst_idx[i] < N[i]+i*num_atoms) & (src_idx[i] < N[i]+i*num_atoms)
        # We'll compute this for all i and concatenate.
        N = batch["N"]
        # Expand N and offsets for comparison
        expanded_n = N[:, None] + offsets[:, None]
        valid_dst = dst_idx < expanded_n
        valid_src = src_idx < expanded_n
        good_pairs = (valid_dst & valid_src).astype(jnp.int32)
        good_indices = good_pairs.reshape(-1)

        # Add metadata to the batch
        atom_mask = jnp.where(batch["Z"] > 0, 1, 0)
        batch.update(
            {
                "dst_idx": dst_idx.flatten(),
                "src_idx": src_idx.flatten(),
                "batch_mask": good_indices,
                "batch_segments": batch_segments,
                "atom_mask": atom_mask,
            }
        )

        # Debug checks
        if debug_mode:
            # Check expected shapes
            assert batch["R"].shape == (
                batch_size * num_atoms,
                3,
            ), f"R shape mismatch: {batch['R'].shape}"
            assert batch["F"].shape == (
                batch_size * num_atoms,
                3,
            ), f"F shape mismatch: {batch['F'].shape}"
            assert batch["E"].shape == (
                batch_size,
                1,
            ), f"E shape mismatch: {batch['E'].shape}"
            assert batch["Z"].shape == (
                batch_size * num_atoms,
            ), f"Z shape mismatch: {batch['Z'].shape}"
            assert batch["N"].shape == (
                batch_size,
            ), f"N shape mismatch: {batch['N'].shape}"
            # Optional: print or log if needed

        output.append(batch)

    return output


# Example of optional JIT compilation if desired (and arguments are stable):
prepare_batches = jax.jit(
    prepare_batches_jit, static_argnames=("batch_size", "num_atoms", "data_keys")
)
