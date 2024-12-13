import ase.data
import e3x
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def cut_vdw(grid, xyz, elements, vdw_scale=1.4):
    """ """
    if type(elements[0]) == str:
        elements = [ase.data.atomic_numbers[s] for s in elements]
    vdw_radii = [ase.data.vdw_radii[s] for s in elements]
    vdw_radii = np.array(vdw_radii) * vdw_scale
    distances = cdist(grid, xyz)
    mask = distances < vdw_radii
    closest_atom = np.argmin(distances, axis=1)
    closest_atom_type = elements[closest_atom]
    mask = ~mask.any(axis=1)
    return mask, closest_atom_type, closest_atom


def prepare_multiple_datasets(
    key,
    train_size=0,
    valid_size=0,
    filename=None,
    clean=False,
    verbose=False,
    esp_mask=False,
    clip_esp=False,
    natoms=60,
):
    """
    Prepare multiple datasets for training and validation.

    Args:
        key: Random key for dataset shuffling.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.
        filename (list): List of filenames to load datasets from.

    Returns:
        tuple: A tuple containing the prepared data and keys.
    """
    # Load the datasets
    datasets = [np.load(f) for f in filename]
    # datasets_keys() = datasets.keys()
    if verbose:
        for dataset in datasets:
            for k, v in dataset.items():
                print(k, v.shape)

    if "id" in datasets[0].keys():
        dataid = np.concatenate([dataset["id"] for dataset in datasets])
    not_failed = None  # np.array(range().reshape(-1,6,3).shape[0]))
    data = []
    keys = []

    if clean:
        failed = pd.read_csv(
            "/pchem-data/meuwly/boittier/home/jaxeq/data/qm9-fails.csv"
        )
        failed = list(failed["0"])
        not_failed = [i for i in range(len(dataid)) if str(dataid[i]) not in failed]
        n_failed = len(dataid) - len(not_failed)
        print("n_failed:", n_failed)
        num_train = int(max([0, num_train - n_failed]))
        if num_train == 0:
            num_valid = int(max([0, num_valid - n_failed]))
        print(num_train, num_valid)

    shape = (
        np.concatenate([dataset["R"] for dataset in datasets])
        .reshape(-1, natoms, 3)
        .shape
    )
    not_failed = np.array(range(shape[0]))
    print("shape", shape, "not failed", not_failed)

    if "id" in datasets[0].keys():
        dataid = dataid[not_failed]
        data.append(dataid)
        keys.append("id")
    if "R" in datasets[0].keys():
        dataR = np.concatenate([dataset["R"] for dataset in datasets])
        print("dataR", dataR.shape)
        dataR = dataR.reshape(shape[0], natoms, 3)[not_failed]
        data.append(dataR.squeeze())
        keys.append("R")
    if "Z" in datasets[0].keys():
        dataZ = np.concatenate([dataset["Z"] for dataset in datasets]).reshape(
            shape[0], natoms
        )[not_failed]
        data.append(dataZ.squeeze())
        keys.append("Z")
    if "F" in datasets[0].keys():
        dataF = np.concatenate([dataset["F"] for dataset in datasets]).reshape(
            shape[0], natoms, 3
        )[not_failed]
        data.append(dataF.squeeze())
        keys.append("F")
    if "E" in datasets[0].keys():
        dataE = np.concatenate([dataset["E"] for dataset in datasets])[not_failed]
        # print("Subtracting mean energy:", np.mean(dataE))
        # dataE = dataE - np.mean(dataE)
        # print(np.mean(dataE))
        data.append(dataE.reshape(shape[0], 1))
        keys.append("E")
    if "N" in datasets[0].keys():
        dataN = np.concatenate([dataset["N"] for dataset in datasets])[not_failed]
        data.append(dataN.reshape(shape[0], 1))
        keys.append("N")
    if "mono" in datasets[0].keys():
        dataMono = np.concatenate([dataset["mono"] for dataset in datasets])[not_failed]
        data.append(dataMono.reshape(shape[0], natoms))
        keys.append("mono")
    if "esp" in datasets[0].keys():
        if clip_esp:
            dataEsp = np.concatenate([dataset["esp"][:1000] for dataset in datasets])[
                not_failed
            ].reshape(shape[0], 1000)
        else:
            dataEsp = np.concatenate([dataset["esp"] for dataset in datasets])[
                not_failed
            ]
        data.append(dataEsp)
        keys.append("esp")
    if "vdw_surface" in datasets[0].keys():
        dataVDW = np.concatenate([dataset["vdw_surface"] for dataset in datasets])[
            not_failed
        ]
        data.append(dataVDW)
        keys.append("vdw_surface")
    if "n_grid" in datasets[0].keys():
        dataNgrid = np.concatenate([dataset["n_grid"] for dataset in datasets])[
            not_failed
        ]
        data.append(dataNgrid)
        keys.append("n_grid")
    if "D" in datasets[0].keys():
        dataD = np.concatenate([dataset["D"] for dataset in datasets])[not_failed]
        print("D", dataD.shape)
        try:
            data.append(dataD.reshape(shape[0], 1))
        except:
            try:
                data.append(dataD.reshape(shape[0], natoms, 3))
            except Exception:
                data.append(dataD.reshape(shape[0], 3))
        keys.append("D")
    if "dipole" in datasets[0].keys():
        dipole = np.concatenate([dataset["dipole"] for dataset in datasets]).reshape(
            shape[0], 3
        )
        # [not_failed]
        print("dipole.shape", dipole.shape)
        data.append(dipole)
        keys.append("dipole")
    if "Dxyz" in datasets[0].keys():
        dataDxyz = np.concatenate([dataset["Dxyz"] for dataset in datasets])[not_failed]
        data.append(dataDxyz)
        keys.append("Dxyz")
    if "com" in datasets[0].keys():
        dataCOM = np.concatenate([dataset["com"] for dataset in datasets])[not_failed]
        data.append(dataCOM)
        keys.append("com")
    if "polar" in datasets[0].keys():
        polar = np.concatenate([dataset["polar"] for dataset in datasets]).reshape(
            shape[0], 3, 3
        )
        print("polar", polar.shape)
        polar = polar  # [not_failed,:,:]
        # print(polar)
        data.append(polar)
        keys.append("polar")
    if "esp_grid" in datasets[0].keys():
        esp_grid = np.concatenate(
            [dataset["esp_grid"][:1000] for dataset in datasets]
        ).reshape(-1, 1000, 3)
        esp_grid = esp_grid  # [not_failed,:,:]
        # print(polar)
        data.append(esp_grid)
        keys.append("esp_grid")
    if "quadrupole" in datasets[0].keys():
        quadrupole = np.concatenate(
            [dataset["quadrupole"] for dataset in datasets]
        ).reshape(-1, 3, 3)
        quadrupole = quadrupole  # [not_failed,:,:]
        # print(polar)
        data.append(quadrupole)
        keys.append("quadrupole")

    for k in datasets[0].keys():
        if k not in keys:
            print(
                k,
                len(datasets[0][k].shape),
                datasets[0][k].shape,
                datasets[0][k].shape[0],
            )
            _ = np.concatenate([dataset[k] for dataset in datasets])
            # [not_failed]
            # if
            print(k, _.shape)
            data.append(_)
            keys.append(k)

    if esp_mask:
        if verbose:
            print("creating_mask")
        dataESPmask = np.array(
            [cut_vdw(dataVDW[i], dataR[i], dataZ[i])[0] for i in range(len(dataZ))]
        )
        data.append(dataESPmask)
        keys.append("espMask")

    print("R", dataR.shape)

    assert_dataset_size(dataR.squeeze(), train_size, valid_size)

    return (
        data,
        keys,
        train_size,
        valid_size,
    )


def prepare_datasets(
    key,
    train_size=0,
    valid_size=0,
    files=None,
    clean=False,
    esp_mask=False,
    clip_esp=False,
    natoms=60,
):
    """
    Prepare datasets for training and validation.

    Args:
        key: Random key for dataset shuffling.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.
        filename (str or list): Filename(s) to load datasets from.

    Returns:
        tuple: A tuple containing train_data and valid_data dictionaries.
    """
    # Load the datasets
    if isinstance(files, str):
        filename = [files]
    elif isinstance(files, list):
        filename = files
    elif files is None:
        # exit and Warning
        raise ValueError("No filename(s) provided")

    data, keys, num_train, num_valid = prepare_multiple_datasets(
        key,
        train_size=train_size,
        valid_size=valid_size,
        filename=filename,
        clean=clean,
        natoms=natoms,
        clip_esp=clip_esp,
        esp_mask=esp_mask,
        # dataset_keys
    )
    print(data[0].shape)
    print(keys)
    print(len(data[0]))
    train_choice, valid_choice = get_choices(
        key, len(data[0]), int(num_train), int(num_valid)
    )

    train_data, valid_data = make_dicts(data, keys, train_choice, valid_choice)

    return train_data, valid_data


def assert_dataset_size(dataR, num_train, num_valid):
    """
    Assert that the dataset contains enough entries for training and validation.

    Args:
        dataR: The dataset to check.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.

    Raises:
        RuntimeError: If the dataset doesn't contain enough entries.
    """
    assert num_train >= 0
    assert num_valid >= 0
    # Make sure that the dataset contains enough entries.
    num_data = len(dataR)
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f"datasets only contains {num_data} points, "
            f"requested num_train={num_train}, num_valid={num_valid}"
        )


def get_choices(key, num_data, num_train, num_valid):
    """
    Randomly draw train and validation sets from the dataset.

    Args:
        key: Random key for shuffling.
        num_data (int): Total number of data points.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.

    Returns:
        tuple: A tuple containing train_choice and valid_choice arrays.
    """
    # Randomly draw train and validation sets from dataset.
    choice = np.asarray(
        jax.random.choice(key, num_data, shape=(num_data,), replace=False)
    )
    train_choice = choice[:num_train]
    valid_choice = choice[num_train : num_train + num_valid]
    return train_choice, valid_choice


def make_dicts(data, keys, train_choice, valid_choice):
    """
    Create dictionaries for train and validation data.

    Args:
        data (list): List of data arrays.
        keys (list): List of keys for the data arrays.
        train_choice (array): Indices for training data.
        valid_choice (array): Indices for validation data.

    Returns:
        tuple: A tuple containing train_data and valid_data dictionaries.
    """
    train_data, valid_data = dict(), dict()

    for i, k in enumerate(keys):
        print(i, k, len(data[i]), data[i].shape)
        train_data[k] = data[i][train_choice]
        valid_data[k] = data[i][valid_choice]

    return train_data, valid_data


def print_shapes(train_data, valid_data):
    """
    Print the shapes of train and validation data.

    Args:
        train_data (dict): Dictionary containing training data.
        valid_data (dict): Dictionary containing validation data.

    Returns:
        tuple: A tuple containing train_data and valid_data dictionaries.
    """
    print("...")
    print("...")
    for k, v in train_data.items():
        print(k, v.shape)
    print("...")
    for k, v in valid_data.items():
        print(k, v.shape)

    return train_data, valid_data


import jax
import jax.numpy as jnp
import e3x.ops


# def prepare_batches(
#     key,
#     data,
#     batch_size,
#     data_keys=None,
#     num_atoms=60,
#     dst_idx=None,
#     src_idx=None,
# ) -> list:
#     """
#     Efficiently prepare batches for training.
#
#     Args:
#         key: Random key for shuffling.
#         data (dict): Dictionary containing the dataset.
#         batch_size (int): Size of each batch.
#         include_id (bool, optional): Whether to include ID in the output.
#         data_keys (list, optional): List of keys to include in the output.
#         num_atoms (int, optional): Number of atoms per batch. Defaults to 60.
#
#     Returns:
#         list: A list of dictionaries, each representing a batch.
#     """
#     # Validate inputs
#     if data_keys is None:
#         data_keys = list(data.keys())
#
#     # Determine the number of training steps per epoch
#     data_size = len(data["R"])
#     steps_per_epoch = data_size // batch_size
#
#     # Optimize random permutation and batch selection
#     perms = jax.random.permutation(key, data_size)[: steps_per_epoch * batch_size]
#     perms = perms.reshape((steps_per_epoch, batch_size))
#
#     # Precompute sparse indices and offsets
#     batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
#     offsets = jnp.arange(batch_size) * num_atoms
#
#     # Compute dst_idx and src_idx only if not provided
#     if dst_idx is None or src_idx is None:
#         dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
#
#     dst_idx = dst_idx + offsets[:, None]
#     src_idx = src_idx + offsets[:, None]
#
#     # Create output list to match original function's behavior
#     output = []
#
#     for perm in perms:
#         # Efficiently select and reshape data
#         batch = {
#             k: (
#                 v[perm].reshape(batch_size * num_atoms)
#                 if k in ["Z", "mono"]
#                 else (
#                     v[perm].reshape(batch_size * num_atoms, -1)
#                     if k in ["R", "F"]
#                     else v[perm].reshape(batch_size, -1) if k == "E" else v[perm]
#                 )
#             )
#             for k, v in data.items()
#             if k in data_keys
#         }
#
#         # Compute good indices similar to the original implementation
#         good_indices = []
#         for i, nat in enumerate(batch["N"]):
#             cond = (dst_idx[i] < (nat + i * num_atoms)) * (
#                 src_idx[i] < (nat + i * num_atoms)
#             )
#             good_indices.append(
#                 jnp.where(
#                     cond,
#                     1,
#                     0,
#                 )
#             )
#         good_indices = jnp.concatenate(good_indices).flatten()
#
#         # Add additional batch metadata
#         batch.update(
#             {
#                 "dst_idx": dst_idx.flatten(),
#                 "src_idx": src_idx.flatten(),
#                 "batch_mask": good_indices,
#                 "batch_segments": batch_segments.reshape(-1),
#                 "atom_mask": jnp.where(batch["Z"] > 0, 1, 0),
#             }
#         )
#
#         output.append(batch)
#
#     return output


def prepare_batches(
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

        if True:
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

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional


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
        expanded_N = N[:, None] + offsets[:, None]
        valid_dst = dst_idx < expanded_N
        valid_src = src_idx < expanded_N
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
prepare_batches_jit = jax.jit(
    prepare_batches_jit, static_argnames=("batch_size", "num_atoms", "data_keys")
)
