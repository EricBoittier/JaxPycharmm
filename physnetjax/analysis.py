import ase
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress
from tqdm import tqdm
from physnetjax.loss import dipole_calc
import numpy as np


def plot(x, y, ax, units="kcal/mol", _property="", kde=True, s=1, diag=True):
    x = x.flatten()
    y = y.flatten()
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_value = "$1 - r^2$: {:.3E}".format(1 - r_value)
    except:
        r_value = ""

    ERROR = x - y
    RMSE = np.mean(ERROR**2) ** 0.5
    MAE = np.mean(abs(ERROR))
    ax.set_aspect("equal")
    ax.text(
        0.4,
        0.85,
        f"{RMSE:.2f}/{MAE:.2f} [{units}]\n{r_value}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    if kde:
        if kde == "y":
            xy = np.vstack([y])
        else:
            # Calculate the point density
            xy = np.vstack([x, y])
        z = gaussian_kde(xy[:, ::10])(xy)
        # Sort the points by density (optional, for better visualization)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    else:
        z = "k"
    plt.set_cmap("plasma")
    ax.scatter(x, y, alpha=0.8, c=z, s=s)
    # plt.scatter(Fs, predFs, alpha=1, color="k")
    ax.set_aspect("equal")
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    ax.set_ylim(min(x_min, y_min), max(x_max, y_max))
    ax.set_xlim(min(x_min, y_min), max(x_max, y_max))
    ax.set_title(_property)
    ax.set_xlabel(f"ref. [{units}]")
    ax.set_ylabel(f"pred. [{units}]")
    if diag:
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="gray")
    else:
        ax.axhline(0, color="gray")
    return ax


def eval(batches, model, params, batch_size=500):
    Es, Eeles, predEs, Fs, predFs, Ds, predDs, charges, outputs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, batch in tqdm(enumerate(batches)):
        output = model.apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
            batch_mask=batch["batch_mask"],
            atom_mask=batch["atom_mask"],
        )
        # nonzero = np.nonzero(batch["Z"])
        # print(nonzero)
        Ds.append(batch["D"])
        D = dipole_calc(
            batch["R"],
            batch["Z"],
            output["charges"],
            batch["batch_segments"],
            batch_size,
        )
        # print(D,batch["D"])
        predDs.append(D)
        Es.append(batch["E"])
        predEs.append(output["energy"])
        _f = np.take(batch["F"], batch["atom_mask"], axis=0)
        _predf = np.take(np.array(output["forces"]), batch["atom_mask"], axis=0)
        Fs.append(_f)
        predFs.append(_predf)
        charges.append(output["charges"])
        Eeles.append(output["electrostatics"])
    Es = np.array(Es).flatten()
    Eeles = np.array(Eeles).flatten()
    predEs = np.array(predEs).flatten()
    Fs = np.concatenate(Fs).flatten()

    predFs = np.concatenate(predFs).flatten()
    Ds = np.array(Ds)  # .flatten()
    predDs = np.array(predDs)  # .flatten()
    outputs.append(output)
    return Es, Eeles, predEs, Fs, predFs, Ds, predDs, charges, outputs


def plot_stats(batches, model, params, _set="", do_kde=False, batch_size=500):
    Es, Eeles, predEs, Fs, predFs, Ds, predDs, charges, outputs = eval(
        batches, model, params, batch_size=batch_size
    )
    charges = np.concatenate(charges)
    Es = Es / (ase.units.kcal / ase.units.mol)
    predEs = predEs / (ase.units.kcal / ase.units.mol)
    Fs = Fs / (ase.units.kcal / ase.units.mol)
    predFs = predFs / (ase.units.kcal / ase.units.mol)
    Eeles = Eeles / (ase.units.kcal / ase.units.mol)
    summed_q = charges.reshape(len(batches) * batch_size, model.natoms).sum(axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    plot(Es, predEs, axes[0, 0], _property="$E$", s=10, kde=do_kde)
    plot(Fs, predFs, axes[0, 1], _property="$F$", kde=do_kde)
    plot(Ds, predDs, axes[0, 2], _property="$D$", units=r"$e \AA$", kde=do_kde)
    plot(
        Es - Es.mean(),
        Eeles - Eeles.mean(),
        axes[1, 0],
        _property="$E$ vs $E_{\\rm ele}$",
        kde=do_kde,
    )
    # axes[1,1].axis("off")

    plot(predFs, abs(predFs - Fs), axes[1, 1], _property="$F$", kde=do_kde, diag=False)
    q_sum_kde = "y" if do_kde else False
    plot(
        np.zeros_like(summed_q),
        summed_q,
        axes[1, 2],
        _property="$Q$",
        units="$e$",
        kde=q_sum_kde,
    )
    plt.subplots_adjust(hspace=0.55)
    plt.suptitle(_set + f" (n={len(predEs.flatten())})", fontsize=20)
    plt.show()

    output = {
        "Es": Es,
        "Eeles": Eeles,
        "predEs": predEs,
        "Fs": Fs,
        "predFs": predFs,
        "Ds": Ds,
        "predDs": predDs,
        "charges": charges,
        "outputs": outputs,
        "batches": batches,
    }

    return output


# import matplotlib.pyplot as plt
# import numpy as np
# from ase import Atoms
# from dscribe.descriptors import SOAP
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler


# # Function to compute SOAP descriptors
# def compute_soap_descriptors(
#     positions, atomic_numbers, species, r_cut=5.0, n_max=8, l_max=6, sigma=0.5
# ):
#     """Compute SOAP descriptors for a list of structures."""
#     soap = SOAP(
#         species=species,
#         r_cut=r_cut,
#         n_max=n_max,
#         l_max=l_max,
#         sigma=sigma,
#         periodic=False,
#         sparse=False,
#         average="inner",
#     )
#     descriptors = []
#     ase_atoms = []
#     for i, (pos, nums) in tqdm(enumerate(zip(positions, atomic_numbers))):
#         atoms = Atoms(
#             numbers=nums[: Nele[i]],
#             positions=pos[: Nele[i]] - pos[: Nele[i]].mean(axis=0),
#         )
#         ase_atoms.append(atoms)
#         desc = soap.create(atoms)
#         descriptors.append(desc)
#     # Convert list of arrays to a single numpy array
#     return np.array(descriptors), ase_atoms


# # Function to flatten 3D descriptors to 2D
# def flatten_descriptors(descriptors):
#     """Flatten 3D descriptors (structures x centers x features) to 2D."""
#     return descriptors.reshape(descriptors.shape[0], -1)


# # Function to apply PCA
# def apply_pca(data, n_components=2):
#     """Apply PCA to the data."""
#     pca = PCA(n_components=n_components)
#     reduced_data = pca.fit_transform(data)
#     return reduced_data, pca


# # Function to apply t-SNE
# def apply_tsne(data, n_components=2, perplexity=30, random_state=42):
#     """Apply t-SNE to the data."""
#     tsne = TSNE(
#         n_components=n_components, perplexity=perplexity, random_state=random_state
#     )
#     reduced_data = tsne.fit_transform(data)
#     return reduced_data, tsne


# # Function to visualize the projection
# def visualize_projection(data, labels=None, title="Projection", c=None):
#     """Visualize 2D projection of the data."""
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(data[:, 0], data[:, 1], c=c, cmap="jet", s=15)
#     if labels is not None:
#         plt.legend(*scatter.legend_elements(), title="Classes")
#     plt.colorbar(scatter)
#     plt.title(title)
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     # plt.show()


# def get_desc(
#     positions,
#     atomic_numbers,
#     species,
# ):
#     """Process data through SOAP and the chosen dimensionality reduction method."""
#     print("Computing SOAP descriptors...")
#     descriptors, ase_atoms = compute_soap_descriptors(
#         positions, atomic_numbers, species
#     )
#     return descriptors, ase_atoms


# # Main processing function
# def process_data(descriptors, method, **kwargs):

#     print("Flattening descriptors...")
#     flattened_descriptors = flatten_descriptors(descriptors)
#     print("Scaling data...")
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(flattened_descriptors)
#     print(f"Applying {method.__name__}...")
#     reduced_data, model = method(scaled_data, **kwargs)
#     return (
#         reduced_data,
#         model,
#     )
