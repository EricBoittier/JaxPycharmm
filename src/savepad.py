from pathlib import Path
import numpy as np
import ase
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

files = list(Path("/pchem-data/meuwly/boittier/home/testala/").glob("*opt*npz"))[:]


def sort_func(x):
    x = str(x)
    spl = x.split("/")
    x = spl[-1]
    spl = x.split("xyz")
    spl = spl[1].split("_")
    spl = [_ for _ in spl if len(_) > 0]
    return abs(int(spl[0]))


from ase.units import Hartree, Bohr

# Force in Hartree/Bohr
hartree_per_bohr_to_ev_per_angstrom = Hartree / Bohr

files.sort(key=sort_func)
print(files[:5])
outdata = files

atom_energies_hartree = np.array(
    [0, -0.500273, 0, 0, 0, 0, -37.846772, -54.583861, -75.064579]
)

Ndata = len(outdata)
print(len(outdata))

for i_ in range(1):
    elements = []
    coordinates = []
    # monopoles = []
    quads = []
    esp = []
    vdw_surface = []
    ids = []
    molecular_dipoles = []
    coms = []
    read_data = []
    Fs = []
    Es = []
    dipole_components = []
    for _ in tqdm(outdata[i_ * Ndata : (i_ + 1) * Ndata]):
        with np.load(_) as load:
            if load is not None:

                # read_data.append(load)
                R = load["R"]
                # print(R.sum(axis=1))
                n_atoms = len(np.nonzero(R.sum(axis=1))[0])
                if int(len(R)) != n_atoms:
                    print(load["R"])
                    break

                fn = np.linalg.norm(
                    np.linalg.norm(
                        load["F"]
                        * (-627.509474 / 0.529177)
                        * (ase.units.kcal / ase.units.mol),
                        axis=(1),
                    )
                )
                # if fn > 7.5:
                # print(fn)
                # print(n_atoms)
                if 3 < n_atoms < 1000:  # and (0.005 < fn < 5.5):
                    # mag, Dxyz = read_output(_.parent / "output.dat")
                    dipole_components.append(load["dipole"])
                    # molecular_dipoles.append(mag)
                    quads.append(load["quadrupole"])
                    ids.append(str(_).split("/")[-2])
                    # print(elements[-1])
                    # print(elements[-1], atom_energies)
                    R = load["R"]
                    coordinates.append(R[np.nonzero(R.sum(axis=1))])
                    _r = coordinates[-1]
                    _z = load["Z"][np.nonzero(R.sum(axis=1))]  # [-_r.shape[0]:]
                    elements.append(_z)
                    # _nonz = np.nonzero(R.sum(axis=1))
                    # zz = np.take(_z, )
                    atom_energies = np.take(atom_energies_hartree, _z)
                    if n_atoms == 3:
                        print(_z, atom_energies)
                    # monopoles.append(load["monopoles"])
                    esp.append(load["esp"])
                    vdw_surface.append(load["esp_grid"])
                    Fs.append(load["F"][np.nonzero(R.sum(axis=1))])
                    Es.append(load["E"] - np.sum(atom_energies))
                    mol = ase.Atoms(_z, coordinates[-1])
                    coms.append(mol.get_center_of_mass())
                else:
                    pass
                    # print(n_atoms, load["Z"])

    N_grid = [_.shape[0] for _ in esp]
    max_N_atoms = 37
    max_grid_points = 10000  # max(N_grid)
    N = len(elements)
    print("max Z", max([len(_) for _ in elements]))
    Z = [np.array([int(_) for _ in elements[i]]) for i in range(N)]
    pad_Z = np.array([np.pad(Z[i], ((0, max_N_atoms - len(Z[i])))) for i in range(N)])
    padE = np.array([[Es[i] * Hartree] for i in range(N)])
    pad_coords = np.array(
        [
            np.pad(coordinates[i], ((0, max_N_atoms - len(coordinates[i])), (0, 0)))
            for i in range(N)
        ]
    )
    # pad_mono = [np.pad(monopoles[i],((0,max_N_atoms - len(monopoles[i])),(0,0))) for i in range(N)]
    pad_esp = [np.pad(esp[i], ((0, max_grid_points - len(esp[i])))) for i in range(N)]
    pad_esp = np.array(pad_esp)
    bohbb = 0.529177
    padF = np.array(
        [
            np.pad(
                Fs[i] * (-hartree_per_bohr_to_ev_per_angstrom),
                ((0, max_N_atoms - len(Z[i])), (0, 0)),
                "constant",
                constant_values=(0, 0),
            )
            for i in range(N)
        ]
    )

    pad_vdw_surface = []
    for i in range(N):
        try:
            _ = np.pad(
                vdw_surface[i],
                ((0, max_grid_points - len(vdw_surface[i])), (0, 0)),
                "constant",
                constant_values=(0, 10000),
            )
            pad_vdw_surface.append(_)
        except ValueError:
            print(i, vdw_surface[i])
            pad_vdw_surface.append(10000 * jnp.ones((max_grid_points, 3)))

    pad_vdw_surface = np.array(pad_vdw_surface)
    pad_vdw_surface.shape
    N_elements = [len(_) for _ in Z]

    print("R", pad_coords.shape)
    print("Z", pad_Z.shape)
    print("E", padE.shape)
    print("F", padF.shape)
    print(len(Z))
    print(f"ala-esp-dip-{i_}.npz")
    # np.savez(f'ala-esp-dip-{N}-{i_}.npz',
    np.savez(
        f"ala-esp-dip-{i_}.npz",
        R=pad_coords,
        Z=pad_Z,
        F=padF,
        E=padE,
        N=N_elements,
        # D=molecular_dipoles,
        com=coms,
        D=np.array(dipole_components) * ase.units.Debye,
        # mono=pad_mono,
        # esp=pad_esp,
        # id=np.array(ids),
        # n_grid=np.array(N_grid),
        # vdw_surface=pad_vdw_surface
    )
