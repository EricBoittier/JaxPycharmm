import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm/"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"


import ase
import ase.io as io
import ase.units as units
import jax
import numpy as np
import pandas as pd
import pycharmm
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.lingo as stream
import pycharmm.minimize as minimize
import pycharmm.read as read
import pycharmm.settings as settings
import pycharmm.write as write

from physnetjax.calc.helper_mlp import get_ase_calc, get_pyc
from physnetjax.restart.restart import get_params_model_with_ase

# Environment settings


def print_device_info():
    """Print JAX device information."""
    devices = jax.local_devices()
    print(devices)
    print(jax.default_backend())
    print(jax.devices())


def initialize_system(pdb_file, pkl_path, model_path):
    """Initialize the system with PDB file and model parameters."""
    atoms = io.read(pdb_file)
    params, model = get_params_model_with_ase(pkl_path, model_path, atoms)

    # Read topology and parameter files
    read.rtf("/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_prot.rtf")
    read.prm(
        "/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36m_prot.prm", flex=True
    )
    pycharmm.lingo.charmm_script(
        "stream /pchem-data/meuwly/boittier/home/charmm/toppar/toppar_water_ions.str"
    )

    return atoms, params, model


def setup_coordinates(pdb_file, psf_file, atoms):
    """Setup system coordinates and parameters."""
    settings.set_bomb_level(-2)
    settings.set_warn_level(-1)

    read.pdb(pdb_file, resid=True)
    read.psf_card(psf_file)
    coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))

    stats = coor.stat()
    stream.charmm_script("print coor")


##########################


def setup_calculator(atoms, params, model):
    """Setup the calculator and verify energies."""
    Z = atoms.get_atomic_numbers()
    Z = [_ if _ < 9 else 6 for _ in Z]
    stream.charmm_script(f"echo {Z}")
    R = atoms.get_positions()
    atoms = ase.Atoms(Z, R)

    calculator = get_ase_calc(params, model, atoms)
    atoms.calc = calculator
    ml_selection = pycharmm.SelectAtoms().by_res_id("1")

    energy.show()
    U = atoms.get_potential_energy() / (units.kcal / units.mol)
    stream.charmm_script(f"echo {U}")

    Model = get_pyc(params, model, atoms)
    Z = np.array(Z)

    # Initialize PhysNet calculator
    _ = pycharmm.MLpot(
        Model,
        Z,
        ml_selection,
        ml_fq=False,
    )

    return verify_energy(U)


def verify_energy(U, atol=1e-4):
    """Verify that energies match within tolerance."""
    energy.show()
    userE = energy.get_energy()["USER"]
    print(userE)
    assert np.isclose(float(U.squeeze()), float(userE), atol=atol)
    print(f"Success! energies are close, within {atol} kcal/mol")
    return True


def run_minimization(output_pdb):
    """Run energy minimization and save results."""
    minimize.run_sd(**{"nstep": 10000, "tolenr": 1e-5, "tolgrd": 1e-5})
    energy.show()
    minimize.run_abnr(**{"nstep": 10000, "tolenr": 1e-5, "tolgrd": 1e-5})
    energy.show()
    stream.charmm_script("print coor")
    write.coor_pdb(output_pdb)


def main():
    """Main function to run the simulation."""
    print_device_info()

    # File paths
    pdb_file = "/pchem-data/meuwly/boittier/home/pycharmm_test/md/adp.pdb"
    psf_file = "/pchem-data/meuwly/boittier/home/pycharmm_test/md/adp.psf"
    pkl_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a/params.pkl"
    model_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a/model_kwargs.pkl"
    output_pdb = "/pchem-data/meuwly/boittier/home/pycharmm_test/md/adp_min.pdb"

    # Initialize and setup
    atoms, params, model = initialize_system(pdb_file, pkl_path, model_path)
    setup_coordinates(pdb_file, psf_file, atoms)

    # Setup calculator and run minimization
    if setup_calculator(atoms, params, model):
        run_minimization(output_pdb)


if __name__ == "__main__":
    main()
