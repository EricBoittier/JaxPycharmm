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

    return verify_energy(U), _


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


def get_base_dynamics_dict():
    """Return the base dictionary for dynamics simulations."""
    return {
        "leap": False,
        "verlet": True,
        "cpt": False,
        "new": False,
        "langevin": False,
        "nsavv": 0,
        "inbfrq": 10,
        "ihbfrq": 0,
        "iunldm": -1,
        "ilap": -1,
        "ilaf": -1,
        "TEMINC": 10,
        "TWINDH": 10,
        "TWINDL": -10,
        "iasors": 1,
        "iasvel": 1,
        "ichecw": 0,
    }


def setup_charmm_files(prefix, phase):
    """Setup CHARMM files for different simulation phases."""
    files = {}
    if phase == "heating":
        files["res"] = pycharmm.CharmmFile(
            file_name=f"{prefix}.res", file_unit=2, formatted=True, read_only=False
        )
        files["dcd"] = pycharmm.CharmmFile(
            file_name=f"{prefix}.dcd", file_unit=1, formatted=False, read_only=False
        )
    else:
        files["str"] = pycharmm.CharmmFile(
            file_name=f"{prefix}.{phase}.res",
            file_unit=3,
            formatted=True,
            read_only=False,
        )
        files["res"] = pycharmm.CharmmFile(
            file_name=f"{prefix}.{phase}.res",
            file_unit=2,
            formatted=True,
            read_only=False,
        )
        files["dcd"] = pycharmm.CharmmFile(
            file_name=f"{prefix}.{phase}.dcd",
            file_unit=1,
            formatted=False,
            read_only=False,
        )
    return files


def run_heating(
    timestep=0.0001,
    tottime=5.0,
    savetime=0.10,
    initial_temp=10,
    final_temp=300,
    prefix="restart",
    nprint=10,
):
    """
    Run the heating phase of molecular dynamics.

    Args:
        timestep (float): Timestep in ps (default: 0.001 = 0.5 fs)
        tottime (float): Total simulation time in ps (default: 5.0 = 10 ps)
        savetime (float): Save frequency in ps (default: 0.10 = 100 fs)
        initial_temp (float): Initial temperature in K (default: 10)
        final_temp (float): Final temperature in K (default: 300)
        prefix (str): Prefix for output files (default: "restart")
        nprint (int): Print frequency (default: 10)
    """
    files = setup_charmm_files(prefix, "heating")
    nstep = int(tottime / timestep)
    nsavc = int(savetime / timestep)
    nstep = 1000
    nprint = 1
    nsavc = 1

    energy.show()

    dynamics_dict = get_base_dynamics_dict()
    dynamics_dict.update(
        {
            "timestep": timestep,
            "start": True,
            "nstep": nstep,
            "nsavc": nsavc,
            "ilbfrq": 50,
            "imgfrq": 0,
            "ixtfrq": 0,
            "iunrea": -1,
            "iunwri": files["res"].file_unit,
            "iuncrd": files["dcd"].file_unit,
            "nsavl": 0,
            "nprint": nprint,
            "iprfrq": 1000,
            "isvfrq": 1000,
            "ntrfrq": 0,
            "ihtfrq": 500,
            "ieqfrq": 100,
            "firstt": initial_temp,
            "finalt": final_temp,
            "echeck": 1000,
        }
    )

    dyn_heat = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_heat.run()
    write.coor_pdb(f"{prefix}.pdb")

    for file in files.values():
        file.close()


def run_equilibration(
    timestep=0.001, tottime=5.0, savetime=0.01, temp=300, prefix="mm"
):
    """
    Run the equilibration phase of molecular dynamics.

    Args:
        timestep (float): Timestep in ps (default: 0.001 = 0.2 fs)
        tottime (float): Total simulation time in ps (default: 5.0 = 50 ps)
        savetime (float): Save frequency in ps (default: 0.01 = 10 fs)
        temp (float): Temperature in K (default: 300)
        prefix (str): Prefix for output files (default: "mm")
    """
    files = setup_charmm_files(prefix, "equi")
    nstep = int(tottime / timestep)
    nsavc = int(savetime / timestep)

    dynamics_dict = get_base_dynamics_dict()
    dynamics_dict.update(
        {
            "timestep": timestep,
            "start": False,
            "restart": True,
            "nstep": nstep,
            "nsavc": nsavc,
            "imgfrq": 10,
            "iunrea": files["str"].file_unit,
            "iunwri": files["res"].file_unit,
            "iuncrd": files["dcd"].file_unit,
            "nsavl": 0,
            "nprint": 10,
            "iprfrq": 100,
            "ieqfrq": 100,
            "firstt": temp,
            "finalt": temp,
        }
    )

    dyn_equi = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_equi.run()

    for file in files.values():
        file.close()

    write.coor_pdb(f"{prefix}.equi.pdb")
    write.coor_card(f"{prefix}.equi.cor")
    write.psf_card(f"{prefix}.equi.psf")


def run_production(timestep=0.001, nsteps=1000000, temp=300, prefix="mm"):
    """
    Run the production phase of molecular dynamics.

    Args:
        timestep (float): Timestep in ps (default: 0.001 = 0.2 fs)
        nsteps (int): Number of simulation steps (default: 1000000)
        temp (float): Temperature in K (default: 300)
        prefix (str): Prefix for output files (default: "mm")
    """
    files = setup_charmm_files(prefix, "dyna")

    dynamics_dict = get_base_dynamics_dict()
    dynamics_dict.update(
        {
            "timestep": timestep,
            "start": False,
            "restart": True,
            "nstep": nsteps,
            "nsavc": 500,
            "imgfrq": 10,
            "iunrea": files["str"].file_unit,
            "iunwri": files["res"].file_unit,
            "iuncrd": files["dcd"].file_unit,
            "nsavl": 0,
            "nprint": 10,
            "iprfrq": 100,
            "ieqfrq": 0,
            "firstt": temp,
            "finalt": temp,
        }
    )

    dyn_prod = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_prod.run()

    for file in files.values():
        file.close()

    write.coor_pdb(f"{prefix}.dyna.pdb")
    write.coor_card(f"{prefix}.dyna.cor")
    write.psf_card(f"{prefix}.dyna.psf")


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
    calc_setup, _ = setup_calculator(atoms, params, model)
    if calc_setup:
        pass
    else:
        print("Error in setting up calculator.")
    run_minimization(output_pdb)
    run_heating()


if __name__ == "__main__":
    main()
