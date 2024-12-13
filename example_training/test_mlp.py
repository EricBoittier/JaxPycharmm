# Basics
import os

# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import pandas
import numpy as np

# ASE
from ase import Atoms
from ase import io
import ase.units as units

# PyCHARMM
os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm/"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"
import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.lingo as stream
import pycharmm.select as select
import pycharmm.shake as shake
import pycharmm.cons_fix as cons_fix
import pycharmm.cons_harm as cons_harm
from pycharmm.lib import charmm as libcharmm
import pycharmm.lib as lib


import jax

devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())

from physnetjax.helper_mlp import *

# from physnetjax.helper_mlp import Model

# os.sleep(1)
with open("i_", "w") as f:
    print("...")

# Read in the topology (rtf) and parameter file (prm) for proteins
# equivalent to the CHARMM scripting command: read rtf card name toppar/top_all36_prot.rtf
read.rtf("/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_prot.rtf")
# equivalent to the CHARMM scripting command: read param card flexible name toppar/par_all36m_prot.prm
read.prm(
    "/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36m_prot.prm", flex=True
)
pycharmm.lingo.charmm_script(
    "stream /pchem-data/meuwly/boittier/home/charmm/toppar/toppar_water_ions.str"
)


# Step 0: Load parameter files
# -----------------------------------------------------------
# add = """generate ADP setup
# open read card unit 10 name input/ala_-154.0_26.0.pdb
# read coor pdb  unit 10 resid
# """
# stream.charmm_script(add)
read.sequence_pdb("input/ala_-154.0_26.0.pdb")
read.psf_card("input/ala_-147.0_180.0.psf")
read.pdb("input/ala_-154.0_26.0.pdb")
settings.set_bomb_level(-2)
settings.set_warn_level(-1)

##########################
R = coor.get_positions()
ase_mol = ase.io.read("input/ala_-154.0_26.0.pdb")
Z = ase_mol.get_atomic_numbers()
Z = [_ if _ < 9 else 6 for _ in Z]
energy.show()
atoms = ase.Atoms(Z, R)

atoms.set_calculator(MessagePassingCalculator())

atoms1 = atoms.copy()

ml_selection = pycharmm.SelectAtoms(seg_id="ADP")

print(atoms.get_potential_energy())

# Model unit is eV and Angstrom
econv = 1.0 / (units.kcal / units.mol)
fconv = 1.0 / (units.kcal / units.mol)

charge = 0
print(dir(Model))
# Initialize PhysNet calculator
_ = pycharmm.MLpot(
    Model,
    Z,
    ml_selection,
    ml_fq=False,
)

with open("i_", "w") as f:
    print("...")


print(_)


energy.show()


minimize.run_sd(nstep=2000, tolenr=1e-3, tolgrd=1e-3)


energy.show()


atoms = ase.Atoms(Z, coor.get_positions())


print(atoms1.get_positions() - atoms.get_positions())

# Step 4: Heating - CHARMM, PhysNet
# -----------------------------------------------------------

if True:

    timestep = 0.0001  # 0.5 fs

    res_file = pycharmm.CharmmFile(
        file_name="heat.res", file_unit=2, formatted=True, read_only=False
    )
    dcd_file = pycharmm.CharmmFile(
        file_name="heat.dcd", file_unit=1, formatted=False, read_only=False
    )

    # Run some dynamics
    dynamics_dict = {
        "leap": False,
        "verlet": True,
        "cpt": False,
        "new": False,
        "langevin": False,
        "timestep": timestep,
        "start": True,
        # 'nstep': 5.*1./timestep,
        "nstep": 100000,
        "nsavc": 100,  # *1./timestep,
        "nsavv": 0,
        "inbfrq": -1,
        "ihbfrq": 50,
        "ilbfrq": 50,
        "imgfrq": 50,
        "ixtfrq": 1000,
        "iunrea": -1,
        "iunwri": res_file.file_unit,
        "iuncrd": dcd_file.file_unit,
        "nsavl": 0,  # frequency for saving lambda values in lamda-dynamics
        "iunldm": -1,
        "ilap": -1,
        "ilaf": -1,
        "nprint": 100,  # Frequency to write to output
        "iprfrq": 1000,  # Frequency to calculate averages
        "isvfrq": 1000,  # Frequency to save restart file
        "ntrfrq": 1000,
        "ihtfrq": 200,
        "ieqfrq": 1000,
        "firstt": 100,
        "finalt": 300,
        "tbath": 300,
        "iasors": 0,
        "iasvel": 1,
        "ichecw": 0,
        "iscale": 0,  # scale velocities on a restart
        "scale": 1,  # scaling factor for velocity scaling
        "echeck": -1,
    }

    dyn_heat = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_heat.run()

    res_file.close()
    dcd_file.close()


# Step 5: NVE - CHARMM, PhysNet
# -----------------------------------------------------------

if True:

    timestep = 0.0002  # 0.2 fs

    str_file = pycharmm.CharmmFile(
        file_name="heat.res", file_unit=3, formatted=True, read_only=False
    )
    res_file = pycharmm.CharmmFile(
        file_name="nve.res", file_unit=2, formatted=True, read_only=False
    )
    dcd_file = pycharmm.CharmmFile(
        file_name="nve.dcd", file_unit=1, formatted=False, read_only=False
    )

    # Run some dynamics
    dynamics_dict = {
        "leap": False,
        "verlet": True,
        "cpt": False,
        "new": False,
        "langevin": False,
        "timestep": timestep,
        "start": False,
        "restart": True,
        "nstep": 5 * 1.0 / timestep,
        "nsavc": 0.01 * 1.0 / timestep,
        "nsavv": 0,
        "inbfrq": -1,
        "ihbfrq": 50,
        "ilbfrq": 50,
        "imgfrq": 50,
        "ixtfrq": 1000,
        "iunrea": str_file.file_unit,
        "iunwri": res_file.file_unit,
        "iuncrd": dcd_file.file_unit,
        "nsavl": 0,  # frequency for saving lambda values in lamda-dynamics
        "iunldm": -1,
        "ilap": -1,
        "ilaf": -1,
        "nprint": 100,  # Frequency to write to output
        "iprfrq": 500,  # Frequency to calculate averages
        "isvfrq": 1000,  # Frequency to save restart file
        "ntrfrq": 0,
        "ihtfrq": 0,
        "ieqfrq": 0,
        "firstt": 300,
        "finalt": 300,
        "tbath": 300,
        "iasors": 0,
        "iasvel": 1,
        "ichecw": 0,
        "iscale": 0,  # scale velocities on a restart
        "scale": 1,  # scaling factor for velocity scaling
        "echeck": -1,
    }

    dyn_nve = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_nve.run()

    str_file.close()
    res_file.close()
    dcd_file.close()

# Step 6: Equilibration - CHARMM, PhysNet
# -----------------------------------------------------------

if False:

    timestep = 0.0002  # 0.2 fs

    pmass = int(np.sum(fad_water[: fad_atoms + Ntip3 * 3].get_masses()) / 50.0)
    tmass = int(pmass * 10)

    str_file = pycharmm.CharmmFile(
        file_name="heat.res", file_unit=3, formatted=True, read_only=False
    )
    res_file = pycharmm.CharmmFile(
        file_name="equi.res", file_unit=2, formatted=True, read_only=False
    )
    dcd_file = pycharmm.CharmmFile(
        file_name="equi.dcd", file_unit=1, formatted=False, read_only=False
    )

    # Run some dynamics
    dynamics_dict = {
        "leap": True,
        "verlet": False,
        "cpt": True,
        "new": False,
        "langevin": False,
        "timestep": timestep,
        "start": False,
        "restart": True,
        "nstep": 20 * 1.0 / timestep,
        "nsavc": 0.01 * 1.0 / timestep,
        "nsavv": 0,
        "inbfrq": -1,
        "ihbfrq": 50,
        "ilbfrq": 50,
        "imgfrq": 50,
        "ixtfrq": 1000,
        "iunrea": str_file.file_unit,
        "iunwri": res_file.file_unit,
        "iuncrd": dcd_file.file_unit,
        "nsavl": 0,  # frequency for saving lambda values in lamda-dynamics
        "iunldm": -1,
        "ilap": -1,
        "ilaf": -1,
        "nprint": 100,  # Frequency to write to output
        "iprfrq": 500,  # Frequency to calculate averages
        "isvfrq": 1000,  # Frequency to save restart file
        "ntrfrq": 1000,
        "ihtfrq": 200,
        "ieqfrq": 0,
        "firstt": 300,
        "finalt": 300,
        "tbath": 300,
        "pint pconst pref": 1,
        "pgamma": 5,
        "pmass": pmass,
        "hoover reft": 300,
        "tmass": tmass,
        "iasors": 0,
        "iasvel": 1,
        "ichecw": 0,
        "iscale": 0,  # scale velocities on a restart
        "scale": 1,  # scaling factor for velocity scaling
        "echeck": -1,
    }

    dyn_equi = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_equi.run()

    str_file.close()
    res_file.close()
    dcd_file.close()


# Step 7: Production - CHARMM, PhysNet
# -----------------------------------------------------------

if False:

    timestep = 0.0002  # 0.2 fs

    pmass = int(np.sum(fad_water[: fad_atoms + Ntip3 * 3].get_masses()) / 50.0)
    tmass = int(pmass * 10)

    for ii in range(0, 10):

        if ii == 0:

            str_file = pycharmm.CharmmFile(
                file_name="equi.res", file_unit=3, formatted=True, read_only=False
            )
            res_file = pycharmm.CharmmFile(
                file_name="dyna.{:d}.res".format(ii),
                file_unit=2,
                formatted=True,
                read_only=False,
            )
            dcd_file = pycharmm.CharmmFile(
                file_name="dyna.{:d}.dcd".format(ii),
                file_unit=1,
                formatted=False,
                read_only=False,
            )

        else:

            str_file = pycharmm.CharmmFile(
                file_name="dyna.{:d}.res".format(ii - 1),
                file_unit=3,
                formatted=True,
                read_only=False,
            )
            res_file = pycharmm.CharmmFile(
                file_name="dyna.{:d}.res".format(ii),
                file_unit=2,
                formatted=True,
                read_only=False,
            )
            dcd_file = pycharmm.CharmmFile(
                file_name="dyna.{:d}.dcd".format(ii),
                file_unit=1,
                formatted=False,
                read_only=False,
            )

        # Run some dynamics
        dynamics_dict = {
            "leap": True,
            "verlet": False,
            "cpt": True,
            "new": False,
            "langevin": False,
            "timestep": timestep,
            "start": False,
            "restart": True,
            "nstep": 100 * 1.0 / timestep,
            "nsavc": 0.01 * 1.0 / timestep,
            "nsavv": 0,
            "inbfrq": -1,
            "ihbfrq": 50,
            "ilbfrq": 50,
            "imgfrq": 50,
            "ixtfrq": 1000,
            "iunrea": str_file.file_unit,
            "iunwri": res_file.file_unit,
            "iuncrd": dcd_file.file_unit,
            "nsavl": 0,  # frequency for saving lambda values in lamda-dynamics
            "iunldm": -1,
            "ilap": -1,
            "ilaf": -1,
            "nprint": 1,  # Frequency to write to output
            "iprfrq": 500,  # Frequency to calculate averages
            "isvfrq": 1000,  # Frequency to save restart file
            "ntrfrq": 1000,
            "ihtfrq": 0,
            "ieqfrq": 0,
            "firstt": 300,
            "finalt": 300,
            "tbath": 300,
            "pint pconst pref": 1,
            "pgamma": 5,
            "pmass": pmass,
            "hoover reft": 300,
            "tmass": tmass,
            "iasors": 0,
            "iasvel": 1,
            "ichecw": 0,
            "iscale": 0,  # scale velocities on a restart
            "scale": 1,  # scaling factor for velocity scaling
            "echeck": -1,
        }

        dyn_prod = pycharmm.DynamicsScript(**dynamics_dict)
        dyn_prod.run()

        str_file.close()
        res_file.close()
        dcd_file.close()
