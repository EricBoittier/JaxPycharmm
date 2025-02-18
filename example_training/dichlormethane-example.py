import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

Eref = np.zeros([21], dtype=float)
Eref[2] = -0.498232909223
Eref[7] = -37.731440432799
Eref[9] = -74.878159582108
Eref[18] = -459.549260062932

data_path = Path("/pchem-data/meuwly/boittier/home/dcm-lj-data")
restart_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-d17aaa54-65e1-415e-94ae-980521fcd2b1"
TRAIN_ML = False
ANALYSE_ML = False
RESTART_ML = False
NATOMS = 10

import os
os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"
# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import jax
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())
import sys
import e3x
import jax
import numpy as np
import optax
import orbax
from pathlib import Path
import pandas as pd
# Add custom path
sys.path.append("/pchem-data/meuwly/boittier/home/pycharmm_test")
import physnetjax
sys.path.append("/pchem-data/meuwly/boittier/home/dcm-lj-data")
from pycharmm_lingo_scripts import script1, script2, script2, script3, load_dcm

from physnetjax.data.data import prepare_datasets
from physnetjax.training.loss import dipole_calc
from physnetjax.models.model import EF
from physnetjax.training.training import train_model  # from model import dipole_calc
from physnetjax.data.batches import _prepare_batches as prepare_batches #prepare_batches, prepare_datasets
from physnetjax.calc.helper_mlp import get_ase_calc
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
data_key, train_key = jax.random.split(jax.random.PRNGKey(43), 2)
from physnetjax.restart.restart import get_last, get_files, get_params_model
from physnetjax.analysis.analysis import plot_stats

import pycharmm

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
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.select as select
import pycharmm.shake as shake

from pycharmm.lib import charmm as libcharmm

import ase
from ase.io import read as read_ase
from ase import visualize
from ase.visualize import view

def parse_non_int(s):
    return "".join([_ for _ in s if _.isalpha()]).lower().capitalize()


def calc_physnet(atomic_numbers, atom_positions, 
                 ase_calc_monomer, ase_calc_dimer, 
                 n_monomer = 20,
                 dimer_permutations=[(2,2)]):

    ase_monomers_dict = {}
    e_monomers_dict = {}
    
    for i in range(n_monomer):
        start, stop = i*5, i*5 + 5
        Z, R = atomic_numbers, atom_positions
        Z = [_ for i, _ in enumerate(Z) if start <= i < stop  ]
        R = [_ for i, _ in enumerate(R) if start <= i < stop  ]
        ase_atoms_monomer = ase.Atoms(Z, R)
        ase_atoms_monomer.calc = ase_calc_monomer
        if i == 0:
            monomer_atom_energies = sum( 
        [Eref[z] for z in ase_atoms_monomer.get_atomic_numbers()]
    )
        ase_monomers_dict[i] = ase_atoms_monomer
        e_monomers_dict[i] = ase_atoms_monomer.get_potential_energy()


    
    ase_dimers_dict = {}
    e_dimers_dict = {}

    for perm in dimer_permutations:
        a,b = perm
        monomer1 = ase_monomers_dict[a]
        monomer2 = ase_monomers_dict[b]
        ase_atoms_dimer = monomer1 + monomer2
        ase_atoms_dimer.calc = ase_calc_dimer
        ase_dimers_dict[i] = ase_atoms_dimer
        e_dimers_dict[perm] = ase_atoms_dimer.get_potential_energy()

    eint_dimers_dict = {}
    emonomers_dimers_dict = {}
    for perm in dimer_permutations:
        a,b = perm
        tot_e_dimer = e_dimers_dict[perm]
        monomer1_tot_e = e_monomers_dict[a]
        monomer2_tot_e = e_monomers_dict[b]
        dimer_monomers_sum = (monomer1_tot_e + monomer2_tot_e)
        eint_dimers_dict[perm] = tot_e_dimer - dimer_monomers_sum
        emonomers_dimers_dict[perm] = dimer_monomers_sum

    sum_of_monomers = float(sum(list(e_monomers_dict.values())))
    sum_of_dimers = float(sum(list(e_dimers_dict.values())))
    dimer_intE = float(sum(list(eint_dimers_dict.values())))
    
    energies = {
        "dimer_intE": dimer_intE, 
        "sum_of_monomers": sum_of_monomers, 
        "monomer_atom_energies": monomer_atom_energies
    }
    per_dimer_energies = {
        "emonomers_dimers_dict": emonomers_dimers_dict,
        "e_dimers_dict": e_dimers_dict, 
        "eint_dimers_dict": eint_dimers_dict
    }
    per_monomer = {"e_monomers": e_monomers_dict}
    
    output = {"per_cluster": energies, 
              "per_dimer": per_dimer_energies, 
              "per_monomer": per_monomer }
    return output




model = EF(
    # attributes
    features=129,
    max_degree=1,
    num_iterations=6,
    num_basis_functions=65,
    cutoff=11.0,
    max_atomic_number=19,
    charges=True,
    natoms=NATOMS,
    total_charge=1,
    n_res=4,
    zbl=False,
    debug=False,
)



files = [
    Path("/pchem-data/meuwly/boittier/home/dcm_dimers_MP3_20999.npz"),

]


restart = None
if RESTART_ML:
    restart = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-d18aaa54-65e1-415e-94ae-980521fcd2b1"


if TRAIN_ML:
    train_data, valid_data = prepare_datasets(
        data_key,
        16801,
         4200,
        files,
        clip_esp=False,
        natoms=NATOMS,
        clean=False,
        subtract_atom_energies=False,
        verbose=True,
    )

    ntest = len(valid_data["E"]) // 3
    test_data = {k: v[ntest:] for k, v in valid_data.items()}
    valid_data = {k: v[:ntest] for k, v in valid_data.items()}

    DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "F", "N"]
    batch_size = 11

    test_batches = prepare_batches(data_key, test_data, batch_size,
                                  num_atoms=NATOMS,
                                  data_keys=DEFAULT_DATA_KEYS)

    train_batches = prepare_batches(data_key, train_data, batch_size,
                                  num_atoms=NATOMS,
                                  data_keys=DEFAULT_DATA_KEYS)

    valid_batches = prepare_batches(data_key, valid_data, batch_size,
                                  num_atoms=NATOMS,
                                  data_keys=DEFAULT_DATA_KEYS)

    params = train_model(
        train_key,
        model,
        train_data,
        valid_data,
        num_epochs=int(2e6),
        learning_rate=1.0001,
        energy_weight=2,
        dipole_weight=2,
        charges_weight=2,
        forces_weight=2,
        schedule_fn="constant",
        optimizer="amsgrad",
        batch_size=batch_size,
        num_atoms=NATOMS,
        data_keys=DEFAULT_DATA_KEYS,
        restart=restart,
        name="dichloromethane",
        print_freq=2,
        objective="valid_loss",
        best=2e6,
        batch_method="default",
    )

if ANALYSE_ML:
    output = plot_stats(test_batches, model, params, _set="Test",
                   do_kde=True, batch_size=batch_size)


    _ = plt.hist(output["Es"] - output["predEs"], bins=101)
    plt.scatter(output["Es"] , output["predEs"], alpha=1.1)
    plt.xlim(-799, -750)
    plt.ylim(-799, -750)
    plt.plot([-799, -750], [-800, -750])
    plt.show()
    plt.scatter(output["Es"] , output["predEs"], alpha=1.1)
    plt.xlim(-389, -375)
    plt.ylim(-389, -375)
    plt.plot([-389, -375], [-390, -375])




ev2kcalmol = 1/(ase.units.kcal / ase.units.mol)


def set_pycharmm_xyz(atom_positions):
    xyz = pd.DataFrame(atom_positions, columns=["x", "y", "z"])
    coor.set_positions(xyz)

def capture_neighbour_list():
    # Print something
    distance_command = """
    open unit 2 write form name total.dmat

    COOR DMAT SINGLE UNIT 2 SELE ALL END SELE ALL END

    close unit 2"""
    _ = pycharmm.lingo.charmm_script(distance_command)

    with open("total.dmat") as f:
        output_dmat = f.read()

    atom_number_type_dict = {}
    atom_number_resid_dict = {}

    pair_distance_dict = {}
    pair_resid_dict = {}

    for _ in output_dmat.split("\n"):
        if _.startswith("*** "):
            _, n, resid, resname, at, _ = _.split()

            n = int(n.split("=")[0]) - 1
            atom_number_type_dict[n] = at
            atom_number_resid_dict[n] = int(resid) - 1

    for _ in output_dmat.split("\n"):
        if _.startswith("  "):
            a, b, dist = _.split()
            a = int(a) - 1
            b = int(b) - 1
            dist = float(dist)
            if (atom_number_resid_dict[a] < atom_number_resid_dict[b]):
                pair_distance_dict[(a, b)] = dist
                pair_resid_dict[(a, b)] = (atom_number_resid_dict[a],
                                           atom_number_resid_dict[b])

    return     {"atom_number_type_dict" : atom_number_type_dict,
    "atom_number_resid_dict" : atom_number_resid_dict,
    "pair_distance_dict" : pair_distance_dict,
    "pair_resid_dict" : pair_resid_dict,}

def get_forces_pycharmm():
    positions = coor.get_positions()
    force_command = """coor force sele all end"""
    _ = pycharmm.lingo.charmm_script(force_command)
    forces = coor.get_positions()
    coor.set_positions(positions)
    return forces

def prnlv(i):
    printlevel = f"PRNLev {i}"
    _ = pycharmm.lingo.charmm_script(printlevel)

def wrnlev(i):
    printlevel = f"WRNLev {i}"
    _ = pycharmm.lingo.charmm_script(printlevel)
    
def view_atoms(atoms):
    return view(atoms, viewer="x3d")


from itertools import combinations

def dimer_permutations(n_mol):
    dimer_permutations = list(combinations(range(n_mol), 2))
    return dimer_permutations

def calc_pycharmm_dimers(n_mol = 20):

    RANGE = len(dimer_permutations(n_mol))

    ele_energies = np.zeros(RANGE)
    evdw_energies = np.zeros(RANGE)

    for i, (a,b) in enumerate(dimer_permutations(20)):
        a += 1
        b += 1
        block = f"""BLOCK 2
        CALL 1 SELE .NOT. (RESID {a} .OR. RESID {b}) END
          CALL 2 SELE (RESID {a} .OR. RESID {b}) END
          COEFF 1 1 0.0
          COEFF 2 2 1.0
          COEFF 1 2 0.0
        END
        """
        _ = pycharmm.lingo.charmm_script(block)
        
        energy.show()
        evdw = energy.get_vdw()
        evdw_energies[i] = evdw
        e = energy.get_elec()
        ele_energies[i] = e

    return {"ele_energies": ele_energies, "evdw_energies": evdw_energies}

def reset_block():
    block = f"""BLOCK 1
    CALL 1 SELE ALL END
      COEFF 1 1 1.0
    END
    """
    _ = pycharmm.lingo.charmm_script(block)


import MDAnalysis as mda

def load_pdb_data(pdb_file):

    atypes = psf.get_atype()
    atc = pycharmm.param.get_atc()
    residues = psf.get_res()
    psf.get_natom()
    
    # TODO: this assumes a pure system, need to update
    atoms_per_res = int(len(atypes) / len(residues))
    n_res = len(residues)
    resids = np.array([[i]*atoms_per_res for i in range(n_res)]).flatten()

    
    loaded_pdb = mda.coordinates.PDB.PDBReader(pdb_file)
    loaded_pdb = mda.topology.PDBParser.PDBParser(pdb_file)

    u = mda.Universe(pdb_file)
    atom_names = [s for s in list(u.atoms.names)]
    atom_positions = list(u.atoms.positions)
    atomic_numbers = np.array([ase.data.atomic_numbers[parse_non_int(s)] for s in atom_names])
    mda_resids = [s for s in list(u.atoms.resids)]
    mda_res_at_dict = {(a-1,b): i for i, (a,b) in enumerate(zip(mda_resids, atom_names))}

    
    charmm_res_at_dict = {(a,b): i for i, (a,b) in enumerate(zip(resids, atypes))}
    an_charmm_res_at_dict = {v: k for k,v in charmm_res_at_dict.items()}
    an_mda_res_at_dict = {v: k for k,v in mda_res_at_dict.items()}
    atom_positions = np.array(atom_positions)
    reorder = np.array([charmm_res_at_dict[an_mda_res_at_dict[i]] for i in range(len(atom_positions))])
    atom_positions = atom_positions[reorder]
    atomic_numbers = atomic_numbers[reorder]

    return {"atom_names" : atom_names,
    "atom_positions" : atom_positions,
    "atomic_numbers" : atomic_numbers,
    "mda_resids" : mda_resids,
    "mda_res_at_dict" : mda_res_at_dict,
    "charmm_res_at_dict" : charmm_res_at_dict,
    "an_charmm_res_at_dict" : an_charmm_res_at_dict,
    "an_mda_res_at_dict" : an_mda_res_at_dict,
    "atom_positions" : atom_positions,
    "reorder" : reorder,
    "atom_positions" : atom_positions,
    "atomic_numbers" : atomic_numbers,
    }

def get_data_mda(fn):
    pdb_file = data_path / "dcmk" / fn
    pdb_data_mda = load_pdb_data(pdb_file)
    return pdb_data_mda

def return_ase_atoms_n_atoms(N, atomic_numbers, atom_positions):
    Z, R = atomic_numbers, atom_positions
    Z = [_ for i, _ in enumerate(Z) if i < N  ]
    R = [_ for i, _ in enumerate(R) if i < N  ]
    return ase.Atoms(Z, R)

def job_procedure(key, per_monomer, per_dimer, per_cluster, per_neighbour):
    fn = f"{key.lower()}_modified.pdb"
    pdb_data_mda = get_data_mda(fn)
    atomic_numbers  = pdb_data_mda["atomic_numbers"]
    atom_positions = pdb_data_mda["atom_positions"]
    
    set_pycharmm_xyz(atom_positions)
    neighbour_list_data = capture_neighbour_list()
    
    res = calc_pycharmm_dimers()
    reset_block()
    ele_energies, evdw_energies = res["ele_energies"], res["evdw_energies"]
    sum_ele, sum_vdw = ele_energies.sum(), evdw_energies.sum()
    charmm_energies = {"sum_ele": sum_ele, "sum_vdw": sum_vdw}
    charmm_energies.update({"filename": fn, "key": key})
    
    physnet_energies = calc_physnet(atomic_numbers, atom_positions,
                                    ase_calc_monomer, ase_calc_dimer,
                                    dimer_permutations=dimer_permutations(20))
    

    _tmp_per_monomer = {}
    _tmp_per_monomer.update(physnet_energies["per_monomer"])
    per_monomer.append(_tmp_per_monomer)
    
    _tmp_per_dimer = {
        "ele_energies": ele_energies,
        "evdw_energies": evdw_energies,
    }
    _tmp_per_dimer.update(physnet_energies["per_dimer"])
    per_dimer.append(_tmp_per_dimer)
    
    _tmp_per_cluster = {}
    _tmp_per_cluster.update(charmm_energies)
    _tmp_per_cluster.update(physnet_energies["per_cluster"])
    per_cluster.append(_tmp_per_cluster)
    
    _tmp_per_neighbour = {
        "pair_distance_dict": neighbour_list_data["pair_distance_dict"],
        "pair_resid_dict": neighbour_list_data["pair_resid_dict"],
    }
    per_neighbour.append(_tmp_per_neighbour)
    
    return per_monomer, per_dimer, per_cluster, per_neighbour


def main_job_loop(df):
    # Output
    per_monomer = []
    per_dimer = []
    per_cluster = []
    per_neighbour = []

    # make charmm a bit more quiet
    prnlv(0)
    wrnlev(0)
    
    # progress bar
    bar = tqdm(df["key"])
    for key in bar:
        _ = job_procedure(key, 
                          per_monomer,
                          per_dimer,
                          per_cluster,
                          per_neighbour)
        per_monomer, per_dimer, per_cluster, per_neighbour = _
    
    # pickle the extra
    # Save extras array as pickle
    with open('per_monomer.pkl', 'wb') as f:
        pickle.dump(per_monomer, f)
    
    with open('per_cluster.pkl', 'wb') as f:
        pickle.dump(per_cluster, f)

    with open('per_dimer.pkl', 'wb') as f:
        pickle.dump(per_dimer, f)

    with open('per_neighbour.pkl', 'wb') as f:
        pickle.dump(per_neighbour, f)


if __name__ == "__main__":
    #####################################################################
    """
    Initialize pycharmm
    """
    pycharmm.lingo.charmm_script(script1)
    pycharmm.lingo.charmm_script(script2)
    pycharmm.lingo.charmm_script(load_dcm)
    energy.show()
    vdw = energy.get_vdw()
    elec = energy.get_elec()
    # reset the positions array
    fn = "100_lig_1_modified.pdb"
    pdb_data_mda = get_data_mda(fn)
    atomic_numbers, atom_positions = pdb_data_mda["atomic_numbers"], pdb_data_mda["atom_positions"]
    set_pycharmm_xyz(atom_positions)
    energy.show()
    #####################################################################
    
    #####################################################################
    """
    Load the models and calculators
    """
    restart = get_last(restart_path)
    params, dimer_model = get_params_model(restart)
    dimer_model.natoms = 10
    print(dimer_model)
    restart = get_last(restart_path)
    params, monomer_model = get_params_model(restart)
    monomer_model.natoms = 5
    print(monomer_model)
    conversion={"energy": ev2kcalmol, "dipole": 1, "forces": ev2kcalmol}
    ase_atoms_monomer = return_ase_atoms_n_atoms(5, atomic_numbers, atom_positions)
    ase_calc_monomer = get_ase_calc(params, monomer_model, ase_atoms_monomer, conversion=conversion)
    ase_atoms_monomer.calc = ase_calc_monomer
    ase_atoms_dimer = return_ase_atoms_n_atoms(10, atomic_numbers, atom_positions)
    ase_calc_dimer = get_ase_calc(params, dimer_model, ase_atoms_dimer, conversion=conversion)
    #####################################################################
    
    
    #####################################################################
    """
    Load the data
    """
    data_path = Path("/pchem-data/meuwly/boittier/home/dcm-lj-data")
    df = pd.read_csv(data_path / "formation_energies_kcal_mol.csv", sep="\t")
    df["key"] = df["Cluster"].apply(lambda x: "_".join(x.split("_")[:3]).upper() )
    #####################################################################
    
    main_job_loop(df)

