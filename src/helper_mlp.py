import jax
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())
#from jax import config
#config.update('jax_enable_x64', True)
import ase
import ase.calculators.calculator as ase_calc
import ase.io as ase_io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
import ase.optimize as ase_opt
import matplotlib.pyplot as plt
import py3Dmol
#import numpy as np
from pycharmm_calculator import PyCharmm_Calculator
from model import *
from model import EF, train_model
import pandas as pd
import e3x
import numpy as np


@jax.jit
def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
  return model.apply(params,
    atomic_numbers=atomic_numbers,
    positions=positions,
    dst_idx=dst_idx,
    src_idx=src_idx,
  )


class MessagePassingCalculator(ase_calc.Calculator):
  implemented_properties = ["energy", "forces", "dipole"]

  def calculate(self, atoms, properties, system_changes = ase.calculators.calculator.all_changes):
    ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))
    output = evaluate_energies_and_forces(
      atomic_numbers=atoms.get_atomic_numbers(),
      positions=atoms.get_positions(),
      dst_idx=dst_idx,
      src_idx=src_idx
    )
    if model.charges:
        dipole = dipole_calc(atoms.get_positions(), 
                             atoms.get_atomic_numbers(), 
                             output["charges"],
                    np.zeros_like(atoms.get_atomic_numbers()),
                    1)
        self.results["dipole"] = dipole
    self.results['energy'] = output["energy"].squeeze() #* (ase.units.kcal/ase.units.mol)
    self.results['forces'] = output["forces"] #* (ase.units.kcal/ase.units.mol) #/ase.units.Angstrom

# params = pd.read_pickle("checkpoints/test.pkl")
# params = pd.read_pickle("checkpoints/test2.pkl")
# #print(params)

# model = EF(
#     # attributes
#     features = 32,
#     max_degree = 4,
#     num_iterations = 2,
#     num_basis_functions = 16,
#     cutoff = 6.0,
#     max_atomic_number = 32,
# )

# model = EF(
#     # attributes
#     features = 128,
#     max_degree = 1,
#     num_iterations = 2,
#     num_basis_functions = 32,
#     cutoff = 6.0,
#     max_atomic_number = 32,
# )

# ase_mol = ase.io.read("input/ala_-154.0_26.0.pdb")
ase_mol = ase.io.read("aaa.pdb")
Z = ase_mol.get_atomic_numbers()
Z = [_ if _ < 9 else 6 for _ in Z ]
R = ase_mol.get_positions()
atoms = ase.Atoms(Z, R )
NATOMS = len(Z)

conversion = {"energy": 1/(ase.units.kcal/ase.units.mol), "forces": 1/(ase.units.kcal/ase.units.mol)}

def get_ase_calc(params, model, ase_mol, conversion=conversion):
    @jax.jit
    def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
      return model.apply(params,
        atomic_numbers=atomic_numbers,
        positions=positions,
        dst_idx=dst_idx,
        src_idx=src_idx,
      )
    class MessagePassingCalculator(ase_calc.Calculator):
      implemented_properties = ["energy", "forces", "dipole"]
    
      def calculate(self, atoms, properties, system_changes = ase.calculators.calculator.all_changes):
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))
        output = evaluate_energies_and_forces(
          atomic_numbers=atoms.get_atomic_numbers(),
          positions=atoms.get_positions(),
          dst_idx=dst_idx,
          src_idx=src_idx
        )
        if model.charges:
            dipole = dipole_calc(atoms.get_positions(), 
                                 atoms.get_atomic_numbers(), 
                                 output["charges"],
                        np.zeros_like(atoms.get_atomic_numbers()),
                        1)
            self.results["dipole"] = dipole
        self.results['energy'] = output["energy"].squeeze() #* (ase.units.kcal/ase.units.mol)
        self.results['forces'] = output["forces"] #* (ase.units.kcal/ase.units.mol) #/ase.units.Angstrom
    return MessagePassingCalculator()

def get_pyc(params, model, ase_mol, conversion=conversion):
    Z = ase_mol.get_atomic_numbers()
    Z = [_ if _ < 9 else 6 for _ in Z ]
    R = ase_mol.get_positions()
    atoms = ase.Atoms(Z, R )
    NATOMS = len(Z)
    print(atoms)

    @jax.jit
    def model_calc(batch):
      output = model.apply(params,
        atomic_numbers=jax.numpy.array(batch["atomic_numbers"]),
        positions=jax.numpy.array(batch["positions"]),
        dst_idx=jax.numpy.array(batch["dst_idx"]),
        src_idx=jax.numpy.array(batch["src_idx"]),
      )
      output["energy"] *= conversion["energy"]
      output["forces"] *= conversion["forces"]
      return output
    
    
    pyc = PyCharmm_Calculator(
            model_calc,   
            ml_atom_indices=np.arange(model.natoms),
            ml_atomic_numbers = Z,
            ml_charge = Z,
            # ml_fluctuating_charges = model.charges
    )

    blah = np.array(list(range(NATOMS)))
    blah1 = np.array(list(range(3200)))
    blah2 = np.arange(NATOMS) * 1.0
    print("...", dir(pyc)), pyc, "pyc?"
    _ = pyc.calculate_charmm(
            Natom=NATOMS,
            Ntrans=0,
            Natim=0,
            idxp=blah,
            x=blah2,
            y=blah2,
            z=blah2,
            dx=blah2,
            dy=blah2,
            dz=blah2,
            Nmlp=NATOMS,
            Nmlmmp=NATOMS,
            idxi= blah1,
            idxj= blah1,
            idxjp= blah,
            idxu= blah,
            idxv= blah,
            idxup= blah,
            idxvp= blah,
        )
    print("_", _)
    
    class pyCModel:
        def __init__():
            pass
    
        def get_pycharmm_calculator(        
            ml_atom_indices = None,
            ml_atomic_numbers = None,
            ml_charge= None,
            ml_fluctuating_charges = False,
            mlmm_atomic_charges= None,
        mlmm_cutoff = None,
        mlmm_cuton = None,
        **kwargs):
            return pyc 
    return pyCModel


