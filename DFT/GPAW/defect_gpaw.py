import re
from turtle import position
from ase.build import bulk
import ase.build
import ase.lattice
from shakenbreak.input import Distortions
from ase.spacegroup import crystal
from ase.optimize import BFGS, QuasiNewton
from ase.visualize import view
from collections import OrderedDict
from ase.vibrations import Vibrations
from ase.filters import FrechetCellFilter
from ase.transport.calculators import TransportCalculator
from gpaw import GPAW, PW, restart, Mixer, MixerSum, MixerDif
from gpaw.defects import charged_defect_corrections
from ase.eos import calculate_eos
from matplotlib.pylab import eig
from doped.generation import DefectsGenerator
import numpy as np
#from atomistics.calculators.ase import evaluate_with_ase
#from atomistics.workflows import ElasticMatrixWorkflow
#from atomistics.workflows import EnergyVolumeCurveWorkflow
#import atomistics
from ase import Atoms
import os
from ase.io import Trajectory, read
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union
from pathlib import Path
from pymatgen.analysis.defects.core import DefectType
from pymatgen.io.cif import CifWriter

def relax(atoms: Atoms,
          calculator_params: Dict[str, Any],
          fmax: float = 0.01,
          fixcell: bool = True,
          logname: str = 'opt.log',
          trajname: str = 'opt.traj',
          gpwname: str = 'rlx.gpw') -> Union[Atoms, FrechetCellFilter]:
    """
    Relax atomic structure using GPAW.
    
    Parameters:
    -----------
    atoms : Atoms
        ASE Atoms object to relax
    calculator_params : dict
        Parameters for GPAW calculator
    fmax : float
        Maximum force tolerance (eV/Å)
    fixcell : bool
        If True, only relax atomic positions (ISIF=2 equivalent)
        If False, relax both atoms and cell (ISIF=3 equivalent)
    logname : str
        Name of optimization log file
    trajname : str
        Name of trajectory file
    gpwname : str
        Name of GPAW restart file
    
    Returns:
    --------
    Atoms or FrechetCellFilter
        Relaxed atomic structure
    """
    
    # -------------------------
    # Restart or initialize GPAW
    # -------------------------
    calc_dft = None
    
    if os.path.exists(gpwname) and os.path.getsize(gpwname) > 100:
        try:
            atoms, calc_dft = restart(gpwname)
            atoms.calc = calc_dft
            print(f"Successfully restarted from {gpwname}")
        except Exception as e:
            print(f"Restart failed ({e}). Starting fresh calculation.")
            calc_dft = GPAW(**calculator_params)
            atoms.calc = calc_dft
    else:
        calc_dft = GPAW(**calculator_params)
        atoms.calc = calc_dft
        print("Starting fresh calculation")

    # -------------------------
    # Choose relaxation mode
    # -------------------------
    if fixcell:
        # ISIF = 2 equivalent - only relax atoms
        opt_atoms = atoms
        print("Relaxation mode: fixed cell (ISIF=2 equivalent)")
    else:
        # ISIF = 3 equivalent - relax both atoms and cell
        opt_atoms = FrechetCellFilter(atoms)
        opt_atoms.atoms
        print("Relaxation mode: variable cell (ISIF=3 equivalent)")

    # -------------------------
    # Run optimizer
    # -------------------------
    # Add observer to save checkpoints during optimization

    # Create optimizer
    opt = BFGS(opt_atoms, 
               logfile=logname, 
               trajectory=trajname)
        
    # Run relaxation
    print(f"Starting relaxation with fmax={fmax} eV/Å")
    opt.run(fmax=fmax, steps=500)  # Added steps limit for safety
    opt._traj_write_image
    
    # -------------------------
    # Save final state
    # -------------------------
    try:
        # Write final calculator state
        atoms.calc.write(gpwname, mode='all')
        print(f"Final state saved to {gpwname}")
        # Save trajectory and log are automatically handled by ASE
    except Exception as e:
        print(f"Warning: Could not save final state: {e}")
    
    # Get final forces for reporting
    if hasattr(opt_atoms, 'get_forces'):
        forces = opt_atoms.get_forces()
        max_force = max(np.linalg.norm(forces, axis=1))
        print(f"Relaxation completed. Maximum force: {max_force:.6f} eV/Å")
    
    return opt_atoms

defect_type_map = {
    DefectType.Vacancy.value: "Vacancy",
    DefectType.Interstitial.value: "Interstitial",
    DefectType.Substitution.value: "Substitution"  # In case substitutions are enabled later
}
symbols = ('La','La','La','La','Ni','Ni','O','O','O','O','O','O','O','O')

# Structure setup
a0 = 3.800742
c0 = 12.455721
pris_cell = [
    [a0, 0, 0],
    [0, a0, 0],
    [0, 0, c0]
]

symbols = ('La','La','La','La','Ni','Ni','O','O','O','O','O','O','O','O')
pris_frac_positions = [  
    (0.50,  0.50,  0.14),
    (0.00,  0.00,  0.36),
    (0.00,  0.00,  0.64),
    (0.50,  0.50,  0.86),
    (0.00,  0.00,  0.00),
    (0.50,  0.50,  0.50),
    (0.50,  0.00,  0.00),
    (0.00,  0.50,  0.00),
    (0.00,  0.00,  0.18),
    (0.50,  0.50,  0.32),
    (0.50,  0.00,  0.50),
    (0.00,  0.50,  0.50),
    (0.50,  0.50,  0.68),
    (0.00,  0.00,  0.82)]


magmoms = [0.6, -0.6, 0.6, -0.6, 2.0, -2.0, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6]

pris_atoms = Atoms(symbols=symbols,
              scaled_positions=pris_frac_positions,
              cell=pris_cell,
              pbc=True)
pris_atoms.set_initial_magnetic_moments(magmoms)
pris_atoms.write('La2NiO4_pris.cif', format='cif')
view(pris_atoms, repeat=(2, 2, 1))
extrinsic = {"P": ['Ca', 'Sr'], "Pb": ['Cu', 'Mn']}
substitution_elements = ['La', 'Ni', 'Ca', 'Sr', 'Co', 'Mn']

defect_gen = DefectsGenerator(
    pris_atoms,
    extrinsic=extrinsic,
    interstitial_gen_kwargs=True,
    interstitial_elements=['O'],
    vacancy_gen_kwargs=True,
    vacancy_elements=['La', 'Ni', 'O'],
    substitution_gen_kwargs=False,
    substitution_elements=substitution_elements,
    generate_supercell=False,
    supercell_gen_kwargs={'force_diagonal': True},
)

calculator_params = {
    "convergence": {"density": 1e-4,
                    "eigenstates": 1e-6,
                    "energy": 1e-5,
                    "forces": 1e-2},
    "eigensolver": {"name": "dav",
                    "niter":5},
    "kpts": {"gamma": True,
            "size": [2, 2, 1]},
    "maxiter": 500,
    "mixer":{"backend": "pulay",
            "beta": 0.1,
            "method": "fullspin",
            "nmaxold": 5,
            "weight":100},
    "mode": {"ecut": 520,
            "name": "pw"},
    "nbands":"nao",
    "occupations": {"name": "fermi-dirac",
                    "width": 0.1},
    "setups": {"Ni": ':d, 6.2'},
    "symmetry":"off",
    "txt": "rlx.txt",
    "xc": "PBE"
}

# Run relaxation
pris_relax_atoms = relax(pris_atoms, 
                           calculator_params,
                      fmax=0.01,
                      fixcell=False, 
                      logname='La2NiO4_pris_opt.log',
                      trajname='La2NiO4_pris_opt.traj',
                      gpwname='La2NiO4_pris_rlx.gpw')

pris = Path('La2NiO4_pris_rlx.gpw')

for defect_entry in defect_gen.defect_entries.values():
    defect_name = defect_entry.name
    defect_type_value = defect_entry.defect.defect_type.value
    defect_type = defect_type_map.get(defect_type_value, "Unknown")
    charge = defect_entry.charge_state    
    sc = defect_entry.defect_supercell
    defect_dict, defect_metadata = Distortions(defect_entry)
    pris_relax_atoms = pris_relax_atoms.copy()
    defect_elements = list(OrderedDict.fromkeys([site.species.elements[0].symbol for site in sc]))
    pris_relax_atoms.calc = GPAW(**calculator_params)
    def_atoms = Atoms(symbols=defect_elements,
                      scaled_positions= pris_frac_positions,
                      pbc = True,
                      charge=charge,
                      magmom = magmoms)
    def_atoms.set_initial_magnetic_moments(magmoms)
    def_atoms.write(f'{defect_name}_{defect_type}_q{charge}.cif', format='cif')
    def_atoms.calc = GPAW(**calculator_params)
    
# Get results
E_pris = pris_relax_atoms.get_potential_energy()
pris_pos = pris_relax_atoms.get_positions()
pris_relax_cell = pris_relax_atoms.get_cell()
print("Relaxed cell:")
print(pris_relax_cell)
print("Potential energy (eV):", E_pris)
print("All positions (Angstrom):")
print(pris_pos)
