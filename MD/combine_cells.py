from ase.md.analysis import DiffusionCoefficient
from ase.md import MDLogger
from ase.optimize import BFGS
from ase.build import bulk, make_supercell
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from ase.calculators.lammpsrun import LAMMPS
from ase.visualize import view
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.npt import NPT
from ase import Atoms
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.spacegroup import crystal

# Set working directory to current script location
if 'ipykernel' in sys.modules:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        print(f"Changed working directory to: {os.getcwd()}")
    except NameError:
        print(f"Working directory: {os.getcwd()}")

# ============================================================
#  CREATE INDIVIDUAL LATTICES
# ============================================================
print("="*60)
print("Creating crystal lattices")
print("="*60)

# NaF - Rocksalt structure
a_naf = 4.57  # Å
naf_crystal = crystal('NaF',basis=[(0,0,0),(0,0,0.5)],spacegroup=225,cellpar=[a_naf, a_naf, a_naf, 90, 90, 90],pbc=False,size=(3, 3, 3))
print(f"NaF crystal: {len(naf_crystal)} atoms")
print(f"  Cell: {naf_crystal.cell}")
print(f"  Volume: {naf_crystal.get_volume():.2f} Å³")

# AlF3 
a_alf3 = 4.87  # Å
c_alf3 = 12.47

# Make a supercell
alf3_crystal = crystal("AlF3", basis=[(1/3,2/3,2/3),(1/3,0.254145,0.416667)], spacegroup=167, cellpar=[a_alf3, a_alf3, c_alf3, 90, 90, 120], pbc=False, size=(3, 3, 1))
print(f"\nAlF3 crystal: {len(alf3_crystal)} atoms")
print(f"  Cell: {alf3_crystal.cell}")
print(f"  Volume: {alf3_crystal.get_volume():.2f} Å³")

# ============================================================
#  COMBINE THE TWO LATTICES
# ============================================================
print("\n" + "="*60)
print("Combining lattices")
print("="*60)

# Get the dimensions of each crystal
naf_positions = naf_crystal.get_positions()
naf_cell = naf_crystal.cell

alf3_positions = alf3_crystal.get_positions()
alf3_cell = alf3_crystal.cell

# Calculate offsets to place them side by side
naf_width = np.linalg.norm(naf_cell[0])  # Width in x-direction
alf3_width = np.linalg.norm(alf3_cell[0])

# Offset BeF2 in x-direction by the width of NaF plus some gap
gap = 0.0  # Å gap between crystals
x_offset = naf_width + gap

# Translate BeF2 positions
alf3_positions_translated = alf3_positions + [x_offset, 0, 0]

# Combine atoms
combined_symbols = list(naf_crystal.symbols) + list(alf3_crystal.symbols)
combined_positions = np.vstack([naf_positions, alf3_positions_translated])

# Create combined cell that contains both
combined_cell = [
    [naf_width + gap + alf3_width, 0, 0],
    [0, max(naf_cell[1][1], alf3_cell[1][1]), 0],
    [0, 0, max(naf_cell[2][2], alf3_cell[2][2])]
]

combined_atoms = Atoms(symbols=combined_symbols,
                       positions=combined_positions,
                       cell=combined_cell,
                       pbc=False)
combined_atoms.write(filename='combined_crystal.xyz')
# view(combined_atoms)  # disabled: spawns a lingering background ase-gui process
print(f"Combined crystal: {len(combined_atoms)} atoms")
print(f"  Cell: {combined_atoms.cell}")
print(f"  Volume: {combined_atoms.get_volume():.2f} Å³")