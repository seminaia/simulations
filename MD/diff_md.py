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

# LiF - Rocksalt structure
a_lif = 4.03  # Å
lif_unit = bulk('LiF', crystalstructure='rocksalt', a=a_lif, cubic=True)
# Make a supercell
lif_crystal = make_supercell(lif_unit, 3 * np.eye(3, dtype=int))  # 3x3x3
print(f"LiF crystal: {len(lif_crystal)} atoms")
print(f"  Cell: {lif_crystal.cell}")
print(f"  Volume: {lif_crystal.get_volume():.2f} Å³")

# BeF2 - Create a tetragonal structure (simplified)
a_bef2 = 4.77
c_bef2 = 5.18
bef2_cell = [[a_bef2, 0, 0],
             [0, a_bef2, 0],
             [0, 0, c_bef2]]

# Simple tetragonal structure (approximate)
bef2_scaled_positions = [
    (0.0, 0.0, 0.0),    # Be
    (0.5, 0.5, 0.5),    # Be
    (0.25, 0.25, 0.25), # F
    (0.75, 0.75, 0.75), # F
    (0.25, 0.75, 0.0),  # F
    (0.75, 0.25, 0.5),  # F
]
bef2_symbols = ['Be', 'Be', 'F', 'F', 'F', 'F']
#bef2_unit = Atoms(symbols=bef2_symbols,
#                  scaled_positions=bef2_scaled_positions,
#                  cell=bef2_cell,
#                  pbc=True)

bef2_unit = bulk('BeF2', crystalstructure='fluorite', a=a_bef2, c=c_bef2,cubic=True)

# Make a supercell
bef2_crystal = make_supercell(bef2_unit, 3 * np.eye(3, dtype=int))
print(f"\nBeF2 crystal: {len(bef2_crystal)} atoms")
print(f"  Cell: {bef2_crystal.cell}")
print(f"  Volume: {bef2_crystal.get_volume():.2f} Å³")

# ============================================================
#  COMBINE THE TWO LATTICES
# ============================================================
print("\n" + "="*60)
print("Combining lattices")
print("="*60)

# Get the dimensions of each crystal
lif_positions = lif_crystal.get_positions()
lif_cell = lif_crystal.cell

bef2_positions = bef2_crystal.get_positions()
bef2_cell = bef2_crystal.cell

# Calculate offsets to place them side by side
lif_width = np.linalg.norm(lif_cell[0])  # Width in x-direction
bef2_width = np.linalg.norm(bef2_cell[0])

# Offset BeF2 in x-direction by the width of LiF plus some gap
gap = 0.0  # Å gap between crystals
x_offset = lif_width + gap

# Translate BeF2 positions
bef2_positions_translated = bef2_positions + [x_offset, 0, 0]

# Combine atoms
combined_symbols = list(lif_crystal.symbols) + list(bef2_crystal.symbols)
combined_positions = np.vstack([lif_positions, bef2_positions_translated])

# Create combined cell that contains both
combined_cell = [
    [lif_width + gap + bef2_width, 0, 0],
    [0, max(lif_cell[1][1], bef2_cell[1][1]), 0],
    [0, 0, max(lif_cell[2][2], bef2_cell[2][2])]
]

combined_atoms = Atoms(symbols=combined_symbols,
                       positions=combined_positions,
                       cell=combined_cell,
                       pbc=True)
traj_file = "LiF_BeF2_nvt.traj"

diff = DiffusionCoefficient(traj_file,timestep=0.5).calculate()
print(f"Diffusion coefficient: {diff:.4e} Å²/ps")