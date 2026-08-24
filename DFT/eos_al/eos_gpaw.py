import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.eos import EquationOfState
from gpaw import GPAW, PW, restart
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.spacegroup import crystal
from ase.visualize import view
from gpaw_helpers import relax, assign_magmoms, pbe_params, mgga_params
print("\nStarting Birch–Murnaghan EOS scan for Fe...\n")
magmom_map = {'Fe': 2.0, 'Al': 0.1}
a_bcc = 2.86
bcc_fe_atoms = crystal(symbols=['Fe'], basis=[0,0,0],spacegroup=229,cellpar=[a_bcc,a_bcc,a_bcc, 90, 90, 90])
assign_magmoms(bcc_fe_atoms, magmom_map)
a_al = 4.05
fcc_al_atoms = crystal(symbols=['Al'], basis=[0,0,0],spacegroup=225,cellpar=[a_al,a_al,a_al, 90, 90, 90])
assign_magmoms(fcc_al_atoms, magmom_map)

strain = 0.05
# Generate scaling factors
scales = np.linspace(1 - strain, 1 + strain, 9)

volumes = []
energies = []

for s in scales:
    a = a_al* s
    if os.path.exists(f'eos_{a:.3f}.gpw'):
        print(f"Loading existing calculation for a = {a:.3f} Å")
        calc = GPAW(f'eos_{a:.3f}.gpw')
        atoms = fcc_al_atoms.copy()
        atoms.set_cell([[a,0,0],[0,a,0],[0,0,a]])
        atoms.calc = calc
        E = atoms.get_potential_energy()
        V = atoms.get_volume()
        volumes.append(V)
        energies.append(E)

    else:
        
        # Build scaled structure
        atoms = fcc_al_atoms.copy()
        atoms.set_cell([[a,0,0],[0,a,0],[0,0,a]])
        bulk_mgga_params = mgga_params(txt= f"eos_{a:.3f}.txt")
        # Fresh calculator each time
        calc = GPAW(**bulk_mgga_params)
        atoms.calc = calc
        E = atoms.get_potential_energy()
        V = atoms.get_volume()
        volumes.append(V)
        energies.append(E)
        print(f"a = {a:.4f} Å   V = {V:.4f} Å^3   E = {E:.6f} eV")
        calc.write(f'eos_{a:.3f}.gpw')

volumes = np.array(volumes)
energies = np.array(energies)

# ---- Fit Birch–Murnaghan EOS ----
eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
V0, E0, B = eos.fit()

# Convert equilibrium volume → lattice constant (cubic)
a0_eq = (V0)**(1/3)

print("\n===== Birch–Murnaghan Fit Results =====")
print(f"Equilibrium volume V0 = {V0:.6f} Å^3")
print(f"Equilibrium lattice constant a0 = {a0_eq:.6f} Å")
print(f"Bulk modulus B = {B:.2f} eV/Å^3")

# ---- Plot ----
eos.plot('Al_EOS.png', show=True)
