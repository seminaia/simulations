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

symbols = {'Fe': 'Fe'}
print(f"\nStarting Birch–Murnaghan EOS scan for {symbols['Fe']}...\n")
magmom_map = {'Fe': 2.0}
a_bcc = 2.86
afm_bcc_atoms = crystal(symbols=[symbols['Fe']], basis=[0,0,0],spacegroup=229,cellpar=[a_bcc,a_bcc,a_bcc, 90, 90, 90])
assign_magmoms(afm_bcc_atoms, magmom_map, magnetization='AFM')
fm_bcc_atoms = crystal(symbols=[symbols['Fe']], basis=[0,0,0],spacegroup=229,cellpar=[a_bcc,a_bcc,a_bcc, 90, 90, 90])
assign_magmoms(fm_bcc_atoms, magmom_map, magnetization='FM')
atoms_con= [afm_bcc_atoms, fm_bcc_atoms]
strain = 0.1
# Generate scaling factors
scales = np.linspace(1 - strain, 1 + strain, 5)

volumes = []
energies = []
for atoms in atoms_con:
    print(f"\n{'='*60}")
    print(f"Starting EOS scan for {atoms.get_chemical_symbols()[0]} with magnetization: {'AFM' if atoms.get_initial_magnetic_moments()[0] > 0 else 'FM'}")
    print(f"{'='*60}")
    os.makedirs('afm_eos', exist_ok=True)
    os.makedirs('fm_eos', exist_ok=True)
    for s in scales:
        a = a_bcc * s
        if os.path.exists(f'eos_{a:.3f}.gpw'):
            print(f"Loading existing calculation for a = {a:.3f} Å")
            calc = GPAW(f'eos_{a:.3f}.gpw')
            atoms = afm_bcc_atoms.copy()
            atoms.set_cell([[a,0,0],[0,a,0],[0,0,a]])
            atoms.calc = calc
            E = atoms.get_potential_energy()
            V = atoms.get_volume()
            volumes.append(V)
            energies.append(E)
        else:

            # Build scaled structure
            atoms = afm_bcc_atoms.copy()
            atoms.set_cell([[a,0,0],[0,a,0],[0,0,a]])
            bulk_mgga_params = mgga_params(txt= f"afm_eos_{a:.3f}.txt")
            # Fresh calculator each time
            calc = GPAW(**bulk_mgga_params)
            atoms.calc = calc
            E = atoms.get_potential_energy()
            V = atoms.get_volume()
            volumes.append(V)
            energies.append(E)
            print(f"a = {a:.4f} Å   V = {V:.4f} Å^3   E = {E:.6f} eV")
            calc.write(f'afm_eos_{a:.3f}.gpw')

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
eos.plot(f"{symbols['Fe']}_EOS.png", show=True)
