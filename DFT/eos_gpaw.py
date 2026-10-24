import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ase.eos import EquationOfState
from gpaw import GPAW
import numpy as np
import matplotlib.pyplot as plt
from ase.spacegroup import crystal
from gpaw_helpers import assign_magmoms, pbe_params

symbols = {'Fe': 'Fe'}
a_fcc = 3.66
a_bcc = 2.86
magmom_map = {'Fe': 2.2}
print(f"\nStarting EOS scans for {symbols['Fe']}...\n")

# Build starting structures
fe_fcc_atoms = crystal(
    symbols=[symbols['Fe']],
    basis=[(0, 0, 0)],
    spacegroup=225,
    cellpar=[a_fcc, a_fcc, a_fcc, 90, 90, 90]
)
fe_bcc_atoms = crystal(
    symbols=[symbols['Fe']],
    basis=[(0, 0, 0)],
    spacegroup=229,
    cellpar=[a_bcc, a_bcc, a_bcc, 90, 90, 90]
)
assign_magmoms(fe_fcc_atoms, magmom_map, magnetization='AFM')
assign_magmoms(fe_bcc_atoms, magmom_map, magnetization='AFM')
fcc_reference_atoms = fe_fcc_atoms.copy()
bcc_reference_atoms = fe_bcc_atoms.copy()

strain = 0.1
scales = np.linspace(1 - strain, 1 + strain, 7)

results = {}
structures = {
    "fcc": (fcc_reference_atoms, a_fcc),
    "bcc": (bcc_reference_atoms, a_bcc),
}

for phase, (reference_atoms, a0) in structures.items():
    print(f"\n{'='*60}")
    print(f"Starting {phase.upper()} EOS scan for Fe")
    print(f"{'='*60}")
    volumes = []
    energies = []
    outdir = f"{symbols['Fe']}_eos_pbe_{phase}"
    os.makedirs(outdir, exist_ok=True)
    gpw_file = lambda a: os.path.join(outdir, f"eos_{a:.3f}.gpw")
    txt_file = lambda a: os.path.join(outdir, f"eos_{a:.3f}.txt")
    for s in scales:
        a = a0 * s
        atoms = reference_atoms.copy()
        atoms.set_cell([[a, 0, 0], [0, a, 0], [0, 0, a]], scale_atoms=True)
        if os.path.exists(gpw_file(a)):
            print(f"Loading existing {phase.upper()} calculation for Fe, a = {a:.3f} Å")
            calc = GPAW(gpw_file(a))
            atoms.calc = calc
        else:
            params = pbe_params(txt=txt_file(a))
            calc = GPAW(**params)
            atoms.calc = calc
            atoms.get_potential_energy()
            calc.write(gpw_file(a))
        E = atoms.get_potential_energy()
        V = atoms.get_volume()
        volumes.append(V)
        energies.append(E)
        print(f"Fe {phase.upper()}  a = {a:.4f} Å   V = {V:.4f} Å^3   E = {E:.6f} eV")
    volumes = np.array(volumes)
    energies = np.array(energies)
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    V0, E0, B = eos.fit()
    a0_eq = V0 ** (1 / 3)
    eos.plot(os.path.join(outdir, f"fe_{phase}_EOS.png"))
    results[phase] = {
        "volumes": volumes,
        "energies": energies / len(reference_atoms),
        "V0": V0,
        "E0": E0 / len(reference_atoms),
        "B": B,
        "a0": a0_eq,
        "B_GPa": B * 160.21766208,
    }
    print(f"\n{phase.upper()} EOS scan completed. Results saved in {outdir}.")
    print(f"Equilibrium volume: {V0:.6f} Å^3")
    print(f"Equilibrium energy: {E0 / len(reference_atoms):.6f} eV/atom")
    print(f"Bulk modulus: {B:.6f} eV/Å^3 ({B * 160.21766208:.2f} GPa)")
    print(f"Equilibrium lattice constant: {a0_eq:.6f} Å")