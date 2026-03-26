import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ase.eos import EquationOfState
from gpaw import GPAW
import numpy as np
import matplotlib.pyplot as plt
from ase.spacegroup import crystal
from gpaw_helpers import assign_magmoms, mgga_params

symbols = {'Fe': 'Fe'}
magmom_map = {'Fe': 2.0}
a_bcc = 2.86

print(f"\nStarting EOS scans for {symbols['Fe']}...\n")

# Build starting structures
afm_bcc_atoms = crystal(
    symbols=[symbols['Fe']],
    basis=[(0, 0, 0)],
    spacegroup=229,
    cellpar=[a_bcc, a_bcc, a_bcc, 90, 90, 90]
)
assign_magmoms(afm_bcc_atoms, magmom_map, magnetization='AFM')

fm_bcc_atoms = crystal(
    symbols=[symbols['Fe']],
    basis=[(0, 0, 0)],
    spacegroup=229,
    cellpar=[a_bcc, a_bcc, a_bcc, 90, 90, 90]
)
assign_magmoms(fm_bcc_atoms, magmom_map, magnetization='FM')

configs = {
    "AFM": afm_bcc_atoms,
    "FM": fm_bcc_atoms,
}

strain = 0.03
scales = np.linspace(1 - strain, 1 + strain, 9)

results = {}

for mag_label, atoms0 in configs.items():
    print(f"\n{'='*60}")
    print(f"Starting EOS scan for Fe ({mag_label})")
    print(f"{'='*60}")

    outdir = f"{mag_label.lower()}_eos"
    os.makedirs(outdir, exist_ok=True)

    volumes = []
    energies = []

    for s in scales:
        a = a_bcc * s
        gpw_file = os.path.join(outdir, f"eos_{mag_label.lower()}_{a:.3f}.gpw")
        txt_file = os.path.join(outdir, f"eos_{mag_label.lower()}_{a:.3f}.txt")

        atoms = atoms0.copy()
        atoms.set_cell([[a, 0, 0], [0, a, 0], [0, 0, a]], scale_atoms=True)

        if os.path.exists(gpw_file):
            print(f"Loading existing calculation for {mag_label}, a = {a:.3f} Å")
            calc = GPAW(gpw_file)
            atoms.calc = calc
        else:
            params = mgga_params(txt=txt_file)
            calc = GPAW(**params)
            atoms.calc = calc
            atoms.get_potential_energy()
            calc.write(gpw_file)

        E = atoms.get_potential_energy()
        V = atoms.get_volume()

        volumes.append(V)
        energies.append(E)

        print(f"{mag_label}  a = {a:.4f} Å   V = {V:.4f} Å^3   E = {E:.6f} eV")

    volumes = np.array(volumes)
    energies = np.array(energies)

    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    V0, E0, B = eos.fit()

    a0_eq = V0 ** (1 / 3)
    B_GPa = B * 160.21766208

    print(f"\n{mag_label} results:")
    print(f"V0 = {V0:.6f} Å^3")
    print(f"a0 = {a0_eq:.6f} Å")
    print(f"E0 = {E0:.6f} eV")
    print(f"B  = {B_GPa:.2f} GPa")

    eos.plot(os.path.join(outdir, f"{mag_label.lower()}_EOS.png"))

    results[mag_label] = {
        "volumes": volumes,
        "energies": energies,
        "V0": V0,
        "E0": E0,
        "B": B,
        "a0": a0_eq,
        "B_GPa": B_GPa,
    }

print("\nComparison:")
print(f"FM  E0 = {results['FM']['E0']:.6f} eV")
print(f"AFM E0 = {results['AFM']['E0']:.6f} eV")
print(f"AFM - FM = {results['AFM']['E0'] - results['FM']['E0']:.6f} eV")