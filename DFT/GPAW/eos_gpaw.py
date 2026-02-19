from ase.eos import EquationOfState
from gpaw import GPAW, PW
import numpy as np
import matplotlib.pyplot as plt

print("\nStarting Birch–Murnaghan EOS scan for Al...\n")

# ---- Settings ----
a_ref = a_al              # initial guess (4.05)
n_points = 9              # number of scan points
strain = 0.03             # ±3%
ecut = 600                # use well-converged cutoff
kpts = (16, 16, 16)       # dense k-grid for metal
smearing = 0.1            # good for metal

# Generate scaling factors
scales = np.linspace(1 - strain, 1 + strain, n_points)

volumes = []
energies = []

for s in scales:
    a = a_ref * s
    
    # Build scaled structure
    atoms = al_conv_atoms.copy()
    atoms.set_cell([[a,0,0],[0,a,0],[0,0,a]], scale_atoms=True)
    
    # Fresh calculator each time
    calc = GPAW(mode=PW(ecut),
                xc='PBE',
                kpts=kpts,
                occupations={'name':'fermi-dirac','width':smearing},
                txt=f'eos_{a:.3f}.txt')
    
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
print(f"Bulk modulus B = {B/1e9:.2f} GPa (if GPAW in eV/Å^3 units)")

# ---- Plot ----
eos.plot('Al_EOS.png')
plt.show()
