import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pymatgen.io.vasp import Poscar

# -----------------------------
# Reference structure
# -----------------------------
ref_dir = Path("Bond_Distortion_0.0%")
ref_struct = Poscar.from_file(ref_dir / "POSCAR").structure
R0 = ref_struct.cart_coords
n_atoms = len(ref_struct)

Q_vals = []
E_vals = []

# -----------------------------
# Loop over distortions
# -----------------------------
for d in sorted(Path(".").glob("Bond_Distortion_*%")):
    struct = Poscar.from_file(d / "POSCAR").structure

    # Sanity checks
    assert struct.lattice == ref_struct.lattice
    assert struct.species == ref_struct.species

    # Configuration coordinate Q (Å)
    R = struct.cart_coords
    Q = np.sqrt(((R - R0) ** 2).sum())

    # Read total energy (E0) from OSZICAR
    with open(d / "OSZICAR") as f:
        for line in f:
            if "E0=" in line:
                E_tot = float(line.split("E0=")[1].split()[0])

    # Convert to eV/atom
    E_atom = E_tot / n_atoms

    Q_vals.append(Q)
    E_vals.append(E_atom)

Q_vals = np.array(Q_vals)
E_vals = np.array(E_vals)
print("Q values (Å):", Q_vals)
print("E values (eV/atom):", E_vals)
# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(Q_vals, E_vals, "o-", lw=2)
plt.xlabel("Configuration Coordinate Q (Å)")
plt.ylabel("Energy (eV/atom)")
plt.title("Configuration Coordinate Diagram")
plt.grid(True)
plt.tight_layout()
plt.show()
