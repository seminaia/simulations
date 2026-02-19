import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.vasp import Vasprun
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

# 1. Load structure and initialize Phonopy
structure = Structure.from_file("POSCAR")
unitcell = PhonopyAtoms(
    symbols=[str(sp) for sp in structure.species],
    cell=structure.lattice.matrix,
    scaled_positions=structure.frac_coords
)


labels = ["GAMMA","X","L","W","W_2","K","U"]
path = [
   [0.0000000000, 0.0000000000, 0.0000000000],
   [0.5000000000, 0.0000000000, 0.5000000000],
   [0.5000000000, 0.5000000000, 0.5000000000],
   [0.5000000000, 0.2500000000, 0.7500000000],
   [0.7500000000, 0.2500000000, 0.5000000000],
   [0.3750000000, 0.3750000000, 0.7500000000],
   [0.6250000000, 0.2500000000, 0.6250000000]
]
 
# Generate q-points path
bands = [path]  # Wrap path in list for phonopy
qpoints, connections = get_band_qpoints_and_path_connections(bands, npoints=151)

# Initialize Phonopy with correct supercell matrix (match your VASP calculation)
phonon = Phonopy(
    unitcell=unitcell,
#    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # MUST match your supercell!
)

# 2. Load force constants from vasprun.xml
print("Reading force constants from vasprun.xml...")
vr = Vasprun("vasprun.xml")
fc, _ = vr.read_force_constants()
phonon.force_constants = fc
phonon.symmetrize_force_constants()
if fc is None:
    raise RuntimeError("Force constants not found. Ensure VASP calculation used IBRION=8")

# 3. Calculate DOS
print("Calculating DOS...")
phonon.run_mesh([20, 20, 20],with_eigenvectors=True, is_mesh_symmetry=False)
#phonon.run_total_dos(sigma=0.02)
phonon.auto_total_dos(plot=True,filename="LK99_DOS.png")
#TDOS = phonon.plot_total_dos(xlabel='Frequency (THz)',ylabel='DOS (states/eV)')
#TDOS.savefig('Al_DOS.png')

# 4. Calculate projected DOS

print("Calculating projected DOS...")
#phonon.run_mesh([20, 20, 20],with_eigenvectors=True, is_mesh_symmetry=False)
phonon.run_projected_dos()
#phonon.auto_projected_dos(plot=True).savefig("LK99_PDOS.png", dpi=300)

# Set force constants and run band structure
print("Calculating band structure...")
#phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
# Band structure and total DOS
phonon.auto_band_structure(plot=True).savefig("LK99_BS.png", dpi=300)
#BS = phonon.plot_band_structure()
#BS.savefig('Al_BS.png')

# 5. Plot results
print("Plotting results...")
plt.figure(figsize=(12, 8))
plt.tight_layout()

# Band structure with projected DOS
BSDOS = phonon.plot_band_structure_and_dos(pdos_indices=[[0], [1]])
BSDOS.savefig("LK99_BS_PDOS.png", dpi=300)
# 6. Save phonopy data
print("Saving data...")
phonon.save()
print("Done!")