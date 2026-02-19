from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.symmetry.bandstructure import HighSymmKpath
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Load VASP output file (vasprun.xml) to extract the band structure and structure
path = 'Bulk_static/vasprun.xml'
vasprun = Vasprun(path, parse_potcar_file=False, parse_projected_eigen=True)

# Get the final structure used in the DOS calculation
structure = vasprun.final_structure
print(structure)

# Create a KPathSeek instance using the structure
kpath_seek = HighSymmKpath(structure)

# Generate the high-symmetry k-path
kpoints, labels = kpath_seek.get_kpoints(line_density=200, coords_are_cartesian=False)

# Write the high-symmetry k-points to a different file
output_filename = 'NSCF/KPOINTS_high_symmetry.dat'
out = Kpoints().automatic_linemode(divisions=200, ibz=kpath_seek)
out.write_file(filename=output_filename)

# Get the symmetry information
sga = SpacegroupAnalyzer(structure)
spacegroup = sga.get_space_group_symbol()
print(f"Spacegroup: {spacegroup}")

# Plot the high-symmetry k-path
fig, ax = plt.subplots(figsize=(10, 6))
for i, kp in enumerate(kpoints):
    ax.scatter(kp[0], kp[1], c='b', marker='o')
    if labels[i]:
        ax.text(kp[0], kp[1], f"{labels[i]}", fontsize=12, ha='right')

plt.xlabel('kx')
plt.ylabel('ky')
plt.title('High-Symmetry K-Path in Reciprocal Space')
plt.grid(True)
plt.savefig('high_symmetry_kpath.png', dpi=300)
plt.show()


