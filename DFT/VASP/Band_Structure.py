
import pymatgen
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.vasp.outputs import BSVasprun
import matplotlib.pyplot as plt
import numpy as np

# Load VASP output
path = 'vasprun.xml'
run = BSVasprun(path, parse_projected_eigen=False, separate_spins=True)
bs = run.get_band_structure("./KPATH.in", line_mode=True)
fermi_energy = bs.efermi

# Print band structure details
vbm = run.eigenvalue_band_properties[2]
cbm = run.eigenvalue_band_properties[1]
band_gap = np.array(cbm) - np.array(vbm)
print(f'VBM: {vbm} eV')
print(f'CBM: {cbm} eV')
print(f'Band Gap: {band_gap} eV')
print("Number of Bands:", bs.nb_bands)
print("Number of k-points:", len(bs.kpoints))

# Check if the material is metallic
if bs.is_metal():
    print("The material is metallic.")
else:
    print("The material is semiconducting.")

# Generate plot
bsplot = BSPlotter(bs).get_plot(smooth=True)

# Customize plot
bsplot.set_title("LNO Band Structure", fontsize=20)
bsplot.set_ylabel("Energy (eV)", fontsize=14)

# Calculate relative energies
vbm_rel = np.array(vbm) - fermi_energy
cbm_rel = np.array(cbm) - fermi_energy

# Add spin legend
bsplot.plot([], [], "b-", label="Spin Up")
bsplot.plot([], [], "r-", label="Spin Down")

# Fill band gap region
if not bs.is_metal():
    xlim = bsplot.get_xlim()
    bsplot.fill_between(
        xlim,
        0,
        band_gap,
        color='gray',
        alpha=0.3,
        label='Band Gap'
    )

# Add grid and legend
bsplot.grid(True, which='both', linestyle='--', linewidth=0.5)
bsplot.legend(fontsize=12, loc='best')
plt.show()
plt.savefig('LNO_BS.png')
