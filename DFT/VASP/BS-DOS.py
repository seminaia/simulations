import os
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import BSDOSPlotter, BSPlotter
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun

def read_kpoints(file_path):
    """
    Read k-points from a KPOINTS file and return them as a NumPy ndarray.
    Supports Line-mode, Explicit mode, and Reciprocal mode.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Determine the mode of the KPOINTS file
    mode = lines[2].strip()
    kpoints = []
    
    if mode == "Line_mode" or mode == "Line-Mode":
        for line in lines[4:]:
            if line.strip() == "":
                continue
            kpoint = list(map(float, line.split()[:3]))
            kpoints.append(kpoint)
    elif mode == "Explicit":
        num_kpoints = int(lines[1].strip())
        for line in lines[3:3 + num_kpoints]:
            kpoint = list(map(float, line.split()[:3]))
            kpoints.append(kpoint)
    elif mode == "Reciprocal":
        for line in lines[3:]:
            if line.strip() == "":
                continue
            kpoint = list(map(float, line.split()[:3]))
            kpoints.append(kpoint)
    else:
        raise ValueError("Unsupported KPOINTS file format.")
    
    return np.array(kpoints)

# Base directory containing all U directories and HSE directory

# Create a figure for combined plots
fig, ax = plt.subplots(figsize=(12, 8))

try:
            
    # Paths to vasprun.xml and KPOINTS files
    vasprun_path = "./vasprun.xml"
    kpoints_path = "./KPOINTS"

    try:
        # Load VASP output with projected eigenstates
        result = Vasprun(vasprun_path,separate_spins=True, parse_projected_eigen=True)
        print(f"Processing directory: {os.getcwd()}")
        
        # Get complete DOS and band structure
        complete_dos = result.complete_dos
        band_structure = result.get_band_structure(kpoints_filename=kpoints_path, force_hybrid_mode=True,line_mode=True)
        band_structure.get_projections_on_elements_and_orbitals(el_orb_spec={'La':['d','f'], 'Ni':['d'], 'O':['p']})
        #band_structure.save('band_structure.dat')
        # Plot combined BS and DOS
        bsd_plotter = BSDOSPlotter(bs_projection='elements', dos_projection='elements', font='DejaVu Sans')
        bsd_plotter.get_plot(band_structure, complete_dos)
        print("Band Gap:", result.eigenvalue_band_properties[0], "eV",
              "VBM:", result.eigenvalue_band_properties[1], "eV",
              "CBM:", result.eigenvalue_band_properties[2], "eV")
        ax.set_title('Combined Band Structure and Density of States')
        ax.set_xlabel('Wave Vector')
        ax.set_ylabel('Energy (eV)')
        plt.tight_layout()
        plt.savefig('./LNO_BS-DOS.pdf', dpi=300)

        # Plot band structure separately
        bs = BSPlotter(band_structure)
        #bs.bs_plot_data()
        bs_plot = bs.get_plot(vbm_cbm_marker=True,smooth=True)
        bs_plot.set_title('Band Structure')
        bs_plot.set_ylabel('Energy (eV)')
        bs_plot.set_xlabel('Wave Vector')
        plt.savefig('./LNO_BS.png', dpi=300)
        plt.close()
        
        # Plot spin-resolved DOS
        dos = DosPlotter()
        dos.add_dos("Total DOS", complete_dos)
        dos_plot = dos.get_plot()
        plt.savefig('./LNO_DOS.png', dpi=300)
        plt.close()
        print("Band Gap: {:.2f} eV".format(band_structure.get_band_gap().get('energy', 0)))
        
    except Exception as e:
        print(f"Error processing files: {e}")
        
except FileNotFoundError:
    print(f"Directory {os.getcwd()} does not contain required files. Skipping...")