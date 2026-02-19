import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from doped.utils.plotting import format_defect_name
from monty.serialization import loadfn
from doped.thermodynamics import FermiSolver
from pymatgen.io.vasp import Vasprun
# Define defect colors
defect_color = {
    "v_O_C4v": "C0", "v_La": "C1", "v_Ni": "C2",
    "v_O_D2h": "C3", "O_i_C2v": "C4", "O_i_Cs": "C5",
    "O_i_D4h": "C6", "O_i_D2d": "C7", "Electrons": "C8", "Holes": "C9"
}

# Configuration
CONFIG = {
    "material": "La2NiO4",
    "elements": ["La", "Ni", "O"],
    "defect_categories": ['Vacancy','Interstitial', 'Combined'],
    "e_above_hull": 0.08,
    "processes": 1,
    "plot_style": {
        "interstitials": {"color": "blue", "marker": "o"},
        "vacancies": {"color": "red", "marker": "s"}
    }
}
def setup_directories(material, defect_categories):
    """Create directory structure and return paths"""
    base_dir = f"{material}_results"
    paths = {
        "base": base_dir,
        "dielectric": os.path.join(base_dir, "dielectric"),
        "chempot": os.path.join(base_dir, "chemical_potentials"),
        "defects": {cat: os.path.join(base_dir, cat) for cat in defect_categories},
    }
    for path in paths.values():
        if isinstance(path, dict):
            for p in path.values():
                os.makedirs(p, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    return paths

def main():
    # Setup paths
    material = CONFIG["material"]
    base_dir = f"{material}_results"
    paths = setup_directories(material, CONFIG["defect_categories"])
    
    cp_dir = paths['chempot']
    cp_chempots_json = os.path.join(cp_dir, "chemical_potentials.json")
    chempots_all = loadfn(cp_chempots_json)
    el_refs = chempots_all["elemental_refs"]
    limit_name = next(iter(chempots_all["limits"].keys()))
    #chempot1 = chempots_all["limits"][limit_name]  
    chempot1 = {'La': -10, 'Ni': -10, 'O': -10}
    #chempot2 = {'La': -4.48, 'Ni': 3.87, 'O': -2.85}
    chempot2 = {'La': 10, 'Ni': 10, 'O': 10}
    #limit_name = chempot2
    print(f"Chemical potentials loaded: {chempot1} and {chempot2}")
    # Prepare plot
    fig, ax = plt.subplots(figsize=(12, 8))
    # Load data
    combined_path = os.path.join(paths['base'], f"Combined/combined_{limit_name}_thermo.json")
    thermo = loadfn(combined_path)
    bulk_vr = Vasprun(os.path.join(f"{material}_defects2", "La2NiO4_bulk/vasp_gam/vasprun.xml"))
    
    # Initialize FermiSolver
    fs = FermiSolver(thermo, bulk_dos=bulk_vr,chempots=chempots_all)
    # Interpolate chemical potentials 
    mu_df = fs.interpolate_chempots(
        n_points=20, 
        annealing_temperature=1000,
        quenched_temperature=300,
        el_refs=el_refs,
        chempots=[chempot1, chempot2],
    )
    mu_df.to_csv(os.path.join(cp_dir, "chempot_range.csv"), index=True)
    
    # Plot defect concentrations
    for defect in mu_df.index.unique():
        defect_df = mu_df.loc[defect]
        print(f"defect:{defect}, μ_O range: {defect_df['μ_O (eV)'].min()} to {defect_df['μ_O (eV)'].max()}")
        color = defect_color[defect]
        
        # Plot defect concentration
        ax.plot(defect_df['μ_O (eV)'], defect_df['Concentration (cm^-3)'], 
                label=format_defect_name(defect, include_site_info_in_name=True, wout_charge=True),
                color=color, 
                linestyle='-',
                marker = 's',
                alpha=0.7)

    # Plot holes concentration
    ax.plot(defect_df['μ_O (eV)'], defect_df['Holes (cm^-3)'], 
            color = "#999999",
            label = 'Holes',
            linestyle='--',
            marker = 'o',
            alpha=0.8)
    # Plot electrons concentration
    ax.plot(defect_df['μ_O (eV)'], defect_df['Electrons (cm^-3)'], 
            color="#333333",
            label = 'Electrons',
            marker ='x',
            linestyle='--',
            alpha=0.8)
    cp_dir = paths['chempot']
    unique_defects = mu_df.index.unique()

    custom_lines = [plt.Line2D([0], [0], color=defect_color[defect], lw=1) for defect in unique_defects]
    custom_lines.append(plt.Line2D([0], [0], color="#999999", lw=1, linestyle="--", label="holes",marker='o'))
    custom_lines.append(plt.Line2D([0], [0], color="#333333", lw=1, linestyle="--", label="electrons",marker='x'))
    # Set plot properties
    ax.set_xlabel('Oxygen Chemical Potential (μ_O, eV)')
    ax.set_ylabel('Concentration (cm$^{-3}$)')
    ax.set_title(f"Defect Concentrations vs Chemical Potential for {material}")
    #ax.set_ylim(1e-10, 1e20)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    
    # Add legend
    ax.legend(loc='best', frameon=False)
    
    # Save and show plot
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"Brouwer_diagram.png"), dpi=300)

if __name__ == '__main__':
    main()