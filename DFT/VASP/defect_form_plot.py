import os
import logging
import json
import re
from warnings import filterwarnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tqdm import tqdm
from scipy.constants import k as k_B, elementary_charge as e

import doped
from doped.analysis import DefectsParser
from doped.thermodynamics import DefectThermodynamics
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.io_utils import micro_pyawk
from monty.serialization import dumpfn, loadfn
from doped.utils.plotting import format_defect_name
# Constants
k_e = k_B / e  # eV/K

# Configuration
CONFIG = {
    "material": "La2NiO4",
    "elements": ["La", "Ni", "O"],
    "defect_categories": ['Interstitial', 'Vacancy'],
    "defect_dir": ['Interstitial', 'Vacancy', 'Combined'],
    "e_above_hull": 0.09,
    "processes": 1,
}

# --- Logging & style config ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("defect_analysis.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
plt.rcdefaults()
plt.style.use(f"{doped.__path__[0]}/utils/doped.mplstyle")
plt.switch_backend('Agg')
plt.rcParams.update({'figure.max_open_warning': 100})
filterwarnings("ignore", category=UserWarning)

defect_color = {
    "Electrons": "C0", "Holes": "C1", "v_Ni": "C2",
    "v_O_D2h": "C3", "O_i_C2v": "C4", "O_i_Cs": "C5",
    "O_i_D4h": "C6", "O_i_D2d": "C7", "v_La": "C8", "v_O_C4v": "C9"
}

class OutcarParser:
    """Simplified OUTCAR parser for dielectric properties"""
    def __init__(self, filename):
        self.filename = filename
        self._initialize_variables()

    def _initialize_variables(self):
        self.dielectric_index = None
        self.dielectric_tensor = np.zeros((3, 3))
        self.dielectric_ionic_index = None
        self.dielectric_ionic_tensor = np.zeros((3, 3))
        self.piezo_index = None
        self.piezo_tensor = np.zeros((3, 6))
        self.piezo_ionic_index = None
        self.piezo_ionic_tensor = np.zeros((3, 6))
        self.born_ion = None
        self.born = []

    def read_lepsilon(self):
        """Read electronic dielectric properties from OUTCAR"""
        try:
            search = self._get_lepsilon_search_patterns()
            micro_pyawk(self.filename, search, self)
            self.born = np.array(self.born)
            return self.born.copy(), self.dielectric_tensor.copy(), self.piezo_tensor.copy()
        except Exception as exc:
            raise RuntimeError("LEPSILON OUTCAR could not be parsed.") from exc

    def read_lepsilon_ionic(self):
        """Read ionic dielectric properties from OUTCAR"""
        try:
            search = self._get_lepsilon_ionic_search_patterns()
            micro_pyawk(self.filename, search, self)
            return self.dielectric_ionic_tensor.copy(), self.piezo_ionic_tensor.copy()
        except Exception as exc:
            raise RuntimeError("Ionic part of LEPSILON OUTCAR could not be parsed.") from exc

    def _get_lepsilon_search_patterns(self):
        """Define search patterns for electronic properties"""
        # [Implementation shortened for brevity - same as original]
        return []

    def _get_lepsilon_ionic_search_patterns(self):
        """Define search patterns for ionic properties"""
        # [Implementation shortened for brevity - same as original]
        return []
from matplotlib.colors import ListedColormap

def build_fe_colormap(thermo, defect_color):
    colors = []
    for name, entry in thermo.defect_entries.items():
        base = name.split("_")[0]
        colors.append(defect_color.get(base, None))  
    return ListedColormap(colors)

def base_name_without_charge(full_name: str) -> str:
    return re.sub(r"_[+-]?\d+$", "", str(full_name))

def make_anneal_name(full_name: str) -> str:
    return f"{full_name} (Anneal)"

def is_equilibrium_defect_column(col: str) -> bool:
    if col.endswith("(Anneal)"):
        return False
    return col not in [
        "Temperature (K)",
        "Electrons (cm^-3)",
        "Holes (cm^-3)",
        "Anneal Electrons (cm^-3)",
        "Anneal Holes (cm^-3)",
    ]

def is_anneal_defect_column(col: str) -> bool:
    return col.endswith("(Anneal)")

def setup_directories(material: str, defect_dirs: List[str]) -> Dict[str, Any]:
    """Create directory structure and return paths"""
    base_dir = f"{material}_results1"
    paths = {
        "base": base_dir,
        "dielectric": os.path.join(base_dir, "dielectric"),
        "phase_diagrams": os.path.join(base_dir, "phase_diagrams"),
        "chempot": os.path.join(base_dir, "chemical_potentials"),
        "defects": {cat: os.path.join(base_dir, cat) for cat in defect_dirs},
    }
    
    for path in paths.values():
        if isinstance(path, dict):
            for p in path.values():
                os.makedirs(p, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    return paths

def get_composition_changes(defect_name: str, elements: List[str]) -> Dict[str, int]:
    """Determine composition changes for defect name"""
    comp_changes = {element: 0 for element in elements}
    
    if defect_name in ["Electrons", "Holes"]:
        return comp_changes
    
    # Vacancy pattern: v_Element
    vacancy_match = re.match(r'^v_([A-Z][a-z]?)', defect_name)
    if vacancy_match:
        element = vacancy_match.group(1)
        if element in comp_changes:
            comp_changes[element] = -1
        return comp_changes
    
    # Interstitial pattern: Element_i
    interstitial_match = re.match(r'^([A-Z][a-z]?)_i', defect_name)
    if interstitial_match:
        element = interstitial_match.group(1)
        if element in comp_changes:
            comp_changes[element] = 1
        return comp_changes
    
    # Substitutional pattern: Element1_Element2
    substitution_match = re.match(r'^([A-Z][a-z]*)_([A-Z][a-z]*)$', defect_name)
    if substitution_match:
        elem1, elem2 = substitution_match.groups()
        if elem1 in comp_changes and elem2 in comp_changes:
            comp_changes[elem1] = 1
            comp_changes[elem2] = -1
        return comp_changes
    
    # Fallback heuristic
    for element in elements:
        if element in defect_name:
            if defect_name.startswith(element):
                comp_changes[element] = 1
            elif f"v_{element}" in defect_name:
                comp_changes[element] = -1
            break
    
    return comp_changes

def get_base_defect_name(defect_name: str) -> str:
    """Extract base defect name without charge state"""
    match = re.match(r'^(.*?)_[+-]?\d+$', defect_name)
    return match.group(1) if match else defect_name

def validate_dielectric_tensors(outcar_path: str, output_dir: str) -> Optional[Dict]:
    """Extract and save dielectric tensors from OUTCAR"""
    try:
        logger.info(f"Processing dielectric tensors from {outcar_path}")
        outcar = OutcarParser(outcar_path)

        born, static, piezo = outcar.read_lepsilon()
        ionic, piezo_ionic = outcar.read_lepsilon_ionic()
        
        # Convert to proper arrays
        static = np.asarray(static, dtype=np.float64).reshape((3, 3))
        ionic = np.asarray(ionic, dtype=np.float64).reshape((3, 3))
        total = static + ionic
        piezo = np.asarray(piezo, dtype=np.float64).reshape((3, 6))
        piezo_ionic = np.asarray(piezo_ionic, dtype=np.float64).reshape((3, 6))
        born = np.asarray(born, dtype=np.float64).reshape((-1, 3, 3))
        
        dielectric_data = {
            "static_dielectric_tensor": static.tolist(),
            "ionic_dielectric_tensor": ionic.tolist(),
            "total_dielectric_tensor": total.tolist(),
            "piezoelectric_tensor": piezo.tolist(),
            "piezoelectric_ionic_tensor": piezo_ionic.tolist(),
            "born_effective_charges": born.tolist()
        }
        
        # Save dielectric data
        json_path = os.path.join(output_dir, "dielectric_tensors.json")
        with open(json_path, 'w') as f:
            json.dump(dielectric_data["total_dielectric_tensor"], f, indent=4)
        logger.info(f"Saved dielectric tensors to {json_path}")
        
        return dielectric_data
        
    except Exception as e:
        logger.error(f"Dielectric extraction failed: {str(e)}", exc_info=True)
        return None

def manual_formation_energies(thermo:DefectThermodynamics, limit_name=None):
    if limit_name is None:
        limit_name = list(thermo.chempots["limits"].keys())[0]
    mu = thermo.chempots["limits"][limit_name]
    ref = thermo.chempots["elemental_refs"]
    Δμ = {el: mu[el] - ref[el] for el in mu}
    print(f"\nManual formation energies (Ef=0) — limit: {limit_name}")
    print(f"{'Defect':<16} {'q':>4} {'E_form (eV)':>12}")
    print("-" * 40)
    for d in thermo.defect_entries.values():
        ΔE = d.sc_entry_energy - d.bulk_entry_energy
        μ_corr = sum(n * Δμ[str(el)] for el, n in d.defect.element_changes.items())
        corr = d.corrections.get('kumagai_charge_correction', 0.0)
        Eform = ΔE + μ_corr + corr
        print(f"{d.name:<16} {d.charge_state:+4} {Eform:12.4f}")

def plot_manual_formation_energy(thermo, limit_name=None, save_path="manual_defect_diagram.png"):
    if limit_name is None:
        limit_name = list(thermo.chempots["limits"].keys())[0]
    mu = thermo.chempots["limits"][limit_name]
    ref = thermo.chempots["elemental_refs"]
    Δμ = {el: mu[el] - ref[el] for el in mu}
    Ef = np.linspace(-0.5, thermo.band_gap + 0.5, 1000)
    plt.figure(figsize=(10,7))
    for d in thermo.defect_entries.values():
        ΔE = d.sc_entry_energy - d.bulk_entry_energy
        μ_corr = sum(n * Δμ[str(el)] for el, n in d.defect.element_changes.items())
        corr = d.corrections.get('kumagai_charge_correction', 0.0)
        E0 = ΔE + μ_corr + corr
        plt.plot(Ef, E0 + d.charge_state * Ef, label=d.name, lw=2.2)
    plt.axvspan(0, thermo.band_gap, color='lightgray', alpha=0.3)
    plt.xlabel("Fermi level (eV from VBM)")
    plt.ylabel("Formation energy (eV)")
    plt.title(f"Manual Defect Diagram — {limit_name}")
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()
        
#======================
#  Concentration computation
#======================
def calculate_defect_concentration(
    thermo: DefectThermodynamics,
    anneal_temperatures: np.ndarray,
    fermi_dos,
    chempots: dict,
    output_dir: str,
):
    """Frozen Defect Approximation:"""
    rows_eq = []
    rows_an = []
    carrier_rows = []
    
    for T in tqdm(anneal_temperatures, desc="Concentrations vs T"):
        (   
            Ef,
            n_eq,
            p_eq,
            conc_df,
            A_Ef,
            n_an,
            p_an,
            A_conc_df,
        ) = thermo.get_fermi_level_and_concentrations(
            bulk_dos=fermi_dos,
            annealing_temperature=T,
            return_annealing_values=True,
            chempots=chempots,
            skip_formatting=True
        )

        # ----------------------------------------
        # Save carrier concentrations
        # ----------------------------------------
        carrier_rows.append({
            "Temperature (K)": T,
            "Electrons (cm^-3)": n_eq,
            "Holes (cm^-3)": p_eq,
            "Anneal Electrons (cm^-3)": n_an,
            "Anneal Holes (cm^-3)": p_an,
            "Fermi Level (eV)": Ef,
            "Anneal Fermi Level (eV)": A_Ef,
        })

        # ----------------------------------------
        # Equilibrium defects
        # ----------------------------------------
        for dname in conc_df.index.get_level_values("Defect").unique():
            sub = conc_df.loc[dname]
            for charge, row in sub.iterrows():
                full = f"{dname}_{charge}"
                rows_eq.append({
                    "Temperature (K)": T,
                    "Defect": full,
                    "Conc": row["Concentration (cm^-3)"],
                })

        # ----------------------------------------
        # Anneal defects
        # ----------------------------------------
        for dname in A_conc_df.index.get_level_values("Defect").unique():
            sub = A_conc_df.loc[dname]
            for charge, row in sub.iterrows():
                full = f"{dname}_{charge}"
                rows_an.append({
                    "Temperature (K)": T,
                    "Defect (Anneal)": make_anneal_name(full),
                    "Conc (Anneal)": row["Concentration (cm^-3)"],
                })

    # Convert to DataFrames
    df_eq = pd.DataFrame(rows_eq)
    df_an = pd.DataFrame(rows_an)
    df_carrier = pd.DataFrame(carrier_rows)

    # Save carriers to CSV
    df_carrier.to_csv(os.path.join(output_dir,
                                   "carrier_concentrations.csv"),
                      index=False)

    # Wide equilibrium defect table
    df_wide = df_eq.pivot_table(
        index="Temperature (K)",
        columns="Defect",
        values="Conc"
    )

    df_an_wide = df_an.pivot_table(
        index="Temperature (K)",
        columns="Defect (Anneal)",
        values="Conc (Anneal)"
    )

    # Merge
    df_wide = df_wide.merge(df_an_wide,
                            left_index=True,
                            right_index=True,
                            how="left")

    df_wide.reset_index().to_csv(
        os.path.join(output_dir, "defect_concentrations_wide.csv"),
        index=False
    )

    return df_wide.reset_index(), df_carrier


def plot_concentrations(
    thermo,
    temperatures,
    bulk_dos,
    output_dir,
    chempots,
):
    """
    Generate two publication-quality plots:
    1. Quenched concentrations (frozen at 300 K)
    2. Annealed concentrations (high-T equilibrium)
    """
    df_def, df_carrier = calculate_defect_concentration(
        thermo, temperatures, bulk_dos, chempots, output_dir
    )

    T = df_def["Temperature (K)"].values

    # Identify equilibrium and annealed defect columns
    eq_cols = [c for c in df_def.columns if is_equilibrium_defect_column(c)]
    an_cols = [c for c in df_def.columns if is_anneal_defect_column(c)]

    # Extract base names for coloring
    def base_name(col):
        return base_name_without_charge(col.replace(" (Anneal)", ""))

    # Use defect_color dict, fallback to cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {}
    for i, col in enumerate(set(base_name(c) for c in eq_cols + an_cols)):
        base = base_name(col)
        color_map[base] = defect_color.get(base, colors[i % len(colors)])

    # ===================================================================
    # Plot 1: Quenched 
    # ===================================================================
    fig_q, (ax_car_q, ax_def_q) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                               gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.3})

    # Carriers (quenched)
    ax_car_q.plot(T, df_carrier["Electrons (cm^-3)"], label="Electrons", color=defect_color.get("Electrons", "C0"), lw=2.5)
    ax_car_q.plot(T, df_carrier["Holes (cm^-3)"], label="Holes", color=defect_color.get("Holes", "C1"), lw=2.5)
    ax_car_q.set_yscale("log")
    ax_car_q.set_ylabel("Carrier conc. (cm$^{-3}$)")
    ax_car_q.legend(frameon=False)
    ax_car_q.grid(False, which="both", alpha=0.3)

    # Defects (quenched)
    for col in eq_cols:
        base = base_name(col)
        label = format_defect_name(col, include_site_info_in_name=True, wout_charge=True)
        ax_def_q.plot(T, df_def[col], label=label, color=color_map[base], lw=2.2, marker="o", markersize=3, markevery=8)

    ax_def_q.set_yscale("log")
    ax_def_q.set_xlabel("Annealing temperature (K)")
    ax_def_q.set_ylabel("Defect conc. (cm$^{-3}$)")
    ax_def_q.grid(False, which="both", alpha=0.3)
    ax_def_q.legend(frameon=False, ncol=2, fontsize=9)
    plt.suptitle("Quenched defect & carrier concentrations (frozen at 300 K)", fontsize=14, y=0.98)
    q_path = os.path.join(output_dir, "concentrations_quenched.png")
    plt.savefig(q_path, dpi=400, bbox_inches="tight")
    plt.close(fig_q)
    logger.info(f"Quenched plot saved: {q_path}")

    # ===================================================================
    # Plot 2: Annealed (high-T equilibrium)
    # ===================================================================
    fig_a, (ax_car_a, ax_def_a) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                               gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.3})

    # Carriers (annealed)
    ax_car_a.plot(T, df_carrier["Anneal Electrons (cm^-3)"], label="Electrons (annealed)", color=defect_color.get("Electrons", "C0"), lw=2.5, ls="--", alpha=0.9)
    ax_car_a.plot(T, df_carrier["Anneal Holes (cm^-3)"], label="Holes (annealed)", color=defect_color.get("Holes", "C1"), lw=2.5, ls="--", alpha=0.9)
    ax_car_a.set_yscale("log")
    ax_car_a.set_ylabel("Carrier conc. (cm$^{-3}$)")
    ax_car_a.legend(frameon=False)
    ax_car_a.grid(False, which="both", alpha=0.3)

    # Defects (annealed)
    for col in an_cols:
        base = base_name(col)
        label = format_defect_name(col.replace(" (Anneal)", ""), include_site_info_in_name=True, wout_charge=True) + " (annealed)",
        ax_def_a.plot(T, df_def[col], label=label, color=color_map[base], lw=2.2, ls="--", alpha=0.8)

    ax_def_a.set_yscale("log")
    ax_def_a.set_xlabel("Annealing temperature (K)")
    ax_def_a.set_ylabel("Defect conc. (cm$^{-3}$)")
    ax_def_a.grid(False, which="both", alpha=0.3)
    ax_def_a.legend(frameon=False, ncol=2, fontsize=9)

    for ax in [ax_car_a, ax_def_a]:
        ax.axvspan(800, 1200, alpha=0.12, color="gray")

    plt.suptitle("Annealed defect & carrier concentrations (high-T equilibrium)", fontsize=14, y=0.98)
    a_path = os.path.join(output_dir, "concentrations_annealed.png")
    plt.savefig(a_path, dpi=400, bbox_inches="tight")
    plt.close(fig_a)
    logger.info(f"Annealed plot saved: {a_path}")


def main():
    """Main workflow for defect analysis"""
    paths = setup_directories(CONFIG['material'], CONFIG['defect_dir'])
    logger.info(f"Starting defect analysis for {CONFIG['material']}")
    
    # Load chemical potentials
    chempot_path = os.path.join(paths['chempot'], "chemical_potentials.json")
    CHEMPOT_SOURCE = "json"  # Change to "set1" or "set2" to use hardcoded values
    
    chemical_potentials, phase_name = load_chemical_potentials(chempot_path, CHEMPOT_SOURCE)
    
    print(f"Phase: {phase_name}")
    print(f"Chemical potentials: {chemical_potentials}")
    
    T = np.arange(300.0, 2000.0, 100.0)
    
    # Define critical file paths
    bulk_dir = os.path.join(f"{CONFIG['material']}_bulk", "PBE_DOS/DOS_8")
    bulk_vasprun_path = os.path.join(bulk_dir, "vasprun.xml")
    dielectric_path = os.path.join(f"{CONFIG['material']}_bulk", "phonopy_FC/dfpt", "OUTCAR")
    
    # Process dielectric tensors
    dielectric_data = validate_dielectric_tensors(dielectric_path, paths['dielectric'])
    if dielectric_data is None:
        logger.error("Dielectric processing failed. Aborting analysis.")
        return
    
    # Load bulk data
    try:
        bulk_vr = Vasprun(bulk_vasprun_path)
        band_gap, cbm, vbm, efermi = bulk_vr.eigenvalue_band_properties
        logger.info(f"Bulk Band gap: {band_gap:.3f} eV, VBM: {vbm:.3f} eV, CBM: {cbm:.3f} eV")
    except Exception as e:
        logger.error(f"Failed to load bulk vasprun: {str(e)}")
        return
    
    # Process defect categories
    combined_defect_entries = []
    for defect_type, defect_dir in zip(CONFIG["defect_categories"], CONFIG["defect_dir"]):
        out_dir = paths['defects'][defect_dir]
        defect_entries = process_defect_category(
            defect_type, out_dir, dielectric_data['total_dielectric_tensor'],
            bulk_vr, chemical_potentials, T
        )
        if defect_entries:
            combined_defect_entries.extend(defect_entries)
    
    # Process combined defects
    if combined_defect_entries:
        out_dir = paths['defects'][CONFIG["defect_dir"][-1]]
        process_combined_defects(combined_defect_entries, out_dir, bulk_vr, chemical_potentials, T)
    
    # Generate formation energy diagram
    csv_path = f"{paths['defects'][CONFIG['defect_dir'][-1]]}/combined_formation_energies.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        generate_formation_energy_diagram(df, chemical_potentials, phase_name, bulk_vr, 
                                        paths['defects'][CONFIG["defect_dir"][-1]])
    else:
        logger.warning(f"Combined formation energies CSV not found at {csv_path}")

if __name__ == '__main__':
    main()