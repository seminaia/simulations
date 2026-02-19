from enum import unique
from math import comb
import os
import logging
from tkinter import font
from unittest.mock import Base
from warnings import filterwarnings
from weakref import ref
from narwhals import col
import numpy as np
import pandas as pd
import doped
from doped.core import DefectEntry
import json
from monty.serialization import dumpfn, loadfn
from doped.analysis import DefectsParser
from doped.generation import get_defect_name_from_entry
from doped.utils.plotting import format_defect_name
from matplotlib.colors import ListedColormap
from doped.thermodynamics import DefectThermodynamics, FermiSolver
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.io_utils import micro_pyawk
import matplotlib.pyplot as plt
import matplotlib.colors 
import re
from pyparsing import C
from sympy import N, im, limit
from tqdm import tqdm
from scipy.constants import k as k_B, elementary_charge as e
from PyFunc.NRroots import df
from typing import Dict
k_e = k_B / e # eV/K

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
    def __init__(self, filename):
        self.filename = filename

    def read_lepsilon(self):
        """Read a LEPSILON run.

        TODO: Document the actual variables.
        """
        try:
            search = []

            def dielectric_section_start(results, match):
                results.dielectric_index = -1

            search.append(
                [
                    r"MACROSCOPIC STATIC DIELECTRIC TENSOR \(",
                    None,
                    dielectric_section_start,
                ]
            )

            def dielectric_section_start2(results, match):
                results.dielectric_index = 0

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.dielectric_index == -1,
                    dielectric_section_start2,
                ]
            )

            def dielectric_data(results, match):
                results.dielectric_tensor[results.dielectric_index, :] = np.array(
                    [float(match[i]) for i in range(1, 4)]
                )
                results.dielectric_index += 1

            search.append(
                [
                    r"^ *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *$",
                    lambda results, _line: results.dielectric_index >= 0
                    if results.dielectric_index is not None
                    else None,
                    dielectric_data,
                ]
            )

            def dielectric_section_stop(results, match):
                results.dielectric_index = None

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.dielectric_index >= 1
                    if results.dielectric_index is not None
                    else None,
                    dielectric_section_stop,
                ]
            )

            self.dielectric_index = None
            self.dielectric_tensor = np.zeros((3, 3))

            def piezo_section_start(results, _match):
                results.piezo_index = 0

            search.append(
                [
                    r"PIEZOELECTRIC TENSOR  for field in x, y, z        \(C/m\^2\)",
                    None,
                    piezo_section_start,
                ]
            )

            def piezo_data(results, match):
                results.piezo_tensor[results.piezo_index, :] = np.array([float(match[i]) for i in range(1, 7)])
                results.piezo_index += 1

            search.append(
                [
                    r"^ *[xyz] +([-0-9.Ee+]+) +([-0-9.Ee+]+)"
                    r" +([-0-9.Ee+]+) *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)*$",
                    lambda results, _line: results.piezo_index >= 0 if results.piezo_index is not None else None,
                    piezo_data,
                ]
            )

            def piezo_section_stop(results, _match):
                results.piezo_index = None
            self.born = []

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.piezo_index >= 1 if results.piezo_index is not None else None,
                    piezo_section_stop,
                ]
            )

            self.piezo_index = None
            self.piezo_tensor = np.zeros((3, 6))

            def born_section_start(results, _match):
                results.born_ion = -1

            search.append([r"BORN EFFECTIVE CHARGES ", None, born_section_start])

            def born_ion(results, match):
                results.born_ion = int(match[1]) - 1
                results.born.append(np.zeros((3, 3)))

            search.append(
                [
                    r"ion +([0-9]+)",
                    lambda results, _line: results.born_ion is not None,
                    born_ion,
                ]
            )

            def born_data(results, match):
                results.born[results.born_ion][int(match[1]) - 1, :] = np.array([float(match[i]) for i in range(2, 5)])

            search.append(
                [
                    r"^ *([1-3]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)$",
                    lambda results, _line: results.born_ion >= 0 if results.born_ion is not None else results.born_ion,
                    born_data,
                ]
            )

            def born_section_stop(results, _match):
                results.born_ion = None

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.born_ion >= 1 if results.born_ion is not None else results.born_ion,
                    born_section_stop,
                ]
            )

            self.born_ion = None
            self.born: np.array(self.born,dtype=float) 

            micro_pyawk(self.filename, search, self)

            self.born = np.array(self.born)
            self.dielectric_tensor = self.dielectric_tensor.tolist()
            self.piezo_tensor = self.piezo_tensor.tolist()
            return self.born.copy(), self.dielectric_tensor.copy(), self.piezo_tensor.copy()

        except Exception as exc:
            raise RuntimeError("LEPSILON OUTCAR could not be parsed.") from exc

    def read_lepsilon_ionic(self):
        """Read the ionic component of a LEPSILON run.

        TODO: Document the actual variables.
        """
        try:
            search = []

            def dielectric_section_start(results, _match):
                results.dielectric_ionic_index = -1

            search.append(
                [
                    r"MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC",
                    None,
                    dielectric_section_start,
                ]
            )

            def dielectric_section_start2(results, _match):
                results.dielectric_ionic_index = 0

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.dielectric_ionic_index == -1
                    if results.dielectric_ionic_index is not None
                    else results.dielectric_ionic_index,
                    dielectric_section_start2,
                ]
            )

            def dielectric_data(results, match):
                results.dielectric_ionic_tensor[results.dielectric_ionic_index, :] = np.array(
                    [float(match[i]) for i in range(1, 4)]
                )
                results.dielectric_ionic_index += 1

            search.append(
                [
                    r"^ *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *$",
                    lambda results, _line: results.dielectric_ionic_index >= 0
                    if results.dielectric_ionic_index is not None
                    else results.dielectric_ionic_index,
                    dielectric_data,
                ]
            )

            def dielectric_section_stop(results, _match):
                results.dielectric_ionic_index = None

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.dielectric_ionic_index >= 1
                    if results.dielectric_ionic_index is not None
                    else results.dielectric_ionic_index,
                    dielectric_section_stop,
                ]
            )

            self.dielectric_ionic_index = None
            self.dielectric_ionic_tensor = np.zeros((3, 3))

            def piezo_section_start(results, _match):
                results.piezo_ionic_index = 0

            search.append(["PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        ", None, piezo_section_start])

            def piezo_data(results, match):
                results.piezo_ionic_tensor[results.piezo_ionic_index, :] = np.array(
                    [float(match[i]) for i in range(1, 7)]
                )
                results.piezo_ionic_index += 1

            search.append(
                [
                    r"^ *[xyz] +([-0-9.Ee+]+) +([-0-9.Ee+]+)"
                    r" +([-0-9.Ee+]+) *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)*$",
                    lambda results, _line: results.piezo_ionic_index >= 0
                    if results.piezo_ionic_index is not None
                    else results.piezo_ionic_index,
                    piezo_data,
                ]
            )

            def piezo_section_stop(results, _match):
                results.piezo_ionic_index = None

            search.append(
                [
                    "-------------------------------------",
                    lambda results, _line: results.piezo_ionic_index >= 1
                    if results.piezo_ionic_index is not None
                    else results.piezo_ionic_index,
                    piezo_section_stop,
                ]
            )

            self.piezo_ionic_index = None
            self.piezo_ionic_tensor = np.zeros((3, 6))

            micro_pyawk(self.filename, search, self)

            self.dielectric_ionic_tensor = self.dielectric_ionic_tensor.tolist()
            self.piezo_ionic_tensor = self.piezo_ionic_tensor.tolist()
            return self.dielectric_ionic_tensor.copy(), self.piezo_ionic_tensor.copy()
        except Exception as exc:
            raise RuntimeError("ionic part of LEPSILON OUTCAR could not be parsed.") from exc
        
def create_custom_cmap(color_list):
    names = list(color_list.keys())
    colors = [defect_color[name] for name in names if name in defect_color]
    n_colors = len(colors)
    cmap = {}
    for defect in color_list:
        if defect in defect_color:
            cmap[defect] = matplotlib.colors.to_rgba(defect_color[defect])
        else:
            cmap[defect] = matplotlib.colors.to_rgba(color_list.get(defect, 'gray'))  # Default color if not specified
    return cmap, names

def create_xmgrace_file(data, columns, filename):
    """Create an xmgrace-compatible space-delimited file"""
    with open(filename, 'w') as f:
        for _, row in data.iterrows():
            formatted = [f"{row[col]:.6e}" if isinstance(row[col], float) else str(row[col]) for col in columns]
            f.write(" ".join(formatted) + "\n")

def validate_dielectric_tensors(outcar_path, output_dir):
    """Extract and save dielectric tensors from OUTCAR"""
    try:
        logger.info(f"Processing dielectric tensors from {outcar_path}")
        outcar = OutcarParser(outcar_path)
        born, static, piezo = outcar.read_lepsilon()
        ionic, piezo_ionic = outcar.read_lepsilon_ionic()
        static = np.asarray(static, dtype=np.float64).reshape((3,3))
        ionic = np.asarray(ionic, dtype=np.float64).reshape((3,3))
        total = np.array(static) + np.array(ionic)
        total = np.asarray(total, dtype=np.float64).reshape((3,3))
        piezo = np.asarray(piezo, dtype=np.float64).reshape((3,6))
        piezo_ionic = np.asarray(piezo_ionic, dtype=np.float64).reshape((3,6))
        born = np.asarray(born, dtype=np.float64).reshape((-1,3,3))
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

def build_fe_colormap(thermo, defect_color):
    colors = []
    for name, entry in thermo.defect_entries.items():
        base = name.split("_")[0]
        colors.append(defect_color.get(base))  
    return ListedColormap(colors)

def base_name_without_charge(full_name: str) -> str:
    return re.sub(r"_[+-]?\d+$", "", str(full_name))

def process_defect_category(defect_type: str, defect_path: str, output_dir: str, dielectric,
                            bulk_vr : Vasprun, chempots, temperatures, el_refs: Dict):
    try:
        logger.info(f"Processing {defect_type} defects in {defect_path}")
        dp = DefectsParser(
            output_path=defect_path,
            dielectric=dielectric,
            processes=CONFIG["processes"],
            json_filename=f"{defect_type}_defect_dict.json",
            bulk_band_gap_vr=bulk_vr,
        )
        ref_dict    = el_refs
        if "limits_wrt_el_refs" in chempots:
            limit_name = list(chempots["limits_wrt_el_refs"].keys())[0]   
            dmu = chempots["limits_wrt_el_refs"][limit_name]
        elif "Phases" in chempots:
            limit_name = chempots['Phases']
            dmu ={el: chempots[el] for el in ref_dict}
        else:
            if "Phases" not in chempots:
                limit_name = "User-Specified-Limits"
            dmu ={el: chempots[el] for el in ref_dict}
        mu_abs = {el: float(dmu[el]) + float(ref_dict[el]) for el in ref_dict}
        thermo = dp.get_defect_thermodynamics(bulk_dos=bulk_vr.complete_dos,
                                              chempots=dmu, 
                                              el_refs=ref_dict,
                                              dist_tol=0.5,
                                              check_compatibility=True)
        vbm = bulk_vr.eigenvalue_band_properties[2]
        fermi=bulk_vr.efermi
        print(f"Bulk Energy: {bulk_vr.final_energy} eV")
        print(f"Using chemical potentials: {dmu}")
        print(f"Δμ (formal, relative to elemental refs):{dmu}")
        print(f"Elemental Reference Energies {ref_dict}")
        print(f"Absolute Chemical Potential {mu_abs}")
        print(f"Limit: {limit_name}")

        for el, dmu_val in dmu.items():
            print(f"   {el}: {dmu_val:+.3f} eV")
            
        #print(f"{'Defect':<14} {'q':>3} {'ΔE':>9} {'μ·ΔN':>9} {'Corr':>9} {'E_form':>12} {'E_form_doped':>12} {'Bulk Energy':>12} {'Defect Energy (correction)':>12} {'Defect Energy (raw)':>12}")
        #print("-" * 100)
        
        for d in thermo.defect_entries.values():
            bulk_energy = d.bulk_entry_energy
            raw_defect_energy = d.sc_entry_energy
            defect_energy = d.corrected_energy
            dE = d.get_ediff()
            mu_term =d._get_chempot_term(mu_abs)
            trans_level = thermo.transition_levels
            q = d.charge_state
            corr = d.corrections.get('kumagai_charge_correction', 0.0)
            E_form_doped = d.formation_energy(chempots=dmu, el_refs=ref_dict)
            E_form = dE + mu_term + q*fermi
            #print(
            #    f"{d.name:<14} {q:+3d} "
            #    f"{dE:9.3f} {mu_term:9.3f} {corr:9.3f} "
            #    f"{E_form:12.3f} {E_form_doped:12.3f} "
            #    f"{bulk_energy:12.3f} {defect_energy:12.3f} {raw_defect_energy:12.3f}"
            #)
        print(f"VBM: {vbm:.3f} eV, Equil. Fermi level: {fermi:.3f} eV\n")
        

        fe_df = thermo.get_formation_energies(chempots=dmu, el_refs=ref_dict,skip_formatting=True)
        fe_df.drop(columns=["Path"], inplace=True)
        fe_df.reset_index(inplace=True)
        dumpfn(thermo, os.path.join(output_dir, f"{defect_type}_{limit_name}_thermo.json"))

        fe_df = fe_df.rename(columns={
            "ΔEʳᵃʷ": r"$\Delta E_{\mathrm{raw}}$",
            "qE_VBM": r"$qE_{\mathrm{VBM}}$",
            "qE_F": r"$qE_F$",
            "Σμ_ref": r"$\sum_i n_i \mu_i^{\mathrm{ref}}$",
            "Σμ_formal": r"$\sum_i n_i \mu_i$",
            "E_corr": r"$E_{\mathrm{corr}}$",
            "Eᶠᵒʳᵐ": r"$E_{\mathrm{form}}$ (eV)",
            "Δ[E_corr]": r"$\Delta E_{\mathrm{corr}}$",
        })
        fe_df["Defect"] = fe_df["Defect"].apply(
            lambda d: format_defect_name(d,include_site_info_in_name=True, wout_charge=True)
        )
        fe_df.drop(columns=[r"$\Delta E_{\mathrm{raw}}$",r"$qE_{\mathrm{VBM}}$",r"$qE_F$",r"$\sum_i n_i \mu_i^{\mathrm{ref}}$",r"$\sum_i n_i \mu_i$",r"$E_{\mathrm{corr}}$",r"$\Delta E_{\mathrm{corr}}$"], 
                   inplace=True)

        fe_df.to_latex(
            os.path.join(output_dir, f"{defect_type}_{limit_name}_formation_energies.tex"), 
            escape=False,
            index=False,
            columns=["Defect","q",r"$E_{\mathrm{form}}$ (eV)"],
            longtable=True,
            caption=f"{defect_type} Defect Formation Energies under {limit_name} conditions",
            float_format="%.3f",
        )
        fe_df.to_csv(os.path.join(output_dir, f"{defect_type}_{limit_name}_formation_energies.csv"),
                      index=False,
                      columns=["Defect","q",r"$E_{\mathrm{form}}$ (eV)"]
                    )
        print(f"Formation Energies:\n",
              f" {fe_df}")
        
        # Symmetry table
        symm_df = thermo.get_symmetries_and_degeneracies(skip_formatting=True)
        symm_df.reset_index(inplace=True)        
        def format_transition(s):
            s = s.replace("ε(", "").replace(")", "")
            s = s.replace("*", r"^{*}")
            return rf"$\varepsilon({s})$"
        symm_df = symm_df.rename(columns={
            "Site_Symm": r"$G_{\mathrm{site}}$",
            "Defect_Symm": r"$G_{\mathrm{defect}}$",
            "g_Orient": r"$g_{\mathrm{orient}}$",
            "g_Spin": r"$g_{\mathrm{spin}}$",
            "g_Total": r"$g_{\mathrm{tot}}$",
        })

        symm_df["Defect"] = symm_df["Defect"].apply(
            lambda d: format_defect_name(d,include_site_info_in_name=True, wout_charge=True)
        )
        symm_df.drop(columns=[r"$G_{\mathrm{site}}$",r"$G_{\mathrm{defect}}$"], inplace=True)
        symm_df.to_latex(
            os.path.join(output_dir, f"{defect_type}_{limit_name}_symmetries.tex"),escape=False,
            index=False,
            columns=["Defect","q", r"$g_{\mathrm{orient}}$", r"$g_{\mathrm{spin}}$", r"$g_{\mathrm{tot}}$","Mult"],
            longtable=True,
            caption=f"{defect_type} Defect Symmetries and Degeneracies under {limit_name} conditions"
        )
        symm_df.to_csv(os.path.join(output_dir, f"{defect_type}_{limit_name}_symmetries.csv"),
                       index=False,
                       columns=["Defect","q", r"$g_{\mathrm{orient}}$", r"$g_{\mathrm{spin}}$", r"$g_{\mathrm{tot}}$","Mult"])
        
        print(f"Symmetries and Degeneracies:\n",
              f"{symm_df}")
        
        # Transition levels table
        trans_level = thermo.get_transition_levels()
        trans_level.reset_index(inplace=True)
        trans_level = trans_level.rename(columns={
            "eV from VBM": r"$\varepsilon(q/q') - E_{\mathrm{VBM}}$~(eV)",
            "In Band Gap?": r"In gap?",
        })
        trans_level["Charges"] = trans_level["Charges"].apply(format_transition)
        trans_level["Defect"] = trans_level["Defect"].apply(
            lambda d: format_defect_name(d,include_site_info_in_name=True, wout_charge=True)
        )
        trans_level.drop(columns = [ r"$\varepsilon(q/q') - E_{\mathrm{VBM}}$~(eV)",r"In gap?"], inplace=True)
        trans_level.to_latex(
            os.path.join(output_dir, f"{defect_type}_{limit_name}_transition_levels.tex"),
            escape=False,
            columns=["Defect", "Charges"],
            index=False,
            longtable=True,
            caption=f"{defect_type} Defect Transition Levels under {limit_name} conditions",
            float_format="%.3f",
        )
        trans_level.to_csv(os.path.join(output_dir, 
                                        f"{defect_type}_{limit_name}_transition_levels.csv"),
                           index=False,
                           columns=["Defect", "Charges"])
        print(f"Transition Levels:\n", 
              f"{trans_level}")
        
        # Formation energy plot
        fe_path = os.path.join(output_dir, f"{defect_type}_{limit_name}_formation_energies.png")
        # Build colormap using the user-defined defect_color mapping
        fe_cmap = build_fe_colormap(thermo, defect_color)
        fig = thermo.plot(all_entries='faded',
                    chempot_table=True,
                    unstable_entries=True,
                    colormap=fe_cmap,
                    fermi_level=bulk_vr.eigenvalue_band_properties[0]/2,
                    include_site_info=True)

        ax = fig.gca()
        ax.set_title(f"{defect_type} Defect Formation Energies\nLimit: {limit_name}",y=1.2)
        ax.title.set_fontsize(12)
        fig.savefig(fe_path, dpi=300, bbox_inches="tight")
        
                 
        plot_fs_concentrations(
            thermo,
            temperatures,
            bulk_vr,
            output_dir,
            chempots,
            ref_dict
        )
        return dp.defect_dict, list(thermo.defect_entries.values())
    except Exception as e:
        logger.error(f"{defect_type} processing failed: {e}", exc_info=True)
        return {}, []
# ===============================================================
#  Combined-defect processing
# ===============================================================
def process_combined_defects(defect_entries, output_dir, bulk_vr,
                             chempots, temperatures, el_refs: Dict):
    try:
        logger.info("Processing COMBINED defect set")
        ref_dict    = el_refs
        fermi=bulk_vr.efermi
        vbm = bulk_vr.eigenvalue_band_properties[2]
        print(f"VBM: {vbm:.3f} eV, Equil. Fermi level: {fermi:.3f} eV\n")
        if "limits_wrt_el_refs" in chempots:
            limit_name = list(chempots["limits_wrt_el_refs"].keys())[0]   
            dmu = chempots["limits_wrt_el_refs"][limit_name]
            
        elif "Phases" in chempots:
            limit_name = chempots['Phases']
            dmu ={el: chempots[el] for el in ref_dict}
        
        else:
            if "Phases" not in chempots:
                limit_name = "User-Specified-Limits"
            dmu ={el: chempots[el] for el in ref_dict}
        mu_abs = {el: float(dmu[el]) + float(ref_dict[el]) for el in ref_dict}  
        
        combined_thermo = DefectThermodynamics(
            defect_entries=defect_entries,
            bulk_dos=bulk_vr.complete_dos,
            chempots=dmu,
            el_refs=ref_dict,
            dist_tol=0.5
        )
        print(f"Using chemical potentials: {dmu}")
        print(f"Δμ (formal, relative to elemental refs):{dmu}")
        print(f"Elemental Reference Energies {ref_dict}")
        print(f"Absolute Chemical Potential {mu_abs}")
        print(f"Limit: {limit_name}")

        for el, dmu_val in dmu.items():
            print(f"   {el}: {dmu_val:+.3f} eV")
        #print(f"{'Defect':<14} {'q':>3} {'ΔE':>9} {'μ·ΔN':>9} {'Corr':>9} {'E_form':>12} {'E_form_doped':>12} {'Bulk Energy':>12} {'Defect Energy(correction)':>12} {'Defect Energy (raw)':>12}")
        #print("-" * 100)
        for d in combined_thermo.defect_entries.values():
            bulk_energy = d.bulk_entry_energy
            defect_energy = d.corrected_energy
            raw_defect_energy = d.sc_entry_energy
            dE = d.get_ediff()
            q = d.charge_state
            E_form_doped = d.formation_energy(chempots=dmu, el_refs=ref_dict)
            mu_term = d._get_chempot_term(mu_abs)
            corr = d.corrections.get('kumagai_charge_correction', 0.0)
            trans_level = combined_thermo.transition_levels
            E_form = dE + mu_term + q * fermi
            #print(
            #    f"{d.name:<14} {q:+3d} "
            #    f"{dE:9.3f} {mu_term:9.3f} {corr:9.3f} "
            #    f"{E_form:12.3f} {E_form_doped:12.3f} "
            #    f"{bulk_energy:12.3f} {defect_energy:12.3f} {raw_defect_energy:12.3f}"
            #)
        print(f"VBM: {vbm:.3f} eV, Equil. Fermi level: {fermi:.3f} eV\n")
        # Formation energy CSV
        fe_df = combined_thermo.get_formation_energies(chempots=dmu, el_refs=ref_dict,skip_formatting=True)
        fe_df.drop(columns=["Path"], inplace=True)
        fe_df.reset_index(inplace=True)
        dumpfn(combined_thermo, os.path.join(output_dir, f"combined_{limit_name}_thermo.json"))

        fe_df = fe_df.rename(columns={
            "ΔEʳᵃʷ": r"$\Delta E_{\mathrm{raw}}$",
            "qE_VBM": r"$qE_{\mathrm{VBM}}$",
            "qE_F": r"$qE_F$",
            "Σμ_ref": r"$\sum_i n_i \mu_i^{\mathrm{ref}}$",
            "Σμ_formal": r"$\sum_i n_i \mu_i$",
            "E_corr": r"$E_{\mathrm{corr}}$",
            "Eᶠᵒʳᵐ": r"$E_{\mathrm{form}}$ (eV)",
            "Δ[E_corr]": r"$\Delta E_{\mathrm{corr}}$",
        })
        fe_df["Defect"] = fe_df["Defect"].apply(
            lambda d: format_defect_name(d,include_site_info_in_name=True, wout_charge=True)
        )
        fe_df.drop(columns=[r"$\Delta E_{\mathrm{raw}}$",r"$qE_{\mathrm{VBM}}$",r"$qE_F$",r"$\sum_i n_i \mu_i^{\mathrm{ref}}$",r"$\sum_i n_i \mu_i$",r"$E_{\mathrm{corr}}$",r"$\Delta E_{\mathrm{corr}}$"], 
                   inplace=True)
        fe_df.to_latex(
            os.path.join(output_dir, f"combined_{limit_name}_formation_energies.tex"), 
            escape=False,
            index=False,
            columns=["Defect","q",r"$E_{\mathrm{form}}$ (eV)"],
            longtable=True,
            caption=f"Combined Defect Formation Energies under {limit_name} conditions",
            float_format="%.3f",
        )
        fe_df.to_csv(os.path.join(output_dir, f"combined_{limit_name}_formation_energies.csv"), 
                    index=False,
                    columns=["Defect","q",r"$E_{\mathrm{form}}$ (eV)"]
        )
        print(f"Formation Energies:\n",
              f" {fe_df}")
        
        # Symmetry table
        symm_df = combined_thermo.get_symmetries_and_degeneracies(skip_formatting=True)
        symm_df.reset_index(inplace=True)        
        def format_transition(s):
            s = s.replace("ε(", "").replace(")", "")
            s = s.replace("*", r"^{*}")
            return rf"$\varepsilon({s})$"
        symm_df = symm_df.rename(columns={
            "Site_Symm": r"$G_{\mathrm{site}}$",
            "Defect_Symm": r"$G_{\mathrm{defect}}$",
            "g_Orient": r"$g_{\mathrm{orient}}$",
            "g_Spin": r"$g_{\mathrm{spin}}$",
            "g_Total": r"$g_{\mathrm{tot}}$",
        })
        symm_df["Defect"] = symm_df["Defect"].apply(
            lambda d: format_defect_name(d,include_site_info_in_name=True, wout_charge=True)
        )
        symm_df.drop(columns=[r"$G_{\mathrm{site}}$",r"$G_{\mathrm{defect}}$"], inplace=True)
        symm_df.to_latex(
            os.path.join(output_dir, f"combined_{limit_name}_symmetries.tex"),escape=False,
            index=False,
            columns=["Defect","q", r"$g_{\mathrm{orient}}$", r"$g_{\mathrm{spin}}$", r"$g_{\mathrm{tot}}$","Mult"],
            longtable=True,
            caption=f"Combined Defect Symmetries and Degeneracies under {limit_name} conditions",
            float_format="%.3f",
            )
        symm_df.to_csv(os.path.join(output_dir, f"combined_{limit_name}_symmetries.csv"),
                       index=False,
                        columns=["Defect","q", r"$g_{\mathrm{orient}}$", r"$g_{\mathrm{spin}}$", r"$g_{\mathrm{tot}}$","Mult"]
        )
        print(f"Symmetries and Degeneracies:\n",
              f"{symm_df}")
        # Transition levels table
        trans_level = combined_thermo.get_transition_levels()
        trans_level.reset_index(inplace=True)
        trans_level = trans_level.rename(columns={
            "eV from VBM": r"$\varepsilon(q/q') - E_{\mathrm{VBM}}$~(eV)",
            "In Band Gap?": r"In gap?",
        })
        trans_level["Charges"] = trans_level["Charges"].apply(format_transition)
        trans_level["Defect"] = trans_level["Defect"].apply(
            lambda d: format_defect_name(d,include_site_info_in_name=True, wout_charge=True)
        )
        trans_level.drop(columns = [ r"$\varepsilon(q/q') - E_{\mathrm{VBM}}$~(eV)",r"In gap?"], inplace=True)
        trans_level.to_latex(
            os.path.join(output_dir, f"combined_{limit_name}_transition_levels.tex"),
            escape=False,
            index=False,
            float_format="%.3f",
            columns=["Defect", "Charges"],
            longtable=True,
            caption=f"Combined Defect Transition Levels under {limit_name} conditions",
        )
        trans_level.to_csv(os.path.join(output_dir, f"combined_{limit_name}_transition_levels.csv"), 
                           index=False,
                           columns=["Defect", "Charges"])
        print(f"Transition Levels:\n", 
              f"{trans_level}")

        # Formation energy plot
        fe_path = os.path.join(output_dir, f"combined_{limit_name}_formation_energies.png")
        fe_cmap = build_fe_colormap(combined_thermo, defect_color)
        fig = combined_thermo.plot(all_entries='faded',
                                   chempot_table=True, 
                                   colormap=fe_cmap, 
                                   fermi_level= bulk_vr.eigenvalue_band_properties[0]/2,
                                   include_site_info=True)
        ax = fig.gca()
        ax.set_title(f"Combined Defect Formation Energies\nLimit: {limit_name}", y=1.2)
        ax.title.set_fontsize(12)
        fig.savefig(fe_path, dpi=300, bbox_inches="tight")
        
        plot_fs_concentrations(
            combined_thermo,
            temperatures,
            bulk_vr,
            output_dir,
            chempots,
            ref_dict
        )
        return combined_thermo.defect_entries

    except Exception as e:
        logger.error(f"Combined defect processing failed: {e}", exc_info=True)
        return 

def calculate_concentrations_fs(
        thermo: DefectThermodynamics,
        temperatures,
        bulk_dos,
        chempots: dict,
        output_dir: str,
        el_refs: Dict):
    if "limits_wrt_el_refs" in chempots:
        limit_name = list(chempots["limits_wrt_el_refs"].keys())[0]
        single_chempot = chempots["limits_wrt_el_refs"][limit_name]
    elif "Phases" in chempots:
        limit_name = chempots["Phases"]
        single_chempot = {el: chempots[el] for el in el_refs}
    else:
        limit_name = "User-Specified"
        single_chempot = {el: chempots[el] for el in el_refs}
    print(f"Using chemical potential limit: {limit_name}")
    print(f"Δμ (relative chempots): {single_chempot}")
    mu_abs = {el: float(single_chempot[el]) + float(el_refs[el]) for el in el_refs}
    # -----------------------------
    # Run Fermi solver
    # -----------------------------
    fs = FermiSolver(
        thermo,
        bulk_dos=bulk_dos,
        backend="doped",
        chempots=mu_abs,
        el_refs=el_refs
    )
    temp_df = fs.scan_temperature(temperature_range=temperatures)
    temp_df.to_csv(os.path.join(output_dir, f"{limit_name}_fermi_solver.csv"))
    return temp_df

def plot_fs_concentrations(
    thermo,
    temperatures,
    bulk_dos,
    output_dir,
    chempots,
    el_refs,
):
    """
    Generate plots using FermiSolver concentrations
    """
    # Determine limit name
    if "limits_wrt_el_refs" in chempots:
        limit_name = list(chempots["limits_wrt_el_refs"].keys())[0]
        dmu = chempots["limits_wrt_el_refs"][limit_name]
    elif "Phases" in chempots:
        limit_name = chempots["Phases"]
        dmu ={el: chempots[el] for el in el_refs}
    else:
        limit_name = "User-Specified"
        dmu ={el: chempots[el] for el in el_refs}

    # Get concentration data
    df_temp = calculate_concentrations_fs(
        thermo,
        temperatures,
        bulk_dos,
        dmu,
        output_dir,
        el_refs,
    )

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # ---- Plot defects (exclude electrons/holes) ----
    unique_defects = [
        d for d in df_temp.index.unique()
        if d not in ("Electrons (cm^-3)", "Holes (cm^-3)")
    ]


    for defect_name in unique_defects:
        defect_df = df_temp.loc[defect_name]

        ax.plot(
            defect_df["Temperature (K)"],
            defect_df["Concentration (cm^-3)"],
            label=format_defect_name(defect_name, include_site_info_in_name=True, wout_charge=True),
            color=defect_color.get(format_defect_name(defect_name, wout_charge=True)),
            lw=2.2,
        )

    # ---- Electrons & Holes ----
    ax.plot(df_temp["Temperature (K)"], df_temp["Electrons (cm^-3)"],
            label="Electrons", color=defect_color.get("Electrons", "C0"), lw=2.5)
    ax.plot(df_temp["Temperature (K)"], df_temp["Holes (cm^-3)"],
            label="Holes", color=defect_color.get("Holes", "C1"), lw=2.5)

    # Final formatting
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Concentration (cm$^{-3}$)")
    ax.grid(False, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    plt.title(f"Defect Concentrations via Fermi Solver\nLimit: {limit_name}", fontsize=14)

    # Save
    out_png = os.path.join(output_dir, f"concentrations_fermi_solver_{limit_name}.png")
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.show()

    print(f"Saved Fermi solver concentration plot → {out_png}")

CONFIG = {
    "material": "La2NiO4",
    "elements": ["La", "Ni", "O"],
    "defect_categories": ['Interstitial', 'Vacancy'],
    "defect_dir": ['Interstitial', 'Vacancy', 'Combined'],
    "e_above_hull": 0.09,
        "processes": 4,
}
        
def setup_directories(material, defect_dir):
    """Create directory structure and return paths"""
    base_dir = f"{material}_results2"
    paths = {
        "base": base_dir,
        "dielectric": os.path.join(base_dir, "dielectric"),
        "chempot": os.path.join(base_dir, "chemical_potentials"),
        "defects": {cat: os.path.join(base_dir, cat) for cat in defect_dir},
    }
    for path in paths.values():
        if isinstance(path, dict):
            for p in path.values():
                os.makedirs(p, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    return paths

def main():
    """Main workflow for defect analysis"""
    logger.info(f"Starting defect analysis for {CONFIG['material']}")
    paths = setup_directories(CONFIG['material'], CONFIG['defect_dir'])
    chempots = loadfn(os.path.join(paths['chempot'], "chemical_potentials.json"))
    chempots1 = loadfn(os.path.join(paths['chempot'], "chemical_potentials1.json"))
    chempots2 = loadfn(os.path.join(paths['chempot'], "chemical_potentials2.json"))
    chempots3 = loadfn(os.path.join(paths['chempot'], "chemical_potentials3.json"))
    T = np.arange(800, 1500, 100)
    el_refs = chempots['elemental_refs']

    chempots4 = {
                  'Phases': 'La2NiO4-La2O3-O2 (Songge)',
                  'La': -8.820154,
                  'Ni': 1.317339,
                  'O': 0.000000
                  }
    
    chempots5 = { 
                  'Phases': 'La2NiO4-LaNiO3-NiO (Songge)',
                  'La': -4.477836,
                  'Ni': 3.870668,
                  'O': -2.846086
                  }
    
    # Define critical file paths
    bulk_dir = os.path.join(f"{CONFIG['material']}_defects4", "U",'bulk', "U_7")
    bulk_vasprun_path = os.path.join(bulk_dir, "vasprun.xml")
    dielectric_path = os.path.join(f"{CONFIG['material']}_defects3", "vib","fd", "OUTCAR")
    dielectric_data = validate_dielectric_tensors(dielectric_path, paths['dielectric'])
    if dielectric_data is None:
        logger.error("Dielectric processing failed. Aborting analysis.")
        return
    print('Dielectric Tensor:', dielectric_data['total_dielectric_tensor'])

    # Load required bulk data
    try:
        bulk_vr = Vasprun(bulk_vasprun_path)
        print('Bulk Band gap:', bulk_vr.eigenvalue_band_properties[0])
        print('Bulk VBM:', bulk_vr.eigenvalue_band_properties[2])
        print('Bulk CBM:', bulk_vr.eigenvalue_band_properties[1])
        print('Bulk Fermi Level:', bulk_vr.efermi)
        logger.info(f"Loaded bulk vasprun from {bulk_vasprun_path}")
    except Exception as e:
        logger.error(f"Failed to load bulk vasprun: {str(e)}")
        return
    
    # Process defect data for each category
    combined_defect_entries = []
    for defect_type, defect_dir in zip(CONFIG["defect_categories"], CONFIG["defect_dir"]):
        out_dir = paths['defects'][defect_dir]
        defect_path = os.path.join(f"{CONFIG['material']}_defects2", defect_type)
        defect_dict, defect_entries = process_defect_category(
            defect_type,
            defect_path,
            out_dir,
            dielectric_data["total_dielectric_tensor"],
            bulk_vr,
            chempots4,
            temperatures=T,
            el_refs=el_refs
        )
        if defect_entries:
            combined_defect_entries.extend(defect_entries)
    
    # Process combined defects 
    if combined_defect_entries:
        out_dir = paths['defects'][CONFIG["defect_dir"][-1]]
        process_combined_defects(combined_defect_entries,
                                 out_dir,
                                 bulk_vr,
                                 chempots4,
                                 temperatures=T,
                                 el_refs=el_refs
                                )        

if __name__ == '__main__':
    main()
