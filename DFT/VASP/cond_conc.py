#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ast import mod
import os
import re
import json
import logging
from warnings import filterwarnings
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
import matplotlib
from sympy import numer
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm
import doped
from doped.thermodynamics import DefectThermodynamics, FermiSolver
from doped.analysis import DefectsParser
from pymatgen.io.vasp.outputs import Outcar, Waveder
from pymatgen.electronic_structure.boltztrap2 import (
    BztTransportProperties, Vasprun, BztInterpolator, BztPlotter, VasprunBSLoader
)
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from pymatgen.util.io_utils import micro_pyawk
from monty.serialization import dumpfn, loadfn
from doped.utils.plotting import format_defect_name

from scipy.constants import (
    k as k_B,
    elementary_charge as e,
    h,
    hbar,
    m_e,
    epsilon_0
)

# --------------------------------------------------------------------------------------
# Globals & plotting
# --------------------------------------------------------------------------------------
k_e   = k_B / e      # eV/K
h_e   = h / e        # eV*s
hbar_e = hbar / e    # eV*s
logger = logging.getLogger(__name__)

plt.rcdefaults()
plt.style.use(f"{doped.__path__[0]}/utils/doped.mplstyle")
plt.switch_backend('Agg')
plt.rcParams.update({'figure.max_open_warning': 100})
filterwarnings("ignore", category=UserWarning)

# Consistent colors for quick-looks
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
# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def create_xmgrace_file(data, columns, filename, include_header=True, separate_datasets=False):
    """Space-delimited data for xmgrace."""
    with open(filename, 'w') as f:
        if include_header:
            f.write("# " + " ".join(columns) + "\n")
        if separate_datasets:
            for col in columns[1:]:
                f.write(f"@ s{columns.index(col)} legend \"{col}\"\n")
                for _, row in data.iterrows():
                    formatted = [f"{row[columns[0]]:.6e}", f"{row[col]:.6e}"]
                    f.write(" ".join(formatted) + "\n")
                f.write("\n")
        else:
            for _, row in data.iterrows():
                out = []
                for col in columns:
                    val = row[col]
                    if pd.notna(val) and isinstance(val, (int, float, np.floating)):
                        out.append(f"{val:.6e}")
                    else:
                        out.append(str(val))
                f.write(" ".join(out) + "\n")
                
def calculate_elastic_tensor(outcar_dir: str, modulus_type=None) -> dict:
    """Working version with proper error handling."""
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    
    outcar_file = os.path.join(outcar_dir, "OUTCAR")
    outcar = Outcar(outcar_file)
    
    # Read elastic tensor
    if "elastic_tensor" not in outcar.data:
        outcar.read_elastic_tensor()
    
    if "elastic_tensor" not in outcar.data:
        raise ValueError("No elastic tensor in OUTCAR")
    
    # Get in kBar, convert to GPa
    tensor_kbar = np.array(outcar.data["elastic_tensor"])
    tensor_gpa = tensor_kbar * 0.1
    
    # Create ElasticTensor using from_voigt
    et = ElasticTensor.from_voigt(tensor_gpa)
    
    # Build results dictionary - include all standard elastic constants
    results = {
        "tensor_kbar": tensor_kbar,
        "tensor_gpa": tensor_gpa,
        # Add ALL Cij constants using a loop
    }
    
    # Add all Cij constants
    for i in range(1, 7):
        for j in range(1, 7):
            results[f"C{i}{j}"] = tensor_gpa[i-1, j-1]
    
    # Add bulk/shear moduli
    results.update({
        "K_vrh": et.k_vrh,
        "G_vrh": et.g_vrh,
        "K_voigt": et.k_voigt,
        "G_voigt": et.g_voigt,
        "K_reuss": et.k_reuss,
        "G_reuss": et.g_reuss,
    })
    
    
    # Handle modulus_type with error checking
    if modulus_type:
        output = {}
        for k in modulus_type:
            if k in results:
                output[k] = results[k]
            else:
                print(f"Warning: Key '{k}' not found in results. Available keys:")
                for key in sorted(results.keys()):
                    if 'C' in key or 'K' in key or 'G' in key:
                        print(f"  {key}")
                raise KeyError(f"'{k}' not in results. Available Cij keys: C11, C12, ..., C66")
        return output
    else:
        return results
    
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
    plt.show()

    print(f"Saved Fermi solver concentration plot → {out_png}")


def tau_ac_imp(T, E_eV, m_eff, E_F, C11, U, eps, Z_I, N_I):
    """
    Acoustic deformation-potential scattering time and Impurity scattering time.
    E_eV: (nE,nK) energy grid in eV
    m_eff: (nT,nD,3) effective masses in m_e
    Cij: (3,) elastic constants in GPa
    U: deformation potential in eV
    eps: total dielectric constant
    Z_I: impurity charge number
    N_I: impurity concentration in cm^-3
    """
    T = np.asarray(T, float)                         # (nT,)
    kT_J = (k_B * T).reshape(-1,1,1,1)               # (nT,1,1,1)

    C_vec = np.atleast_1d(np.asarray(C11, float))
    if C_vec.size == 1:
        C_vec = np.repeat(C_vec, 3)
    elif C_vec.size != 3:
        raise ValueError("C11 must be scalar or length-3 [C11,C22,C33].")
    C_pa = (C_vec * 1e9)[None,None,:,None]           # (1,1,3,1) Pa
    U_J = float(U) * e                               # scalar J
    m_eff = np.asarray(m_eff, float)                 # (nT,nD,3)
    m_kg  = (m_eff[:,:,:,None] * m_e)                # (nT,nD,3,1)
    E_eV = np.asarray(E_eV, float)                 # (nE,nK)
    E_mean = np.mean(E_eV, axis=1)                     # (nE,)
    E_J = (E_mean * e)[None,None,None,:]           # (1,1,1,nE)
    E_F = float(E_F) * e
    
    # Transport weight: -df/dE in Joules
    x = (E_J-E_F) / kT_J                                   # (nT,1,1,nE)
    exm = np.exp(-x)
    dfdE = exm / (kT_J * (1.0 + exm)**2)             # (nT,1,1,nE)

    # --- Build a strictly-positive energy mesh near the band edge ---
    # τ^{-1}(E) ∝ kT * (2 m)^{3/2} * U^2 * E^{1/2} / (π ħ^4 C)
    denom_ac = (np.pi * hbar**4 * C_pa)                 # (1,1,3,1)
    numer_ac = kT_J * (2.0*np.abs(m_kg))**1.5 * (U_J**2)
    tau_inv_ac = (numer_ac/denom_ac) * np.sqrt(E_J)         # (nT,nD,3,nE)
    
    # --- Impurity scattering time ---
    # τ^{-1}(E) ∝ N_i * Z^2 * e^4 * (log(1+1/x)-1/(1+x))/(sqrt(2m*)*4πε^2)
    
    # FIX: Reshape N_I to match broadcasting with x
    N_I = np.asarray(N_I, dtype=float)
    if N_I.ndim == 0:  # scalar
        N_I = N_I.reshape(1,1,1,1)
    elif N_I.ndim == 1:  # 1D array
        # Reshape to (nT,1,1,1) to broadcast with x which is (nT,1,1,nE)
        N_I = N_I.reshape(-1,1,1,1)
    
    # Convert eps to scalar if it's a tensor
    if hasattr(eps, '__len__'):
        if isinstance(eps, np.ndarray) and eps.ndim == 2:
            eps_scalar = np.mean(np.diag(eps))
        else:
            eps_scalar = np.mean(eps)
    else:
        eps_scalar = float(eps)
    
    # Calculate with proper broadcasting
    denom_imp = (np.sqrt(2.0 * np.abs(m_kg)) * 4 * np.pi * (eps_scalar**2))
    numer_imp = N_I * (Z_I**2) * e**4 * np.log(1+1/x)  # Now shapes match
    tau_inv_imp = numer_imp / denom_imp                # (nT,nD,3,1)
    
    tau_inv_E = tau_inv_ac + tau_inv_imp               
    xgrid = E_J[0,0,0,:]                             # 1(nE,)
    num = np.trapezoid(tau_inv_E * dfdE, x=xgrid, axis=3)   # (nT,nD,3)
    den = np.trapezoid(dfdE,           x=xgrid, axis=3)     # (nT,1,1)

    # Guard against 0/0 → NaN
    tau_inv_avg = num / den                     # (nT,nD,3)
    tau_inv_avg = np.where(np.isfinite(tau_inv_avg), tau_inv_avg, 1e-12)
    
    return 1.0 / tau_inv_avg

def setup_directories(material, defect_dir):
    """Create directory structure and return paths."""
    base_dir = f"{material}_results"
    paths = {
        "base": base_dir,
        "dielectric": os.path.join(base_dir, "dielectric"),
        "phase_diagrams": os.path.join(base_dir, "phase_diagrams"),
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


# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
CONFIG = {
    "material": "La2NiO4",
    "elements": ["La", "Ni", "O"],
    "defect_categories": ['Interstitial', 'Vacancy'],
    "defect_dirs": ['Interstitial', 'Vacancy', 'Combined'],
    "e_above_hull": 0.09,
    "processes": 1,
}

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    paths = setup_directories(CONFIG["material"], CONFIG["defect_dirs"])
    base_dir = paths["base"]
    output_dir = paths["defects"][CONFIG["defect_dirs"][-1]]
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------------------------
    # Band structure & BoltzTraP2
    # ----------------------------------------------------------------------------------
    vrun = Vasprun("La2NiO4_bulk/BS/vasprun.xml")
    st   = vrun.initial_structure
    nele = vrun.parameters['NELECT']

    p_doping = np.array([0.375e19, -0.25e19, 0.25e19, -0.25e19])  # cm^-3 signs follow BZT convention
    n_doping = np.array([-0.375e19, 0.25e19, -0.25e19, 0.25e19])  # unused here, but kept for completeness
    temp_r   = np.arange(800, 1500, 100)

    data = VasprunBSLoader(vrun, st, nele)
    
    bzt_int = BztInterpolator(
        data, lpfac=10, 
        energy_range=2, 
        curvature=True,
        save_bztInterp=True, save_bands=False, fname='LNO_BZT_interp.json.gz'
    )
    bzt_trans = BztTransportProperties(
        bzt_int, temp_r=temp_r, doping=p_doping,
        save_bztTranspProps=True, fname='bztTranspProps.json.gz'
    )
    bzt_trans.compute_properties_doping(doping=p_doping, temp_r=temp_r)

    # ----------------------------------------------------------------------------------
    # Experimental data (optional)
    # ----------------------------------------------------------------------------------
    def maybe_load_xy(path):
        if os.path.exists(path):
            arr = np.loadtxt(path)
            return np.array(arr[:,0]), np.array(arr[:,1])
        return np.array([]), np.array([])

    T_exp,  sigma_exp  = maybe_load_xy("T1_LNO1.dat")
    T_exp1, sigma_exp1 = maybe_load_xy("T2_LNO2.dat")
    T_exp2, sigma_exp2 = maybe_load_xy("T3_LNO3.dat")
    T_exp3, sigma_exp3 = maybe_load_xy("T4_LNO4.dat")

    # ----------------------------------------------------------------------------------
    # Defect probabilities (use columns, not hard indices)
    # ----------------------------------------------------------------------------------
    limit_name = "La2NiO4-La2O3-O2"
    df_p = pd.read_csv(os.path.join(base_dir, f"Combined/defect_type_probabilities_{limit_name}.csv"))
    Pcols = [c for c in df_p.columns if c != 'T (K)']
    Pmap  = {name: df_p[name].to_numpy() for name in Pcols}

    def sum_cols(regex):
        r = re.compile(regex)
        arrs = [v for k, v in Pmap.items() if r.fullmatch(k)]
        return np.sum(np.stack(arrs, axis=0), axis=0) if arrs else np.zeros(len(df_p))

    P_Oi  = sum_cols(r"O_i_.*")      # all oxygen interstitials
    P_VLa = sum_cols(r"v_La.*")
    P_VNi = sum_cols(r"v_Ni.*")
    P_VO  = sum_cols(r"v_O_.*")

    # Doping dimension (nD) is 4 in our BZT run; we group to 4 classes here
    P_def = np.stack((P_VLa, P_VNi, P_VO, P_Oi), axis=1)  # (nT, 4)
    print(f"P_def shape: {P_def.shape}, head row: {P_def[0, :]}")

    # ----------------------------------------------------------------------------------
    # Shapes from BZT
    # ----------------------------------------------------------------------------------
    cond_over_tau_p = bzt_trans.Conductivity_doping['p']  # (nT, nD, nR, nC), SI (S/m per second)
    print(f"cond_over_tau_p shape: {cond_over_tau_p.shape}")
    print(bzt_trans.CRTA)

    eff_mass_p = bzt_trans.Effective_mass_doping["p"]    # (nT, nD, 3, 3)
    print(f"Effective_mass_doping[p] shape: {eff_mass_p.shape}")

    # Diagonal effective masses (nT, nD, 3)
    def _diag_m(m):  # (nT,nD,3,3) -> (nT,nD,3)
        return np.stack((m[:,:,0,0], m[:,:,1,1], m[:,:,2,2]), axis=2)
    m_eff_p_diag = _diag_m(eff_mass_p)
    print(f"m_eff_p_diag shape: {m_eff_p_diag.shape}")

    # Energy grid (nE, nK)
    E_grid = np.asarray(bzt_int.data.ebands, float)
    print(f"E_grid shape: {E_grid.shape}")
    E_F = vrun.efermi  # eV
    chempots = loadfn(os.path.join(paths['chempot'], "chemical_potentials.json"))

    # ----------------------------------------------------------------------------------
    # Elastic constants & τ_ac
    # ----------------------------------------------------------------------------------
    bulk_outcar = os.path.join(f"{CONFIG['material']}_bulk", "phonopy", "dfpt")
    C11 = calculate_elastic_tensor(bulk_outcar, ['C11'])['C11']
    C22 = calculate_elastic_tensor(bulk_outcar, ['C22'])['C22']
    C33 = calculate_elastic_tensor(bulk_outcar, ['C33'])['C33']
    K_vrh = calculate_elastic_tensor(bulk_outcar, ['K_vrh'])['K_vrh']
    C_diag = np.array([C11, C22, C33], float)
    eps_0 = validate_dielectric_tensors(
        os.path.join(bulk_outcar, "OUTCAR"),
        output_dir
    )["total_dielectric_tensor"]
    eps = np.array(eps_0, float)
    print(f"C11, C22, C33 (GPa) = {C11:.3f}, {C22:.3f}, {C33:.3f}")
    print(f"Bulk modulus K_vrh (GPa): {K_vrh:.3f}")
    print(f"Dielectric tensor:\n{eps}")

    # User-provided deformation potentials (eV) – sign matters
    U_p = -0.369  # holes
    # τ_ac for holes
    el_refs = chempots['elemental_refs']

   # Get your concentration data
    thermo = DefectThermodynamics.from_json(os.path.join(output_dir,"combined_La2NiO4-La2O3-O2_thermo.json"))
    bulk_dos = Vasprun("La2NiO4_bulk/PBE_DOS/DOS_7/vasprun.xml")
    conc = calculate_concentrations_fs(
        thermo=thermo,
        bulk_dos=bulk_dos,
        chempots=chempots,
        temperatures=temp_r,
        output_dir=output_dir,
        el_refs=el_refs
    )
    print(f"Concentration DataFrame shape: {conc.shape}")
    plot_fs_concentrations(
        thermo=thermo,
        temperatures=temp_r,
        bulk_dos=bulk_dos,
        output_dir=output_dir,
        chempots=chempots,
        el_refs=el_refs,
    )
    # Get unique defects
    unique_defects = [
        d for d in conc.index.unique()
        if d not in ("Electrons (cm^-3)", "Holes (cm^-3)")
    ]
    
    # Simple mapping dictionary
    defect_name_map = {
        'V_La': ['v_La'],
        'V_Ni': ['v_Ni'],
        'V_O': ['v_O_C4v', 'v_O_D2h', 'v_O'],  # All oxygen vacancy types
        'O_i': ['O_i_D2d', 'O_i_Cs', 'O_i_C2v', 'O_i']  # All oxygen interstitial types
    }
    
    # Initialize for 4 defects
    tau_p = np.zeros((len(temp_r), 4, 3))
    
    # Loop through 4 defect types
    for i, defect_type in enumerate(['V_La', 'V_Ni', 'V_O', 'O_i']):
        # Find the right concentration data
        possible_names = defect_name_map[defect_type]
        N_i = None
        
        for name in possible_names:
            if name in conc.index:
                defect_data = conc.loc[name]
                N_i = defect_data['Concentration (cm^-3)'].values
                print(f"Using {name} for {defect_type}, N_i = {N_i} cm⁻³")
                break
            
        if N_i is None:
            print(f"Warning: No data found for {defect_type}")
            N_i = np.ones(len(temp_r)) * 1e17  # Default
        
        # Set charge (adjust based on your defect charges)
        if defect_type == 'V_La':
            Z_i = 3
        else:
            Z_i = 2
        
        # Calculate tau
        tau_single = tau_ac_imp(
            temp_r, E_grid, m_eff_p_diag[:, i:i+1, :],
            E_F=E_F, C11=C_diag, U=U_p, eps=eps,
            Z_I=Z_i, N_I=N_i
        )
        
        tau_p[:, i, :] = tau_single[:, 0, :]
    
    print(f"tau_p shape: {tau_p.shape}")  # Should be (7, 4, 3)

    # ----------------------------------------------------------------------------------
    # Calculate actual conductivity σ = (σ/τ) * τ
    # ----------------------------------------------------------------------------------
    # Average tau over directions for each defect
    tau_scalar = tau_p.mean(axis=2)  # Shape: (7, 4)
    tau_expanded = tau_scalar[:, :, np.newaxis, np.newaxis]  # Shape: (7, 4, 1, 1)

    # Calculate actual conductivity
    sigma_si = cond_over_tau_p * tau_expanded  # σ = (σ/τ) * τ, shape: (7, 4, 3, 3)

    # Now continue with your conductivity calculations but use sigma_si instead of cond_over_tau_p
    cond_La = sigma_si[:,0,:,:]  # Use sigma_si, not cond_over_tau_p
    cond_Ni = sigma_si[:,1,:,:]
    cond_VO = sigma_si[:,2,:,:]
    cond_Oi = sigma_si[:,3,:,:]

    tot_cond = cond_La + cond_Ni + cond_VO + cond_Oi

    # Convert to scalar conductivity (S/cm)
    cond_La_trace = cond_La.trace(axis1=1, axis2=2) / 3.0 / 100.0
    cond_Ni_trace = cond_Ni.trace(axis1=1, axis2=2) / 3.0 / 100.0
    cond_VO_trace = cond_VO.trace(axis1=1, axis2=2) / 3.0 / 100.0
    cond_Oi_trace = cond_Oi.trace(axis1=1, axis2=2) / 3.0 / 100.0
    tot_cond_trace = tot_cond.trace(axis1=1, axis2=2) / 3.0 / 100.0

    print("Actual conductivity with scattering times (S/cm):")
    print("  V_La: ", cond_La_trace)
    print("  V_Ni: ", cond_Ni_trace)
    print("  V_O: ", cond_VO_trace)
    print("  O_i: ", cond_Oi_trace)
    print("  Total: ", tot_cond_trace)

    # Also calculate probability-weighted conductivity if you want
    if 'P_def' in locals():
        # Multiply by defect probabilities
        sigma_defect_prob = sigma_si * P_def[:, :, None, None]  # (nT, nD, nR, nC)
        cond_prob_La = sigma_defect_prob[:,0,:,:].trace(axis1=1, axis2=2) / 3.0 / 100.0
        cond_prob_Ni = sigma_defect_prob[:,1,:,:].trace(axis1=1, axis2=2) / 3.0 / 100.0
        cond_prob_VO = sigma_defect_prob[:,2,:,:].trace(axis1=1, axis2=2) / 3.0 / 100.0
        cond_prob_Oi = sigma_defect_prob[:,3,:,:].trace(axis1=1, axis2=2) / 3.0 / 100.0
        cond_prob_total = cond_prob_La + cond_prob_Ni + cond_prob_VO + cond_prob_Oi

        print("\nProbability-weighted conductivity (S/cm):")
        print("  V_La × P: ", cond_prob_La)
        print("  V_Ni × P: ", cond_prob_Ni)
        print("  V_O × P: ", cond_prob_VO)
        print("  O_i × P: ", cond_prob_Oi)
        print("  Total × P: ", cond_prob_total)
    
    print(f"\nDEBUG - Array shapes before calculation:")
    print(f"  sigma_si shape: {sigma_si.shape}")  # Should be (nT, nD, 3, 3)
    print(f"  P_def shape: {P_def.shape}")        # Should be (nT, nD)

    # Check temperature dimensions match
    if sigma_si.shape[0] != P_def.shape[0]:
        print(f"ERROR: Temperature mismatch! sigma_si has {sigma_si.shape[0]} temps, P_def has {P_def.shape[0]} temps")
        print("Aligning to minimum...")
        n_temp = min(sigma_si.shape[0], P_def.shape[0])
        sigma_si = sigma_si[:n_temp, :, :, :]
        P_def = P_def[:n_temp, :]
        print(f"  New shapes: sigma_si={sigma_si.shape}, P_def={P_def.shape}")

    # Check defect dimensions match  
    if sigma_si.shape[1] != P_def.shape[1]:
        print(f"ERROR: Defect count mismatch! sigma_si has {sigma_si.shape[1]} defects, P_def has {P_def.shape[1]} defects")
        print("Aligning to minimum...")
        n_defects = min(sigma_si.shape[1], P_def.shape[1])
        sigma_si = sigma_si[:, :n_defects, :, :]
        P_def = P_def[:, :n_defects]
        print(f"  New shapes: sigma_si={sigma_si.shape}, P_def={P_def.shape}")
    
    # Check probability broadcast (nT matches temp points in df_p)
    assert P_def.shape[0] == sigma_si.shape[0], "Temperature grid mismatch for P_def and sigma."
    assert P_def.shape[1] == sigma_si.shape[1], "P_def second dim must equal # dopings (nD)."

    sigma_defect_prob = sigma_si * P_def[:, :, None, None]         # (nT, nD, nR, nC)
    
    trace_def_prob    = sigma_defect_prob.trace(axis1=2, axis2=3) / 3.0  # (nT, nD)
    conductivity_prob_p = trace_def_prob / 100.0                   # S/cm
    conductivity_sum_p  = conductivity_prob_p.sum(axis=1)          # (nT,) S/cm

    print("conductivity_prob_p shape:", conductivity_prob_p.shape)
    print("conductivity_sum_p shape:",  conductivity_sum_p.shape)

    # ----------------------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(10,6))
    labels = ['V_La', 'V_Ni', 'V_O', 'O_i']
    colors = ['red', 'blue', 'green', 'orange']
    #for d, (lab, col) in enumerate(zip(labels, colors)):
    #    plt.plot(temp_r, conductivity_prob_p[:, d], label=lab, lw=3, color=col)
    #plt.plot(temp_r, conductivity_sum_p, label='Total', lw=4, color='black', linestyLa2NiO4_results/Combined3/prob_type.csvle='--')
    plt.plot(temp_r, cond_La_trace, label='V_La', lw=3, color='red')
    plt.plot(temp_r, cond_Ni_trace, label='V_Ni', lw=3, color='blue')
    plt.plot(temp_r, cond_VO_trace,  label='V_O',  lw=3, color='green')
    plt.plot(temp_r, cond_Oi_trace, label='O_i', lw=3, color='orange')
    plt.plot(temp_r, tot_cond_trace, label='Total', lw=4, color='black', linestyle='--')
    
    if T_exp.size:
        plt.plot(T_exp,  sigma_exp,  'o', label='Exp. T1', color='red',   markersize=6)
    if T_exp1.size:
        plt.plot(T_exp1, sigma_exp1, 'o', label='Exp. T2', color='blue',  markersize=6)
    if T_exp2.size:
        plt.plot(T_exp2, sigma_exp2, 'o', label='Exp. T3', color='green', markersize=6)
    if T_exp3.size:
        plt.plot(T_exp3, sigma_exp3, 'o', label='Exp. T4', color='orange', markersize=6)

    plt.yscale('log')
    plt.xlabel("Temperature (K)", fontsize=14)
    plt.ylabel(r"$\sigma \times P$ (S/cm)", fontsize=14)
    plt.title("La$_2$NiO$_4$: Defect Conductivity × Probability", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('LNO_Conductivity_Prob.png', dpi=300)
    plt.show()
    print("Saved figure: LNO_Conductivity_Prob.png")

if __name__ == "__main__":
    main()
