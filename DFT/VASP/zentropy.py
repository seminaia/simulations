#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zentropy_clean.py – Clean, optimized La₂NiO₄ defect thermodynamics
Generates:
 • Formation energies
 • Defect-type probabilities
 • Charge-state probabilities (all defects, automatic grid)
 • Concentrations
 • Carrier concentrations
"""

import os
import re
import json
import logging
from warnings import filterwarnings
import doped
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from scipy.constants import k as k_B, elementary_charge as e
from matplotlib.colors import to_hex
import matplotlib.cm as cm
from monty.serialization import loadfn, dumpfn
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from doped.thermodynamics import DefectThermodynamics, FermiSolver
from doped.generation import get_defect_name_from_entry
from doped.utils.plotting import format_defect_name
from PyFunc.Trap import f
from matplotlib.colors import ListedColormap

def build_fe_colormap(thermo, defect_color):
    colors = []
    for name, entry in thermo.defect_entries.items():
        base = name.split("_")[0]
        colors.append(defect_color.get(base, None))  
    return ListedColormap(colors)

def base_name_without_charge(full_name: str) -> str:
    return re.sub(r"_[+-]?\d+$", "", str(full_name))

# ============================================================================
# 0. GLOBAL SETTINGS
# ============================================================================
limit_name = "La2NiO4-La2O3-O2"
OUT_DIR = "La2NiO4_results/Combined"
THERMO_JSON = os.path.join(OUT_DIR, f"combined_{limit_name}_thermo.json")
BULK_VASPRUN = "La2NiO4_bulk/PBE_DOS/DOS_8/vasprun.xml"
k_eV = k_B / e 

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

def calculate_concentrations_fs(thermo:DefectThermodynamics,
                             temperatures,
                             bulk_dos,
                             chempots:dict,
                             output_dir:str):
    fs = FermiSolver(thermo, bulk_dos=bulk_dos,chempots=chempots ,backend= "doped")
    temp_df = fs.scan_temperature(temperature_range=temperatures,chempots=chempots)
    temp_df.to_csv(os.path.join(output_dir, "fermi_solver_temperatures.csv"))
    return temp_df 

def plot_fs_concentrations(
    thermo,
    temperatures,
    bulk_dos,
    output_dir,
    chempots,
):
    """
    Generate plots using FermiSolver concentrations
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    df_temp = calculate_concentrations_fs(thermo,
                                            temperatures,
                                            bulk_dos,
                                            chempots,
                                            output_dir) 
    unique_defects = df_temp.index.unique()
    for defects in unique_defects:
        defect_df = df_temp.loc[defects]
        ax.plot(defect_df["Temperature (K)"], defect_df['Concentration (cm^-3)'], label=format_defect_name(defects, include_site_info_in_name=True, wout_charge=True),
                color=defect_color.get(base_name_without_charge(defects), None), lw=2.2)
    ax.plot(df_temp["Temperature (K)"], df_temp['Electrons (cm^-3)'], label="Electrons", color=defect_color.get("Electrons", "C0"), lw=2.5)
    ax.plot(df_temp["Temperature (K)"], df_temp['Holes (cm^-3)'], label="Holes", color=defect_color.get("Holes", "C1"), lw=2.5)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Defect conc. (cm$^{-3}$)")
    ax.grid(False, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    plt.title("Defect concentrations Fermi Solver", fontsize=14, y=1.02)
    plt.savefig(os.path.join(output_dir, "concentrations_fermi_solver.png"), dpi=400, bbox_inches="tight")
    plt.show()
    
# ============================================================================
# 8. PLOTS SETTINGS
# ============================================================================
def outside_legend(ax):
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.,
        fontsize=8
    )

def no_grid(ax):
    ax.grid(False)

no_grid(plt.gca())
outside_legend(plt.gca())

# ============================================================================
# 1. LOAD THERMODYNAMICS
# ============================================================================
print("Loading thermodynamics...")
thermo = loadfn(THERMO_JSON)
temp_r = np.arange(800, 1500, 100)
beta = 1.0 / (k_eV * temp_r)

bulk_vr = Vasprun(BULK_VASPRUN)

# ============================================================================
# 2. NORMALIZATION
# ============================================================================
def normalize_base(name: str) -> str:
    base = re.sub(r'_[+-]?\d+$', '', name)
    base = re.sub(r'_site\d+$', '', base)
    base = re.sub(r'_\d+$', '', base)
    return base

# ============================================================================
# 3. COLLECT DEFECT ENTRIES
# ============================================================================
print("Collecting defect entries...")
defect_data = defaultdict(list)

for entry in thermo.defect_entries.values():
    base = normalize_base(entry.name)
    g_spin = float(entry.degeneracy_factors.get("spin degeneracy", 1))
    g_orient = float(entry.degeneracy_factors.get("orientational degeneracy", 1))
    N_site = float(entry.bulk_site_concentration)
    mult = float(entry.defect.multiplicity)
    weight = g_spin * g_orient * N_site

    E_form = float(entry.formation_energy(
        chempots=thermo.chempots,
        fermi_level=0.0
    ))

    defect_data[base].append({
        "name": entry.name,
        "q": entry.charge_state,
        "E": E_form,
        "w": weight,
    })

base_names = sorted(defect_data.keys())
print("Defect families:", base_names)

# color maps
charge_cmap = matplotlib.colormaps["Set3"]

# ============================================================================
# 4. PARTITION FUNCTIONS
# ============================================================================
Z_total = np.zeros_like(temp_r, dtype=float)
Z_base = {b: np.zeros_like(temp_r,dtype=float) for b in base_names}
Z_q_base = {b: defaultdict(lambda: np.zeros_like(temp_r,dtype=float)) for b in base_names}

for b in base_names:
    for d in defect_data[b]:
        E = d["E"]
        q = d["q"]
        w = d["w"]
        boltz = np.exp(-beta * E)
        Z_base[b] += w * boltz
        Z_q_base[b][q] += w * boltz
for b in base_names:
    Z_total += Z_base[b]
# ============================================================================
# 5. PROBABILITIES 
# ============================================================================
temp_r = np.asarray(temp_r, dtype=np.float64)   # your temperature array

P_type = {b: Z_base[b] / Z_total for b in base_names}       
# ============================================================================
# SAVE P_type DATA TO CSV
# ============================================================================
print("Saving defect-type probabilities to CSV...")

# Create DataFrame
P_type_df = pd.DataFrame({'Temperature_K': temp_r})

# Add each defect probability as a column
for b in base_names:
    P_type_df[b] = P_type[b]

# Save to CSV
csv_path = os.path.join(OUT_DIR, f"defect_type_probabilities_{limit_name}.csv")
P_type_df.to_csv(csv_path, index=False)
print(f" Saved defect-type probabilities to {csv_path}")
print(f"  Shape: {P_type_df.shape}, Columns: {list(P_type_df.columns)}")

# Optional: Also save in long format (tidy data)
long_data = []
for i, temp in enumerate(temp_r):
    for b in base_names:
        long_data.append({
            'Temperature_K': temp,
            'Defect_Type': b,
            'Probability': P_type[b][i]
        })

P_type_long_df = pd.DataFrame(long_data)
csv_long_path = os.path.join(OUT_DIR, "defect_type_probabilities_long.csv")
P_type_long_df.to_csv(csv_long_path, index=False)
print(f"Saved long format to {csv_long_path}")
                         
P_charge_cond = {
    b: {q: Z_q_base[b][q] / Z_base[b] for q in Z_q_base[b]}
        for b in base_names
    }                                                                                    

P_joint = {
    b: {q: P_type[b] * P_charge_cond[b][q] for q in P_charge_cond[b]}
    for b in base_names
}                                                                                    

all_q = sorted({q for b in base_names for q in P_charge_cond[b]})
charge_color = {q: charge_cmap(i / max(len(all_q)-1, 1)) for i, q in enumerate(all_q)}

# Pre-allocate P_q as proper numpy arrays of floats (instead of dict of zeros_like that inherited int dtype)
P_q = {q: np.zeros_like(temp_r, dtype=np.float64) for q in all_q}

# Accumulate total probability for each charge state
for b in base_names:
    for q in P_joint[b]:
        P_q[q] += P_joint[b][q]          # now both sides are float64 → no error

# Conditional probability P(type | q)
P_type_given_q = {}
for q in all_q:
    P_q_safe = np.maximum(P_q[q], 1e-30)           # avoid div-by-zero
    P_type_given_q[q] = {}
    for b in base_names:
        if q in P_joint[b]:
            P_type_given_q[q][b] = P_joint[b][q] / P_q_safe
        else:
            P_type_given_q[q][b] = np.zeros_like(temp_r)
# ============================================================================
# 6. FERMI SOLVER CONCENTRATIONS
# ============================================================================
print("Computing Fermi solver concentrations...")
fs = calculate_concentrations_fs(
    thermo,
    temperatures=temp_r,
    bulk_dos=bulk_vr,
    chempots=thermo.chempots,
    output_dir=OUT_DIR,
)
print(f"Fermi Solver: {fs.head()}")

plot_fs_concentrations(
    thermo,
    temperatures=temp_r,
    bulk_dos=bulk_vr,
    output_dir=OUT_DIR,
    chempots=thermo.chempots)

# ============================================================================
# 9. PLOT 1 — DEFECT-TYPE PROBABILITIES
# ============================================================================
plt.figure(figsize=(8, 6))
for b in base_names:
    plt.plot(temp_r, P_type[b], lw=2, color=defect_color[b], label=b)

no_grid(plt.gca())
outside_legend(plt.gca())

plt.xlabel("Temperature (K)")
plt.ylabel("P(type)")
plt.title("Defect-Type Probabilities")
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.savefig(os.path.join(OUT_DIR, "prob_type.png"), dpi=300)
plt.close()

# ============================================================================
# 11. CHARGE-STATE PROBABILITIES 
# ============================================================================
n_def = len(base_names)
n_cols = 3
n_rows = int(np.ceil(n_def / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
axes = axes.ravel()

for idx, b in enumerate(base_names):
    ax = axes[idx]
    for q in sorted(P_charge_cond[b]):
        ax.plot(temp_r, P_charge_cond[b][q],
                lw=2, color=charge_color[q], label=f"q={q:+d}")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(f"P(q| {b})")
    ax.set_title(b)
    no_grid(ax)
    outside_legend(ax)

# empty subplots
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Charge-State Probabilities for All Defects", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.82, 0.95])
plt.savefig(os.path.join(OUT_DIR, "prob_charge_all.png"), dpi=300)
