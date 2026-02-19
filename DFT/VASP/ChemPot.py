from enum import unique
from math import e
from operator import mul
import os
import logging
from turtle import color
from nbconvert import export
import numpy as np
import pandas as pd
from pandas import DataFrame
import doped
from monty.serialization import loadfn
import json
from doped.utils.plotting import format_defect_name
from monty.serialization import dumpfn, loadfn
from doped.analysis import DefectsParser
from doped.chemical_potentials import CompetingPhases, CompetingPhasesAnalyzer
from doped.thermodynamics import DefectThermodynamics
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, Element
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp import Vasprun, Outcar
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core import Composition, Structure
import matplotlib.pyplot as plt
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from sympy import false
import plotly.graph_objects as go
from pathlib import Path


defect_color = {
    "Electrons": "C0", "Holes": "C1", "v_Ni": "C2",
    "v_O_D2h": "C3", "O_i_C2v": "C4", "O_i_Cs": "C5",
    "O_i_D4h": "C6", "O_i_D2d": "C7", "v_La": "C8", "v_O_C4v": "C9"
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

# --- Config ---
CONFIG = {
    "material": "La2NiO4",
    "elements": [Element("La"), Element("Ni"), Element("O")],
    "defect_categories": ['Interstitial', 'Vacancy'],
    "e_above_hull": 0.5,
    "processes": 1,
}

plt.switch_backend('Agg')
plt.rcParams.update({'figure.max_open_warning': 100})

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
                logger.info("Processing COMBINED defect set")

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
        
def validate_dielectric_tensors(outcar_path, output_dir):
    """
    Processes dielectric tensors from OUTCAR and saves to JSON
    Parameters:
    outcar_path (str): Full path to OUTCAR file
    output_dir (str): Directory to save JSON output
    Returns:
    np.array: Total dielectric tensor
    """
    try:
        # Extract dielectric tensors from OUTCAR
        logger.info(f"Extracting dielectric tensors from OUTCAR: {outcar_path}")
        outcar = OutcarParser(outcar_path)
        
        # Get dielectric components
        static = np.array(outcar.dielectric_tensor)
        ionic = np.array(outcar.dielectric_ionic_tensor)
        total = static + ionic
        
        # Create output data
        dielectric_data = {
            "static_dielectric_tensor": static.tolist(),
            "ionic_dielectric_tensor": ionic.tolist(),
            "total_dielectric_tensor": total.tolist()
        }
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "total_dielectric.json")
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(dielectric_data, f, indent=4)
        logger.info(f"Saved complete dielectric data to: {output_path}")
        
        # Log results
        logger.info(f"Staic Dielectric Tensor:\n{static}")
        logger.info(f"Ionic Dielectric Tensor:\n{ionic}")
        logger.info(f"Total Dielectric Tensor:\n{total}")
        
        return total
        
    except Exception as e:
        logger.critical(f"Dielectric processing failed: {str(e)}", exc_info=True)
        raise RuntimeError("Dielectric tensor validation failed") from e
    
def plot_mu_binary(
    elements,
    cpd: ChemicalPotentialDiagram,
    mu_floor=-10.0,
    mu_third_fixed=0.0,
    out_png=None,
    tol=1e-8,
):
    """
    Binary chemical-potential diagram from CPD hyperplanes.
    elements = ["La","Ni"] or ["La","O"] or ["Ni","O"]
    cpd      = ChemicalPotentialDiagram(...)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    assert len(elements) == 2
    E1, E2 = elements

    # --- CPD elements ---
    elems = [str(e) for e in cpd.elements]
    idx1 = elems.index(E1)
    idx2 = elems.index(E2)
    idx3 = [i for i in range(3) if i not in (idx1, idx2)][0]
    E3 = elems[idx3]

    # --- Hyperplanes (N,4) and entries list ---
    hyperplanes, entries = cpd._get_hyperplanes_and_entries()

    # ------------------------------------------------------------------
    # Compute binary stability region (intersection of half-spaces)
    # ------------------------------------------------------------------
    A, b = [], []

    for hp in hyperplanes:
        a1, a2, a3, c = hp[idx1], hp[idx2], hp[idx3], hp[3]
        A.append([a1, a2])
        b.append(-(c + a3 * mu_third_fixed))

    # bounding walls
    A += [[1, 0], [0, 1], [-1, 0], [0, -1]]
    b += [0, 0, -mu_floor, -mu_floor]
    A, b = map(np.array, (A, b))

    # intersection vertices
    verts = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            M = np.vstack([A[i], A[j]])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            mu = np.linalg.solve(M, np.array([b[i], b[j]]))
            if np.all(A @ mu <= b + tol):
                verts.append(mu)
    verts = np.unique(np.round(verts, 6), axis=0) if len(verts) else np.zeros((0, 2))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    x_range = np.linspace(mu_floor, 0, 1500)

    ALLOWED = {"La2NiO4", "LaNiO3", "La2O3", "NiO"}   

    for hp, entry in zip(hyperplanes, entries):
        a1, a2, a3, c = hp[idx1], hp[idx2], hp[idx3], hp[3]
        formula = entry.composition.reduced_formula

        color, lw, alpha, z = ("red",4,1.0,50) if formula=="La2NiO4" else ("#666666",1.6,0.75,5)

        if abs(a1)<tol and abs(a2)<tol:                       # skip degenerate
            continue

        # ---------- vertical line ----------
        if abs(a2)<tol:
            mu1 = -(c + a3*mu_third_fixed)/a1
            if mu_floor <= mu1 <= 0:
                ax.axvline(mu1, color=color, lw=lw, alpha=alpha, zorder=z)
                if formula in ALLOWED:
                    ax.annotate(
                        formula,
                        xy=(mu1, 0.5*(mu_floor + 0)),        # middle y
                        ha='center', va='center',
                        fontsize=9, color=color,
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none')
                    )
            continue

        # ---------- generic line ----------
        mu2 = -(c + a1*x_range + a3*mu_third_fixed)/a2
        mask = (mu2 >= mu_floor) & (mu2 <= 0)
        if not np.any(mask):
            continue
        x_vis, y_vis = x_range[mask], mu2[mask]
        ax.plot(x_vis, y_vis, color=color, lw=lw, alpha=alpha, zorder=z)

        if formula in ALLOWED:
            mid_idx = len(x_vis)//2
            ax.annotate(
                formula,
                xy=(x_vis[mid_idx], y_vis[mid_idx]),
                ha='center', va='center',
                fontsize=12, color=color,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none')
            )

    # ------------------------------------------------------------------
    # Fill stable region
    # ------------------------------------------------------------------
    if len(verts)>=3:
        ctr = verts.mean(axis=0)
        ang = np.arctan2(verts[:,1]-ctr[1], verts[:,0]-ctr[0])
        order = np.argsort(ang)
        V = verts[order]
        ax.fill(V[:,0], V[:,1], color="#dddddd", alpha=0.7, zorder=-5)
        ax.plot(V[:,0], V[:,1], "k-", lw=1.4, zorder=-3)
        ax.text(ctr[0], ctr[1], "Stable region",
                fontsize=11, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'), zorder=20)

    # axes cosmetics
    ax.axvline(0, color='k', lw=1.5)
    ax.axhline(0, color='k', lw=1.5)
    ax.set_xlim(mu_floor, 5)
    ax.set_ylim(mu_floor, 5)
    ax.set_xlabel(f"$\\Delta\\mu_{{{E1}}}$ (eV)", fontsize=15)
    ax.set_ylabel(f"$\\Delta\\mu_{{{E2}}}$ (eV)", fontsize=15)
    ax.set_title(f"Binary Chemical Potential – {E1}-{E2} slice at Δμ({E3})={mu_third_fixed} eV", fontsize=17)
    ax.set_aspect("equal", adjustable='box')
    ax.grid(False)
    ax.figure.show()

    if out_png:
        ax.figure.savefig(out_png, dpi=400, bbox_inches='tight')
        print("Saved:", out_png)



def export_chempot_limits(
    PD_entry: PhaseDiagram,
    phases: list[str],
    limits: dict[str, tuple[float, float]],
    outfile: str = "chemical_potentials.json",
    decimals: int = 5,
):
    """
    General chemical potential solver.

    - Works for ANY number of phases (n phases = n elements)
    - Automatically finds hyperplanes for the requested phase names
    - Computes absolute and relative chemical potentials

    Args:
        PD_entry: PhaseDiagram object
        phases: list of phase chemical formulas (e.g. ["La2NiO4","La2O3","O2"])
        limits: dict of element chemical potential bounds
        outfile: JSON output filename
        decimals: rounding precision

    Returns:
        dict containing absolute and relative chemical potentials
    """

    # ------------------------------------------------------------
    # 1. Build chemical potential diagram
    # ------------------------------------------------------------
    cpd = ChemicalPotentialDiagram(PD_entry.entries, limits=limits)
    hyperplanes, entries = cpd._get_hyperplanes_and_entries()

    # All system elements sorted alphabetically
    elements = sorted({el.symbol for el in PD_entry.elements})

    if len(phases) != len(elements):
        raise ValueError(
            f"Need exactly {len(elements)} phases to solve "
            f"for {len(elements)} chemical potentials. Got: {len(phases)}"
        )

    # ------------------------------------------------------------
    # 2. Find hyperplane for each requested phase
    # ------------------------------------------------------------
    phase_to_hp = {}
    for p in phases:
        comp = Composition(p).reduced_composition

        match = None
        for hp, e in zip(hyperplanes, entries):
            if e.composition.reduced_composition == comp:
                match = hp
                break

        if match is None:
            raise ValueError(f"Phase {p} not found in ChemicalPotentialDiagram.")

        phase_to_hp[p] = match

    # ------------------------------------------------------------
    # 3. Build linear system A μ = b
    # ------------------------------------------------------------
    A = []
    b = []

    for p in phases:
        hp = phase_to_hp[p]

        # hp = (a1, a2, ..., aN, b)
        A.append(hp[:len(elements)])        # stoichiometric coefficients
        b.append(-hp[len(elements)])        # RHS = -constant term

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # ------------------------------------------------------------
    # 4. Solve for μ
    # ------------------------------------------------------------
    mu = np.linalg.solve(A, b)

    mu_rel = {el: mu[i] for i, el in enumerate(elements)}

    # ------------------------------------------------------------
    # 5. Extract elemental reference energies
    # ------------------------------------------------------------
    mu0 = {str(el): PD_entry.el_refs[el].energy_per_atom for el in PD_entry.elements}

    mu0 = {el: mu0[el] for el in elements}

    # relative μ
    mu_abs = {el: mu_rel[el] + mu0[el] for el in elements}

    # ------------------------------------------------------------
    # 6. Rounding helper
    # ------------------------------------------------------------
    def rnd(x):
        s = f"{x:.{decimals}f}".rstrip("0").rstrip(".")
        return float(s) if s else 0.0

    # ------------------------------------------------------------
    # 7. Construct output dictionary
    # ------------------------------------------------------------
    key = "-".join(phases)

    out = {
        "limits": {key: {el: rnd(mu_abs[el]) for el in elements}},
        "elemental_refs": {el: rnd(mu0[el]) for el in elements},
        "limits_wrt_el_refs": {
            key: {el: rnd(mu_rel[el]) for el in elements}
        },
    }

    Path(outfile).write_text(json.dumps(out, indent=2))
    return out

def main():
    api_key = "zBYiak7h6ies4ziNlAWqzXXHhc7rMBDB"

    # -------------------------
    # Load bulk data
    # -------------------------
    contcar_path = os.path.join(f"{CONFIG['material']}_bulk", "PBE_relax", "CONTCAR")
    structure = Structure.from_file(contcar_path)

    paths = setup_directories(CONFIG["material"], CONFIG["defect_categories"])
    logger.info(f"Output directories created at: {paths['base']}")

    # Bulk vasprun
    try:
        bulk_vr_path = os.path.join(f"{CONFIG['material']}_bulk", "PBE_DOS/DOS_7", "vasprun.xml")
        bulk_vr = Vasprun(bulk_vr_path)
        bulk_entry = ComputedStructureEntry(structure=structure, energy=bulk_vr.final_energy)
        logger.info(f"Bulk energy: {bulk_vr.final_energy:.4f} eV")
    except Exception:
        logger.error("Bulk structure processing failed", exc_info=True)
        return

    # Output file locations
    cp_dir = paths['chempot']
    cp_chempots_json = os.path.join(cp_dir, "chemical_potentials.json")
    cp_chempots1_json = os.path.join(cp_dir, "chemical_potentials1.json")
    cp_chempots2_json = os.path.join(cp_dir, "chemical_potentials2.json")
    cp_chempots3_json = os.path.join(cp_dir, "chemical_potentials3.json")

    cp_form_energies_csv = os.path.join(cp_dir, f"{CONFIG['material']}_formation_energies.csv")
    pd_file = os.path.join(cp_dir, f"{CONFIG['material']}_phase_diagram.png")
    cp_plot_file = os.path.join(cp_dir, f"{CONFIG['material']}_chemical_potentials.png")

    logger.info("Running CompetingPhasesAnalyzer")
    cpa = CompetingPhasesAnalyzer(composition=Composition(CONFIG["material"]),
                                  entries="CompetingPhases",
                                  check_compatibility=True)
    pd_entry = PhaseDiagram(cpa.entries)

    for entry in pd_entry.entries:
        comp = entry.composition
        formulas_per_unit = comp.get_reduced_composition_and_factor()[1]

    print(f"Chemical potentials: {cpa.intrinsic_phase_diagram.get_all_chempots(cpa.bulk_entry.composition)}")
    cpa.calculate_chempots()
    
    df_cpa_pd = pd.DataFrame({
        "Formula": [e.composition.reduced_formula for e in cpa.intrinsic_phase_diagram.entries],
        "Energy_form":[pd_entry.get_form_energy(e)/formulas_per_unit for e in cpa.intrinsic_phase_diagram.entries]
        }).sort_values(["Formula", "Energy_form"]).reset_index(drop=True)
    print("\n=== CPA→PD ENTRIES ===")
    print(df_cpa_pd)
    print(f"CPA→PD entry count: {len(df_cpa_pd)}")
    
    # ------------------------------------------------
    # CPA ENTRY TABLE
    # ------------------------------------------------
    df_cpa = pd.DataFrame({
        "Formula": [e.composition.reduced_formula for e in cpa.entries],
        "Energy_per_atom": [e.energy_per_atom for e in cpa.entries],
    }).sort_values(["Formula", "Energy_per_atom"]).reset_index(drop=True)

    print("\n=== CPA ENTRIES ===")
    print(df_cpa)
    print(f"CPA entry count: {len(df_cpa)}")


    df_pd = pd.DataFrame({
        "Formula": [e.composition.reduced_formula for e in pd_entry.entries],
        "Formation Energy (eV/fu)": [pd_entry.get_form_energy(e)/formulas_per_unit for e in pd_entry.entries],
        "Formation Energy (eV/atom)": [pd_entry.get_form_energy_per_atom(e) for e in pd_entry.entries],
        "DFT Energy (eV/fu)": [e.energy for e in pd_entry.entries],
        "DFT Energy (eV/atom)": [e.energy_per_atom for e in pd_entry.entries]
    }).sort_values(["Formula", "Formation Energy (eV/atom)"]).reset_index(drop=True)
    print("\n=== PHASE DIAGRAM ENTRIES ===")
    print(df_pd)
    print(f"PD entry count : {len(df_pd)}")

    # ------------------------------------------------
    # POLYMORPH TABLE (REAL DFT formation energies)
    # ------------------------------------------------
    formation_energy_df = (
        cpa.get_formation_energy_df(prune_polymorphs=False, include_dft_energies=True)
        .reset_index()
        .drop_duplicates(subset=["Formula", "Space Group"])
        .sort_values(["Formula", "Formation Energy (eV/atom)"])
        .reset_index(drop=True)
    )
    formation_energy_df.to_csv(cp_form_energies_csv)

    print("\n=== POLYMORPHS BY FORMULA ===")
    for _, row in formation_energy_df.iterrows():
        print(
            f"{row['Formula']:<12} | "
            f"SG: {row['Space Group']:<10} | "
            f"E_form = {row['Formation Energy (eV/atom)']:.6f} eV/atom | "
            f"E_DFT = {row['DFT Energy (eV/atom)']:.6f} eV/atom"
        )

    # ------------------------------------------------
    # PHASE DIAGRAM PLOT
    # -------------------------------------------------
    plotter = PDPlotter(pd_entry, show_unstable=False, backend="matplotlib")
    
    # get_plot returns Axes, not Figure
    ax = plotter.get_plot()
    
    # Extract the tuple: (lines, stable, unstable)
    lines, stable_entries, unstable_entries = plotter.pd_plot_data
    # Find La2NiO4
    for entry, coords in unstable_entries.items():
        if entry.composition.reduced_formula == "La2NiO4":
            # coords is a 2-tuple like (x, y)
            plt.scatter(coords[0], coords[1],
                       color="red", s=130, marker="X", zorder=10)
            plt.text(coords[0], coords[1],
                    "La2NiO4", color="red", fontsize=12,
                    ha="center", va="bottom")
    
    # Fix misplaced labels once
    for text in ax.texts:
        txt = text.get_text()
        x, y = text.get_position()

        if "La$_{2}$NiO$_{4}$" in txt:
            text.set_position((x + 80, y - 30))
        elif "La$_{3}$Ni$_{2}$O$_{7}$" in txt:
            text.set_position((x + 55, y))
        elif "La$_{7}$Ni$_{16}$" in txt:
            text.set_position((x - 65, y))
        elif "LaNi" in txt:
            text.set_position((x - 20, y))

    ax.figure.savefig(pd_file, bbox_inches="tight", dpi=300)
    ax.figure.show()
    print(f"\nPhase diagram saved to: {pd_file}")

    # ------------------------------------------------
    # CHEMICAL POTENTIAL DIAGRAM
    # ------------------------------------------------
    limits = {"La": (-15, 5), "Ni": (-15, 5), "O": (-15, 5)}
    cpd = ChemicalPotentialDiagram(pd_entry.entries,
                                   limits=limits)
    hyperplane, entries = cpd._get_hyperplanes_and_entries()
    lanio3_hp = hyperplane[2]
    la2o3_hp = hyperplane[3]
    lno_hp = hyperplane[4]
    o2_hp = hyperplane[17]
    stoich_matrix = np.array([lno_hp[0:3], la2o3_hp[0:3], o2_hp[0:3]])
    b = -np.array([ lno_hp[3], la2o3_hp[3], o2_hp[3]])
    mu_la, mu_ni, mu_o = np.linalg.solve(stoich_matrix, b)
    print(f"Chemical potentials at intersection of La2NiO4, La2O3, O2 planes:")
    print(f"mu_La: {mu_la:.4f} eV, mu_Ni: {mu_ni:.4f} eV, mu_O: {mu_o:.4f} eV")
    
    print("\n=== CPD HYPERPLANES AND ENTRIES ===")
    for i in range(len(entries)):
        print(f"Entry {i}:")
        print(f"Composition {entries[i].composition.reduced_composition} Hyper-plane Energy: {entries[i].energy_per_atom}, Hyperplane data: {hyperplane[i]}, length: {len(hyperplane)}")
    x_la_la2o3 = la2o3_hp[0]
    x_ni_la2o3 = la2o3_hp[1]
    x_o_la2o3 = la2o3_hp[2]
    b_la2o3 = la2o3_hp[3]
    
    x_la_lanio3 = lanio3_hp[0]
    x_ni_lanio3 = lanio3_hp[1]
    x_o_lanio3 = lanio3_hp[2]
    b_lanio3 = lanio3_hp[3]
    
    x_la_lno = lno_hp[0]
    x_ni_lno = lno_hp[1]
    x_o_lno = lno_hp[2]
    b_lno = lno_hp[3]
    
    x_la_o2 = o2_hp[0]
    x_ni_o2 = o2_hp[1]
    x_o_o2 = o2_hp[2]
    b_o2 = o2_hp[3]
    mu_la = np.linspace(-15,5,100)
    mu_ni = np.linspace(-15,5,100)
    MU_LA, MU_NI = np.meshgrid(mu_la, mu_ni)
    
    MU_O_lno = -(b_lno  + x_la_lno*MU_LA + x_ni_lno*MU_NI)/x_o_lno
    MU_O_la2o3 = -(b_la2o3 + x_la_la2o3*MU_LA + x_ni_la2o3*MU_NI)/x_o_la2o3
    MU_O_o2 = -(b_o2 + x_la_o2*MU_LA + x_ni_o2*MU_NI)/x_o_o2
    MU_O_lanio3 = -(b_lanio3 + x_la_lanio3*MU_LA + x_ni_lanio3*MU_NI)/x_o_lanio3
    print(f"mu_O2:" f"x_la_o2: {x_la_o2}, x_ni_o2: {x_ni_o2}, x_o_o2: {x_o_o2}, b_o2: {b_o2}")
    print(f"mu_O_lno:" f"x_la_lno: {x_la_lno}, x_ni_lno: {x_ni_lno}, x_o_lno: {x_o_lno}, b_lno: {b_lno}")
    print(f"mu_O_la2o3:" f"x_la_la2o3: {x_la_la2o3}, x_ni_la2o3: {x_ni_la2o3}, x_o_la2o3: {x_o_la2o3}, b_la2o3: {b_la2o3}")
    print(f"mu_O_lanio3:" f"x_la_lanio3: {x_la_lanio3}, x_ni_lanio3: {x_ni_lanio3}, x_o_lanio3: {x_o_lanio3}, b_lanio3: {b_lanio3}")

    cp_plot = cpd.get_plot(
        elements=CONFIG["elements"],
        formulas_to_draw=["LaNiO3", "La2O3", "NiO", "O2"],
        formula_colors=["red", "blue", "green", "orange", "violet"],
        label_stable=True,
        draw_formula_lines=True,
        draw_formula_meshes=True,
    )
    cp_plot.update_layout(
        title=f"Chemical Potential Diagram for {CONFIG['material']}",
        scene=dict(
            xaxis_showgrid = False,
            yaxis_showgrid = False,
            zaxis_showgrid = False,    
        )
    )
    cp_plot.update_xaxes(range=[-15, 5])
    cp_plot.update_yaxes(range=[-15, 5])
    
    # ------------------------------------------------------------------
    # 3.  Pick a centre-point of the visible patch we already meshed.
    # ------------------------------------------------------------------
    centre_x_lno = MU_LA.mean()
    centre_y_lno = MU_NI.mean()
    centre_z_lno = (-b_lno - x_la_lno * centre_x_lno - x_ni_lno * centre_y_lno) / x_o_lno

    cp_plot.add_trace(
        go.Scatter3d(
            x=[centre_x_lno],
            y=[centre_y_lno],
            z=[centre_z_lno],
            text=["La<sub>2</sub>NiO<sub>4</sub> plane"],
            mode="text",
            textfont=dict(size=18, color="black"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    centre_x_lanio3 = centre_x_lno
    centre_y_lanio3 = centre_y_lno
    centre_z_lanio3 = (-b_lanio3 - x_la_lanio3 * centre_x_lanio3 - x_ni_lanio3 * centre_y_lanio3) / x_o_lanio3
    #cp_plot.add_trace(
    #    go.Scatter3d(
    #        x=[centre_x_lanio3],
    #        y=[centre_y_lanio3],
    #        z=[centre_z_lanio3],
    #        text=["LaNiO<sub>3</sub> plane"],
    #        mode="text",
    #        textfont=dict(size=18, color="black"),
    #        showlegend=False,
    #        hoverinfo="skip",
    #    )
    #)

    centre_x_la2o3 = centre_x_lno
    centre_y_la2o3 = centre_y_lno
    centre_z_la2o3 = (-b_la2o3 - x_la_la2o3 * centre_x_la2o3 - x_ni_la2o3 * centre_y_la2o3) / x_o_la2o3
    #cp_plot.add_trace(
    #    go.Scatter3d(
    #        x=[centre_x_la2o3],
    #        y=[centre_y_la2o3],
    #        z=[centre_z_la2o3],
    #        text=["La<sub>2</sub>O<sub>3</sub> plane"],
    #        mode="text",
    #        textfont=dict(size=18, color="black"),
    #        showlegend=False,
    #        hoverinfo="skip",
    #    )
    #)
    #cp_plot.add_trace(
    #    go.Scatter3d(
    #        x=[mu_la],
    #        y=[mu_ni],
    #        z=[mu_o],
    #        marker = dict(size=5, color="red"),
    #        text=["Intersection point"],
    #    )
    #)
    
    centre_x_o2 = centre_x_lno
    centre_y_o2 = centre_y_lno
    centre_z_o2 = (-b_o2 - x_la_o2 * centre_x_o2 - x_ni_o2 * centre_y_o2) / x_o_o2
    #cp_plot.add_trace(
    #    go.Scatter3d(
    #        x=[centre_x_o2],
    #        y=[centre_y_o2],
    #        z=[centre_z_o2],
    #        text=["O<sub>2</sub> plane"],
    #        mode="text",
    #        textfont=dict(size=18, color="black"),
    #        showlegend=False,
    #        hoverinfo="skip",
    #    )
    #)
    cp_plot.add_surface(x=MU_LA, y=MU_NI, z=MU_O_lno, opacity=0.5, showscale=False, name="La2NiO4 Hyperplane", colorscale="reds")
    #cp_plot.add_surface(x=MU_LA, y=MU_NI, z=MU_O_lanio3, opacity=0.5, showscale=False, name="LaNiO3 Hyperplane", colorscale="purples")
    #cp_plot.add_surface(x=MU_LA, y=MU_NI, z=MU_O_la2o3, opacity=0.5, showscale=False, name="La2O3 Hyperplane", colorscale="blues")
    #cp_plot.add_surface(x=MU_LA, y=MU_NI, z=MU_O_o2, opacity=0.5, showscale=False, name="O2 Hyperplane", colorscale="greens")
    cp_plot.add_trace(
        go.Scatter3d(
            x=[mu_la],
            y=[mu_ni],
            z=[mu_o],
            marker = dict(size=5, color="red"),
            text=["Intersection point"],
            mode="markers+text",
            textposition="top center",
            showlegend=False,
        )
    )

    # ------------------------------------------------------------------
    # 4.  Hard-code a large, square PNG so the label is not squashed
    # ------------------------------------------------------------------
    cp_plot.write_image(
        cp_plot_file,
        width=900,
        height=900,
        scale=2,        # 2× 900 ⇒ 1800 px final
    )
    print(f"Chemical potential diagram saved to: {cp_plot_file}")
    cp_plot.show()
    element_pairs = [('La', 'Ni'), ('La', 'O'), ('Ni', 'O')]
    out = export_chempot_limits(
        PD_entry=cpa.phase_diagram,
        phases=["La2NiO4", "La2O3", "Ni"],
        limits=limits,
        outfile=cp_chempots_json,
        decimals=5,
    )
    print(f"Chemical Potential for {out}")
    
    out1 = export_chempot_limits(
        PD_entry=pd_entry,
        phases=["La2NiO4", "La2O3", "O2"],
        limits=limits,
        outfile=cp_chempots1_json,
        decimals=5,
    )
    print(f"Chemical Potential for {out1}")

    out2 = export_chempot_limits(
        PD_entry=pd_entry,
        phases=["La2NiO4", "LaNiO3", "O2"],
        limits=limits,
        outfile=cp_chempots2_json,
        decimals=5,
    )
    print(f"Chemical Potential for {out2}")
    out3 = export_chempot_limits(
        PD_entry=pd_entry,
        phases=["La2NiO4", "LaNiO3", "NiO"],
        limits=limits,
        outfile=cp_chempots3_json,
        decimals=5,
    )
    print(f"Chemical Potential for {out3}")
    print(f"Chemical potential limits saved to: {cp_chempots_json}")

    for pair in element_pairs:
        out_file1 = f"{paths['chempot']}/chemical_potential_{pair[0]}_{pair[1]}.png"
        plot_mu_binary(pair,
                       cpd=cpd,
                       mu_floor=-10.0, out_png=out_file1 )

    print(f"Chemical potential diagram saved to: {cp_plot_file}")
if __name__ == '__main__':
    main()
