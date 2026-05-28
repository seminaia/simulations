"""
Competing phases for GPAW defect workflows
Fetches structures from Materials Project, relaxes with PBE, computes chemical potentials.

Outputs (in OUTPUT_DIR):
  - entries.json                   Raw MP entries
  - phase_list.txt                 Competing phases summary
  - chempot_limits.json            Full doped chempots dict (for DefectThermodynamics)
  - chempot_limits.csv             All stability region vertices as CSV
  - formation_energies.csv         Formation energies of all competing phases
  - elemental_refs.json            Elemental reference energies (eV/atom)
  - chemical_potentials.json       All limits in {limit_name: {el: mu}} format
  - chempot_diagram.pdf            Chemical potential phase diagram plot
"""

import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doped.chemical_potentials import CompetingPhases, CompetingPhasesAnalyzer
from monty.serialization import dumpfn, loadfn
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from gpaw_helpers import relax, assign_magmoms, pbe_params

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

MATERIAL = "FeMn"
API_KEY = "zBYiak7h6ies4ziNlAWqzXXHhc7rMBDB"
ENERGY_ABOVE_HULL = 0.01
OUTPUT_DIR = "competing_phases"


# ---------------------------------------------------------------------------
#  Step 1: Fetch and set up
# ---------------------------------------------------------------------------

def setup_competing_phases(
    material: str = MATERIAL,
    api_key: str = API_KEY,
    energy_above_hull: float = ENERGY_ABOVE_HULL,
    output_dir: str = OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)

    # Fetch
    print(f"Fetching competing phases for {material} "
          f"(energy_above_hull <= {energy_above_hull} eV/atom) ...")
    cp = CompetingPhases(
        material,
        energy_above_hull=energy_above_hull,
        full_phase_diagram=False,
        api_key=api_key,
    )
    entries = cp.entries
    print(f"  {len(entries)} competing phases fetched")

    # Save entries + phase list before relaxation
    entries_json = os.path.join(output_dir, "entries.json")
    dumpfn(entries, entries_json)

    phase_list_txt = os.path.join(output_dir, "phase_list.txt")
    with open(phase_list_txt, "w") as f:
        f.write(f"Competing phases for {material}  "
                f"(energy_above_hull <= {energy_above_hull} eV/atom)\n")
        f.write(f"  {'Formula':<20}  {'mp_id':<25}  {'Spacegroup':<12}  SG#  N_atoms\n")
        f.write("  " + "-" * 80 + "\n")
        for ent in entries:
            try:
                sga = SpacegroupAnalyzer(ent.structure, symprec=0.1)
                sg_sym = sga.get_space_group_symbol()
                sg_num = sga.get_space_group_number()
            except Exception:
                sg_sym, sg_num = "?", "?"
            f.write(f"  {ent.composition.reduced_formula:<20}  "
                    f"{str(getattr(ent, 'entry_id', 'N/A')):<25}  "
                    f"{sg_sym:<12}  {sg_num:<4}  {ent.structure.num_sites}\n")
    print(f"  Phase list -> {phase_list_txt}")

    # Relax each phase and store energies
    for entry in entries:
        reduced = entry.composition.reduced_formula
        mp_id = str(getattr(entry, "entry_id", reduced)).replace("/", "_")
        phase_dir = os.path.join(output_dir, f"{reduced}_{mp_id}")
        os.makedirs(phase_dir, exist_ok=True)
        energy_file = os.path.join(phase_dir, "energy.txt")

        if os.path.exists(energy_file):
            with open(energy_file) as f:
                e = float(f.read().strip())
            print(f"  {reduced:<20s}  E = {e:.6f} eV  [cached]")
            continue

        atoms = AseAtomsAdaptor.get_atoms(entry.structure)
        mag_map = {"Fe": 5.0, "Mn": 5.0}
        assign_magmoms(atoms, mag_map, magnetization="FM")
        atoms = relax(
            atoms,
            pbe_params(kpts={"gamma": True, "density": 2.0},
                       txt=os.path.join(phase_dir, "scf.log")),
            fmax=0.01, fixcell=True,
            logname=os.path.join(phase_dir, "relax.log"),
            trajname=os.path.join(phase_dir, "relax.traj"),
            gpwname=os.path.join(phase_dir, "pbe_relax.gpw"),
        )
        e_total = float(atoms.get_potential_energy())
        with open(energy_file, "w") as f:
            f.write(f"{e_total:.10f}\n")
        print(f"  {reduced:<20s}  E = {e_total:.6f} eV  ({e_total/len(atoms):.4f} eV/atom)")

    print(f"  All {len(entries)} phases complete.")


# ---------------------------------------------------------------------------
#  Step 3: Collect energies and compute chemical potentials
# ---------------------------------------------------------------------------

def collect_chempots(
    material: str = MATERIAL,
    output_dir: str = OUTPUT_DIR,
    host_energy: Optional[float] = None,
    host_structure=None,
    output_json: str = "chemical_potentials.json",
):
    entries = loadfn(os.path.join(output_dir, "entries.json"))
    host_reduced = Composition(material).reduced_formula
    gpaw_entries: List[ComputedStructureEntry] = []
    missing = []

    for entry in entries:
        reduced = entry.composition.reduced_formula
        mp_id = str(getattr(entry, "entry_id", reduced)).replace("/", "_")
        phase_dir = os.path.join(output_dir, f"{reduced}_{mp_id}")
        energy_file = os.path.join(phase_dir, "energy.txt")

        if reduced == host_reduced and host_energy is not None:
            e_total = host_energy
            structure = host_structure if host_structure is not None else entry.structure
            print(f"  {reduced:<20s}  E = {e_total/entry.structure.num_sites:+.4f} eV/atom  [host]")
        elif os.path.exists(energy_file):
            with open(energy_file) as f:
                e_total = float(f.read().strip())
            print(f"  {reduced:<20s}  E = {e_total/entry.structure.num_sites:+.4f} eV/atom")
            structure = entry.structure
        else:
            print(f"  {reduced:<20s}  MISSING - using MP energy")
            missing.append(reduced)
            e_total = entry.energy
            structure = entry.structure

        gpaw_entries.append(ComputedStructureEntry(
            structure=structure,
            energy=e_total,
            entry_id=getattr(entry, "entry_id", reduced),
        ))

    if missing:
        print(f"\n  WARNING: {len(missing)} phases not converged: {missing}")

    cpa = CompetingPhasesAnalyzer(material, gpaw_entries)
    chempots_df = cpa.calculate_chempots()
    chempots_all = cpa.chempots

    # --- 1. Full doped chempots dict (usable with DefectThermodynamics) ---
    limits_json = os.path.join(output_dir, "chempot_limits.json")
    dumpfn(chempots_all, limits_json)
    print(f"\n  Full doped chempots dict -> {limits_json}")

    # --- 2. Chemical potential limits as CSV ---
    limits_csv = os.path.join(output_dir, "chempot_limits.csv")
    chempots_df.to_csv(limits_csv)
    print(f"  Chempot limits table -> {limits_csv}")
    print(chempots_df.to_string())

    # --- 3. Formation energies ---
    form_df = cpa.get_formation_energy_df(include_dft_energies=True)
    form_csv = os.path.join(output_dir, "formation_energies.csv")
    form_df.to_csv(form_csv)
    print(f"\n  Formation energies -> {form_csv}")
    print(form_df.to_string())

    # --- 4. Elemental reference energies ---
    el_refs = {str(el): float(entry.energy_per_atom)
               for el, entry in cpa.elemental_energies.items()}
    el_refs_path = os.path.join(output_dir, "elemental_refs.json")
    with open(el_refs_path, "w") as f:
        json.dump(el_refs, f, indent=2)
    print(f"\n  Elemental references (eV/atom) -> {el_refs_path}")
    for el, mu in el_refs.items():
        print(f"    {el}: {mu:.6f} eV/atom")

    # --- 5. All limits as {limit_name: {element: mu}} JSON ---
    limits: dict = chempots_all.get("limits", {})
    all_limits_clean = {}
    for name, mus in limits.items():
        all_limits_clean[name] = {str(el): float(mu) for el, mu in mus.items()}
    output_path = os.path.join(output_dir, output_json)
    with open(output_path, "w") as f:
        json.dump(all_limits_clean, f, indent=2)
    print(f"\n  All chemical potential limits -> {output_path}")

    # --- 6. Phase diagram plot ---
    try:
        import matplotlib.pyplot as plt
        fig = cpa.plot_chempot_heatmap()
        diagram_path = os.path.join(output_dir, "chempot_diagram.pdf")
        fig.savefig(diagram_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"  Phase diagram plot -> {diagram_path}")
    except Exception as e:
        print(f"  WARNING: Could not generate phase diagram plot: {e}")

    # --- 7. LaTeX table ---
    try:
        latex_str = cpa.to_LaTeX_table()
        latex_path = os.path.join(output_dir, "formation_energies.tex")
        with open(latex_path, "w") as f:
            f.write(latex_str)
        print(f"  LaTeX table -> {latex_path}")
    except Exception as e:
        print(f"  WARNING: Could not generate LaTeX table: {e}")

    return chempots_all


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_competing_phases()
    collect_chempots()
