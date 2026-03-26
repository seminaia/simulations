import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from ase import Atoms
from ase.spacegroup import crystal
from ase.visualize import view
from doped.core import DefectEntry
from doped.generation import DefectsGenerator
from gpaw import GPAW, restart
from gpaw.hybrids.energy import non_self_consistent_energy
from pymatgen.analysis.defects.core import DefectType
from pymatgen.io.ase import AseAtomsAdaptor
from shakenbreak.input import Distortions
from gpaw_helpers import relax, assign_magmoms, pbe_params, mgga_params


def _get_vbm(atoms: Atoms) -> float:
    try:
        homo, _ = atoms.calc.get_homo_lumo()
        return float(homo)
    except Exception:
        ef = float(atoms.calc.get_fermi_level())
        print(f"  Warning: get_homo_lumo failed; using Fermi level ({ef:.4f} eV) as VBM.")
        return ef


def _hse_singlepoint(
    atoms: Atoms,
    base_dir: str,
    hse_params_template: Dict[str, Any],
) -> Tuple[float, float]:
    os.makedirs(base_dir, exist_ok=True)
    pbe_scf_gpw = os.path.join(base_dir, "pbe_scf.gpw")
    hse_energy_txt = os.path.join(base_dir, "hse_energy.txt")
    hse_log = os.path.join(base_dir, "hse_scf.log")

    if not (os.path.exists(pbe_scf_gpw) and os.path.getsize(pbe_scf_gpw) > 100):
        scf_p = hse_params_template.copy()
        scf_p["xc"] = "PBE"
        scf_p["txt"] = os.path.join(base_dir, "pbe_scf.log")
        a = atoms.copy()
        assign_magmoms(a)
        a.calc = GPAW(**scf_p)
        a.get_potential_energy()
        a.calc.write(pbe_scf_gpw, mode='all')
        print(f"  PBE SCF saved to {pbe_scf_gpw}")
        atoms_for_vbm = a
    else:
        print(f"  Reusing existing PBE SCF: {pbe_scf_gpw}")
        atoms_for_vbm, _ = restart(pbe_scf_gpw)

    vbm = _get_vbm(atoms_for_vbm)

    if os.path.exists(hse_energy_txt):
        with open(hse_energy_txt) as f:
            e_hse = float(f.read().strip())
        print(f"  Reusing cached HSE energy: {e_hse:.6f} eV")
    else:
        components = non_self_consistent_energy(pbe_scf_gpw, 'HSE06')
        e_hse = float(components.sum())
        with open(hse_energy_txt, 'w') as f:
            f.write(f"{e_hse:.10f}\n")
        labels = [
            "DFT total free energy", "-DFT XC", "Hybrid semi-local XC",
            "EXX core-core", "EXX core-valence", "EXX valence-valence",
        ]
        with open(hse_log, 'w') as f:
            f.write("HSE06 non-self-consistent energy components (eV):\n")
            for lbl, val in zip(labels, components):
                f.write(f"  {lbl}: {float(val):.10f}\n")
            f.write(f"  TOTAL: {e_hse:.10f}\n")
        print(f"  HSE06 E = {e_hse:.6f} eV  (details: {hse_log})")

    return e_hse, vbm


def _get_natoms_change(defect_entry) -> Dict[str, int]:
    defect = defect_entry.defect
    if hasattr(defect, 'element_changes') and defect.element_changes:
        return {str(el): int(n) for el, n in defect.element_changes.items()}
    d_comp = defect_entry.defect_supercell.composition.as_dict()
    b_comp = defect_entry.bulk_supercell.composition.as_dict()
    all_els = set(d_comp) | set(b_comp)
    return {
        el: int(d_comp.get(el, 0) - b_comp.get(el, 0))
        for el in all_els
        if int(d_comp.get(el, 0) - b_comp.get(el, 0)) != 0
    }


def _formation_energy(
    e_hse_defect: float,
    e_corr: float,
    e_hse_bulk_sc: float,
    natoms_change: Dict[str, int],
    chempots: Dict[str, float],
    charge: int,
    vbm: float,
    fermi_level: float,
) -> float:
    mu_sum = sum(n * chempots.get(el, 0.0) for el, n in natoms_change.items())
    return e_hse_defect + e_corr - e_hse_bulk_sc - mu_sum + charge * (vbm + fermi_level)


def _kumagai_correction(
    defect_entry: DefectEntry,
    defect_atoms: Atoms,
    bulk_atoms: Atoms,
    dielectric: np.ndarray,
    corr_dir: str,
) -> float:
    os.makedirs(corr_dir, exist_ok=True)
    corr_txt = os.path.join(corr_dir, "kumagai.txt")

    if defect_entry.charge_state == 0:
        with open(corr_txt, 'w') as f:
            f.write("charge=0: no correction needed\ncorrection_energy=0.0 eV\n")
        return 0.0

    try:
        defect_pots = defect_atoms.calc.get_atomic_electrostatic_potentials().tolist()
        bulk_pots = bulk_atoms.calc.get_atomic_electrostatic_potentials().tolist()
        if not hasattr(defect_entry, 'calculation_metadata') or defect_entry.calculation_metadata is None:
            defect_entry.calculation_metadata = {}
        defect_entry.calculation_metadata['defect_site_potentials'] = defect_pots
        defect_entry.calculation_metadata['bulk_site_potentials'] = bulk_pots

        defect_entry.get_kumagai_correction(dielectric=dielectric, plot=False)
        e_corr = float(defect_entry.corrections.get('kumagai_charge_correction', 0.0))

        with open(corr_txt, 'w') as f:
            f.write(f"Kumagai (eFNV) charge correction\n")
            f.write(f"correction_energy = {e_corr:.6f} eV\n")
            f.write(f"charge_state      = {defect_entry.charge_state:+d}\n")
            f.write(f"dielectric_tensor =\n{dielectric}\n")
        print(f"  Kumagai correction: {e_corr:+.4f} eV")
        return e_corr

    except Exception as exc:
        print(f"  Warning: Kumagai correction failed ({exc}). Using E_corr = 0.0.")
        with open(corr_txt, 'w') as f:
            f.write(f"ERROR: {exc}\ncorrection_energy=0.0 eV (placeholder)\n")
        return 0.0


def _plot_formation_energies(
    formation_data: List[Dict],
    vbm: float,
    bandgap: float,
    chempots: Dict[str, float],
    output_path: str,
    title: str = "TiO2 defect formation energies",
) -> None:
    ef_range = np.linspace(0.0, bandgap, 400)
    defect_groups: Dict[str, List[Dict]] = defaultdict(list)
    for entry in formation_data:
        base = entry['label'].rsplit('_q', 1)[0]
        defect_groups[base].append(entry)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = cm.tab10.colors

    for idx, (dname, entries) in enumerate(defect_groups.items()):
        color = colors[idx % len(colors)]
        charge_lines: List[Tuple[int, np.ndarray]] = []
        for ent in entries:
            q = ent['charge']
            ef_vals = np.array([
                _formation_energy(
                    ent['e_hse'], ent['e_corr'], ent['e_bulk_sc'],
                    ent['natoms_change'], chempots, q, ent['vbm'], ef,
                )
                for ef in ef_range
            ])
            charge_lines.append((q, ef_vals))

        envelope = np.min(np.stack([l for _, l in charge_lines], axis=0), axis=0)
        ax.plot(ef_range, envelope, color=color, lw=2.0, label=dname)

        cl_sorted = sorted(charge_lines, key=lambda x: x[0])
        for i in range(len(cl_sorted) - 1):
            q1, l1 = cl_sorted[i]
            q2, l2 = cl_sorted[i + 1]
            cross_idx = np.where(np.diff(np.sign(l1 - l2)))[0]
            for ci in cross_idx:
                ef_ctl = float(ef_range[ci])
                e_ctl = float((envelope[ci] + envelope[min(ci + 1, len(envelope) - 1)]) / 2)
                ax.axvline(ef_ctl, ls='--', lw=0.7, color=color, alpha=0.5)
                ax.annotate(
                    f"e({q1:+d}/{q2:+d})\n{ef_ctl:.2f} eV",
                    xy=(ef_ctl, e_ctl), fontsize=5.5, color=color, ha='center',
                )

    ax.axvline(0.0, color='k', lw=1.0, ls=':')
    ax.axvline(bandgap, color='k', lw=1.0, ls=':', label=f'CBM ({bandgap:.2f} eV)')
    ax.axhline(0.0, color='gray', lw=0.6, ls='--')
    ax.set_xlabel(f"Fermi level relative to VBM (eV)  [VBM = {vbm:.3f} eV abs]", fontsize=12)
    ax.set_ylabel("Formation energy (eV)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc='best')
    ax.set_xlim(0, bandgap)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Formation energy diagram -> {output_path}")


def _compute_dielectric_tensor(
    atoms: Atoms,
    pbe_params: Dict[str, Any],
    output_json: str,
    kpts_dense: Tuple[int, int, int] = (4, 4, 4),
) -> np.ndarray:
    from gpaw.response.df import DielectricFunction

    base_dir = os.path.dirname(output_json) or 'defects'
    os.makedirs(base_dir, exist_ok=True)
    scf_gpw = os.path.join(base_dir, 'dielectric_scf.gpw')

    if not (os.path.exists(scf_gpw) and os.path.getsize(scf_gpw) > 100):
        p = pbe_params.copy()
        p['kpts'] = {'gamma': True, 'density': 4.0}
        p['nbands'] = -50
        p['convergence'] = {'density': 1e-8, 'eigenstates': 1e-10, 'energy': 1e-6}
        p['txt'] = scf_gpw.replace('.gpw', '.log')
        a = atoms.copy()
        assign_magmoms(a)
        a.calc = GPAW(**p)
        a.get_potential_energy()
        a.calc.write(scf_gpw, mode='all')
        print(f"  Dielectric SCF -> {scf_gpw}")
    else:
        print(f"  Reusing cached dielectric SCF: {scf_gpw}")

    eps_diag: List[float] = []
    for direction in ['x', 'y', 'z']:
        df = DielectricFunction(
            calc=scf_gpw,
            frequencies={'type': 'nonlinear', 'domega0': 0.1, 'omega2': 10.0, 'omegamax': 15.0},
        )
        eps_re, eps_im = df.get_dielectric_constant(direction=direction)
        eps_diag.append(float(eps_re))
        print(f"  eps_{direction}{direction} = {float(eps_re):.4f}  (imag={float(eps_im):.6f})")

    eps_tensor = np.diag(eps_diag)
    print(f"  Dielectric tensor:\n{eps_tensor}")

    with open(output_json, 'w') as fh:
        json.dump(eps_tensor.tolist(), fh, indent=2)
    print(f"  Dielectric tensor -> {output_json}")
    return eps_tensor


# ---------------------------------------------------------------------------
#  TiO2 phase definitions
# ---------------------------------------------------------------------------

defect_type_map = {
    DefectType.Vacancy.value: "Vacancy",
    DefectType.Interstitial.value: "Interstitial",
    DefectType.Substitution.value: "Substitution",
}

_CHARGE_MAP: Dict[str, float] = {'Ti': 4, 'O': -2}

# --- Rutile TiO2: P4_2/mnm (136) ---
# Experimental: a=4.594 A, c=2.959 A
# Ti at 2a (0,0,0), O at 4f (x,x,0) with x=0.3053
rutile_a, rutile_c = 4.594, 2.959
rutile_atoms = crystal(
    ['Ti', 'O'],
    [(1/2, 1/2, 0), (0.695679, 0.695679, 1/2)],
    spacegroup=136,
    cellpar=[rutile_a, rutile_a, rutile_c, 90, 90, 90],
    size=(1, 1, 1),
)
charges_rutile = [_CHARGE_MAP[s] for s in rutile_atoms.get_chemical_symbols()]
rutile_atoms.set_initial_charges(charges_rutile)
assign_magmoms(rutile_atoms)
view(rutile_atoms,repeat=(2,2,3))
rutile_atoms.write('TiO2_rutile_P42mnm.cif', format='cif')

# --- Anatase TiO2: I4_1/amd (141) ---
# Experimental: a=3.785 A, c=9.514 A
# Ti at 4a (0,0,0), O at 8e (0,0,u) with u=0.2081
anatase_a, anatase_c = 3.785, 9.514
anatase_atoms = crystal(
    ['Ti', 'O'],
    [(1/2, 1/2, 1/2), (0, 1/2, 0.457152)],
    spacegroup=141,
    cellpar=[anatase_a, anatase_a, anatase_c, 90, 90, 90],
    size=(1, 1, 1),
)
charges_anatase = [_CHARGE_MAP[s] for s in anatase_atoms.get_chemical_symbols()]
anatase_atoms.set_initial_charges(charges_anatase)
assign_magmoms(anatase_atoms)
view(anatase_atoms,repeat=(3,3,2))
anatase_atoms.write('TiO2_anatase_I41amd.cif', format='cif')

print(f"Rutile:  {len(rutile_atoms)} atoms, cell = {rutile_atoms.cell.cellpar()[:3]}")
print(f"Anatase: {len(anatase_atoms)} atoms, cell = {anatase_atoms.cell.cellpar()[:3]}")


# ---------------------------------------------------------------------------
#  Run defect workflow for each polymorph
# ---------------------------------------------------------------------------

def run_tio2_defect_workflow(
    phase_name: str,
    tio2_atoms: Atoms,
    out_prefix: str,
):
    """Full defect workflow for one TiO2 polymorph."""

    print(f"\n{'#' * 70}")
    print(f"#  {phase_name} TiO2 defect workflow")
    print(f"{'#' * 70}\n")

    defects_dir = f"defects_{out_prefix}"
    os.makedirs(defects_dir, exist_ok=True)

    # --- r2SCAN relaxation ---
    mgga_p = mgga_params(txt=f"{out_prefix}_scf_mgga.log")

    relaxed_mgga = relax(
        tio2_atoms,
        calculator_params=mgga_p,
        fmax=0.01,
        fixcell=False,
        logname=f'{out_prefix}_opt_mgga.log',
        trajname=f'{out_prefix}_opt_mgga.traj',
        gpwname=f'{out_prefix}_opt_mgga.gpw',
    )
    E_mgga = relaxed_mgga.get_potential_energy()
    cell = relaxed_mgga.cell.cellpar()
    print(f"{phase_name} r2SCAN energy: {E_mgga:.6f} eV")
    print(f"Relaxed lattice: a={cell[0]:.4f} b={cell[1]:.4f} c={cell[2]:.4f} A, "
          f"a={cell[3]:.2f} b={cell[4]:.2f} g={cell[5]:.2f} deg")

    # Update global lattice constants with relaxed values
    if phase_name == "Rutile":
        global rutile_a, rutile_c
        rutile_a, rutile_c = cell[0], cell[2]
    elif phase_name == "Anatase":
        global anatase_a, anatase_c
        anatase_a, anatase_c = cell[0], cell[2]

    # --- Chemical potentials (pre-computed by comp_phases.py) ---
    CHEMPOTS_JSON = os.path.join(defects_dir, "chemical_potentials.json")
    if not os.path.exists(CHEMPOTS_JSON):
        CHEMPOTS_JSON = os.path.join("competing_phases", "chemical_potentials.json")
    if not os.path.exists(CHEMPOTS_JSON):
        sys.exit("No chemical_potentials.json found. Run comp_phases.py first.")
    with open(CHEMPOTS_JSON) as f:
        chempots: Dict[str, float] = {k: v for k, v in json.load(f).items()
                                       if k != "__instructions__"}
    print(f"Loaded chemical potentials from {CHEMPOTS_JSON}")

    # --- Dielectric tensor ---
    DIELECTRIC_JSON = os.path.join(defects_dir, "dielectric_tensor.json")
    if os.path.exists(DIELECTRIC_JSON):
        DIELECTRIC_TENSOR: Optional[np.ndarray] = np.array(json.load(open(DIELECTRIC_JSON)))
        print(f"Loaded dielectric tensor from {DIELECTRIC_JSON}")
    else:
        print("\nComputing dielectric tensor ...")
        DIELECTRIC_TENSOR = _compute_dielectric_tensor(
            atoms=tio2_atoms,
            pbe_params=pbe_params(),
            output_json=DIELECTRIC_JSON,
        )
        print(f"Dielectric tensor:\n{DIELECTRIC_TENSOR}")

    # --- Generate defects ---
    defect_gen = DefectsGenerator(
        relaxed_mgga,
        extrinsic={},
        interstitial_gen_kwargs=True,
        interstitial_elements=['O'],
        vacancy_gen_kwargs=True,
        vacancy_elements=['Ti', 'O'],
        substitution_gen_kwargs=False,
        substitution_elements=[],
        generate_supercell=True,
        supercell_gen_kwargs={'force_diagonal': True, 'min_image_distance': 10},
    )

    # --- Bulk supercell reference ---
    print("\n" + "=" * 60)
    print(f"Bulk supercell reference ({phase_name})")
    print("=" * 60)
    bulk_sc_dir = os.path.join(defects_dir, "bulk_supercell")
    os.makedirs(bulk_sc_dir, exist_ok=True)
    bulk_sc_atoms = AseAtomsAdaptor.get_atoms(defect_gen.bulk_supercell, msonable=False)
    assign_magmoms(bulk_sc_atoms)

    bulk_sc_mgga_p = mgga_params(txt=os.path.join(bulk_sc_dir, "mgga_scf.log"))

    bulk_sc_atoms = relax(
        bulk_sc_atoms,
        calculator_params=bulk_sc_mgga_p,
        fmax=0.01,
        fixcell=True,
        logname=os.path.join(bulk_sc_dir, "mgga_relax.log"),
        trajname=os.path.join(bulk_sc_dir, "mgga_relax.traj"),
        gpwname=os.path.join(bulk_sc_dir, "mgga_relax.gpw"),
    )

    E_bulk_sc = float(bulk_sc_atoms.get_potential_energy())
    vbm_bulk_sc = _get_vbm(bulk_sc_atoms)
    print(f"Bulk SC  E_MGGA = {E_bulk_sc:.4f} eV  VBM = {vbm_bulk_sc:.4f} eV")

    with open(os.path.join(bulk_sc_dir, "reference.txt"), "w") as _f:
        _f.write(f"Bulk supercell reference ({phase_name})\n")
        _f.write(f"  N atoms:    {len(bulk_sc_atoms)}\n")
        _f.write(f"  E_MGGA (eV): {E_bulk_sc:.6f}\n")
        _f.write(f"  VBM (eV):   {vbm_bulk_sc:.6f}\n")

    # --- Loop over defects ---
    summary_path = os.path.join(defects_dir, "defects_summary.txt")
    formation_data: List[Dict] = []

    with open(summary_path, "w") as sf:
        sf.write(f"{phase_name} TiO2 defect summary  {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        sf.write(f"{'Name':<40} {'Type':<15} {'Charge':>6}  "
                 f"{'E_MGGA (eV)':>14}  {'E_corr (eV)':>12}  {'Ef@VBM (eV)':>12}\n")
        sf.write("-" * 105 + "\n")

    for defect_entry in defect_gen.defect_entries.values():
        defect_name = defect_entry.name
        defect_type = defect_type_map.get(defect_entry.defect.defect_type.value, "Unknown")
        charge = defect_entry.charge_state
        sc = defect_entry.defect_supercell
        frac_coords = defect_entry.sc_defect_frac_coords
        natoms_change = _get_natoms_change(defect_entry)

        defect_dir = os.path.join(defects_dir, f"{defect_name}_q{charge:+d}")
        os.makedirs(defect_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Defect: {defect_name}  charge: {charge:+d}  type: {defect_type}")
        print(f"{'='*60}")

        cell_params = sc.lattice.abc + sc.lattice.angles
        with open(os.path.join(defect_dir, "info.txt"), "w") as f:
            f.write(f"Defect:           {defect_name}\n")
            f.write(f"Type:             {defect_type}\n")
            f.write(f"Charge state:     {charge:+d}\n")
            f.write(f"Frac coords:      "
                    f"{frac_coords[0]:.6f}  {frac_coords[1]:.6f}  {frac_coords[2]:.6f}\n")
            f.write(f"Supercell sites:  {len(sc)}\n")
            f.write(f"  a={cell_params[0]:.4f}  b={cell_params[1]:.4f}  c={cell_params[2]:.4f} A\n")
            f.write(f"  a={cell_params[3]:.2f}  b={cell_params[4]:.2f}  g={cell_params[5]:.2f} deg\n")
            f.write(f"Atom count change: {natoms_change}\n")

        snb = Distortions(defect_entry)
        distorted_dict, distortion_meta = snb.apply_distortions()

        with open(os.path.join(defect_dir, "distortions.json"), "w") as f:
            json.dump(distortion_meta, f, indent=2, default=str)

        charge_block = (
            distorted_dict
            .get(defect_name, {})
            .get('charges', {})
            .get(charge, {})
        )
        structs_block = charge_block.get('structures', {})
        unperturbed = structs_block.get('Unperturbed')
        dist_structs = structs_block.get('distortions', {})

        all_structs: Dict[str, Any] = {}
        if unperturbed is not None:
            all_structs['Unperturbed'] = unperturbed
        all_structs.update(dist_structs)

        if not all_structs:
            print(f"  WARNING: no structures for {defect_name} q={charge:+d}. Skipping.")
            continue

        distortion_energies: Dict[str, float] = {}

        for dist_key, pmg_struct in all_structs.items():
            safe_key = dist_key.replace('/', '_').replace(' ', '_')
            dist_dir = os.path.join(defect_dir, safe_key)
            os.makedirs(dist_dir, exist_ok=True)

            energy_cache = os.path.join(dist_dir, "energy.txt")
            if os.path.exists(energy_cache):
                with open(energy_cache) as _ef:
                    distortion_energies[dist_key] = float(_ef.read().strip())
                print(f"  {dist_key:40s}  E_PBE = {distortion_energies[dist_key]:.4f} eV  [cached]")
                continue

            def_atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
            assign_magmoms(def_atoms)

            d_pbe_p = pbe_params(charge=charge, txt=os.path.join(dist_dir, "scf.log"))

            def_atoms = relax(
                def_atoms,
                calculator_params=d_pbe_p,
                fmax=0.05,
                fixcell=True,
                logname=os.path.join(dist_dir, "relax.log"),
                trajname=os.path.join(dist_dir, "relax.traj"),
                gpwname=os.path.join(dist_dir, "pbe_relax.gpw"),
            )
            e_pbe = float(def_atoms.get_potential_energy())
            distortion_energies[dist_key] = e_pbe
            with open(energy_cache, 'w') as _ef:
                _ef.write(str(e_pbe))
            print(f"  {dist_key:40s}  E_PBE = {e_pbe:.4f} eV")

        E_unperturbed = distortion_energies.get('Unperturbed')
        with open(os.path.join(defect_dir, "distortion_energies.txt"), "w") as f:
            f.write(f"# Distortion screening for {defect_name} q={charge:+d}\n")
            f.write(f"# {'Distortion':<40} {'E_PBE (eV)':>14}  {'dE vs Unperturbed (eV)':>24}\n")
            for dk, e in sorted(distortion_energies.items(), key=lambda x: x[1]):
                dE = (e - E_unperturbed) if E_unperturbed is not None else float('nan')
                f.write(f"  {dk:<40} {e:14.6f}  {dE:+24.6f}\n")

        best_key = min(distortion_energies, key=distortion_energies.__getitem__)
        best_e = distortion_energies[best_key]
        dE_best = (best_e - E_unperturbed) if E_unperturbed is not None else float('nan')
        print(f"  Best distortion: {best_key}  dE = {dE_best:+.4f} eV")

        best_safe = best_key.replace('/', '_').replace(' ', '_')
        best_gpw = os.path.join(defect_dir, best_safe, "pbe_relax.gpw")
        best_atoms, _ = restart(best_gpw)

        # --- r2SCAN single-point on best distortion ---
        mgga_dir = os.path.join(defect_dir, "mgga")
        os.makedirs(mgga_dir, exist_ok=True)
        mgga_energy_txt = os.path.join(mgga_dir, "mgga_energy.txt")

        if os.path.exists(mgga_energy_txt):
            with open(mgga_energy_txt) as f:
                e_mgga_defect = float(f.read().strip())
            print(f"  Reusing cached MGGA energy: {e_mgga_defect:.6f} eV")
        else:
            d_mgga_p = mgga_params(charge=charge, txt=os.path.join(mgga_dir, "mgga_scf.log"))
            mgga_atoms = best_atoms.copy()
            assign_magmoms(mgga_atoms)
            mgga_atoms.calc = GPAW(**d_mgga_p)
            e_mgga_defect = float(mgga_atoms.get_potential_energy())
            with open(mgga_energy_txt, 'w') as f:
                f.write(f"{e_mgga_defect:.10f}\n")
            print(f"  MGGA E = {e_mgga_defect:.6f} eV")

        # --- Charge correction ---
        bulk_atoms_corr, _ = restart(os.path.join(bulk_sc_dir, "mgga_relax.gpw"))
        e_corr = 0.0
        if DIELECTRIC_TENSOR is not None:
            e_corr = _kumagai_correction(
                defect_entry=defect_entry,
                defect_atoms=best_atoms,
                bulk_atoms=bulk_atoms_corr,
                dielectric=DIELECTRIC_TENSOR,
                corr_dir=os.path.join(defect_dir, "charge_correction"),
            )

        ef_at_vbm = _formation_energy(
            e_mgga_defect, e_corr, E_bulk_sc, natoms_change,
            chempots, charge, vbm_bulk_sc, fermi_level=0.0,
        )

        formation_data.append({
            'label': f"{defect_name}_q{charge:+d}",
            'charge': charge,
            'e_hse': e_mgga_defect,
            'e_corr': e_corr,
            'e_bulk_sc': E_bulk_sc,
            'natoms_change': natoms_change,
            'vbm': vbm_bulk_sc,
        })

        with open(summary_path, "a") as sf:
            sf.write(f"{defect_name:<40} {defect_type:<15} {charge:>+6d}  "
                     f"{e_mgga_defect:>14.4f}  {e_corr:>+12.4f}  {ef_at_vbm:>12.4f}\n")

        print(f"[DONE] {defect_name} q={charge:+d}  "
              f"E_MGGA={e_mgga_defect:.4f} eV  E_corr={e_corr:+.4f} eV  "
              f"E_f(VBM)={ef_at_vbm:.4f} eV")

    # --- Formation energy diagram ---
    if formation_data:
        try:
            _bulk_ref, _ = restart(os.path.join(bulk_sc_dir, "mgga_relax.gpw"))
            _, _lumo = _bulk_ref.calc.get_homo_lumo()
            bandgap_est = max(float(_lumo) - vbm_bulk_sc, 0.3)
        except Exception:
            bandgap_est = 3.0  # TiO2 ~3 eV gap
            print("Warning: could not extract bandgap; using 3.0 eV placeholder.")

        _plot_formation_energies(
            formation_data=formation_data,
            vbm=vbm_bulk_sc,
            bandgap=bandgap_est,
            chempots=chempots,
            output_path=os.path.join(defects_dir, "formation_energy_diagram.png"),
            title=f"{phase_name} TiO2 defect formation energies (r2SCAN, PBE geometry)",
        )

    with open(summary_path, "a") as sf:
        sf.write(f"\nCompleted: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    print(f"\n{phase_name} done.  Summary -> {summary_path}")


# ---------------------------------------------------------------------------
#  Main: run both polymorphs
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_tio2_defect_workflow("Rutile",  rutile_atoms,  "rutile")
    run_tio2_defect_workflow("Anatase", anatase_atoms, "anatase")

    print("\n" + "=" * 70)
    print("All TiO2 defect workflows complete.")
    print("=" * 70)
