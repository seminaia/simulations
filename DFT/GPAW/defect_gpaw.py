import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Force line-buffered output so every print() appears immediately in SLURM logs.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use('Agg')   # non-interactive; safe on HPC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gpaw.mpi import world
import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import BFGS
from ase.spacegroup import crystal
from ase.units import Bohr
from ase.visualize import view
from doped.generation import DefectsGenerator
from gpaw import GPAW, restart
from gpaw.hybrids.energy import non_self_consistent_energy
from pymatgen.analysis.defects.core import DefectType
from pymatgen.io.ase import AseAtomsAdaptor
from shakenbreak.input import Distortions

# ── Relaxation helper ─────────────────────────────────────────────────────────
def relax(
    atoms: Atoms,
    calculator_params: Dict[str, Any],
    fmax: float = 0.01,
    fixcell: bool = True,
    logname: str = 'opt.log',
    trajname: str | None = None,
    gpwname: str = 'rlx.gpw',
) -> Atoms:
    orig_atoms = atoms
    done_file = gpwname.replace('.gpw', '.traj')
    if os.path.exists(gpwname) and os.path.getsize(gpwname) > 100:
        try:
            atoms, _ = restart(gpwname)
            if len(atoms) != len(orig_atoms):
                print(f"Cached GPW has {len(atoms)} atoms but current structure has {len(orig_atoms)}. Starting fresh.")
                atoms = orig_atoms
            atoms.calc = GPAW(**calculator_params)
            print(f"Restarted positions from {gpwname}")
        except Exception as e:
            print(f"Restart failed ({e}), starting fresh.")
            atoms.calc = GPAW(**calculator_params)
    else:
        atoms.calc = GPAW(**calculator_params)
        print("Starting fresh calculation")
    opt_atoms = atoms if fixcell else FrechetCellFilter(atoms)
    print(f"Relaxation mode: {'fixed cell' if fixcell else 'variable cell'}  fmax={fmax} eV/Å")
    bfgs = BFGS(opt_atoms, logfile=logname, trajectory=trajname)
        
    def _relax_log():
        raw = opt_atoms.atoms if isinstance(opt_atoms, FrechetCellFilter) else opt_atoms
        fmax_cur = float(np.max(np.linalg.norm(raw.get_forces(), axis=1)))
        epot = raw.get_potential_energy() / len(raw)
        print(f"[{datetime.now():%H:%M:%S}]  relax step {bfgs.nsteps:4d}  "
              f"Epot={epot:.4f} eV/atom  fmax={fmax_cur:.4f} eV/Å")
        
    bfgs.attach(_relax_log, interval=1)
    bfgs.run(fmax=fmax, steps=500)
    if isinstance(opt_atoms, FrechetCellFilter):
        opt_atoms = opt_atoms.atoms
    try:
        opt_atoms.calc.write(gpwname, mode='all')
        print(f"Saved to {gpwname}")
    except Exception as e:
        print(f"Warning: could not save state: {e}")
    ase_write(done_file, opt_atoms)
    print(f"Converged structure saved to {done_file}")
    forces = opt_atoms.get_forces()
    print(f"Max force: {np.max(np.linalg.norm(forces, axis=1)):.6f} eV/Å")
    relaxed = opt_atoms

    orig_atoms.set_cell(relaxed.get_cell(), scale_atoms=False)
    orig_atoms.set_positions(relaxed.get_positions())
    cell = relaxed.cell.cellpar()
    print(f"""  Lattice: a={cell[0]:.4f} b={cell[1]:.4f} c={cell[2]:.4f} Å,
          α={cell[3]:.2f} β={cell[4]:.2f} γ={cell[5]:.2f}° Volume={relaxed.get_volume():.2f} Å³""")
    return relaxed


# ── Per-element initial magnetic moment magnitudes for LaNiO3 supercells ──────
_MAGMOM_MAP: Dict[str, float] = {
    'La': 0.6, 'Ni': 2.0, 'O': 0.6,
    'Ca': 0.0, 'Sr': 0.0, 'Co': 2.0, 'Mn': 3.0,
}

def _assign_magmoms(atoms: Atoms) -> None:
    """Assign alternating-sign magnetic moments by element to an Atoms object.

    Sign alternates for each occurrence of a given species, giving an
    antiferromagnetic starting guess that scales with any supercell size.
    """
    counters: Dict[str, int] = defaultdict(int)
    moms: List[float] = []
    for sym in atoms.get_chemical_symbols():
        mag  = _MAGMOM_MAP.get(sym, 0.6)
        sign = 1 if counters[sym] % 2 == 0 else -1
        moms.append(sign * mag)
        counters[sym] += 1
    atoms.set_initial_magnetic_moments(moms)


def _get_vbm(atoms: Atoms) -> float:
    """Return the VBM (HOMO) eigenvalue in eV from a GPAW Atoms object."""
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
    """Non-self-consistent HSE06 energy on top of a converged PBE state.

    Saves the PBE SCF to ``base_dir/pbe_scf.gpw``, then calls
    ``non_self_consistent_energy`` for the HSE06 correction.

    Returns:
        (e_hse, vbm) — HSE06 total energy (eV) and PBE VBM eigenvalue (eV).
    """
    os.makedirs(base_dir, exist_ok=True)
    pbe_scf_gpw    = os.path.join(base_dir, "pbe_scf.gpw")
    hse_energy_txt = os.path.join(base_dir, "hse_energy.txt")
    hse_log        = os.path.join(base_dir, "hse_scf.log")

    # ── PBE single-point SCF (writes wavefunction to disk) ──────────────────
    if not (os.path.exists(pbe_scf_gpw) and os.path.getsize(pbe_scf_gpw) > 100):
        scf_p = hse_params_template.copy()
        scf_p["xc"]  = "PBE"
        scf_p["txt"] = os.path.join(base_dir, "pbe_scf.log")
        a = atoms.copy()
        _assign_magmoms(a)
        a.calc = GPAW(**scf_p)
        a.get_potential_energy()
        a.calc.write(pbe_scf_gpw, mode='all')
        print(f"  PBE SCF saved to {pbe_scf_gpw}")
        atoms_for_vbm = a
    else:
        print(f"  Reusing existing PBE SCF: {pbe_scf_gpw}")
        atoms_for_vbm, _ = restart(pbe_scf_gpw)

    vbm = _get_vbm(atoms_for_vbm)

    # ── HSE06 non-self-consistent ────────────────────────────────────────────
    if os.path.exists(hse_energy_txt):
        with open(hse_energy_txt) as f:
            e_hse = float(f.read().strip())
        print(f"  Reusing cached HSE energy: {e_hse:.6f} eV")
    else:
        components = non_self_consistent_energy(pbe_scf_gpw, 'HSE06')
        e_hse = float(components.sum())
        with open(hse_energy_txt, 'w') as f:
            f.write(f"{e_hse:.10f}\n")
        with open(hse_log, 'w') as f:
            labels = [
                "DFT total free energy", "-DFT XC", "Hybrid semi-local XC",
                "EXX core-core", "EXX core-valence", "EXX valence-valence",
            ]
            f.write("HSE06 non-self-consistent energy components (eV):\n")
            for lbl, val in zip(labels, components):
                f.write(f"  {lbl}: {float(val):.10f}\n")
            f.write(f"  TOTAL: {e_hse:.10f}\n")
        print(f"  HSE06 E = {e_hse:.6f} eV  (details: {hse_log})")

    return e_hse, vbm


def _get_natoms_change(defect_entry) -> Dict[str, int]:
    """Stoichiometry change n_i: negative = removed from bulk, positive = added."""
    defect = defect_entry.defect
    if hasattr(defect, 'element_changes') and defect.element_changes:
        return {str(el): int(n) for el, n in defect.element_changes.items()}
    # Fallback: composition difference
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
    """E_f(q,εF) = E_HSE(defect,q) + E_corr - E_HSE(bulk_SC) - Σ n_i μ_i + q(εVBM + εF)"""
    mu_sum = sum(n * chempots.get(el, 0.0) for el, n in natoms_change.items())
    return e_hse_defect + e_corr - e_hse_bulk_sc - mu_sum + charge * (vbm + fermi_level)


def _kumagai_correction(
    defect_entry,
    defect_atoms: Atoms,
    bulk_atoms: Atoms,
    dielectric: np.ndarray,
    corr_dir: str,
) -> float:
    """Kumagai (eFNV) charge correction via doped, using GPAW site potentials.

    Injects GPAW atomic electrostatic potentials into
    ``defect_entry.calculation_metadata`` so doped's correction routine can
    read them without needing a VASP OUTCAR.

    Returns correction energy in eV (0.0 for neutral defects or on failure).
    """
    os.makedirs(corr_dir, exist_ok=True)
    corr_txt = os.path.join(corr_dir, "kumagai.txt")

    if defect_entry.charge_state == 0:
        with open(corr_txt, 'w') as f:
            f.write("charge=0: no correction needed\ncorrection_energy=0.0 eV\n")
        return 0.0

    try:
        defect_pots = defect_atoms.calc.get_atomic_electrostatic_potentials().tolist()
        bulk_pots   = bulk_atoms.calc.get_atomic_electrostatic_potentials().tolist()
        if not hasattr(defect_entry, 'calculation_metadata') or \
                defect_entry.calculation_metadata is None:
            defect_entry.calculation_metadata = {}
        defect_entry.calculation_metadata['defect_site_potentials'] = defect_pots
        defect_entry.calculation_metadata['bulk_site_potentials']   = bulk_pots

        corr_result = defect_entry.get_kumagai_correction(
            dielectric=dielectric, plot=False,
        )
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
            f.write("Install pydefect and ensure DIELECTRIC_TENSOR is set.\n")
        return 0.0


def _plot_formation_energies(
    formation_data: List[Dict],
    vbm: float,
    bandgap: float,
    chempots: Dict[str, float],
    output_path: str,
) -> None:
    """Plot formation energy vs Fermi level with charge-transition-level markers."""
    ef_range = np.linspace(0.0, bandgap, 400)
    defect_groups: Dict[str, List[Dict]] = defaultdict(list)
    for entry in formation_data:
        # group by base name (strip _q+1 suffix)
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

        # charge transition levels
        cl_sorted = sorted(charge_lines, key=lambda x: x[0])
        for i in range(len(cl_sorted) - 1):
            q1, l1 = cl_sorted[i]
            q2, l2 = cl_sorted[i + 1]
            cross_idx = np.where(np.diff(np.sign(l1 - l2)))[0]
            for ci in cross_idx:
                ef_ctl = float(ef_range[ci])
                e_ctl  = float((envelope[ci] + envelope[min(ci + 1, len(envelope) - 1)]) / 2)
                ax.axvline(ef_ctl, ls='--', lw=0.7, color=color, alpha=0.5)
                ax.annotate(
                    f"ε({q1:+d}/{q2:+d})\n{ef_ctl:.2f} eV",
                    xy=(ef_ctl, e_ctl), fontsize=5.5, color=color, ha='center',
                )

    ax.axvline(0.0, color='k', lw=1.0, ls=':')
    ax.axvline(bandgap, color='k', lw=1.0, ls=':', label=f'CBM (PBE est. {bandgap:.2f} eV)')
    ax.axhline(0.0, color='gray', lw=0.6, ls='--')
    ax.set_xlabel(f"Fermi level relative to VBM (eV)  [VBM ≈ {vbm:.3f} eV abs]", fontsize=12)
    ax.set_ylabel("Formation energy (eV)", fontsize=12)
    ax.set_title("LaNiO₃ defect formation energies (HSE06, PBE geometry)", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc='best')
    ax.set_xlim(0, bandgap)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Formation energy diagram → {output_path}")


defect_type_map = {
    DefectType.Vacancy.value: "Vacancy",
    DefectType.Interstitial.value: "Interstitial",
    DefectType.Substitution.value: "Substitution"
}
symbols = ('La','Ni','O','O')

# ── Calculation parameters ────────────────────────────────────────────────────
ECUT_EV = 520
KPTS    = (1, 1, 1)
SCREEN  = 0.2*Bohr   # HSE06 screening parameter (Å⁻¹)

# Structure setup
a0 = 3.92
c0 = 12.52

magmoms = [0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6,
           0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 
           2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 
           0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6,
           0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6,
           0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6,
           0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6]

lno_atoms = crystal(['La','Ni','O','O'],[(0,0,0.363473),(0,0,0),(1/2,0,0),(0,0,0.178993)], spacegroup=139, cellpar=[a0, a0, c0, 90, 90, 90],size = (2,2,1))

lno_atoms.set_initial_magnetic_moments(magmoms)
lno_atoms.write('LNO_I4mmm.cif', format='cif')
view(lno_atoms)
extrinsic = {"P": ['Ca', 'Sr'], "Pb": ['Cu', 'Mn']}
substitution_elements = ['La', 'Ni', 'Ca', 'Sr', 'Co', 'Mn']


pbe_params = {
    "convergence": {"density":1e-8, "eigenstates":1e-10, "energy": 1e-6, "forces":1e-4},
    "eigensolver": {"name": "dav", "niter": 5},
    "kpts": {"gamma": True, "size": KPTS},
    "maxiter": 1000,
    "mixer": {"backend": "pulay", "beta": 0.25, "method": "fullspin", "nmaxold": 5, "weight": 50.0},
    "mode": {"name": "pw", "ecut": ECUT_EV},
    "nbands": "nao",
    "parallel": {"sl_auto": True, "augment_grids": True},
    "occupations": {"name": "fermi-dirac", "width": 0.01},
    "txt": "LNO_scf_pbe.log",
    "xc": "PBE",
}
hse_params = {
    "convergence": {"density": 1e-6, "eigenstates": 1e-8, "energy": 1e-4, "forces": 1e-2},
    "eigensolver": {"name": "dav", "niter": 5},
    "kpts": {"gamma": True, "size": (1, 1, 1)},
    "maxiter": 1000,
    "mixer": {"backend": "pulay", "beta": 0.25, "method": "fullspin", "nmaxold": 5, "weight": 50.0},
    "mode": {"name": "pw", "ecut": ECUT_EV},
    "nbands": "nao",
    "parallel": {"sl_auto": True, "augment_grids": True},
    "occupations": {"name": "fermi-dirac", "width": 0.01},
    "txt": "LNO_scf_hse.log",
    "xc": {"name": "HYB_GGA_XC_HSE06", "omega": SCREEN, "fraction": 0.25, "backend": "pw"},
}

# ── Dielectric tensor for Kumagai charge correction ───────────────────────────
# LaNiO3 is anisotropic.  Compute via DFPT/finite-differences and save to
# defects/dielectric_tensor.json, or set directly here.
# If the JSON exists it takes priority.
DIELECTRIC_JSON = "defects/dielectric_tensor.json"
if os.path.exists(DIELECTRIC_JSON):
    DIELECTRIC_TENSOR: Optional[np.ndarray] = np.array(json.load(open(DIELECTRIC_JSON)))
    print(f"Loaded dielectric tensor from {DIELECTRIC_JSON}")
else:
    DIELECTRIC_TENSOR = None   # correction will be skipped; set to np.eye(3)*25 as placeholder
    print("WARNING: DIELECTRIC_TENSOR not set — Kumagai correction will be skipped for "
          f"charged defects.  Save your computed 3×3 tensor to {DIELECTRIC_JSON}.")

# ── Chemical potentials (μ_i, eV per atom from competing-phase DFT) ───────────
# Fill in values after running GPAW calculations for La, La2O3, Ni, NiO, ½O2.
# μ = 0.0 placeholder → formation energies will be printed but are not physical.
CHEMPOTS_JSON = "defects/chemical_potentials.json"
_chempot_template = {
    "__instructions__": (
        "Replace 0.0 values with GPAW PBE total energy / atom for each competing "
        "phase reservoir: La→La-metal/atom, Ni→Ni-metal/atom, O→½E(O2), "
        "Ca/Sr/Co/Mn→ their standard-state elemental energies per atom."
    ),
    "La": 0.0, "Ni": 0.0, "O": 0.0,
    "Ca": 0.0, "Sr": 0.0, "Co": 0.0, "Mn": 0.0,
}
if os.path.exists(CHEMPOTS_JSON):
    _raw = json.load(open(CHEMPOTS_JSON))
    chempots: Dict[str, float] = {k: v for k, v in _raw.items() if k != "__instructions__"}
    print(f"Loaded chemical potentials from {CHEMPOTS_JSON}")
else:
    os.makedirs("defects", exist_ok=True)
    with open(CHEMPOTS_JSON, 'w') as _f:
        json.dump(_chempot_template, _f, indent=2)
    chempots = {k: v for k, v in _chempot_template.items() if k != "__instructions__"}
    print(f"WARNING: Chemical potentials not set.  Template written to {CHEMPOTS_JSON}.")
    print("         Formation energies will be inaccurate until you fill in real μ values.")

# Run relaxation
lno_relax_atoms = relax(lno_atoms,
                      calculator_params=pbe_params,
                      fmax=0.01,
                      fixcell=False,
                      logname='LNO_opt.log',
                      trajname='LNO_opt.traj',
                      gpwname='LNO_opt.gpw')
lno_atoms_hse = lno_relax_atoms.copy()
lno_atoms_hse.calc = GPAW(**hse_params)
E_lno_hse = lno_atoms_hse.get_potential_energy()
lno_pos = lno_atoms_hse.get_positions()
lno_relax_cell = lno_atoms_hse.get_cell()
print("Relaxed cell:")
print(lno_relax_cell)
print("PBE Potential energy (eV):", lno_relax_atoms.get_potential_energy())
print("HSE06 Potential energy (eV):", E_lno_hse)
print("All positions (Angstrom):")
print(lno_pos)

defect_gen = DefectsGenerator(
    lno_relax_atoms,                                # use PBE-relaxed structure
    extrinsic=extrinsic,
    interstitial_gen_kwargs=True,
    interstitial_elements=['O'],
    vacancy_gen_kwargs=True,
    vacancy_elements=['La', 'Ni', 'O'],
    substitution_gen_kwargs=False,
    substitution_elements=substitution_elements,
    generate_supercell=True,
    supercell_gen_kwargs={'force_diagonal': True, 'min_image_distance': 10},
)

# ── Bulk supercell reference (same supercell as defects, pristine) ────────────
# Required for E_f = E_HSE(defect) - E_HSE(bulk_SC) - Σ n_i μ_i + ...
print("\n" + "=" * 60)
print("Setting up bulk supercell reference calculation")
print("=" * 60)
os.makedirs("defects/bulk_supercell", exist_ok=True)
bulk_sc_pmg   = defect_gen.bulk_supercell          # pymatgen Structure
bulk_sc_atoms = AseAtomsAdaptor.get_atoms(bulk_sc_pmg)
_assign_magmoms(bulk_sc_atoms)

bulk_sc_pbe_params = pbe_params.copy()
bulk_sc_pbe_params["txt"] = "defects/bulk_supercell/pbe_scf.log"

bulk_sc_atoms = relax(
    bulk_sc_atoms,
    calculator_params=bulk_sc_pbe_params,
    fmax=0.01,
    fixcell=True,                                   # cell fixed — same as defect SCs
    logname="defects/bulk_supercell/pbe_relax.log",
    trajname="defects/bulk_supercell/pbe_relax.traj",
    gpwname="defects/bulk_supercell/pbe_relax.gpw",
)

E_bulk_sc_hse, vbm_bulk_sc = _hse_singlepoint(
    bulk_sc_atoms,
    base_dir="defects/bulk_supercell",
    hse_params_template=hse_params,
)
print(f"Bulk SC  E_HSE = {E_bulk_sc_hse:.4f} eV  VBM = {vbm_bulk_sc:.4f} eV")

with open("defects/bulk_supercell/reference.txt", "w") as _f:
    _f.write(f"Bulk supercell reference\n")
    _f.write(f"  N atoms:    {len(bulk_sc_atoms)}\n")
    _f.write(f"  E_HSE (eV): {E_bulk_sc_hse:.6f}\n")
    _f.write(f"  VBM (eV):   {vbm_bulk_sc:.6f}\n")

# ── Defect loop ────────────────────────────────────────────────────────────────
os.makedirs("defects", exist_ok=True)
summary_path  = "defects/defects_summary.txt"
formation_data: List[Dict] = []

with open(summary_path, "w") as sf:
    sf.write(f"Defect summary  generated {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    sf.write(f"{'Name':<40} {'Type':<15} {'Charge':>6}  "
             f"{'E_HSE (eV)':>14}  {'E_corr (eV)':>12}  {'Ef@VBM (eV)':>12}\n")
    sf.write("-" * 105 + "\n")

for defect_entry in defect_gen.defect_entries.values():
    defect_name      = defect_entry.name
    defect_type_val  = defect_entry.defect.defect_type.value
    defect_type      = defect_type_map.get(defect_type_val, "Unknown")
    charge           = defect_entry.charge_state
    sc               = defect_entry.defect_supercell
    frac_coords      = defect_entry.sc_defect_frac_coords
    natoms_change    = _get_natoms_change(defect_entry)

    defect_dir = os.path.join("defects", f"{defect_name}_q{charge:+d}")
    os.makedirs(defect_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Defect: {defect_name}  charge: {charge:+d}  type: {defect_type}")
    print(f"{'='*60}")

    # ── info.txt: human-readable metadata ─────────────────────────────────────
    cell_params = sc.lattice.abc + sc.lattice.angles
    with open(os.path.join(defect_dir, "info.txt"), "w") as f:
        f.write(f"Defect:           {defect_name}\n")
        f.write(f"Type:             {defect_type}\n")
        f.write(f"Charge state:     {charge:+d}\n")
        f.write(f"Frac coords:      "
                f"{frac_coords[0]:.6f}  {frac_coords[1]:.6f}  {frac_coords[2]:.6f}\n")
        f.write(f"Supercell sites:  {len(sc)}\n")
        f.write(f"  a={cell_params[0]:.4f}  b={cell_params[1]:.4f}  c={cell_params[2]:.4f} Å\n")
        f.write(f"  α={cell_params[3]:.2f}  β={cell_params[4]:.2f}  γ={cell_params[5]:.2f}°\n")
        f.write(f"Atom count change: {natoms_change}\n")

    # ── ShakeNBreak distortions ────────────────────────────────────────────────
    snb = Distortions(defect_entry)
    distorted_dict, distortion_meta = snb.apply_distortions()

    with open(os.path.join(defect_dir, "distortions.json"), "w") as f:
        json.dump(distortion_meta, f, indent=2, default=str)

    # Navigate the nested SnB output:
    # distorted_dict[defect_name]['charges'][charge]['structures']
    #   -> {'Unperturbed': Structure, 'distortions': {key: Structure}}
    charge_block = (
        distorted_dict
        .get(defect_name, {})
        .get('charges', {})
        .get(charge, {})
    )
    structs_block = charge_block.get('structures', {})
    unperturbed   = structs_block.get('Unperturbed')
    dist_structs  = structs_block.get('distortions', {})

    all_structs: Dict[str, Any] = {}
    if unperturbed is not None:
        all_structs['Unperturbed'] = unperturbed
    all_structs.update(dist_structs)

    if not all_structs:
        print(f"  WARNING: no structures found in SnB output for {defect_name} "
              f"q={charge:+d}.  Skipping.")
        continue

    # ── PBE position-only relaxation for each distortion ─────────────────────
    # Electronic convergence: defects/{name}/Unperturbed/scf.log   (GPAW txt)
    # Ionic steps:            defects/{name}/Unperturbed/relax.log  (BFGS log)
    # Trajectory:             defects/{name}/Unperturbed/relax.traj
    distortion_energies: Dict[str, float] = {}
    dist_base = defect_dir                         # distortion dirs live inside defect_dir

    for dist_key, pmg_struct in all_structs.items():
        safe_key = dist_key.replace('/', '_').replace(' ', '_')
        dist_dir = os.path.join(dist_base, safe_key)
        os.makedirs(dist_dir, exist_ok=True)

        energy_cache = os.path.join(dist_dir, "energy.txt")
        if os.path.exists(energy_cache):
            with open(energy_cache) as _ef:
                distortion_energies[dist_key] = float(_ef.read().strip())
            print(f"  {dist_key:40s}  E_PBE = {distortion_energies[dist_key]:.4f} eV  [cached]")
            continue

        def_atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
        _assign_magmoms(def_atoms)

        d_pbe_p = pbe_params.copy()
        d_pbe_p["charge"] = charge          # uniform background charge for charged SC
        d_pbe_p["txt"]    = os.path.join(dist_dir, "scf.log")

        def_atoms = relax(
            def_atoms,
            calculator_params=d_pbe_p,
            fmax=0.05,                      # looser fmax for distortion screening
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

    # ── distortion_energies.txt ────────────────────────────────────────────────
    E_unperturbed = distortion_energies.get('Unperturbed')
    with open(os.path.join(defect_dir, "distortion_energies.txt"), "w") as f:
        f.write(f"# Distortion screening for {defect_name} q={charge:+d}\n")
        f.write(f"# {'Distortion':<40} {'E_PBE (eV)':>14}  {'ΔE vs Unperturbed (eV)':>24}\n")
        for dk, e in sorted(distortion_energies.items(), key=lambda x: x[1]):
            dE = (e - E_unperturbed) if E_unperturbed is not None else float('nan')
            f.write(f"  {dk:<40} {e:14.6f}  {dE:+24.6f}\n")

    # ── best_distortion.txt: lowest-energy distortion ─────────────────────────
    best_key = min(distortion_energies, key=distortion_energies.__getitem__)
    best_e   = distortion_energies[best_key]
    dE_best  = (best_e - E_unperturbed) if E_unperturbed is not None else float('nan')
    print(f"  Best distortion: {best_key}  ΔE = {dE_best:+.4f} eV")
    with open(os.path.join(defect_dir, "best_distortion.txt"), "w") as f:
        f.write(f"Best distortion:   {best_key}\n")
        f.write(f"E_PBE (eV):        {best_e:.6f}\n")
        f.write(f"ΔE vs Unperturbed: {dE_best:+.6f} eV\n")

    # ── Reload best-distortion relaxed Atoms with its GPAW calculator ─────────
    best_safe = best_key.replace('/', '_').replace(' ', '_')
    best_gpw  = os.path.join(dist_base, best_safe, "pbe_relax.gpw")
    best_atoms, _ = restart(best_gpw)

    # ── HSE single-point on the best distortion ───────────────────────────────
    # Electronic relaxation log: defects/{name}/hse/hse_scf.log
    # Energy result:             defects/{name}/hse/hse_energy.txt
    hse_dir = os.path.join(defect_dir, "hse")
    e_hse_defect, _ = _hse_singlepoint(
        best_atoms,
        base_dir=hse_dir,
        hse_params_template=hse_params,
    )
    print(f"  E_HSE = {e_hse_defect:.6f} eV")

    # ── Kumagai charge correction ──────────────────────────────────────────────
    bulk_gpw = "defects/bulk_supercell/pbe_relax.gpw"
    bulk_atoms_corr, _ = restart(bulk_gpw)
    e_corr = 0.0
    if DIELECTRIC_TENSOR is not None:
        e_corr = _kumagai_correction(
            defect_entry=defect_entry,
            defect_atoms=best_atoms,
            bulk_atoms=bulk_atoms_corr,
            dielectric=DIELECTRIC_TENSOR,
            corr_dir=os.path.join(defect_dir, "charge_correction"),
        )
    else:
        os.makedirs(os.path.join(defect_dir, "charge_correction"), exist_ok=True)
        with open(os.path.join(defect_dir, "charge_correction", "kumagai.txt"), 'w') as f:
            f.write(f"Correction skipped: DIELECTRIC_TENSOR is None.\n"
                    f"Set it in defects/dielectric_tensor.json and rerun.\n"
                    f"correction_energy = 0.0 eV (uncorrected)\n")

    # ── Formation energy at VBM (εF = 0) ──────────────────────────────────────
    ef_at_vbm = _formation_energy(
        e_hse_defect, e_corr, E_bulk_sc_hse, natoms_change,
        chempots, charge, vbm_bulk_sc, fermi_level=0.0,
    )

    # ── Accumulate for the plot ────────────────────────────────────────────────
    formation_data.append({
        'label':         f"{defect_name}_q{charge:+d}",
        'charge':        charge,
        'e_hse':         e_hse_defect,
        'e_corr':        e_corr,
        'e_bulk_sc':     E_bulk_sc_hse,
        'natoms_change': natoms_change,
        'vbm':           vbm_bulk_sc,
    })

    # ── Append to running summary ──────────────────────────────────────────────
    with open(summary_path, "a") as sf:
        sf.write(f"{defect_name:<40} {defect_type:<15} {charge:>+6d}  "
                 f"{e_hse_defect:>14.4f}  {e_corr:>+12.4f}  {ef_at_vbm:>12.4f}\n")

    print(f"[DONE] {defect_name} q={charge:+d}  "
          f"E_HSE={e_hse_defect:.4f} eV  E_corr={e_corr:+.4f} eV  "
          f"E_f(VBM)={ef_at_vbm:.4f} eV")

# ── Formation energy diagram ───────────────────────────────────────────────────
if formation_data:
    # Estimate PBE bandgap from bulk SC (proxy only; HSE gap will differ)
    try:
        _bulk_ref, _ = restart("defects/bulk_supercell/pbe_relax.gpw")
        _, _lumo = _bulk_ref.calc.get_homo_lumo()
        bandgap_est = max(float(_lumo) - vbm_bulk_sc, 0.3)
    except Exception:
        bandgap_est = 1.0
        print("Warning: could not extract bandgap from bulk SC; using 1.0 eV placeholder.")

    _plot_formation_energies(
        formation_data=formation_data,
        vbm=vbm_bulk_sc,
        bandgap=bandgap_est,
        chempots=chempots,
        output_path="defects/formation_energy_diagram.png",
    )

with open(summary_path, "a") as sf:
    sf.write(f"\nCompleted: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
print(f"\nAll done.  Summary → {summary_path}")

