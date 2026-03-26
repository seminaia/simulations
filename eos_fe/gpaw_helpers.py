"""
Shared GPAW helpers for TiO2 defect workflows.

Used by: defect_tio2.py, comp_phases.py
"""

import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import BFGS
from ase.units import Bohr
from gpaw import GPAW, restart

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
#  Magnetic moments
# ---------------------------------------------------------------------------


def assign_magmoms(atoms: Atoms, MAGMOM_MAP: Dict[str, float]) -> None:
    counters: Dict[str, int] = defaultdict(int)
    moms: List[float] = []
    for sym in atoms.get_chemical_symbols():
        mag = MAGMOM_MAP.get(sym, 0.1)
        sign = 1 if counters[sym] % 2 == 0 else -1
        moms.append(sign * mag)
        counters[sym] += 1
    atoms.set_initial_magnetic_moments(moms)


# ---------------------------------------------------------------------------
#  Relaxation
# ---------------------------------------------------------------------------

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
    struct_out = gpwname.replace('.gpw', '.traj')

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
    print(f"Relaxation mode: {'fixed cell' if fixcell else 'variable cell'}  fmax={fmax} eV/A")
    bfgs = BFGS(opt_atoms, logfile=logname, trajectory=trajname)

    def _log():
        raw = opt_atoms.atoms if isinstance(opt_atoms, FrechetCellFilter) else opt_atoms
        fmax_cur = float(np.max(np.linalg.norm(opt_atoms.get_forces(), axis=1)))
        epot = raw.get_potential_energy() / len(raw)
        print(f"[{datetime.now():%H:%M:%S}]  relax step {bfgs.nsteps:4d}  "
              f"Epot={epot:.4f} eV/atom  fmax={fmax_cur:.4f} eV/A")

    bfgs.attach(_log, interval=1)
    bfgs.run(fmax=fmax, steps=500)

    if isinstance(opt_atoms, FrechetCellFilter):
        opt_atoms = opt_atoms.atoms

    try:
        opt_atoms.calc.write(gpwname, mode='all')
        print(f"Saved to {gpwname}")
    except Exception as e:
        print(f"Warning: could not save state: {e}")

    ase_write(struct_out, opt_atoms)
    print(f"Converged structure saved to {struct_out}")
    forces = opt_atoms.get_forces()
    print(f"Max force: {np.max(np.linalg.norm(forces, axis=1)):.6f} eV/A")

    orig_atoms.set_cell(opt_atoms.get_cell(), scale_atoms=False)
    orig_atoms.set_positions(opt_atoms.get_positions())
    cell = opt_atoms.cell.cellpar()
    print(f"  Lattice: a={cell[0]:.4f} b={cell[1]:.4f} c={cell[2]:.4f} A, "
          f"a={cell[3]:.2f} b={cell[4]:.2f} g={cell[5]:.2f} deg  Volume={opt_atoms.get_volume():.2f} A^3")
    return opt_atoms


# ---------------------------------------------------------------------------
#  Calculator parameter templates
# ---------------------------------------------------------------------------

ECUT_EV = 520
SCREEN = 0.2 * Bohr  # HSE06 screening (A^-1)

BASE_PARAMS = {
    "eigensolver": {"name": "rmm-diis", "niter": 5},
    "maxiter": 1000,
    "mixer": {"backend": "pulay", "beta": 0.05, "method": "difference", "nmaxold": 5, "weight": 50.0},
    "mode": {"name": "pw", "ecut": ECUT_EV},
    "nbands": "nao",
    "parallel": {"sl_auto": True, "augment_grids": True},
    "occupations": {"name": "fermi-dirac", "width": 0.1},
}


def pbe_params(**overrides) -> Dict[str, Any]:
    p = BASE_PARAMS.copy()
    p["convergence"] = {"density": 1e-8, "eigenstates": 1e-10, "energy": 1e-6, "forces": 1e-4}
    p["xc"] = "PBE"
    p["kpts"] = {"gamma": True, "density": 2.5}
    p.update(overrides)
    return p


def mgga_params(**overrides) -> Dict[str, Any]:
    p = BASE_PARAMS.copy()
    p["convergence"] = {"density": 1e-6, "eigenstates": 1e-8, "energy": 1e-4, "forces": 1e-2}
    p["xc"] = "MGGA_X_R2SCAN+MGGA_C_R2SCAN"
    p["kpts"] = {"gamma": True, "density": 1.5}
    p.update(overrides)
    return p


def hse_params(**overrides) -> Dict[str, Any]:
    p = BASE_PARAMS.copy()
    p["convergence"] = {"density": 1e-6, "eigenstates": 1e-8, "energy": 1e-4, "forces": 1e-2}
    p["xc"] = {"name": "HYB_GGA_XC_HSE06", "omega": SCREEN, "fraction": 0.25, "backend": "pw"}
    p["kpts"] = {"gamma": True, "density": 1.5}
    p["parallel"] = {"sl_auto": True, "augment_grids": True, "band": 2}
    p.update(overrides)
    return p
