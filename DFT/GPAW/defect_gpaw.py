import os
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict
from ase.units import Bohr

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import read, write as ase_write
from ase.optimize import BFGS
from ase.visualize import view
from doped.generation import DefectsGenerator
from gpaw import GPAW, restart
from pymatgen.analysis.defects.core import DefectType
from shakenbreak.input import Distortions
from ase.spacegroup import crystal

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
    if os.path.exists(done_file):
        relaxed = read(done_file, index=0)
        if len(relaxed) != len(orig_atoms):
            print(f"Cached structure has {len(relaxed)} atoms but current structure has {len(orig_atoms)}. Ignoring cache.")
            relaxed = None
        else:
            print(f"Loaded converged structure from {done_file}")
    else:
        relaxed = None
    if relaxed is None:
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


defect_type_map = {
    DefectType.Vacancy.value: "Vacancy",
    DefectType.Interstitial.value: "Interstitial",
    DefectType.Substitution.value: "Substitution"
}
symbols = ('La','Ni','O','O')

# ── Calculation parameters ────────────────────────────────────────────────────
ECUT_EV = 520
KPTS    = (2, 2, 2)
SCREEN  = 0.2*Bohr   # HSE06 screening parameter (Å⁻¹)

# Structure setup
a0 = 3.92
c0 = 12.52

magmoms = [0.6, -0.6, 0.6, -0.6, 2.0, -2.0, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6]
lno_atoms = crystal(['La','Ni','O','O'],[(0,0,0.363473),(0,0,0),(1/2,0,0),(0,0,0.178993)], spacegroup=139, cellpar=[a0, a0, c0, 90, 90, 90])

lno_atoms.set_initial_magnetic_moments(magmoms)
lno_atoms.write('LNO_I4mmm.cif', format='cif')
view(lno_atoms, repeat=(2, 2, 1))
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
    "occupations": {"name": "fermi-dirac", "width": 0.01},
    "txt": "pbe_relax.log",
    "xc": "PBE",
}
hse_params = {
    "convergence": {"density": 1e-4, "eigenstates": 1e-4, "energy": 1e-3},
    "eigensolver": {"name": "dav", "niter": 5},
    "kpts": {"gamma": True, "size": (1, 1, 1)},
    "maxiter": 300,
    "mixer": {"backend": "pulay", "beta": 0.25, "method": "separate", "nmaxold": 5, "weight": 50.0},
    "mode": {"name": "pw", "ecut": ECUT_EV},
    "nbands": "nao",
    "occupations": {"name": "fermi-dirac", "width": 0.01},
    "txt": "hse_relax.log",
    "xc": {"name": "HYB_GGA_XC_HSE06", "omega": SCREEN, "fraction": 0.25, "backend": "pw"},
}
# Run relaxation
lno_relax_atoms = relax(lno_atoms,
                      calculator_params=pbe_params,
                      fmax=0.01,
                      fixcell=False,
                      logname='LNO_opt.log',
                      trajname='LNO_opt.traj',
                      gpwname='LNO_opt.gpw')
lno_relax_hse = relax(lno_relax_atoms.copy(), hse_params,
                      fmax=0.01, fixcell=False,
                      logname='LNO_hse_opt.log',
                      trajname='LNO_hse_opt.traj',
                      gpwname='LNO_hse_rlx.gpw')
E_lno_hse = lno_relax_hse.get_potential_energy()
lno_pos = lno_relax_hse.get_positions()
lno_relax_cell = lno_relax_hse.get_cell()
print("Relaxed cell:")
print(lno_relax_cell)
print("PBE Potential energy (eV):", lno_relax_atoms.get_potential_energy())
print("HSE06 Potential energy (eV):", E_lno_hse)
print("All positions (Angstrom):")
print(lno_pos)

defect_gen = DefectsGenerator(
    lno_atoms,
    extrinsic=extrinsic,
    interstitial_gen_kwargs=True,
    interstitial_elements=['O'],
    vacancy_gen_kwargs=True,
    vacancy_elements=['La', 'Ni', 'O'],
    substitution_gen_kwargs=False,
    substitution_elements=substitution_elements,
    generate_supercell=False,
    supercell_gen_kwargs={'force_diagonal': True},
)


for defect_entry in defect_gen.defect_entries.values():
    defect_name = defect_entry.name
    defect_type_value = defect_entry.defect.defect_type.value
    defect_type = defect_type_map.get(defect_type_value, "Unknown")
    charge = defect_entry.charge_state
    sc = defect_entry.defect_supercell
    frac_coords = defect_entry.sc_defect_frac_coords
    defect_dict, defect_metadata = Distortions(defect_entry)
    defect_elements = list(OrderedDict.fromkeys([site.species.elements[0].symbol for site in sc]))
    def_atoms = Atoms(symbols=defect_elements,
                      scaled_positions=frac_coords,
                      pbc=True)
    def_atoms.set_initial_magnetic_moments(magmoms)
    def_atoms.write(f'{defect_name}_{defect_type}_q{charge}.cif', format='cif')
    def_atoms.calc = GPAW(**pbe_params)

