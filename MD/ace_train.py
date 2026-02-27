"""
ACE Training Data Generator — GPAW → pacemaker
===============================================
Edit the parameters below, then run:  python ace_train.py

Output: <NAME>_data.pckl.gzip  (pandas DataFrame for pacemaker)

After this script finishes:
  pacemaker -t                      # generate input.yaml interactively
  # set filename to <NAME>_data.pckl.gzip in input.yaml
  pacemaker input.yaml              # fit the potential
"""

import numpy as np
import pandas as pd
from ase import units
from ase.build import bulk, make_supercell
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from ase.io import read
from gpaw import GPAW, PW, FermiDirac

# =============================================================================
# ── Parameters — edit here ───────────────────────────────────────────────────
# =============================================================================

ATOMS = bulk('LiF', crystalstructure='rocksalt', a=4.03, cubic=True)
# ATOMS = read('my_structure.cif')

NAME = 'LiF'

RELAX    = True   # relax structure first
RUN_AIMD = True   # include AIMD frames

# GPAW
ECUT_EV     = 400
XC          = 'HSE06'
SMEAR_WIDTH = 0.01
KPTS_RELAX  = (1,1,1)
KPTS_SUPER  = (2, 2, 2)
FMAX        = 0.02         # eV/Å

# Rattled supercell configs
N_RATTLE    = 30
RATTLE_STD  = 0.10         # Å
RATTLE_SEED = 42

# Strained unit-cell configs
STRAIN_AMPS = [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]

# AIMD
AIMD_TEMP       = 1200     # K
AIMD_TIMESTEP   = 1.0      # fs
AIMD_TDAMP      = 50       # fs
AIMD_STEPS      = 300
AIMD_SAMPLE_INT = 10       # collect every N steps

# =============================================================================

def make_gpaw(kpts, txt, convergence=1e-5):
    return GPAW(
        mode        = PW(ECUT_EV),
        xc          = XC,
        kpts        = kpts,
        occupations = FermiDirac(SMEAR_WIDTH),
        symmetry    = 'off',
        txt         = txt,
        convergence = {'energy': convergence},
        maxiter     = 500,
        mixer       = {'backend': 'pulay', 'method': 'separate',
                       'beta': 0.02, 'nmaxold': 10, 'weight': 100},
    )

def compute(atoms, kpts, txt, label=''):
    """Run GPAW single-point, return (atoms_clean, energy, forces)."""
    atoms = atoms.copy()
    atoms.calc = make_gpaw(kpts, txt)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"  [{label}]  E = {energy/len(atoms):.5f} eV/atom  "
          f"|F|max = {np.abs(forces).max():.4f} eV/Å")
    atoms.calc = None
    return atoms, energy, forces

# ── Collect training structures ───────────────────────────────────────────────

records = []   # list of dicts → becomes the DataFrame

def add(atoms, energy, forces):
    records.append({
        'ase_atoms':        atoms,
        'energy':           energy,
        'forces':           forces,
        'energy_corrected': energy,   # no reference correction; adjust if needed
    })

# Step 1: relax
print("=" * 60)
if RELAX:
    print("Step 1: Relaxing structure")
    atoms = ATOMS.copy()
    atoms.calc = make_gpaw(KPTS_RELAX, f'{NAME}_relax.log')
    BFGS(atoms, logfile=f'{NAME}_relax_opt.log').run(fmax=FMAX, steps=200)
    atoms.calc = None
    relaxed = atoms
else:
    relaxed = ATOMS.copy()
print(f"  E = {relaxed.get_potential_energy()/len(relaxed):.5f} eV/atom" if RELAX else "  Skipped")

# Step 2a: rattled supercells
print("\nStep 2a: Rattled supercell configs")
supercell = make_supercell(relaxed, 2 * np.eye(3, dtype=int))
rng = np.random.default_rng(RATTLE_SEED)
for i in range(N_RATTLE):
    sc = supercell.copy()
    sc.positions += rng.normal(0, RATTLE_STD, (len(sc), 3))
    a, e, f = compute(sc, KPTS_SUPER, f'{NAME}_rattle_{i:03d}.log',
                      label=f"rattle {i+1}/{N_RATTLE}")
    add(a, e, f)

# Step 2b: strained unit cells
print("\nStep 2b: Strained unit-cell configs")
cell0 = relaxed.get_cell().copy()

for eps in STRAIN_AMPS:                              # volumetric
    sc = relaxed.copy()
    sc.set_cell((1 + eps) * cell0, scale_atoms=True)
    a, e, f = compute(sc, KPTS_RELAX, f'{NAME}_vol_{eps:+.3f}.log',
                      label=f"vol ε={eps:+.2f}")
    add(a, e, f)

for i, j in [(0,1), (0,2), (1,2)]:                  # shear
    for eps in STRAIN_AMPS:
        sc = relaxed.copy()
        cell = cell0.copy()
        cell[i] += eps * cell0[j]
        sc.set_cell(cell, scale_atoms=True)
        a, e, f = compute(sc, KPTS_RELAX, f'{NAME}_shear{i}{j}_{eps:+.3f}.log',
                          label=f"shear({i}{j}) ε={eps:+.2f}")
        add(a, e, f)

# Step 2c: AIMD
if RUN_AIMD:
    print("\nStep 2c: Short AIMD")
    md_atoms = make_supercell(relaxed, 2 * np.eye(3, dtype=int))
    md_atoms.calc = make_gpaw(KPTS_SUPER, f'{NAME}_aimd.log', convergence=1e-4)

    MaxwellBoltzmannDistribution(md_atoms, temperature_K=AIMD_TEMP)
    Stationary(md_atoms)
    ZeroRotation(md_atoms)

    dyn = NoseHooverChainNVT(
        md_atoms,
        timestep      = AIMD_TIMESTEP * units.fs,
        temperature_K = AIMD_TEMP,
        tdamp         = AIMD_TDAMP * units.fs,
        trajectory    = f'{NAME}_aimd.traj',
        logfile       = f'{NAME}_aimd.log',
    )

    def collect():
        frame = md_atoms.copy()
        frame.calc = None
        energy = md_atoms.get_potential_energy()
        forces = md_atoms.get_forces()
        add(frame, energy, forces)

    dyn.attach(collect, interval=AIMD_SAMPLE_INT)
    dyn.run(AIMD_STEPS)

# ── Save DataFrame ────────────────────────────────────────────────────────────

out = f'{NAME}_data.pckl.gzip'
df = pd.DataFrame(records)
df.to_pickle(out, compression='gzip', protocol=4)

print(f"\nDone: {len(df)} frames → {out}")
print(f"\nNext steps:")
print(f"  pacemaker -t        # generates input.yaml, set filename to {out}")
print(f"  pacemaker input.yaml")
