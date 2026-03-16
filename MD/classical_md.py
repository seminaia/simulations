"""
Classical MD for LiF·BeF2 using LAMMPS via ASE
================================================
Force field : Buckingham + Ewald (Coulomb)
Potentials  : Tosi-Fumi (1964) for LiF, extended to Be-F
Workflow    : load equilibrated structure → NVT production run
              → RDF, MSD, VDOS analysis → plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ase import units, Atoms
from ase.io import read, write as ase_write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md import MDLogger
from ase.calculators.lammpsrun import LAMMPS
from gpaw import GPAW, restart
from ase.optimize import BFGS
from ase.units import Bohr
from ase.spacegroup import crystal
from ase.build.tools import stack
from ase.build import bulk
from typing import Dict, Any
# ── Parameters ────────────────────────────────────────────────────────────────
TEMPERATURE  = 1200        # K
TIMESTEP_FS  = 1.0         # fs
TDAMP_FS     = 50          # thermostat damping (fs)
N_STEPS      = 50_000      # production steps  (25 ps)
LOG_INTERVAL = 10          # steps between log entries
ECUT_EV      = 500         # plane-wave cutoff (eV)
KPTS         = (2, 2, 2)   # Γ-point only — speed priority
SCREEN       = 0.25 * Bohr # HSE06 range-separation parameter

# ── File names ────────────────────────────────────────────────────────────────
LIF_GPW_FILE   = "LiF_aimd_relax.gpw"
LIF_RLX_LOG    = "LiF_aimd_relax_opt.log"
BEF2_GPW_FILE  = "BeF2_aimd_relax.gpw"
BEF2_RLX_LOG   = "BeF2_aimd_relax_opt.log"

MIX_TRAJ_EQUIL = "mix_aimd_equil.traj"
MIX_LOG_EQUIL  = "mix_aimd_equil.log"
MIX_PLOT_FILE  = "mix_aimd_results.png"

def relax(
    atoms: Atoms,
    calculator_params: Dict[str, Any],
    fmax: float = 0.01,
    fixcell: bool = True,
    logname: str = 'opt.log',
    trajname: str | None = None,
    gpwname: str = 'rlx.gpw',
) -> Atoms:
    orig_atoms = atoms  # keep reference so we can update in-place at the end

    # Fast restart: if a converged structure was saved previously, load it directly
    done_file = gpwname.replace('.gpw', '_relaxed.traj')
    if os.path.exists(done_file):
        relaxed = read(done_file, index=0)
        print(f"Loaded converged structure from {done_file}")
    else:
        if os.path.exists(gpwname) and os.path.getsize(gpwname) > 100:
            try:
                atoms, calc = restart(gpwname, txt=calculator_params.get("txt", "gpaw.log"))
                atoms.calc = calc
                print(f"Restarted from {gpwname}")
            except Exception as e:
                print(f"Restart failed ({e}), starting fresh.")
                atoms.calc = GPAW(**calculator_params)
        else:
            atoms.calc = GPAW(**calculator_params)
            print("Starting fresh calculation")

        opt_atoms = atoms if fixcell else FrechetCellFilter(atoms)
        print(f"Relaxation mode: {'fixed cell' if fixcell else 'variable cell'}  fmax={fmax} eV/Å")

        BFGS(opt_atoms, logfile=logname, trajectory=trajname).run(fmax=fmax, steps=500)

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

    # Update the original atoms object's cell and positions in-place
    orig_atoms.set_cell(relaxed.get_cell(), scale_atoms=False)
    orig_atoms.set_positions(relaxed.get_positions())

    cell = relaxed.cell.cellpar()
    print(f"  Lattice: a={cell[0]:.4f} b={cell[1]:.4f} c={cell[2]:.4f} Å  "
          f"α={cell[3]:.2f} β={cell[4]:.2f} γ={cell[5]:.2f}°")
    return relaxed


pbe_params = {
    "convergence": {"density": 1e-8, "eigenstates": 1e-10, "energy": 1e-6, "forces": 1e-6},
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
    "convergence": {"density": 1e-8, "eigenstates": 1e-10, "energy": 1e-6, "forces": 1e-6},
    "eigensolver": {"name": "dav", "niter": 5},
    "kpts": {"gamma": True, "size": (1,1,1)},  
    "maxiter": 1000,
    "mixer": {"backend": "pulay", "beta": 0.25, "method": "fullspin", "nmaxold": 5, "weight": 50.0},
    "mode": {"name": "pw", "ecut": ECUT_EV},
    "nbands": "nao",
    "occupations": {"name": "fermi-dirac", "width": 0.01},
    "txt": "hse_relax.log",
    "xc":{"name":"HYB_GGA_XC_HSE06",
        "omega": SCREEN,
        "fraction": 0.25, 
        "backend": "pw"},
}

# ── Load equilibrated structure ───────────────────────────────────────────────
lif_atoms  = bulk('LiF',
                  crystalstructure='rocksalt', 
                  a=3.9747, cubic=True)
lif_atoms.set_initial_charges([1.0, -1.0, 1.0, -1.0,
                               1.0, -1.0, 1.0, -1.0])
bef2_atoms = crystal('BeF2',
                     spacegroup=152, 
                     cellpar=[4.73, 4.73, 5.18, 90, 90, 120], 
                     basis=[(0.5,0,0.33),(0.41,0.28,0.22)])
bef2_atoms.set_initial_charges([2.0, 2.0, 2.0, 2.0,
                                -1.0, -1.0, -1.0, -1.0,
                                -1.0, -1.0, -1.0, -1.0])
lif_rlx  = relax(lif_atoms,  pbe_params, fmax=0.01, fixcell=False, logname=LIF_RLX_LOG,  gpwname=LIF_GPW_FILE)
bef2_rlx = relax(bef2_atoms, pbe_params, fmax=0.01, fixcell=False, logname=BEF2_RLX_LOG, gpwname=BEF2_GPW_FILE)
rlx = stack(lif_rlx, bef2_rlx, maxstrain=1, distance=2.5)
lif_cell  = lif_rlx.cell.cellpar()
bef2_cell = bef2_rlx.cell.cellpar()
print(f"LiF  : {len(lif_rlx)} atoms  a={lif_cell[0]:.4f} b={lif_cell[1]:.4f} c={lif_cell[2]:.4f} Å"
      f"                             α={lif_cell[3]:.2f} β={lif_cell[4]:.2f} γ={lif_cell[5]:.2f}°")
print(f"BeF2 : {len(bef2_rlx)} atoms  a={bef2_cell[0]:.4f} b={bef2_cell[1]:.4f} c={bef2_cell[2]:.4f} Å  "
      f"                              α={bef2_cell[3]:.2f} β={bef2_cell[4]:.2f} γ={bef2_cell[5]:.2f}°")

# ── Cutoff (must be < L/2 for all box dimensions) ────────────────────────────
cell   = rlx.get_cell()
L_min  = min(cell[0][0], cell[1][1], cell[2][2])
cutoff = round(L_min / 2 - 0.5, 1)
print(f"  Cutoff   : {cutoff} Å  (L_min/2 = {L_min/2:.2f} Å)")

# ── Buckingham pair coefficients ──────────────────────────────────────────────
#  Pair   A (eV)     ρ (Å)     C (eV·Å⁶)
#  Li-Li   0.0       1.0        0.0      cation-cation: Coulomb only
#  Li-Be   0.0       1.0        0.0
#  Li-F  593.72    0.26310      0.0
#  Be-Be   0.0       1.0        0.0
#  Be-F  1389.47   0.23604      0.0
#  F-F   1127.70   0.27533     14.835
#pair_coeff = [
#    '1 1    0.0      1.0      0.0  ',   # Li–Li
#    '1 2    0.0      1.0      0.0  ',   # Li–Be
#    '1 3  593.72   0.26310    0.0  ',   # Li–F
#    '2 2    0.0      1.0      0.0  ',   # Be–Be
#    '2 3 1389.47   0.23604    0.0  ',   # Be–F
#    '3 3 1127.70   0.27533   14.835',   # F–F
#]

#calc = LAMMPS(
#    specorder   = ['Li', 'Be', 'F'],
#    atom_style  = 'charge',
#    pair_style  = f'born/coul/long {cutoff}',
#    kspace_style= 'ewald 1.0e-5',
#    pair_coeff  = pair_coeff,
#)

calc = GPAW(**hse_params)
rlx.calc = calc

# ── Reinitialise velocities at target temperature ─────────────────────────────
MaxwellBoltzmannDistribution(rlx, temperature_K=TEMPERATURE)
Stationary(rlx)
ZeroRotation(rlx)

# ── NVT thermostat (Nosé-Hoover chain) ───────────────────────────────────────
dyn = NoseHooverChainNVT(
    rlx,
    timestep    = TIMESTEP_FS * units.fs,
    temperature_K = TEMPERATURE,
    tdamp       = TDAMP_FS * units.fs,
    trajectory  = MIX_TRAJ_EQUIL,
    logfile     = MIX_LOG_EQUIL,
)
dyn.attach(MDLogger(dyn, rlx, "classical_md.log"), interval=LOG_INTERVAL)

# ── Accumulators ──────────────────────────────────────────────────────────────
time_ps, epot_list, ekin_list, temp_list = [], [], [], []
steps_per_block = 500
n_blocks        = N_STEPS // steps_per_block

def collect():
    time_ps.append(dyn.get_time() / (1000 * units.fs))
    epot_list.append(rlx.get_potential_energy())
    ekin_list.append(rlx.get_kinetic_energy())
    temp_list.append(rlx.get_temperature())
dyn.attach(collect, interval=steps_per_block)

# ── Production run ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"NVT Production  T={TEMPERATURE} K  dt={TIMESTEP_FS} fs  "
      f"steps={N_STEPS}  ({N_STEPS*TIMESTEP_FS/1000:.1f} ps)")
print("=" * 60)

for i in range(n_blocks):
    dyn.run(steps_per_block)
    if (i + 1) % 10 == 0:
        t   = time_ps[-1]
        T   = temp_list[-1]
        Ep  = epot_list[-1] / len(rlx)
        print(f"  block {i+1:4d}/{n_blocks}  t={t:.3f} ps  "
              f"T={T:.1f} K  Epot={Ep:.4f} eV/atom")

etot_arr = np.array(epot_list) + np.array(ekin_list)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Production Summary")
print("=" * 60)
print(f"  T_mean   : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K")
print(f"  E_drift  : {(etot_arr[-1]-etot_arr[0])/len(rlx):.6f} eV/atom")

# ── RDF ───────────────────────────────────────────────────────────────────────
print("\nComputing RDF from trajectory...")
traj   = Trajectory(MIX_TRAJ_EQUIL, 'r')
n_traj = len(traj)

# Sample last 50% of trajectory for RDF
sample_frames = [traj[i] for i in range(n_traj//2, n_traj, max(1, n_traj//200))]

r_max   = cutoff - 0.5
n_bins  = 200
r_edges = np.linspace(0, r_max, n_bins + 1)
r_mid   = 0.5 * (r_edges[:-1] + r_edges[1:])
dr      = r_edges[1] - r_edges[0]

# Collect all pair combinations of interest
pairs = [('Li','F'), ('Be','F'), ('F','F'), ('Li','Li')]
counts = {p: np.zeros(n_bins) for p in pairs}
n_frames_used = 0

for frame in sample_frames:
    pos  = frame.get_positions()
    sym  = frame.get_chemical_symbols()
    cell = np.array(frame.get_cell())
    vol  = frame.get_volume()
    n_frames_used += 1

    for i, (si, pi) in enumerate(zip(sym, pos)):
        for j, (sj, pj) in enumerate(zip(sym, pos)):
            if i >= j:
                continue
            key = None
            if   (si=='Li' and sj=='F')  or (si=='F'  and sj=='Li'):  key=('Li','F')
            elif (si=='Be' and sj=='F')  or (si=='F'  and sj=='Be'):  key=('Be','F')
            elif  si=='F'  and sj=='F':                                key=('F','F')
            elif  si=='Li' and sj=='Li':                               key=('Li','Li')
            if key is None:
                continue
            dr_vec = pj - pi
            # Minimum image
            dr_vec -= cell[2][2] * np.round(dr_vec[2]/cell[2][2]) * np.array([0,0,1])
            dr_vec -= cell[1][1] * np.round(dr_vec[1]/cell[1][1]) * np.array([0,1,0])
            dr_vec -= cell[0][0] * np.round(dr_vec[0]/cell[0][0]) * np.array([1,0,0])
            dist = np.linalg.norm(dr_vec)
            if dist < r_max:
                idx = int(dist / dr)
                if idx < n_bins:
                    counts[key][idx] += 1

# Normalise to g(r)
sym_all   = sample_frames[0].get_chemical_symbols()
species_count = {s: sym_all.count(s) for s in set(sym_all)}
vol       = sample_frames[0].get_volume()
rdf       = {}
for (s1, s2), cnt in counts.items():
    n1, n2 = species_count[s1], species_count[s2]
    rho    = n2 / vol
    shell  = (4/3) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
    norm   = n1 * rho * shell * n_frames_used
    rdf[(s1,s2)] = cnt / np.where(norm > 0, norm, 1)

# ── MSD ───────────────────────────────────────────────────────────────────────
print("Computing MSD...")
msd_Li, msd_Be, msd_F = [], [], []
pos0 = None
idx_Li = idx_Be = idx_F = np.array([], dtype=int)

for k, frame in enumerate(traj):
    sym = np.array(frame.get_chemical_symbols())
    pos = frame.get_positions()
    if pos0 is None:
        pos0    = pos.copy()
        sym0    = sym.copy()
        idx_Li  = np.where(sym0 == 'Li')[0]
        idx_Be  = np.where(sym0 == 'Be')[0]
        idx_F   = np.where(sym0 == 'F' )[0]
    disp = pos - pos0
    msd_Li.append(np.mean(np.sum(disp[idx_Li]**2, axis=1)))
    msd_Be.append(np.mean(np.sum(disp[idx_Be]**2, axis=1)))
    msd_F.append( np.mean(np.sum(disp[idx_F ]**2, axis=1)))

msd_time = np.arange(len(msd_Li)) * TIMESTEP_FS / 1000   # ps

# Diffusion: D = MSD / (6t)  [Å²/ps → cm²/s]
def diffusion_cm2s(msd, time_ps):
    """Linear fit over last 50% to get D in cm²/s."""
    n  = len(msd)
    t  = time_ps[n//2:]
    m  = np.array(msd[n//2:])
    slope = np.polyfit(t, m, 1)[0]           # Å²/ps
    return slope / 6.0 * 1e-4 * 1e12 * 1e-20 # cm²/s

# ── Plots ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))

# 1. Energy
ax1 = fig.add_subplot(2, 3, 1)
n   = len(rlx)
ax1.plot(time_ps, np.array(epot_list)/n, 'b-',  lw=1.2, label='Potential')
ax1.plot(time_ps, np.array(ekin_list)/n, 'r-',  lw=1.2, label='Kinetic')
ax1.plot(time_ps, etot_arr/n,            'k--', lw=1.2, label='Total')
ax1.set_xlabel('Time (ps)'); ax1.set_ylabel('Energy/atom (eV)')
ax1.set_title('Energy'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# 2. Temperature
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(time_ps, temp_list, 'g-', lw=1.2)
ax2.axhline(TEMPERATURE, color='r', ls='--', alpha=0.7, label=f'Target {TEMPERATURE} K')
ax2.set_xlabel('Time (ps)'); ax2.set_ylabel('Temperature (K)')
ax2.set_title('Temperature'); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# 3. RDF
ax3 = fig.add_subplot(2, 3, 3)
for (s1,s2), gr in rdf.items():
    ax3.plot(r_mid, gr, lw=1.5, label=f'{s1}–{s2}')
ax3.set_xlabel('r (Å)'); ax3.set_ylabel('g(r)')
ax3.set_title('Radial Distribution Function')
ax3.legend(fontsize=8); ax3.grid(alpha=0.3); ax3.set_xlim(0, r_max)

# 4. MSD
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(msd_time, msd_Li, 'b-',  lw=1.2, label='Li')
ax4.plot(msd_time, msd_Be, 'r-',  lw=1.2, label='Be')
ax4.plot(msd_time, msd_F,  'g-',  lw=1.2, label='F')
ax4.set_xlabel('Time (ps)'); ax4.set_ylabel('MSD (Å²)')
ax4.set_title('Mean Square Displacement'); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

# 5. Temperature histogram
ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(temp_list, bins=25, color='steelblue', alpha=0.7, edgecolor='k')
ax5.axvline(TEMPERATURE, color='r', ls='--', label='Target')
ax5.axvline(float(np.mean(temp_list)), color='b', ls='-',
            label=f'Mean {np.mean(temp_list):.0f} K')
ax5.set_xlabel('T (K)'); ax5.set_ylabel('Frequency')
ax5.set_title('Temperature Distribution'); ax5.legend(fontsize=8)

# 6. Diffusion summary text
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
lines = [
    f"System:  LiF·BeF2  ({len(rlx)} atoms)",
    f"T:       {TEMPERATURE} K",
    f"Steps:   {N_STEPS}  ({N_STEPS*TIMESTEP_FS/1000:.1f} ps)",
    f"dt:      {TIMESTEP_FS} fs",
    "",
    f"T_mean:  {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K",
    f"E_drift: {(etot_arr[-1]-etot_arr[0])/len(rlx):.5f} eV/atom",
    "",
    "Diffusion (last 50% MSD fit):",
    f"  D(Li) = {diffusion_cm2s(msd_Li, msd_time):.3e} cm²/s",
    f"  D(Be) = {diffusion_cm2s(msd_Be, msd_time):.3e} cm²/s",
    f"  D(F)  = {diffusion_cm2s(msd_F,  msd_time):.3e} cm²/s",
]
ax6.text(0.05, 0.95, "\n".join(lines), transform=ax6.transAxes,
         va='top', fontsize=9, family='monospace')

plt.suptitle(f'Classical MD  LiF·BeF2  T={TEMPERATURE} K  (LAMMPS/ASE)', fontsize=12)
plt.tight_layout()
plt.savefig(MIX_PLOT_FILE, dpi=150, bbox_inches='tight')
print(f"\nPlot saved → {MIX_PLOT_FILE}")
print(f"Trajectory → {MIX_TRAJ_EQUIL}")
