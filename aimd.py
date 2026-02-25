"""
AIMD (Born-Oppenheimer MD) for LiF using GPAW via ASE
======================================================
Method  : DFT/PBE plane-wave basis → forces at each step
System  : LiF rocksalt 2×2×2 supercell (64 atoms)
          (AIMD scales as O(N³) — keep supercell small)
Ensemble: NVT Nosé-Hoover chain
Workflow: build → relax → equilibrate → production → plots

Note on cost vs classical MD
  Classical MD (LAMMPS) : ~10⁶ steps/hour on 1 CPU
  AIMD (GPAW)           : ~10–100 steps/hour on 1 CPU
  → Use AIMD for short runs / validation; classical for statistics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ase import units
from ase.build import bulk, make_supercell
from ase.optimize import BFGS
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from gpaw import GPAW, PW, FermiDirac

# ── Parameters ────────────────────────────────────────────────────────────────
TEMPERATURE    = 1200       # K  (lower than classical run — AIMD is expensive)
TIMESTEP_FS    = 1.0        # fs  (BOMD can use larger dt than CP-MD)
TDAMP_FS       = 50         # thermostat damping (fs)
N_EQUIL        = 200        # NVT equilibration steps
N_PROD         = 500        # NVT production steps
LOG_INTERVAL   = 10         # steps between log entries
ECUT_EV        = 400        # plane-wave cutoff (eV)
KPTS           = (2, 2, 2)  # k-point grid — small for MD speed
SUPERCELL      = 2          # n×n×n supercell of LiF rocksalt unit cell

TRAJ_EQUIL = "LiF_aimd_equil.traj"
TRAJ_PROD  = "LiF_aimd_prod.traj"
LOG_EQUIL  = "LiF_aimd_equil.log"
LOG_PROD   = "LiF_aimd_prod.log"
PLOT_FILE  = "LiF_aimd_results.png"

# ── Build LiF 2×2×2 supercell ────────────────────────────────────────────────
print("=" * 60)
print("Building LiF rocksalt supercell")
print("=" * 60)

lif_unit   = bulk('LiF', crystalstructure='rocksalt', a=4.03, cubic=True)
atoms      = make_supercell(lif_unit, SUPERCELL * np.eye(3, dtype=int))

print(f"  Atoms    : {len(atoms)}")
print(f"  Cell (Å) : {np.diag(atoms.cell)}")
print(f"  Species  : {set(atoms.get_chemical_symbols())}")

# ── GPAW calculator ───────────────────────────────────────────────────────────
# Γ-only or small k-grid for MD (speed priority)
# FermiDirac smearing handles metallic/semi-ionic systems better during MD
def make_gpaw(txt='gpaw_md.log'):
    return GPAW(
        mode        = PW(ECUT_EV),
        xc          = 'PBE',
        kpts        = KPTS,
        occupations = FermiDirac(0.1),     # eV — broadens occupation, aids SCF
        symmetry    = 'off',               # atoms break symmetry during MD — must disable
        txt         = txt,
        convergence = {'energy': 1e-4},    # relaxed tolerance — faster SCF per step
        maxiter     = 300,
    )

# ── Step 1: Geometry relaxation ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 1: Geometry relaxation (BFGS, fmax=0.05 eV/Å)")
print("=" * 60)

atoms.calc = make_gpaw(txt='LiF_aimd_relax.log')
relax = BFGS(atoms, logfile='LiF_aimd_relax_opt.log')
relax.run(fmax=0.05, steps=100)
print(f"  Epot = {atoms.get_potential_energy()/len(atoms):.4f} eV/atom")
atoms.write("LiF_aimd_relaxed.xyz")
print("  Relaxed structure → LiF_aimd_relaxed.xyz")

# ── Step 2: NVT equilibration ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2: NVT equilibration  T={TEMPERATURE} K  steps={N_EQUIL}")
print("=" * 60)

# Fresh calculator for MD (new txt file)
atoms.calc = make_gpaw(txt='LiF_aimd_equil_gpaw.log')

MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
Stationary(atoms)
ZeroRotation(atoms)

dyn_eq = NoseHooverChainNVT(
    atoms,
    timestep      = TIMESTEP_FS * units.fs,
    temperature_K = TEMPERATURE,
    tdamp         = TDAMP_FS * units.fs,
    trajectory    = TRAJ_EQUIL,
    logfile       = LOG_EQUIL,
)
dyn_eq.attach(MDLogger(dyn_eq, atoms, LOG_EQUIL), interval=LOG_INTERVAL)

eq_temp, eq_epot = [], []

def collect_equil():
    eq_epot.append(atoms.get_potential_energy())
    eq_temp.append(atoms.get_temperature())

dyn_eq.attach(collect_equil, interval=LOG_INTERVAL)

for i in range(N_EQUIL // 10):
    dyn_eq.run(10)
    t = dyn_eq.get_time() / (1000 * units.fs)
    T = atoms.get_temperature()
    Ep = atoms.get_potential_energy() / len(atoms)
    print(f"  equil step {(i+1)*10:4d}/{N_EQUIL}  "
          f"t={t:.3f} ps  T={T:.1f} K  Epot={Ep:.4f} eV/atom")

# ── Step 3: NVT production ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 3: NVT production  T={TEMPERATURE} K  steps={N_PROD}")
print("=" * 60)

atoms.calc = make_gpaw(txt='LiF_aimd_prod_gpaw.log')

dyn = NoseHooverChainNVT(
    atoms,
    timestep      = TIMESTEP_FS * units.fs,
    temperature_K = TEMPERATURE,
    tdamp         = TDAMP_FS * units.fs,
    trajectory    = TRAJ_PROD,
    logfile       = LOG_PROD,
)
dyn.attach(MDLogger(dyn, atoms, LOG_PROD), interval=LOG_INTERVAL)

time_ps, epot_list, ekin_list, temp_list = [], [], [], []
vel_list = []   # for VDOS via velocity autocorrelation

def collect_prod():
    time_ps.append(dyn.get_time() / (1000 * units.fs))
    epot_list.append(atoms.get_potential_energy())
    ekin_list.append(atoms.get_kinetic_energy())
    temp_list.append(atoms.get_temperature())
    vel_list.append(atoms.get_velocities().copy())

dyn.attach(collect_prod, interval=1)   # every step for VDOS

for i in range(N_PROD // 10):
    dyn.run(10)
    t  = time_ps[-1]
    T  = temp_list[-1]
    Ep = epot_list[-1] / len(atoms)
    if (i + 1) % 5 == 0:
        print(f"  prod step {(i+1)*10:4d}/{N_PROD}  "
              f"t={t:.3f} ps  T={T:.1f} K  Epot={Ep:.4f} eV/atom")

atoms.write("LiF_aimd_final.xyz")
print("\n  Final configuration → LiF_aimd_final.xyz")

etot_arr = np.array(epot_list) + np.array(ekin_list)

# ── VDOS via velocity autocorrelation function (VACF) ────────────────────────
print("\nComputing VDOS from VACF...")
vels  = np.array(vel_list)          # shape (N_steps, N_atoms, 3)
n_steps, n_atoms, _ = vels.shape

# VACF: C(t) = <v(0)·v(t)>
max_lag = n_steps // 2
vacf    = np.zeros(max_lag)
for lag in range(max_lag):
    vacf[lag] = np.mean(np.sum(vels[:n_steps-lag] * vels[lag:], axis=-1))
vacf /= vacf[0]   # normalise

# Power spectrum → VDOS
from numpy.fft import rfft, rfftfreq
vdos_raw = np.abs(rfft(vacf))**2
freqs    = rfftfreq(max_lag, d=TIMESTEP_FS * 1e-15) / 1e12   # THz
vdos_raw /= vdos_raw.max()

# ── MSD from trajectory ───────────────────────────────────────────────────────
print("Computing MSD...")
traj_prod = Trajectory(TRAJ_PROD)
pos0      = None
msd_Li, msd_F, msd_t = [], [], []

for k, frame in enumerate(traj_prod):
    sym = np.array(frame.get_chemical_symbols())
    pos = frame.get_positions()
    if pos0 is None:
        pos0   = pos.copy()
        idx_Li = np.where(sym == 'Li')[0]
        idx_F  = np.where(sym == 'F' )[0]
    disp = pos - pos0
    msd_Li.append(np.mean(np.sum(disp[idx_Li]**2, axis=1)))
    msd_F.append( np.mean(np.sum(disp[idx_F ]**2, axis=1)))
    msd_t.append(k * TIMESTEP_FS / 1000)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Production Summary")
print("=" * 60)
print(f"  T_target : {TEMPERATURE} K")
print(f"  T_mean   : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K")
print(f"  E_drift  : {(etot_arr[-1]-etot_arr[0])/len(atoms):.6f} eV/atom")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# 1. Energy (production)
ax = axes[0, 0]
n  = len(atoms)
ax.plot(time_ps, np.array(epot_list)/n, 'b-',  lw=1.2, label='Potential')
ax.plot(time_ps, np.array(ekin_list)/n, 'r-',  lw=1.2, label='Kinetic')
ax.plot(time_ps, etot_arr/n,            'k--', lw=1.2, label='Total')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('Energy/atom (eV)')
ax.set_title('Energy (production)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 2. Temperature (production)
ax = axes[0, 1]
ax.plot(time_ps, temp_list, 'g-', lw=1.2)
ax.axhline(TEMPERATURE, color='r', ls='--', alpha=0.7, label=f'Target {TEMPERATURE} K')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('T (K)')
ax.set_title('Temperature (production)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 3. Equilibration temperature
ax = axes[0, 2]
eq_steps = np.arange(len(eq_temp)) * LOG_INTERVAL
ax.plot(eq_steps, eq_temp, 'orange', lw=1.2)
ax.axhline(TEMPERATURE, color='r', ls='--', alpha=0.7)
ax.set_xlabel('Step'); ax.set_ylabel('T (K)')
ax.set_title('Equilibration temperature'); ax.grid(alpha=0.3)

# 4. VDOS
ax = axes[1, 0]
freq_max = 25   # THz — LiF phonon range
mask = freqs < freq_max
ax.plot(freqs[mask], vdos_raw[mask], 'navy', lw=1.5)
ax.set_xlabel('Frequency (THz)'); ax.set_ylabel('VDOS (arb.)')
ax.set_title('VDOS  (velocity autocorrelation)'); ax.grid(alpha=0.3)
ax.fill_between(freqs[mask], vdos_raw[mask], alpha=0.2, color='navy')

# 5. MSD
ax = axes[1, 1]
ax.plot(msd_t, msd_Li, 'b-', lw=1.5, label='Li')
ax.plot(msd_t, msd_F,  'g-', lw=1.5, label='F')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('MSD (Å²)')
ax.set_title('Mean Square Displacement'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 6. Summary text
ax = axes[1, 2]
ax.axis('off')
lines = [
    "AIMD  LiF  (GPAW/PBE)",
    f"Supercell : {SUPERCELL}×{SUPERCELL}×{SUPERCELL}  ({len(atoms)} atoms)",
    f"E_cut     : {ECUT_EV} eV",
    f"k-points  : {KPTS}",
    f"T_target  : {TEMPERATURE} K",
    f"dt        : {TIMESTEP_FS} fs",
    f"Equil     : {N_EQUIL} steps",
    f"Prod      : {N_PROD} steps",
    "",
    f"T_mean    : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K",
    f"E_drift   : {(etot_arr[-1]-etot_arr[0])/len(atoms):.5f} eV/atom",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va='top', fontsize=9, family='monospace')

plt.suptitle(f'AIMD  LiF  T={TEMPERATURE} K  (GPAW/PBE, PW-{ECUT_EV}eV)', fontsize=12)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
print(f"\nPlot saved      → {PLOT_FILE}")
print(f"Prod trajectory → {TRAJ_PROD}")
print(f"Final config    → LiF_aimd_final.xyz")
