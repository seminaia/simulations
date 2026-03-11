"""
AIMD (Born-Oppenheimer MD) for LiF + BeF2 using GPAW via ASE
=============================================================
Method  : DFT/HSE06 plane-wave basis → forces at each step
Ensemble: NVT Nosé-Hoover chain
Workflow: build → relax → equilibrate → production → plots
"""

import os
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import ase
from ase import units, Atoms
from ase.build import bulk, make_supercell
from ase.build.attach import attach
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from ase.units import Bohr
from ase.visualize import view
from gpaw import GPAW, restart
from ase.spacegroup import crystal
from ase.build.tools import cut, stack

# ── Parameters ────────────────────────────────────────────────────────────────
TEMPERATURE  = 1200        # K
TIMESTEP_FS  = 1.0         # fs
TDAMP_FS     = 50          # thermostat damping (fs)
N_EQUIL      = 200         # NVT equilibration steps
N_PROD       = 500         # NVT production steps
LOG_INTERVAL = 10          # steps between log entries
ECUT_EV      = 500         # plane-wave cutoff (eV)
KPTS         = (2, 2, 2)   # Γ-point only — speed priority
SUPERCELL    = 1           # n×n×n supercell multiplier
SCREEN       = 0.25 * Bohr # HSE06 range-separation parameter

# ── File names ────────────────────────────────────────────────────────────────
LIF_GPW_FILE   = "LiF_aimd_relax.gpw"
LIF_RLX_LOG    = "LiF_aimd_relax_opt.log"
BEF2_GPW_FILE  = "BeF2_aimd_relax.gpw"
BEF2_RLX_LOG   = "BeF2_aimd_relax_opt.log"

MIX_TRAJ_EQUIL = "mix_aimd_equil.traj"
MIX_LOG_EQUIL  = "mix_aimd_equil.log"
MIX_TRAJ_PROD  = "mix_aimd_prod.traj"
MIX_LOG_PROD   = "mix_aimd_prod.log"
MIX_PLOT_FILE  = "mix_aimd_results.png"

# ── Build structures ──────────────────────────────────────────────────────────
print("=" * 60)
print("Building LiF and BeF2 supercells")
print("=" * 60)

lif_atoms  = bulk('LiF', crystalstructure='rocksalt', a=4.03, cubic=True)


bef2_atoms = crystal('BeF2',spacegroup=152, cellpar=[4.73, 4.73, 5.18, 90, 90, 120], basis=[(0.5,0,0.33),(0.41,0.28,0.22)])

print(f"LiF  : {len(lif_atoms)} atoms  cell={np.diag(lif_atoms.cell)} Å")
print(f"BeF2 : {len(bef2_atoms)} atoms  cell={np.diag(bef2_atoms.cell)} Å")


# ── GPAW calculator factory ───────────────────────────────────────────────────
def make_gpaw(txt='-', screen=SCREEN, ecut=ECUT_EV, hund=False) -> GPAW:
    return GPAW(
        convergence={"density": 1e-8, "eigenstates": 1e-10, "energy": 1e-6, "forces": 1e-6},
        eigensolver={"name": "dav", "niter": 5},
        hund=hund,
        kpts=KPTS,
        maxiter=1000,
        mixer={"backend": "pulay", "beta": 0.25, "method": "fullspin", "nmaxold": 5, "weight": 50.0},
        mode={"name": "pw", "ecut": ecut},
        nbands="nao",
        occupations={"name": "fermi-dirac", "width": 0.01},
        txt=txt,
        xc={"name":"HYB_GGA_XC_HSE06",
            "omega": screen,
            "fraction": 0.25},
    )


# ── Relaxation helper ─────────────────────────────────────────────────────────
def relax(
    atoms: Atoms,
    calculator_params: Dict[str, Any],
    fmax: float = 0.01,
    fixcell: bool = True,
    logname: str = 'opt.log',
    trajname: str = 'opt.traj',
    gpwname: str = 'rlx.gpw',
) -> Atoms:
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

    forces = opt_atoms.get_forces()
    print(f"Max force: {np.max(np.linalg.norm(forces, axis=1)):.6f} eV/Å")
    return opt_atoms


# ── Step 1: Geometry relaxation ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 1: Geometry relaxation (BFGS, fmax=0.01 eV/Å)")
print("=" * 60)

pbe_params = {
    "convergence": {"density": 1e-8, "eigenstates": 1e-10, "energy": 1e-6, "forces": 1e-6},
    "eigensolver": {"name": "dav", "niter": 5},
    "kpts": KPTS,
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
lif_relax = relax(lif_atoms, hse_params, fmax=0.01, fixcell=False,
                  logname=LIF_RLX_LOG, gpwname=LIF_GPW_FILE)
view(lif_relax, repeat=(2, 2, 2))

bef2_relax = relax(bef2_atoms, hse_params, fmax=0.01, fixcell=False,
                   logname=BEF2_RLX_LOG, gpwname=BEF2_GPW_FILE)
view(bef2_relax, repeat=(2, 2, 2))

print(f"  LiF  Epot = {lif_relax.get_potential_energy()/len(lif_atoms):.4f} eV/atom")
print(f"  BeF2 Epot = {bef2_relax.get_potential_energy()/len(bef2_atoms):.4f} eV/atom")
lif_relax.write("LiF_aimd_relaxed.xyz")
bef2_relax.write("BeF2_aimd_relaxed.xyz")

# ── Step 2: NVT equilibration ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2: NVT equilibration  T={TEMPERATURE} K  steps={N_EQUIL}")
print("=" * 60)


mix = stack(lif_relax, bef2_relax, maxstrain=1, distance=2.5)
mix.calc = relax(mix, pbe_params,fmax=0.01, fixcell=False, logname='mix_relax.log', gpwname='mix_relax.gpw')
view(mix, repeat=(2, 2, 2))
MaxwellBoltzmannDistribution(mix, temperature_K=TEMPERATURE)
Stationary(mix)
ZeroRotation(mix)

dyn_eq = NoseHooverChainNVT(
    mix,
    timestep=TIMESTEP_FS * units.fs,
    temperature_K=TEMPERATURE,
    tdamp=TDAMP_FS * units.fs,
    trajectory=MIX_TRAJ_EQUIL,
    logfile=MIX_LOG_EQUIL,
)
dyn_eq.attach(MDLogger(dyn_eq, mix, MIX_LOG_EQUIL), interval=LOG_INTERVAL)

eq_temp, eq_epot = [], []

def collect_equil():
    eq_epot.append(mix.get_potential_energy())
    eq_temp.append(mix.get_temperature())

dyn_eq.attach(collect_equil, interval=LOG_INTERVAL)

for i in range(N_EQUIL // 10):
    dyn_eq.run(10)
    t  = dyn_eq.get_time() / (1000 * units.fs)
    T  = mix.get_temperature()
    Ep = mix.get_potential_energy() / len(mix)
    print(f"  equil step {(i+1)*10:4d}/{N_EQUIL}  t={t:.3f} ps  T={T:.1f} K  Epot={Ep:.4f} eV/atom")

# ── Step 3: NVT production ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 3: NVT production  T={TEMPERATURE} K  steps={N_PROD}")
print("=" * 60)

mix.calc = make_gpaw(txt='mix_aimd_prod_gpaw.log')

dyn = NoseHooverChainNVT(
    mix,
    timestep=TIMESTEP_FS * units.fs,
    temperature_K=TEMPERATURE,
    tdamp=TDAMP_FS * units.fs,
    trajectory=MIX_TRAJ_PROD,
    logfile=MIX_LOG_PROD,
)
dyn.attach(MDLogger(dyn, mix, MIX_LOG_PROD), interval=LOG_INTERVAL)

time_ps, epot_list, ekin_list, temp_list = [], [], [], []
vel_list = []

def collect_prod():
    time_ps.append(dyn.get_time() / (1000 * units.fs))
    epot_list.append(mix.get_potential_energy())
    ekin_list.append(mix.get_kinetic_energy())
    temp_list.append(mix.get_temperature())
    vel_list.append(mix.get_velocities().copy())

dyn.attach(collect_prod, interval=1)

for i in range(N_PROD // 10):
    dyn.run(10)
    t  = time_ps[-1]
    T  = temp_list[-1]
    Ep = epot_list[-1] / len(mix)
    if (i + 1) % 5 == 0:
        print(f"  prod step {(i+1)*10:4d}/{N_PROD}  t={t:.3f} ps  T={T:.1f} K  Epot={Ep:.4f} eV/atom")

mix.write("mix_aimd_final.xyz")
print("\n  Final configuration → mix_aimd_final.xyz")

etot_arr = np.array(epot_list) + np.array(ekin_list)
n = len(mix)

# ── VDOS via velocity autocorrelation (VACF) ──────────────────────────────────
print("\nComputing VDOS from VACF...")
vels    = np.array(vel_list)   # (N_steps, N_atoms, 3)
n_steps = vels.shape[0]
max_lag = n_steps // 2

vacf = np.zeros(max_lag)
for lag in range(max_lag):
    vacf[lag] = np.mean(np.sum(vels[:n_steps-lag] * vels[lag:], axis=-1))
vacf /= vacf[0]

vdos_raw = np.abs(rfft(vacf))**2
freqs    = rfftfreq(max_lag, d=TIMESTEP_FS * 1e-15) / 1e12   # THz
vdos_raw /= vdos_raw.max()

# ── MSD from trajectory ───────────────────────────────────────────────────────
print("Computing MSD...")
traj_prod = Trajectory(MIX_TRAJ_PROD)
pos0 = None
msd_Li, msd_F, msd_t = [], [], []

for k, frame in enumerate(traj_prod):
    sym = np.array(frame.get_chemical_symbols())
    pos = frame.get_positions()
    if pos0 is None:
        pos0   = pos.copy()
        idx_Li = np.where(sym == 'Li')[0]
        idx_F  = np.where(sym == 'F')[0]
    disp = pos - pos0
    msd_Li.append(np.mean(np.sum(disp[idx_Li]**2, axis=1)))
    msd_F.append(np.mean(np.sum(disp[idx_F]**2, axis=1)))
    msd_t.append(k * TIMESTEP_FS / 1000)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Production Summary")
print("=" * 60)
print(f"  T_target : {TEMPERATURE} K")
print(f"  T_mean   : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K")
print(f"  E_drift  : {(etot_arr[-1]-etot_arr[0])/n:.6f} eV/atom")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

ax = axes[0, 0]
ax.plot(time_ps, np.array(epot_list)/n, 'b-',  lw=1.2, label='Potential')
ax.plot(time_ps, np.array(ekin_list)/n, 'r-',  lw=1.2, label='Kinetic')
ax.plot(time_ps, etot_arr/n,            'k--', lw=1.2, label='Total')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('Energy/atom (eV)')
ax.set_title('Energy (production)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(time_ps, temp_list, 'g-', lw=1.2)
ax.axhline(TEMPERATURE, color='r', ls='--', alpha=0.7, label=f'Target {TEMPERATURE} K')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('T (K)')
ax.set_title('Temperature (production)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 2]
eq_steps = np.arange(len(eq_temp)) * LOG_INTERVAL
ax.plot(eq_steps, eq_temp, 'orange', lw=1.2)
ax.axhline(TEMPERATURE, color='r', ls='--', alpha=0.7)
ax.set_xlabel('Step'); ax.set_ylabel('T (K)')
ax.set_title('Equilibration temperature'); ax.grid(alpha=0.3)

ax = axes[1, 0]
mask = freqs < 25   # THz — LiF phonon range
ax.plot(freqs[mask], vdos_raw[mask], 'navy', lw=1.5)
ax.fill_between(freqs[mask], vdos_raw[mask], alpha=0.2, color='navy')
ax.set_xlabel('Frequency (THz)'); ax.set_ylabel('VDOS (arb.)')
ax.set_title('VDOS (velocity autocorrelation)'); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(msd_t, msd_Li, 'b-', lw=1.5, label='Li')
ax.plot(msd_t, msd_F,  'g-', lw=1.5, label='F')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('MSD (Å²)')
ax.set_title('Mean Square Displacement'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 2]
ax.axis('off')
lines = [
    "AIMD  LiF+BeF2  (GPAW/HSE06)",
    f"Supercell : {SUPERCELL}×{SUPERCELL}×{SUPERCELL}  ({n} atoms)",
    f"E_cut     : {ECUT_EV} eV",
    f"k-points  : {KPTS}",
    f"T_target  : {TEMPERATURE} K",
    f"dt        : {TIMESTEP_FS} fs",
    f"Equil     : {N_EQUIL} steps",
    f"Prod      : {N_PROD} steps",
    "",
    f"T_mean    : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K",
    f"E_drift   : {(etot_arr[-1]-etot_arr[0])/n:.5f} eV/atom",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va='top', fontsize=9, family='monospace')

plt.suptitle(f'AIMD  LiF+BeF2  T={TEMPERATURE} K  (GPAW/HSE06, PW-{ECUT_EV}eV)', fontsize=12)
plt.tight_layout()
plt.savefig(MIX_PLOT_FILE, dpi=150, bbox_inches='tight')
print(f"\nPlot saved      → {MIX_PLOT_FILE}")
print(f"Prod trajectory → {MIX_TRAJ_PROD}")
print(f"Final config    → mix_aimd_final.xyz")
