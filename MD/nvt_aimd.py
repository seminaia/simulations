"""
NVT AIMD for LiF + BeF2 using GPAW via ASE
============================================
Ensemble : NVT  Nosé-Hoover chain thermostat
Method   : DFT/HSE06 plane-wave basis
Workflow : build → relax → equilibrate → production → analysis → plots

Thermophysical properties (Porter et al. 2022, Fig. 3):
  - VDOS        : velocity autocorrelation FFT
  - RDF         : partial g(r) for all species pairs
  - MSD         : mean square displacement per species
  - Diffusion D : Green-Kubo / MSD slope (Eq. 10)
  - Cv          : heat capacity at constant volume (Eq. 8)
"""

import os
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from ase import units, Atoms
from ase.build import bulk
from ase.build.tools import stack
from ase.filters import FrechetCellFilter
from ase.io import read, write as ase_write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from ase.spacegroup import crystal
from ase.units import Bohr
from ase.visualize import view
from gpaw import GPAW, restart

# ── Parameters ────────────────────────────────────────────────────────────────
TEMPERATURE  = 1200        # K
TIMESTEP_FS  = 1.0         # fs
TDAMP_FS     = 50          # thermostat damping (fs)
N_EQUIL      = 200         # NVT equilibration steps
N_PROD       = 500         # NVT production steps
LOG_INTERVAL = 10          # steps between log entries
ECUT_EV      = 500         # plane-wave cutoff (eV)
KPTS         = (2, 2, 2)
SUPERCELL    = 1
SCREEN       = 0.25 * Bohr

# ── File names ────────────────────────────────────────────────────────────────
LIF_GPW_FILE   = "LiF_aimd_relax.gpw"
LIF_RLX_LOG    = "LiF_aimd_relax_opt.log"
BEF2_GPW_FILE  = "BeF2_aimd_relax.gpw"
BEF2_RLX_LOG   = "BeF2_aimd_relax_opt.log"

MIX_TRAJ_EQUIL = "nvt_mix_equil.traj"
MIX_LOG_EQUIL  = "nvt_mix_equil.log"
MIX_TRAJ_PROD  = "nvt_mix_prod.traj"
MIX_LOG_PROD   = "nvt_mix_prod.log"
MIX_PLOT_FILE  = "nvt_results.png"

# ── Build structures ──────────────────────────────────────────────────────────
print("=" * 60)
print("Building LiF and BeF2 supercells")
print("=" * 60)

lif_atoms  = bulk('LiF', crystalstructure='rocksalt', a=4.03, cubic=True)
lif_atoms.set_initial_charges([-1, 1] * (len(lif_atoms) // 2))
lif_atoms.set_initial_magnetic_moments([1, -1, 1, -1, 0.5, -0.5, 0.5, -0.5])
bef2_atoms = crystal('BeF2', spacegroup=152,
                     cellpar=[4.73, 4.73, 5.18, 90, 90, 120],
                     basis=[(0.5, 0, 0.33), (0.41, 0.28, 0.22)])
bef2_atoms.set_initial_magnetic_moments([2, -2, 2, -1, 1, -1, 1, -1, 1, -1, 1, -1])
bef2_atoms.set_initial_charges([-1, -1, 2] * (len(bef2_atoms) // 3))

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
        xc={"name": "HYB_GGA_XC_HSE06", "omega": screen, "fraction": 0.25},
    )


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

    orig_atoms.set_cell(relaxed.get_cell(), scale_atoms=False)
    orig_atoms.set_positions(relaxed.get_positions())
    cell = relaxed.cell.cellpar()
    print(f"  Lattice: a={cell[0]:.4f} b={cell[1]:.4f} c={cell[2]:.4f} Å  "
          f"α={cell[3]:.2f} β={cell[4]:.2f} γ={cell[5]:.2f}°")
    return relaxed


def diffusion_cm2s(msd, time_ps):
    """D from last 50% MSD linear fit; returns cm²/s."""
    n = len(msd)
    t = time_ps[n // 2:]
    m = np.array(msd[n // 2:])
    slope = np.polyfit(t, m, 1)[0]           # Å²/ps
    return slope / 6.0 * 1e-4 * 1e12 * 1e-20  # cm²/s


# ── Step 1: Geometry relaxation ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 1: Geometry relaxation (BFGS, fmax=0.01 eV/Å)")
print("=" * 60)

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
    "kpts": {"gamma": True, "size": (1, 1, 1)},
    "maxiter": 1000,
    "mixer": {"backend": "pulay", "beta": 0.25, "method": "fullspin", "nmaxold": 5, "weight": 50.0},
    "mode": {"name": "pw", "ecut": ECUT_EV},
    "nbands": "nao",
    "occupations": {"name": "fermi-dirac", "width": 0.01},
    "txt": "hse_relax.log",
    "xc": {"name": "HYB_GGA_XC_HSE06", "omega": SCREEN, "fraction": 0.25, "backend": "pw"},
}

lif_relax  = relax(lif_atoms,  pbe_params, fmax=0.01, fixcell=False,
                   logname=LIF_RLX_LOG, gpwname=LIF_GPW_FILE)
view(lif_relax, repeat=(2, 2, 2))
bef2_relax = relax(bef2_atoms, pbe_params, fmax=0.01, fixcell=False,
                   logname=BEF2_RLX_LOG, gpwname=BEF2_GPW_FILE)
view(bef2_relax, repeat=(2, 2, 2))

print(f"  LiF  Epot = {lif_relax.get_potential_energy()/len(lif_relax):.4f} eV/atom")
print(f"  BeF2 Epot = {bef2_relax.get_potential_energy()/len(bef2_relax):.4f} eV/atom")
lif_relax.write("LiF_aimd_relaxed.xyz")
bef2_relax.write("BeF2_aimd_relaxed.xyz")

# ── Step 2: NVT equilibration ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2: NVT equilibration  T={TEMPERATURE} K  steps={N_EQUIL}")
print("=" * 60)

mix = stack(lif_relax, bef2_relax, maxstrain=1, distance=2.5)
mix.calc = relax(mix, hse_params, fmax=0.01, fixcell=True,
                 logname='mix_relax.log', gpwname='mix_relax.gpw')
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

mix.calc = make_gpaw(txt='nvt_prod_gpaw.log')

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

mix.write("nvt_final.xyz")
print("\n  Final configuration → nvt_final.xyz")

etot_arr = np.array(epot_list) + np.array(ekin_list)
n = len(mix)

# ── VDOS via VACF ──────────────────────────────────────────────────────────────
print("\nComputing VDOS from VACF...")
vels    = np.array(vel_list)
n_steps = vels.shape[0]
max_lag = n_steps // 2

vacf = np.zeros(max_lag)
for lag in range(max_lag):
    vacf[lag] = np.mean(np.sum(vels[:n_steps - lag] * vels[lag:], axis=-1))
vacf /= vacf[0]

vdos_raw = np.abs(rfft(vacf)) ** 2
freqs    = rfftfreq(max_lag, d=TIMESTEP_FS * 1e-15) / 1e12   # THz
vdos_raw /= vdos_raw.max()

# ── RDF ────────────────────────────────────────────────────────────────────────
print("Computing RDF from trajectory...")
traj_prod = Trajectory(MIX_TRAJ_PROD)
n_traj    = len(traj_prod)

cell_arr = np.array(traj_prod[0].get_cell())
L_min    = min(cell_arr[0][0], cell_arr[1][1], cell_arr[2][2])
r_max    = round(L_min / 2 - 0.5, 1)

sample_frames = [traj_prod[i] for i in range(n_traj // 2, n_traj, max(1, n_traj // 200))]

n_bins  = 200
r_edges = np.linspace(0, r_max, n_bins + 1)
r_mid   = 0.5 * (r_edges[:-1] + r_edges[1:])
dr      = r_edges[1] - r_edges[0]

pairs  = [('Li', 'F'), ('Be', 'F'), ('F', 'F'), ('Li', 'Li')]
counts = {p: np.zeros(n_bins) for p in pairs}
n_frames_used = 0

for frame in sample_frames:
    pos  = frame.get_positions()
    sym  = frame.get_chemical_symbols()
    cell = np.array(frame.get_cell())
    n_frames_used += 1
    for i, (si, pi) in enumerate(zip(sym, pos)):
        for j, (sj, pj) in enumerate(zip(sym, pos)):
            if i >= j:
                continue
            key = None
            if   (si == 'Li' and sj == 'F')  or (si == 'F'  and sj == 'Li'):  key = ('Li', 'F')
            elif (si == 'Be' and sj == 'F')  or (si == 'F'  and sj == 'Be'):  key = ('Be', 'F')
            elif  si == 'F'  and sj == 'F':                                    key = ('F',  'F')
            elif  si == 'Li' and sj == 'Li':                                   key = ('Li', 'Li')
            if key is None:
                continue
            dr_vec = pj - pi
            dr_vec -= cell[2][2] * np.round(dr_vec[2] / cell[2][2]) * np.array([0, 0, 1])
            dr_vec -= cell[1][1] * np.round(dr_vec[1] / cell[1][1]) * np.array([0, 1, 0])
            dr_vec -= cell[0][0] * np.round(dr_vec[0] / cell[0][0]) * np.array([1, 0, 0])
            dist = np.linalg.norm(dr_vec)
            if dist < r_max:
                idx = int(dist / dr)
                if idx < n_bins:
                    counts[key][idx] += 1

sym_all       = sample_frames[0].get_chemical_symbols()
species_count = {s: sym_all.count(s) for s in set(sym_all)}
vol           = sample_frames[0].get_volume()
rdf           = {}
for (s1, s2), cnt in counts.items():
    n1, n2 = species_count[s1], species_count[s2]
    rho    = n2 / vol
    shell  = (4 / 3) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    norm   = n1 * rho * shell * n_frames_used
    rdf[(s1, s2)] = cnt / np.where(norm > 0, norm, 1)

# ── MSD ────────────────────────────────────────────────────────────────────────
print("Computing MSD...")
msd_Li, msd_Be, msd_F = [], [], []
pos0 = None
idx_Li = idx_Be = idx_F = np.array([], dtype=int)

for k, frame in enumerate(traj_prod):
    sym = np.array(frame.get_chemical_symbols())
    pos = frame.get_positions()
    if pos0 is None:
        pos0   = pos.copy()
        idx_Li = np.where(sym == 'Li')[0]
        idx_Be = np.where(sym == 'Be')[0]
        idx_F  = np.where(sym == 'F' )[0]
    disp = pos - pos0
    msd_Li.append(np.mean(np.sum(disp[idx_Li] ** 2, axis=1)))
    msd_Be.append(np.mean(np.sum(disp[idx_Be] ** 2, axis=1)))
    msd_F.append( np.mean(np.sum(disp[idx_F ] ** 2, axis=1)))

msd_time = np.arange(len(msd_Li)) * TIMESTEP_FS / 1000   # ps

# ── Cv — heat capacity at constant volume (Eq. 8) ─────────────────────────────
# Cv = Var(E_total)_NVT / (kB * T²)
Cv_total     = np.var(etot_arr) / (units.kB * TEMPERATURE ** 2)   # eV/K
Cv_per_atom  = Cv_total / n                                         # eV/K/atom
Cv_J         = Cv_per_atom * 1.602e-19                              # J/K/atom

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NVT Production Summary")
print("=" * 60)
print(f"  T_target : {TEMPERATURE} K")
print(f"  T_mean   : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K")
print(f"  E_drift  : {(etot_arr[-1] - etot_arr[0]) / n:.6f} eV/atom")
print(f"  Cv       : {Cv_per_atom:.4e} eV/K/atom  ({Cv_J:.4e} J/K/atom)")
print(f"  D(Li)    : {diffusion_cm2s(msd_Li, msd_time):.3e} cm²/s")
print(f"  D(Be)    : {diffusion_cm2s(msd_Be, msd_time):.3e} cm²/s")
print(f"  D(F)     : {diffusion_cm2s(msd_F,  msd_time):.3e} cm²/s")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

ax = axes[0, 0]
ax.plot(time_ps, np.array(epot_list) / n, 'b-',  lw=1.2, label='Potential')
ax.plot(time_ps, np.array(ekin_list) / n, 'r-',  lw=1.2, label='Kinetic')
ax.plot(time_ps, etot_arr / n,            'k--', lw=1.2, label='Total')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('Energy/atom (eV)')
ax.set_title('Energy (production)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(time_ps, temp_list, 'g-', lw=1.2)
ax.axhline(TEMPERATURE, color='r', ls='--', alpha=0.7, label=f'Target {TEMPERATURE} K')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('T (K)')
ax.set_title('Temperature'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 2]
mask = freqs < 25
ax.plot(freqs[mask], vdos_raw[mask], 'navy', lw=1.5)
ax.fill_between(freqs[mask], vdos_raw[mask], alpha=0.2, color='navy')
ax.set_xlabel('Frequency (THz)'); ax.set_ylabel('VDOS (arb.)')
ax.set_title('VDOS'); ax.grid(alpha=0.3)

ax = axes[1, 0]
for (s1, s2), gr in rdf.items():
    ax.plot(r_mid, gr, lw=1.5, label=f'{s1}–{s2}')
ax.set_xlabel('r (Å)'); ax.set_ylabel('g(r)')
ax.set_title('Radial Distribution Function')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, r_max)

ax = axes[1, 1]
ax.plot(msd_time, msd_Li, 'b-', lw=1.5, label='Li')
ax.plot(msd_time, msd_Be, 'r-', lw=1.5, label='Be')
ax.plot(msd_time, msd_F,  'g-', lw=1.5, label='F')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('MSD (Å²)')
ax.set_title('Mean Square Displacement'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 2]
ax.axis('off')
lines = [
    "NVT AIMD  LiF+BeF2  (GPAW/HSE06)",
    f"N atoms   : {n}",
    f"E_cut     : {ECUT_EV} eV",
    f"T_target  : {TEMPERATURE} K",
    f"dt        : {TIMESTEP_FS} fs",
    f"Prod      : {N_PROD} steps ({N_PROD*TIMESTEP_FS/1000:.2f} ps)",
    "",
    f"T_mean    : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K",
    f"E_drift   : {(etot_arr[-1] - etot_arr[0]) / n:.5f} eV/atom",
    f"Cv        : {Cv_per_atom:.3e} eV/K/atom",
    f"           ({Cv_J:.3e} J/K/atom)",
    "",
    "Diffusion (last 50% MSD):",
    f"  D(Li) = {diffusion_cm2s(msd_Li, msd_time):.3e} cm²/s",
    f"  D(Be) = {diffusion_cm2s(msd_Be, msd_time):.3e} cm²/s",
    f"  D(F)  = {diffusion_cm2s(msd_F,  msd_time):.3e} cm²/s",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va='top', fontsize=9, family='monospace')

plt.suptitle(f'NVT AIMD  LiF+BeF2  T={TEMPERATURE} K  (GPAW/HSE06, PW-{ECUT_EV}eV)', fontsize=12)
plt.tight_layout()
plt.savefig(MIX_PLOT_FILE, dpi=150, bbox_inches='tight')
print(f"\nPlot saved      → {MIX_PLOT_FILE}")
print(f"Prod trajectory → {MIX_TRAJ_PROD}")
print(f"Final config    → nvt_final.xyz")
