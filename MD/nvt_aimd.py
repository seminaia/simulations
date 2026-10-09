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
  - ADF         : angular distribution function (Eq. 23)
"""

import os
import sys
from datetime import datetime
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
from ase.units import Bohr
from ase.visualize import view
from ase.spacegroup import crystal
from gpaw import GPAW
from gpaw_helpers import relax, lcy_pbe_params


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
SCREEN       = 0.2 * Bohr

# ── File names ────────────────────────────────────────────────────────────────
NAF_GPW_FILE   = "NaF_aimd_relax.gpw"
NAF_RLX_LOG    = "NaF_aimd_relax_opt.log"
ALF3_GPW_FILE  = "AlF3_aimd_relax.gpw"
ALF3_RLX_LOG   = "AlF3_aimd_relax_opt.log"

MIX_TRAJ_EQUIL = "nvt_mix_equil.traj"
MIX_LOG_EQUIL  = "nvt_mix_equil.log"
MIX_TRAJ_PROD  = "nvt_mix_prod.traj"
MIX_LOG_PROD   = "nvt_mix_prod.log"
MIX_PLOT_FILE  = "nvt_results.png"
PROGRESS_LOG   = "nvt_aimd_progress.log"

# ── Progress logger (tee stdout+stderr → PROGRESS_LOG) ───────────────────────
class _Tee:
    """Mirror writes to multiple streams; use for sys.stdout / sys.stderr."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self) -> None:
        for s in self._streams:
            s.flush()
    def fileno(self) -> int:          # needed by some C extensions
        return self._streams[0].fileno()

_progress_fh = open(PROGRESS_LOG, "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _progress_fh)  # type: ignore[assignment]
sys.stderr = _Tee(sys.__stderr__, _progress_fh)  # type: ignore[assignment]
print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Logging to {PROGRESS_LOG}  —  tail -f {PROGRESS_LOG}")

# ── Build structures ──────────────────────────────────────────────────────────
print("=" * 60)
print("Building NaF and AlF3 supercells")
print("=" * 60)
charges = {'Al':3, 'F':-1, 'Na':1}
magmoms = {'Al': 1, 'F':-1,'Na':1}
a_naf = 4.57
naf_crystal = crystal('NaF',basis=[(0,0,0),(0,0,0.5)],spacegroup=225,cellpar=[a_naf, a_naf, a_naf, 90, 90, 90],pbc=False,size=(1, 1, 1))
naf_charges = [charges[s] for s in naf_crystal.get_chemical_symbols()]
naf_magmoms = [magmoms[s] for s in naf_crystal.get_chemical_symbols()]
naf_crystal.set_initial_charges(naf_charges)
naf_crystal.set_initial_magnetic_moments(naf_magmoms)
naf_cell_params = naf_crystal.cell.cellpar()
view(naf_crystal)
a_alf3 = 4.87
c_alf3 = 12.47
alf3_crystal = crystal("AlF3", basis=[(1/3,2/3,2/3),(1/3,0.254145,0.416667)], spacegroup=167, cellpar=[a_alf3, a_alf3, c_alf3, 90, 90, 120], pbc=False, size=(1, 1, 1))
alf3_charges = [charges[s] for s in alf3_crystal.get_chemical_symbols()]
alf3_magmoms = [magmoms[s] for s in alf3_crystal.get_chemical_symbols()]
alf3_crystal.set_initial_charges(alf3_charges)
alf3_crystal.set_initial_magnetic_moments(alf3_magmoms)
alf3_cell_params = alf3_crystal.cell.cellpar()
view(alf3_crystal)

print(f"""
NaF  : {len(naf_crystal)} atoms  a={naf_cell_params[0]:.2f} b={naf_cell_params[1]:.2f} c={naf_cell_params[2]:.2f},
        alpha={naf_cell_params[3]:.2f} beta={naf_cell_params[4]:.2f} gamma={naf_cell_params[5]:.2f},
        initial Volumes: {naf_crystal.get_volume():.2f} Å³,
""")
print(f"""
AlF3 : {len(alf3_crystal)} atoms  a={alf3_cell_params[0]:.2f} b={alf3_cell_params[1]:.2f} c={alf3_cell_params[2]:.2f},
        alpha={alf3_cell_params[3]:.2f} beta={alf3_cell_params[4]:.2f} gamma={alf3_cell_params[5]:.2f},
        initial Volumes: {alf3_crystal.get_volume():.2f} Å³,
        )""")



def diffusion_cm2s(msd, time_ps):
    """D from last 50% MSD linear fit; returns cm²/s."""
    n = len(msd)
    t = time_ps[n // 2:]
    m = np.array(msd[n // 2:])
    slope = np.polyfit(t, m, 1)[0]           # Å²/ps
    return slope / 6.0 * 1e-4 * 1e12 * 1e-20  # cm²/s


def compute_adf(frames, triplets, r_cut=3.0, n_bins=180):
    """Angular Distribution Function (Porter et al. 2022, Eq. 23).

    For each triplet (central, nbr1, nbr2): find all neighbour pairs (j, k)
    within r_cut of centre atom i, with types nbr1 and nbr2 respectively (j≠k);
    compute the j-i-k angle.  Returns (theta_deg array, adf dict peak-normalised).
    """
    theta_edges = np.linspace(0, 180, n_bins + 1)
    theta_mid   = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    adf = {t: np.zeros(n_bins) for t in triplets}
    for frame in frames:
        pos  = frame.get_positions()
        sym  = np.array(frame.get_chemical_symbols())
        cell = np.array(frame.get_cell())
        L    = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
        for (c_type, n1_type, n2_type) in triplets:
            idx_c  = np.where(sym == c_type)[0]
            idx_n1 = np.where(sym == n1_type)[0]
            idx_n2 = np.where(sym == n2_type)[0]
            for ic in idx_c:
                pc = pos[ic]
                vecs_n1 = []
                for j in idx_n1:
                    if j == ic:
                        continue
                    dv = pos[j] - pc
                    dv -= L * np.round(dv / L)
                    d  = np.linalg.norm(dv)
                    if 0 < d < r_cut:
                        vecs_n1.append((j, dv, d))
                vecs_n2 = []
                for k in idx_n2:
                    if k == ic:
                        continue
                    dv = pos[k] - pc
                    dv -= L * np.round(dv / L)
                    d  = np.linalg.norm(dv)
                    if 0 < d < r_cut:
                        vecs_n2.append((k, dv, d))
                for (j, v1, d1) in vecs_n1:
                    for (k, v2, d2) in vecs_n2:
                        if j == k:
                            continue
                        cos_t = np.clip(np.dot(v1, v2) / (d1 * d2), -1.0, 1.0)
                        theta = np.degrees(np.arccos(cos_t))
                        ib    = min(int(theta / 180.0 * n_bins), n_bins - 1)
                        adf[(c_type, n1_type, n2_type)][ib] += 1
    for t in adf:
        peak = adf[t].max()
        if peak > 0:
            adf[t] /= peak
    return theta_mid, adf


# ── Step 1: Geometry relaxation ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 1: Geometry relaxation (BFGS, fmax=0.01 eV/Å)")
print("=" * 60)

naf_relax  = relax(naf_crystal,  lcy_pbe_params(txt="naf_relax.txt"), fmax=0.01, fixcell=False,
                   logname=NAF_RLX_LOG, gpwname=NAF_GPW_FILE)
view(naf_relax, repeat=(2, 2, 2))
naf_relax.write("NaF_aimd_relaxed.xyz")
naf_relax_cellparams = naf_relax.cell.cellpar()
print(f"""
Relaxed NaF  Epot = {naf_relax.get_potential_energy()/len(naf_relax):.4f} eV/atom
       a= {naf_relax_cellparams[0]:.2f}, b={naf_relax_cellparams[1]:.2f}, c={naf_relax_cellparams[2]:.2f}, 
       alpha={naf_relax_cellparams[3]:.2f}, beta={naf_relax_cellparams[4]:.2f}, gamma = {naf_relax_cellparams[5]:.2f}
       Relaxed Volume = {naf_relax.get_volume():.2f} Å³ """)

alf3_relax = relax(alf3_crystal, lcy_pbe_params(txt="alf3_relax.txt"), fmax=0.01, fixcell=False,
                   logname=ALF3_RLX_LOG, gpwname=ALF3_GPW_FILE)
view(alf3_relax, repeat=(2, 2, 2))
alf3_relax.write("AlF3_aimd_relaxed.xyz")
alf3_relax_cellparams = alf3_relax.cell.cellpar()
print(f""" 
Relaxed AlF3  Epot = {alf3_relax.get_potential_energy()/len(alf3_relax):.4f} eV/atom
       a= {alf3_relax_cellparams[0]:.2f}, b={alf3_relax_cellparams[1]:.2f}, c={alf3_relax_cellparams[2]:.2f}, 
       alpha={alf3_relax_cellparams[3]:.2f}, beta={alf3_relax_cellparams[4]:.2f}, gamma = {alf3_relax_cellparams[5]:.2f}
       Relaxed Volume = {alf3_relax.get_volume()} Å³
       """)

# ── Step 2: NVT equilibration ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2: NVT equilibration  T={TEMPERATURE} K  steps={N_EQUIL}")
print("=" * 60)

mix = stack(naf_relax, alf3_relax, maxstrain=1, distance=2.5)
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
    print(f"[{datetime.now():%H:%M:%S}]  equil step {(i+1)*10:4d}/{N_EQUIL}  t={t:.3f} ps  T={T:.1f} K  Epot={Ep:.4f} eV/atom")

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
        print(f"[{datetime.now():%H:%M:%S}]  prod step {(i+1)*10:4d}/{N_PROD}  t={t:.3f} ps  T={T:.1f} K  Epot={Ep:.4f} eV/atom")

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
traj_prod = Trajectory(MIX_TRAJ_PROD, 'r')
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

# ── ADF (Angular Distribution Function, Porter et al. Eq. 23) ─────────────────
print("Computing ADF...")
ADF_TRIPLETS = [('Be', 'F', 'F'), ('Li', 'F', 'F'), ('F', 'Be', 'Be')]
adf_theta, adf_data = compute_adf(sample_frames, ADF_TRIPLETS, r_cut=3.0)

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
fig, axes = plt.subplots(3, 3, figsize=(14, 13))

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
_adf_colors = ['darkorange', 'steelblue', 'green']
for triplet, color in zip(ADF_TRIPLETS, _adf_colors):
    label = f"{triplet[1]}-{triplet[0]}-{triplet[2]}"
    ax.plot(adf_theta, adf_data[triplet], color=color, lw=1.5, label=label)
ax.axvline(109.47, color='gray', ls=':', lw=1, label='109.5° (tet)')
ax.set_xlabel('Angle (°)'); ax.set_ylabel('ADF (norm.)')
ax.set_title('Angular Distribution Function'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_xlim(0, 180)

axes[2, 0].axis('off')
axes[2, 1].axis('off')

ax = axes[2, 2]
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
