"""
NVE AIMD for LiF + BeF2 using GPAW via ASE
============================================
Ensemble : NVE  microcanonical (VelocityVerlet)
Method   : DFT/HSE06 plane-wave basis
Workflow : build → relax → NVT pre-thermalization → NVE production → analysis → plots

Thermophysical properties (Porter et al. 2022, Fig. 3):
  - Energy conservation : total energy drift and std dev (primary NVE diagnostic)
  - VDOS               : velocity autocorrelation FFT
  - MSD / Diffusion    : per species (Eq. 10)
  - Viscosity η        : Green-Kubo stress autocorrelation (Eq. 13)
  - ADF                : angular distribution function (Eq. 23)
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
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from ase.units import Bohr
from ase.visualize import view
from gpaw import GPAW, restart

# ── Parameters ────────────────────────────────────────────────────────────────
TEMPERATURE  = 1200        # K
TIMESTEP_FS  = 1.0         # fs
TDAMP_FS     = 50          # NVT pre-thermalisation damping (fs)
N_EQUIL      = 200         # NVT pre-thermalisation steps
N_PROD       = 2000        # NVE production steps  (2 ps)
LOG_INTERVAL = 10
ECUT_EV      = 500
KPTS         = (2, 2, 2)
SUPERCELL    = 1
SCREEN       = 0.25 * Bohr

# ── File names ────────────────────────────────────────────────────────────────
LIF_GPW_FILE   = "LiF_aimd_relax.gpw"
LIF_RLX_LOG    = "LiF_aimd_relax_opt.log"
BEF2_GPW_FILE  = "BeF2_aimd_relax.gpw"
BEF2_RLX_LOG   = "BeF2_aimd_relax_opt.log"

MIX_TRAJ_EQUIL = "nve_mix_equil.traj"
MIX_LOG_EQUIL  = "nve_mix_equil.log"
MIX_TRAJ_PROD  = "nve_mix_prod.traj"
MIX_LOG_PROD   = "nve_mix_prod.log"
MIX_PLOT_FILE  = "nve_results.png"

# ── Build structures ──────────────────────────────────────────────────────────
print("=" * 60)
print("Building LiF and BeF2 supercells")
print("=" * 60)

lif_atoms  = bulk('LiF', crystalstructure='rocksalt', a=4.03, cubic=True)
lif_atoms.set_initial_charges([-1, 1] * (len(lif_atoms) // 2))
lif_atoms.set_initial_magnetic_moments([1, -1, 1, -1, 0.5, -0.5, 0.5, -0.5])
_bef2_cell = [[4.77, 0, 0],
              [-4.77/2, 4.77*np.sqrt(3)/2, 0],
              [0, 0, 5.18]]
bef2_atoms = Atoms(
    symbols=['Be', 'Be', 'F', 'F', 'F', 'F'],
    scaled_positions=[
        (0.0, 0.0, 0.0),
        (1/3, 2/3, 0.5),
        (0.2, 0.4, 0.25),
        (0.8, 0.6, 0.75),
        (0.4, 0.2, 0.75),
        (0.6, 0.8, 0.25),
    ],
    cell=_bef2_cell,
    pbc=True)
bef2_atoms = bef2_atoms.repeat([1, 1, 2])  # 6 → 12 atoms
bef2_atoms.set_initial_charges([2, 2, -1, -1, -1, -1] * 2)
bef2_atoms.set_initial_magnetic_moments([2, 2, -1, -1, -1, -1] * 2)

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
    slope = np.polyfit(t, m, 1)[0]
    return slope / 6.0 * 1e-4 * 1e12 * 1e-20


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

lif_relax.write("LiF_aimd_relaxed.xyz")
bef2_relax.write("BeF2_aimd_relaxed.xyz")

# ── Step 2: NVT pre-thermalisation ────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2: NVT pre-thermalisation  T={TEMPERATURE} K  steps={N_EQUIL}")
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

for i in range(N_EQUIL // 10):
    dyn_eq.run(10)
    t  = dyn_eq.get_time() / (1000 * units.fs)
    T  = mix.get_temperature()
    Ep = mix.get_potential_energy() / len(mix)
    print(f"  equil step {(i+1)*10:4d}/{N_EQUIL}  t={t:.3f} ps  T={T:.1f} K  Epot={Ep:.4f} eV/atom")

# ── Step 3: NVE production ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 3: NVE production  steps={N_PROD}  ({N_PROD*TIMESTEP_FS/1000:.1f} ps)")
print("=" * 60)

# Replace thermostat with plain Velocity Verlet — velocities from NVT are preserved
mix.calc = make_gpaw(txt='nve_prod_gpaw.log')

dyn = VelocityVerlet(
    mix,
    timestep=TIMESTEP_FS * units.fs,
    trajectory=MIX_TRAJ_PROD,
    logfile=MIX_LOG_PROD,
)
dyn.attach(MDLogger(dyn, mix, MIX_LOG_PROD), interval=LOG_INTERVAL)

time_ps, epot_list, ekin_list, temp_list = [], [], [], []
vel_list   = []
stress_list = []   # off-diagonal stress tensor components [xy, xz, yz] in eV/Å³

def collect_prod():
    time_ps.append(dyn.get_time() / (1000 * units.fs))
    epot_list.append(mix.get_potential_energy())
    ekin_list.append(mix.get_kinetic_energy())
    temp_list.append(mix.get_temperature())
    vel_list.append(mix.get_velocities().copy())
    # Stress tensor (3×3), eV/Å³; off-diagonal = [0,1], [0,2], [1,2]
    s = mix.get_stress(voigt=False)   # (3,3) array
    stress_list.append([s[0, 1], s[0, 2], s[1, 2]])

dyn.attach(collect_prod, interval=1)

for i in range(N_PROD // 10):
    dyn.run(10)
    t  = time_ps[-1]
    T  = temp_list[-1]
    Ep = epot_list[-1] / len(mix)
    if (i + 1) % 50 == 0:
        print(f"  NVE step {(i+1)*10:5d}/{N_PROD}  t={t:.3f} ps  T={T:.1f} K  Epot={Ep:.4f} eV/atom")

mix.write("nve_final.xyz")
print("\n  Final configuration → nve_final.xyz")

etot_arr = np.array(epot_list) + np.array(ekin_list)
n        = len(mix)

# ── Energy conservation diagnostics ───────────────────────────────────────────
E_mean  = np.mean(etot_arr) / n
E_std   = np.std(etot_arr)  / n
E_drift = (etot_arr[-1] - etot_arr[0]) / n
print(f"\n  E_total mean  : {E_mean:.6f} eV/atom")
print(f"  E_total std   : {E_std:.6f} eV/atom")
print(f"  E_total drift : {E_drift:.6f} eV/atom")

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

# ── MSD ────────────────────────────────────────────────────────────────────────
print("Computing MSD...")
traj_prod = Trajectory(MIX_TRAJ_PROD)
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

# ── Viscosity η via Green-Kubo (Eq. 13) ───────────────────────────────────────
# η = V/(kB T) ∫₀^∞ <σ_αβ(0)·σ_αβ(t)> dt
# averaged over off-diagonal components xy, xz, yz
print("Computing viscosity via Green-Kubo stress ACF...")
stress_arr = np.array(stress_list)    # (N_steps, 3) — [xy, xz, yz] in eV/Å³
vol_A3     = mix.get_volume()          # Å³

# Stress ACF: average over 3 components and all time origins (half-trajectory)
max_lag_s  = n_steps // 2
sacf       = np.zeros(max_lag_s)
for lag in range(max_lag_s):
    sacf[lag] = np.mean(
        np.sum(stress_arr[:n_steps - lag] * stress_arr[lag:], axis=1)
    )  # mean over time origins, sum already over 3 components → divide by 3
sacf /= 3.0   # average over xy, xz, yz

# Running integral for convergence check
eta_running = np.array([
    vol_A3 * np.trapz(sacf[:k + 1], dx=TIMESTEP_FS) / (units.kB * TEMPERATURE)
    for k in range(max_lag_s)
])   # units: Å³ · eV/Å³ · fs / (eV/K · K) = fs = 1e-15 s·Å⁻³... need conversion

# Unit conversion: [eV/Å³ · fs] → Pa·s
# 1 eV = 1.602e-19 J; 1 Å³ = 1e-30 m³; 1 fs = 1e-15 s
# eV/Å³ · fs = 1.602e-19 J / 1e-30 m³ · 1e-15 s = 1.602e-4 Pa·s
EV_A3_FS_TO_PAS = 1.602e-4
eta_PaS     = eta_running * EV_A3_FS_TO_PAS   # Pa·s
eta_plateau = float(np.mean(eta_PaS[max_lag_s // 2:]))  # average last half as plateau

lag_time_ps = np.arange(max_lag_s) * TIMESTEP_FS / 1000

# ── ADF (Angular Distribution Function, Porter et al. Eq. 23) ─────────────────
print("Computing ADF...")
n_traj_nve  = len(traj_prod)
sample_frames = [traj_prod[i] for i in range(n_traj_nve // 2, n_traj_nve,
                                              max(1, n_traj_nve // 200))]
ADF_TRIPLETS = [('Be', 'F', 'F'), ('Li', 'F', 'F'), ('F', 'Be', 'Be')]
adf_theta, adf_data = compute_adf(sample_frames, ADF_TRIPLETS, r_cut=3.0)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NVE Production Summary")
print("=" * 60)
print(f"  T_mean    : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K")
print(f"  E_mean    : {E_mean:.6f} eV/atom")
print(f"  E_std     : {E_std:.6f} eV/atom")
print(f"  E_drift   : {E_drift:.6f} eV/atom")
print(f"  η (GK)    : {eta_plateau:.4e} Pa·s")
print(f"  D(Li)     : {diffusion_cm2s(msd_Li, msd_time):.3e} cm²/s")
print(f"  D(Be)     : {diffusion_cm2s(msd_Be, msd_time):.3e} cm²/s")
print(f"  D(F)      : {diffusion_cm2s(msd_F,  msd_time):.3e} cm²/s")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(14, 13))

ax = axes[0, 0]
ax.plot(time_ps, etot_arr / n, 'k-', lw=1.2, label='Total')
ax.plot(time_ps, np.array(epot_list) / n, 'b-', lw=0.8, alpha=0.7, label='Potential')
ax.plot(time_ps, np.array(ekin_list) / n, 'r-', lw=0.8, alpha=0.7, label='Kinetic')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('Energy/atom (eV)')
ax.set_title('Energy conservation (NVE)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(time_ps, temp_list, 'g-', lw=1.2)
ax.axhline(TEMPERATURE, color='r', ls='--', alpha=0.7, label=f'Initial {TEMPERATURE} K')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('T (K)')
ax.set_title('Temperature (NVE — no control)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 2]
mask = freqs < 25
ax.plot(freqs[mask], vdos_raw[mask], 'navy', lw=1.5)
ax.fill_between(freqs[mask], vdos_raw[mask], alpha=0.2, color='navy')
ax.set_xlabel('Frequency (THz)'); ax.set_ylabel('VDOS (arb.)')
ax.set_title('VDOS'); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(msd_time, msd_Li, 'b-', lw=1.5, label='Li')
ax.plot(msd_time, msd_Be, 'r-', lw=1.5, label='Be')
ax.plot(msd_time, msd_F,  'g-', lw=1.5, label='F')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('MSD (Å²)')
ax.set_title('Mean Square Displacement'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(lag_time_ps, sacf / sacf[0], 'purple', lw=1.5)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Lag time (ps)'); ax.set_ylabel('C(τ) / C(0)')
ax.set_title('Stress autocorrelation (off-diag avg)'); ax.grid(alpha=0.3)

ax_twin = ax.twinx()
ax_twin.plot(lag_time_ps, eta_PaS, 'darkorange', lw=1.2, ls='--', label='Running η')
ax_twin.set_ylabel('η (Pa·s)', color='darkorange')
ax_twin.tick_params(axis='y', labelcolor='darkorange')

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
    "NVE AIMD  LiF+BeF2  (GPAW/HSE06)",
    f"N atoms   : {n}",
    f"E_cut     : {ECUT_EV} eV",
    f"T_initial : {TEMPERATURE} K  (NVT pre-thermalised)",
    f"dt        : {TIMESTEP_FS} fs",
    f"Prod      : {N_PROD} steps ({N_PROD*TIMESTEP_FS/1000:.2f} ps)",
    "",
    f"T_mean    : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K",
    f"E_std     : {E_std:.5f} eV/atom",
    f"E_drift   : {E_drift:.5f} eV/atom",
    "",
    f"η (GK)    : {eta_plateau:.3e} Pa·s",
    "",
    "Diffusion (last 50% MSD):",
    f"  D(Li) = {diffusion_cm2s(msd_Li, msd_time):.3e} cm²/s",
    f"  D(Be) = {diffusion_cm2s(msd_Be, msd_time):.3e} cm²/s",
    f"  D(F)  = {diffusion_cm2s(msd_F,  msd_time):.3e} cm²/s",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va='top', fontsize=9, family='monospace')

plt.suptitle(f'NVE AIMD  LiF+BeF2  T≈{TEMPERATURE} K  (GPAW/HSE06, PW-{ECUT_EV}eV)', fontsize=12)
plt.tight_layout()
plt.savefig(MIX_PLOT_FILE, dpi=150, bbox_inches='tight')
print(f"\nPlot saved      → {MIX_PLOT_FILE}")
print(f"Prod trajectory → {MIX_TRAJ_PROD}")
print(f"Final config    → nve_final.xyz")
