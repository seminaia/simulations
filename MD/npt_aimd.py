"""
NPT AIMD for LiF + BeF2 using GPAW via ASE
============================================
Ensemble : NPT  Nosé-Hoover thermostat + Parrinello-Rahman barostat
Method   : DFT/HSE06 plane-wave basis
Workflow : build → relax → NVT equilibration → NPT production → analysis → plots

Thermophysical properties (Porter et al. 2022, Fig. 3):
  - Density ρ          : average equilibrated volume → ρ = NM / (Na <V>) (Eq. 2)
  - Thermal expansion β: volume fluctuation formula β = Var(V)/(kB T <V>)  (Eq. 3/4)
  - Heat capacity Cp   : enthalpy fluctuations Cp = Var(H)/(kB T²)         (Eq. 7)
"""

import os
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
from ase import units, Atoms
from ase.build import bulk
from ase.build.tools import stack
from ase.data import atomic_masses, chemical_symbols
from ase.filters import FrechetCellFilter
from ase.io import read, write as ase_write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from ase.spacegroup import crystal
from ase.units import Bohr
from ase.visualize import view
from gpaw import GPAW, restart

# ── Parameters ────────────────────────────────────────────────────────────────
TEMPERATURE  = 1200        # K
PRESSURE_PA  = 0.0         # Pa  (ambient)
TIMESTEP_FS  = 1.0         # fs
TDAMP_FS     = 50          # thermostat timescale (fs)
PDAMP_FS     = 500         # barostat timescale (fs)   — ~10× thermostat
N_EQUIL      = 200         # NVT equilibration steps
N_PROD       = 500         # NPT production steps
LOG_INTERVAL = 10
ECUT_EV      = 500
KPTS         = (2, 2, 2)
SUPERCELL    = 1
SCREEN       = 0.25 * Bohr

# Barostat coupling constant (pfactor = bulk_modulus × ttime²)
# Approximate bulk modulus for molten LiF+BeF2 ~ 20 GPa
_BULK_MOD_EV_A3 = 20e9 * (1e10 ** 3) / 1.602e19  # convert Pa to eV/Å³  ≈ 0.125 eV/Å³
PFACTOR = _BULK_MOD_EV_A3 * (PDAMP_FS * units.fs) ** 2

# ── File names ────────────────────────────────────────────────────────────────
LIF_GPW_FILE   = "LiF_aimd_relax.gpw"
LIF_RLX_LOG    = "LiF_aimd_relax_opt.log"
BEF2_GPW_FILE  = "BeF2_aimd_relax.gpw"
BEF2_RLX_LOG   = "BeF2_aimd_relax_opt.log"

MIX_TRAJ_EQUIL = "npt_mix_equil.traj"
MIX_LOG_EQUIL  = "npt_mix_equil.log"
MIX_TRAJ_PROD  = "npt_mix_prod.traj"
MIX_LOG_PROD   = "npt_mix_prod.log"
MIX_PLOT_FILE  = "npt_results.png"

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

for i in range(N_EQUIL // 10):
    dyn_eq.run(10)
    t  = dyn_eq.get_time() / (1000 * units.fs)
    T  = mix.get_temperature()
    Ep = mix.get_potential_energy() / len(mix)
    print(f"  equil step {(i+1)*10:4d}/{N_EQUIL}  t={t:.3f} ps  T={T:.1f} K  Epot={Ep:.4f} eV/atom")

# ── Step 3: NPT production ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 3: NPT production  T={TEMPERATURE} K  P={PRESSURE_PA:.0f} Pa  steps={N_PROD}")
print("=" * 60)

mix.calc = make_gpaw(txt='npt_prod_gpaw.log')

dyn = NPT(
    mix,
    timestep=TIMESTEP_FS * units.fs,
    temperature_K=TEMPERATURE,
    externalstress=PRESSURE_PA,
    ttime=TDAMP_FS * units.fs,
    pfactor=PFACTOR,
    trajectory=MIX_TRAJ_PROD,
    logfile=MIX_LOG_PROD,
)
dyn.attach(MDLogger(dyn, mix, MIX_LOG_PROD), interval=LOG_INTERVAL)

time_ps, epot_list, ekin_list, temp_list, vol_list = [], [], [], [], []

def collect_prod():
    time_ps.append(dyn.get_time() / (1000 * units.fs))
    epot_list.append(mix.get_potential_energy())
    ekin_list.append(mix.get_kinetic_energy())
    temp_list.append(mix.get_temperature())
    vol_list.append(mix.get_volume())   # Å³

dyn.attach(collect_prod, interval=1)

for i in range(N_PROD // 10):
    dyn.run(10)
    t  = time_ps[-1]
    T  = temp_list[-1]
    V  = vol_list[-1]
    Ep = epot_list[-1] / len(mix)
    if (i + 1) % 5 == 0:
        print(f"  NPT step {(i+1)*10:4d}/{N_PROD}  t={t:.3f} ps  T={T:.1f} K  "
              f"V={V:.2f} Å³  Epot={Ep:.4f} eV/atom")

mix.write("npt_final.xyz")
print("\n  Final configuration → npt_final.xyz")

etot_arr = np.array(epot_list) + np.array(ekin_list)
vol_arr  = np.array(vol_list)   # Å³
n        = len(mix)

# ── Density ρ (Eq. 2) ─────────────────────────────────────────────────────────
# ρ = N · M_total / (N_A · <V>)
syms       = mix.get_chemical_symbols()
Z_list     = [chemical_symbols.index(s) for s in syms]
mass_total = sum(atomic_masses[Z] for Z in Z_list)   # amu = g/mol
V_mean_cm3 = np.mean(vol_arr) * 1e-24               # Å³ → cm³
rho_gcm3   = (mass_total / 6.022e23) / V_mean_cm3   # g/cm³
print(f"\n  Density ρ = {rho_gcm3:.4f} g/cm³  (<V> = {np.mean(vol_arr):.2f} Å³)")

# ── Thermal expansion β (volume fluctuation, Eq. 4) ──────────────────────────
# β = Var(V) / (kB T <V>)  [K⁻¹]
V_var   = np.var(vol_arr)            # Å⁶
V_mean  = np.mean(vol_arr)           # Å³
kB_eV   = units.kB                   # eV/K
# V_var in Å⁶, kB*T in eV, V_mean in Å³ → β in Å³/eV = Å³/eV
# We need consistent energy units: kB·T in eV, V in Å³ → β [Å³·K/eV·Å³] = K/eV
# Convert: 1 eV = 1.602e-19 J, 1 Å³ = 1e-30 m³ → β [K⁻¹] = β_raw / (kB_J)
# Simplest: use kB in eV/K, V in Å³ → β = Var(V)[Å⁶] / (kB[eV/K] * T[K] * V[Å³])
#           units: Å⁶ / (eV · Å³) = Å³/eV  — needs conversion by × (1.602e-19/1e-30) = × 1.602e11
# → β [K⁻¹] = Var(V) / (kB * T * V_mean) × (1.602e-19 J/eV) / (1e-30 m³/Å³) × ...
# Actually cleaner: β [K⁻¹] = Var(V)[m⁶] / (kB[J/K] * T[K] * V[m³])
kB_J    = 1.380649e-23   # J/K
V_var_m6   = V_var   * 1e-60   # Å⁶ → m⁶
V_mean_m3  = V_mean  * 1e-30   # Å³ → m³
beta_K     = V_var_m6 / (kB_J * TEMPERATURE * V_mean_m3)   # K⁻¹
print(f"  Thermal expansion β = {beta_K:.4e} K⁻¹")

# ── Heat capacity Cp (Eq. 7) ─────────────────────────────────────────────────
# Cp = Var(H)_NPT / (kB T²)  where H = Epot + Ekin + P·V
# At P=0: H = E_total
H_arr    = etot_arr + PRESSURE_PA * vol_arr * 1e-30 / 1.602e-19   # eV (P·V term ~0 at P=0)
Cp_total = np.var(H_arr) / (units.kB * TEMPERATURE ** 2)          # eV/K
Cp_atom  = Cp_total / n                                             # eV/K/atom
Cp_J     = Cp_atom * 1.602e-19                                      # J/K/atom
print(f"  Heat capacity Cp = {Cp_atom:.4e} eV/K/atom  ({Cp_J:.4e} J/K/atom)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NPT Production Summary")
print("=" * 60)
print(f"  T_mean   : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K")
print(f"  E_drift  : {(etot_arr[-1] - etot_arr[0]) / n:.6f} eV/atom")
print(f"  <V>      : {V_mean:.2f} ± {np.std(vol_arr):.2f} Å³")
print(f"  ρ        : {rho_gcm3:.4f} g/cm³")
print(f"  β        : {beta_K:.4e} K⁻¹")
print(f"  Cp       : {Cp_atom:.4e} eV/K/atom  ({Cp_J:.4e} J/K/atom)")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

ax = axes[0, 0]
ax.plot(time_ps, np.array(epot_list) / n, 'b-',  lw=1.2, label='Potential')
ax.plot(time_ps, np.array(ekin_list) / n, 'r-',  lw=1.2, label='Kinetic')
ax.plot(time_ps, etot_arr / n,            'k--', lw=1.2, label='Total')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('Energy/atom (eV)')
ax.set_title('Energy (NPT)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(time_ps, temp_list, 'g-', lw=1.2)
ax.axhline(TEMPERATURE, color='r', ls='--', alpha=0.7, label=f'Target {TEMPERATURE} K')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('T (K)')
ax.set_title('Temperature'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.plot(time_ps, vol_arr, 'teal', lw=1.2)
ax.axhline(V_mean, color='r', ls='--', alpha=0.7, label=f'⟨V⟩={V_mean:.1f} Å³')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('Volume (Å³)')
ax.set_title('Cell Volume (NPT)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
# Density time series
rho_ts = (mass_total / 6.022e23) / (vol_arr * 1e-24)   # g/cm³
ax.plot(time_ps, rho_ts, 'darkorange', lw=1.2)
ax.axhline(rho_gcm3, color='r', ls='--', alpha=0.7, label=f'⟨ρ⟩={rho_gcm3:.4f} g/cm³')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('ρ (g/cm³)')
ax.set_title('Density'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(time_ps, H_arr / n, 'purple', lw=1.2, label='H/atom')
ax.axhline(np.mean(H_arr) / n, color='r', ls='--', alpha=0.7)
ax.set_xlabel('Time (ps)'); ax.set_ylabel('H/atom (eV)')
ax.set_title('Enthalpy fluctuations (→ Cp)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 2]
ax.axis('off')
lines = [
    "NPT AIMD  LiF+BeF2  (GPAW/HSE06)",
    f"N atoms   : {n}",
    f"E_cut     : {ECUT_EV} eV",
    f"T_target  : {TEMPERATURE} K",
    f"P_target  : {PRESSURE_PA:.0f} Pa",
    f"dt        : {TIMESTEP_FS} fs",
    f"Prod      : {N_PROD} steps ({N_PROD*TIMESTEP_FS/1000:.2f} ps)",
    "",
    f"T_mean    : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K",
    f"<V>       : {V_mean:.2f} ± {np.std(vol_arr):.2f} Å³",
    "",
    f"ρ         : {rho_gcm3:.4f} g/cm³",
    f"β         : {beta_K:.3e} K⁻¹",
    f"Cp        : {Cp_atom:.3e} eV/K/atom",
    f"           ({Cp_J:.3e} J/K/atom)",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va='top', fontsize=9, family='monospace')

plt.suptitle(f'NPT AIMD  LiF+BeF2  T={TEMPERATURE} K  P={PRESSURE_PA:.0f} Pa  '
             f'(GPAW/HSE06, PW-{ECUT_EV}eV)', fontsize=12)
plt.tight_layout()
plt.savefig(MIX_PLOT_FILE, dpi=150, bbox_inches='tight')
print(f"\nPlot saved      → {MIX_PLOT_FILE}")
print(f"Prod trajectory → {MIX_TRAJ_PROD}")
print(f"Final config    → npt_final.xyz")
