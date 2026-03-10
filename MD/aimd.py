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
from ase import units, Atoms
from ase.build import bulk, make_supercell
from ase.optimize import BFGS
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from gpaw import GPAW, PW, restart, Mixer, MixerSum, MixerDif
from ase.units import eV, Ang, Bohr
from ase.filters import FrechetCellFilter

# ── Parameters ────────────────────────────────────────────────────────────────
TEMPERATURE    = 1200       # K  (lower than classical run — AIMD is expensive)
TIMESTEP_FS    = 1.0        # fs  (BOMD can use larger dt than CP-MD)
TDAMP_FS       = 50         # thermostat damping (fs)
N_EQUIL        = 200        # NVT equilibration steps
N_PROD         = 500        # NVT production steps
LOG_INTERVAL   = 10         # steps between log entries
ECUT_EV        = 500        # plane-wave cutoff (eV)
KPTS           = (1, 1, 1)  # k-point grid — small for MD speed
SUPERCELL      = 1          # n×n×n supercell of LiF rocksalt unit cell
SCREEN = 0.25*Bohr 


LIF_TRAJ_EQUIL = "LiF_aimd_equil.traj"
LIF_TRAJ_PROD  = "LiF_aimd_prod.traj"
LIF_LOG_EQUIL  = "LiF_aimd_equil.log"
LIF_RLX_LOG    = "LiF_aimd_relax_opt.log"
LIF_LOG_PROD   = "LiF_aimd_prod.log"
LIF_PLOT_FILE  = "LiF_aimd_results.png"
LIF_GPW_FILE   = "LiF_aimd_relax.gpw"
BEF2_TRAJ_EQUIL ="BeF2_aimd_equil.traj"
BEF2_TRAJ_PROD = "BeF2_aimd_prod.traj"
BEF2_LOG_EQUIL = "BeF2_aimd_equil.log"
BEF2_PLOT_FILE = "BeF2_aimd_results.png"
BEF2_GPW_FILE  = "BeF2_aimd_relax.gpw"
BEF2_RLX_LOG   = "BeF2_aimd_relax_opt.log"
# ── Build LiF 2×2×2 supercell ────────────────────────────────────────────────
print("=" * 60)
print("Building LiF rocksalt supercell")
print("=" * 60)
bef2_frac_cell = [   
    (0.4658484850000000,    0.0000000000000000,    0.3333333333333330),
    (0.0000000000000000,    0.4658484850000000,    0.6666666666666661),
    (0.5341515150000000,    0.5341515150000000,    0.0000000000000000),
    (0.4111597766666660,    0.2772815833333330,    0.2222729999999990),
    (0.7227184166666660,    0.1338781933333330,    0.5556063333333331),
    (0.8661218066666660,    0.5888402233333331,    0.8889396666666660),
    (0.1338781933333330,    0.7227184166666660,    0.4443936666666660),
    (0.5888402233333331,    0.8661218066666660,    0.1110603333333330),
    (0.2772815833333330,    0.4111597766666660,    0.7777270000000001)]
bef2_cell =[(2.3336366944146949,   -4.0419773211333370,    0.0000000000000000),
   (2.3336366944146949,    4.0419773211333370,    0.0000000000000000),
   (0.0000000000000000,    0.0000000000000000,    5.1827969293352050)]
lif_unit   = bulk('LiF', crystalstructure='rocksalt', a=4.03, cubic=True)
bef2_symbols = ['Be', 'Be', 'Be', 'F', 'F', 'F', 'F', 'F','F']
bef2_unit = Atoms(  bef2_symbols,
                    cell = bef2_cell,
                    scaled_positions = bef2_frac_cell
                    )
lif_atoms      = make_supercell(lif_unit, SUPERCELL * np.eye(3, dtype=int))
bef2_atoms = make_supercell(bef2_unit,SUPERCELL*np.eye(3,dtype=int))
print(f"LiF Atoms    : {len(lif_atoms)}")
print(f"LiF Cell (Å) : {np.diag(lif_atoms.cell)}")
print(f"BeF2 Atoms   : {len(bef2_atoms)}")
print(f"BeF2 Cell (A): {np.diag(bef2_atoms.cell)}")

def relax(atoms: Atoms,
          calculator_params: Dict[str, Any],
          fmax: float = 0.01,
          fixcell: bool = True,
          logname: str = 'opt.log',
          trajname: str = 'opt.traj',
          gpwname: str = 'rlx.gpw') -> Atoms:
    """
    Relax atomic structure using GPAW.
    
    Parameters:
    -----------
    atoms : Atoms
        ASE Atoms object to relax
    calculator_params : dict
        Parameters for GPAW calculator
    fmax : float
        Maximum force tolerance (eV/Å)
    fixcell : bool
        If True, only relax atomic positions (ISIF=2 equivalent)
        If False, relax both atoms and cell (ISIF=3 equivalent)
    logname : str
        Name of optimization log file
    trajname : str
        Name of trajectory file
    gpwname : str
        Name of GPAW restart file
    
    Returns:
    --------
    Atoms
        Relaxed atomic structure
    """
    
    # -------------------------
    # Restart or initialize GPAW
    # -------------------------
    calc_dft = None
    
    if os.path.exists(gpwname) and os.path.getsize(gpwname) > 100:
        try:
            atoms, calc_dft = restart(gpwname)
            atoms.calc = calc_dft
            print(f"Successfully restarted from {gpwname}")
        except Exception as e:
            print(f"Restart failed ({e}). Starting fresh calculation.")
            calc_dft = GPAW(**calculator_params)
            atoms.calc = calc_dft
    else:
        calc_dft = GPAW(**calculator_params)
        atoms.calc = calc_dft
        print("Starting fresh calculation")
    # -------------------------
    # Choose relaxation mode
    # -------------------------
    if fixcell:
        # ISIF = 2 equivalent - only relax atoms
        opt_atoms = atoms
        print("Relaxation mode: fixed cell (ISIF=2 equivalent)")
    else:
        # ISIF = 3 equivalent - relax both atoms and cell
        opt_atoms = FrechetCellFilter(atoms)
        print("Relaxation mode: variable cell (ISIF=3 equivalent)")

    # -------------------------
    # Run optimizer
    # -------------------------
    # Add observer to save checkpoints during optimization

    # Create optimizer
    opt = BFGS(opt_atoms, 
               logfile=logname,
               trajectory=trajname)
        
    # Run relaxation
    print(f"Starting relaxation with fmax={fmax} eV/Å")
    opt.run(fmax=fmax, steps=500)  # Added steps limit for safety

    if isinstance(opt_atoms, FrechetCellFilter):
        opt_atoms = opt_atoms.atoms

    # -------------------------
    # Save final state
    # -------------------------
    try:
        # Write final calculator state
        opt_atoms.calc.write(gpwname, mode='all')
        print(f"Final state saved to {gpwname}")        
    except Exception as e:
        print(f"Warning: Could not save final state: {e}")
    
    # Get final forces f or reporting
    if hasattr(opt_atoms, 'get_forces'):
        forces = opt_atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        print(f"Relaxation completed. Maximum force: {max_force:.6f} eV/Å")
    return opt_atoms
# ── GPAW calculator ───────────────────────────────────────────────────────────
# Γ-only or small k-grid for MD (speed priority)
# FermiDirac smearing handles metallic/semi-ionic systems better during MD
def make_gpaw(txt='-',SCREEN=SCREEN, ECUT_EV=ECUT_EV, hunds = False):
    """GPAW PW calculator for isolated dimers (Γ-point, mild smearing)."""
    base_params = {
        "convergence": {"density": 1e-8,
                        "eigenstates": 1e-10,
                        "energy": 1e-6, 
                        "forces": 1e-6},
        "eigensolver": {"name": "rmm-diis",
                        "niter": 5},
        "hund": hunds,
        "kpts": (1, 1, 1),
        "maxiter": 1000,
        "mixer": {"backend": "pulay", 
                  "beta": 0.25,
                  "method": "fullspin",
                  "nmaxold": 5,
                  "weight": 50.0},
        "mode": {"name": "pw",
                 "ecut": ECUT_EV},
        "nbands": "nao",
        #"symmetry": "off",
        "occupations": {"name": "fermi-dirac",
                        "width": 0.01},
        "txt": txt,  
        "xc": { 'backend': 'pw',
               'fraction': 0.25,
               'omega': SCREEN * Bohr,  #bohr^-1
               'name': 'HYB_GGA_XC_HSE06',
               },
    }
    
    return GPAW(**base_params)

# ── Step 1: Geometry relaxation ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 1: Geometry relaxation (BFGS, fmax=0.05 eV/Å)")
print("=" * 60)
base_params = {
        "convergence": {"density": 1e-8,
                        "eigenstates": 1e-10,
                        "energy": 1e-6, 
                        "forces": 1e-6},
        "eigensolver": {"name": "rmm-diis",
                        "niter": 5},
        "kpts": (1, 1, 1),
        "maxiter": 1000,
        "mixer": {"backend": "pulay", 
                  "beta": 0.25,
                  "method": "fullspin",
                  "nmaxold": 5,
                  "weight": 50.0},
        "mode": {"name": "pw",
                 "ecut": ECUT_EV},
        "nbands": "nao",
        #"symmetry": "off",
        "occupations": {"name": "fermi-dirac",
                        "width": 0.01},
        "xc": { 'backend': 'pw',
               'fraction': 0.25,
               'omega': SCREEN * Bohr,  #bohr^-1
               'name': 'HYB_GGA_XC_HSE06',
               },
    }
    
lif_relax = relax(lif_atoms,
                  calculator_params=base_params,
                  fmax = 0.01,
                  fixcell = True,
                  logname = LIF_RLX_LOG,
                  gpwname = LIF_GPW_FILE)

bef2_relax = relax(bef2_atoms,
                   calculator_params=base_params,
                   fixcell = True,
                   logname = BEF2_RLX_LOG,
                   gpwname = BEF2_GPW_FILE)

print(f"  LiF Epot = {lif_relax.get_potential_energy()/len(lif_atoms):.4f} eV/atom"
      f"  BeF2 Epot = {bef2_relax.get_potential_energies()/len(bef2_atoms):.4f} eV/atom")
lif_relax.write("LiF_aimd_relaxed.xyz")
bef2_relax.write("BeF2_aimd_relaxed.xyz")
print("  Relaxed structure → LiF_aimd_relaxed.xyz")

# ── Step 2: NVT equilibration ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2: NVT equilibration  T={TEMPERATURE} K  steps={N_EQUIL}")
print("=" * 60)

# Fresh calculator for MD (new txt file)
lif_relax.calc = make_gpaw(txt='LiF_aimd_equil_gpaw.log')
bef2_relax.calc = make_gpaw(txt='BeF2_aimd_equil_gpaw.log')
mix = lif_relax + bef2_relax
MaxwellBoltzmannDistribution(mix, temperature_K=TEMPERATURE)
Stationary(mix)
ZeroRotation(mix)

dyn_eq = NoseHooverChainNVT(
    mix,
    timestep      = TIMESTEP_FS * units.fs,
    temperature_K = TEMPERATURE,
    tdamp         = TDAMP_FS * units.fs,
    trajectory    = LIF_TRAJ_EQUIL,
    logfile       = LIF_LOG_EQUIL,
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
