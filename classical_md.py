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
from ase import units
from ase.io import read
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md import MDLogger
from ase.calculators.lammpsrun import LAMMPS

# ── Parameters ────────────────────────────────────────────────────────────────
TEMPERATURE     = 2000      # K
TIMESTEP_FS     = 0.5       # fs
TDAMP_FS        = 100       # thermostat damping time (fs)
N_STEPS         = 50_000    # production steps  (25 ps)
LOG_INTERVAL    = 100       # steps between log entries
TRAJ_FILE       = "LiF_BeF2_classical_prod.traj"
LOG_FILE        = "LiF_BeF2_classical_prod.log"
PLOT_FILE       = "LiF_BeF2_classical_results.png"
STRUCT_FILE     = "LiF_BeF2_equilibrated.xyz"

# ── Load equilibrated structure ───────────────────────────────────────────────
print("=" * 60)
print("Loading equilibrated LiF·BeF2 structure")
print("=" * 60)

atoms = read(STRUCT_FILE)
print(f"  Atoms    : {len(atoms)}")
print(f"  Species  : {set(atoms.get_chemical_symbols())}")
print(f"  Cell (Å) : {np.diag(atoms.cell)}")

# ── Ionic charges ─────────────────────────────────────────────────────────────
charge_map = {'Li': 1.0, 'Be': 2.0, 'F': -1.0}
charges    = [charge_map[s] for s in atoms.get_chemical_symbols()]
atoms.set_initial_charges(charges)

# ── Cutoff (must be < L/2 for all box dimensions) ────────────────────────────
cell   = atoms.get_cell()
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
pair_coeff = [
    '1 1    0.0      1.0      0.0  ',   # Li–Li
    '1 2    0.0      1.0      0.0  ',   # Li–Be
    '1 3  593.72   0.26310    0.0  ',   # Li–F
    '2 2    0.0      1.0      0.0  ',   # Be–Be
    '2 3 1389.47   0.23604    0.0  ',   # Be–F
    '3 3 1127.70   0.27533   14.835',   # F–F
]

calc = LAMMPS(
    specorder   = ['Li', 'Be', 'F'],
    atom_style  = 'charge',
    pair_style  = f'buck/coul/long {cutoff}',
    kspace_style= 'ewald 1.0e-5',
    pair_coeff  = pair_coeff,
)
atoms.calc = calc

# ── Reinitialise velocities at target temperature ─────────────────────────────
MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
Stationary(atoms)
ZeroRotation(atoms)

# ── NVT thermostat (Nosé-Hoover chain) ───────────────────────────────────────
dyn = NoseHooverChainNVT(
    atoms,
    timestep    = TIMESTEP_FS * units.fs,
    temperature_K = TEMPERATURE,
    tdamp       = TDAMP_FS * units.fs,
    trajectory  = TRAJ_FILE,
    logfile     = LOG_FILE,
)
dyn.attach(MDLogger(dyn, atoms, "LiF_BeF2_classical_md.log"), interval=LOG_INTERVAL)

# ── Accumulators ──────────────────────────────────────────────────────────────
time_ps, epot_list, ekin_list, temp_list = [], [], [], []
# For VDOS: store velocities every step in a block
vel_block, vel_store_every = [], 10    # store every 10 steps
steps_per_block = 500
n_blocks        = N_STEPS // steps_per_block

def collect():
    time_ps.append(dyn.get_time() / (1000 * units.fs))
    epot_list.append(atoms.get_potential_energy())
    ekin_list.append(atoms.get_kinetic_energy())
    temp_list.append(atoms.get_temperature())

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
        Ep  = epot_list[-1] / len(atoms)
        print(f"  block {i+1:4d}/{n_blocks}  t={t:.3f} ps  "
              f"T={T:.1f} K  Epot={Ep:.4f} eV/atom")

etot_arr = np.array(epot_list) + np.array(ekin_list)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Production Summary")
print("=" * 60)
print(f"  T_mean   : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K")
print(f"  E_drift  : {(etot_arr[-1]-etot_arr[0])/len(atoms):.6f} eV/atom")

# ── RDF ───────────────────────────────────────────────────────────────────────
print("\nComputing RDF from trajectory...")
from ase.io.trajectory import Trajectory as Traj

traj   = Traj(TRAJ_FILE)
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
    cell = frame.get_cell()
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
n_atoms   = len(sample_frames[0])
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
n   = len(atoms)
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
colors_rdf = {'Li-F':'blue','Be-F':'red','F-F':'green','Li-Li':'purple'}
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
ax5.axvline(np.mean(temp_list), color='b', ls='-',
            label=f'Mean {np.mean(temp_list):.0f} K')
ax5.set_xlabel('T (K)'); ax5.set_ylabel('Frequency')
ax5.set_title('Temperature Distribution'); ax5.legend(fontsize=8)

# 6. Diffusion summary text
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
lines = [
    f"System:  LiF·BeF2  ({len(atoms)} atoms)",
    f"T:       {TEMPERATURE} K",
    f"Steps:   {N_STEPS}  ({N_STEPS*TIMESTEP_FS/1000:.1f} ps)",
    f"dt:      {TIMESTEP_FS} fs",
    "",
    f"T_mean:  {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K",
    f"E_drift: {(etot_arr[-1]-etot_arr[0])/len(atoms):.5f} eV/atom",
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
plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
print(f"\nPlot saved → {PLOT_FILE}")
print(f"Trajectory → {TRAJ_FILE}")
