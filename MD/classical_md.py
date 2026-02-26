"""
Classical MD for LiF·BeF2 + H₂ using LAMMPS via ASE
=====================================================
Force field : Buckingham + Ewald (Coulomb)
Potentials  : Tosi-Fumi (1964) for LiF, extended to Be-F and H-F
H₂ model   : neutral H atoms (charge 0) — models dissociated H₂ in melt.
              Charge-neutral by construction (no Ewald correction needed).
              H–H and H–F short-range repulsion via Buckingham only
              (no Coulomb since q_H = 0).
H params    : approximate soft-sphere repulsion — verify against ab initio
              for quantitative work.
Workflow    : load equilibrated structure → insert H₂ pairs →
              energy minimise → NVT production → RDF + MSD analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import units, Atom, Atoms
from ase.build import bulk, make_supercell
from ase.optimize import BFGS
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase.calculators.lammpsrun import LAMMPS

# ── Parameters ────────────────────────────────────────────────────────────────
TEMPERATURE     = 1000      # K
TIMESTEP_FS     = 0.5       # fs
TDAMP_FS        = 100       # thermostat damping (fs)
N_STEPS         = 50000    # production steps (25 ps)
LOG_INTERVAL    = 100       # steps between log entries
N_H_PAIRS       = 20        # number of H₂ pairs to insert (2 H atoms each)
MIN_INSERT_DIST = 1.8       # Å — minimum distance when placing new atoms
RNG_SEED        = 42

TRAJ_FILE   = "LiF_BeF2_H2_classical_1000K.traj"
LOG_FILE    = "LiF_BeF2_H2_classical_1000K.log"
PLOT_FILE   = "LiF_BeF2_H2_classical_1000K.png"

# ── Build LiF·BeF2 combined structure ────────────────────────────────────────
print("=" * 60)
print("Building LiF·BeF2 structure")
print("=" * 60)

lif_unit    = bulk('LiF', crystalstructure='rocksalt', a=4.03, cubic=True)
lif_crystal = make_supercell(lif_unit, 3 * np.eye(3, dtype=int))

bef2_unit    = bulk('BeF2', crystalstructure='fluorite', a=4.77, c=5.18, cubic=True)
bef2_crystal = make_supercell(bef2_unit, 3 * np.eye(3, dtype=int))

lif_pos  = lif_crystal.get_positions()
bef2_pos = bef2_crystal.get_positions()
lif_cell = lif_crystal.cell
bef2_cell = bef2_crystal.cell

x_offset = np.linalg.norm(lif_cell[0])
bef2_pos_shifted = bef2_pos + [x_offset, 0, 0]

combined_cell = [
    [x_offset + np.linalg.norm(bef2_cell[0]), 0, 0],
    [0, max(lif_cell[1][1], bef2_cell[1][1]), 0],
    [0, 0, max(lif_cell[2][2], bef2_cell[2][2])],
]
flibe_atoms = Atoms(
    symbols   = list(lif_crystal.symbols) + list(bef2_crystal.symbols),
    positions = np.vstack([lif_pos, bef2_pos_shifted]),
    cell      = combined_cell,
    pbc       = True,
)
print(f"  Atoms before H insertion : {len(flibe_atoms)}")
print(f"  Species                  : {set(flibe_atoms.get_chemical_symbols())}")
print(f"  Cell (Å)                 : {np.diag(flibe_atoms.cell)}")

# ── Insert H⁺ / H⁺ pairs at random interstitial sites ────────────────────────
print(f"\nInserting {N_H_PAIRS} H₂ pairs (neutral H atoms, charge=0)...")

rng       = np.random.default_rng(RNG_SEED)
cell_diag = np.diag(flibe_atoms.cell)

def _mic_distances(new_pos, existing_pos, cell_diag):
    """Minimum-image distances from new_pos to all existing_pos."""
    diff  = existing_pos - new_pos
    diff -= cell_diag * np.round(diff / cell_diag)
    return np.linalg.norm(diff, axis=1)

def random_interstitial(atoms_obj, min_dist, max_tries=2000):
    """Return a random position inside the cell that is ≥ min_dist from all atoms."""
    cd   = np.diag(atoms_obj.cell)
    pos  = atoms_obj.get_positions()
    for _ in range(max_tries):
        trial = rng.random(3) * cd
        if pos.shape[0] == 0 or _mic_distances(trial, pos, cd).min() > min_dist:
            return trial
    raise RuntimeError("Could not find an interstitial site — cell too crowded.")

n_inserted = 0
for _ in range(N_H_PAIRS):
    h_pos = random_interstitial(flibe_atoms, MIN_INSERT_DIST)
    flibe_atoms.append(Atom('H', position=h_pos))
    h2_pos = random_interstitial(flibe_atoms, MIN_INSERT_DIST)
    flibe_atoms.append(Atom('H', position=h2_pos))
    n_inserted += 1

print(f"  Inserted {2*n_inserted} H atoms ({n_inserted} H₂ pairs)")
print(f"  Total atoms after insertion : {len(flibe_atoms)}")
sym_counts = {s: list(flibe_atoms.get_chemical_symbols()).count(s)
              for s in sorted(set(flibe_atoms.get_chemical_symbols()))}
print(f"  Composition : {sym_counts}")

# ── Ionic charges ─────────────────────────────────────────────────────────────
charge_map = {'Li': 1.0, 'Be': 2.0, 'F': -1.0, 'H': 0.0}  # H₂ neutral — charge 0
charges    = [charge_map[s] for s in flibe_atoms.get_chemical_symbols()]
flibe_atoms.set_initial_charges(charges)
net_charge = sum(charges)
print(f"  Net charge   : {net_charge:+.1f}  (must be 0 for Ewald)")

# ── Cutoff ────────────────────────────────────────────────────────────────────
cell   = flibe_atoms.get_cell()
L_min  = min(cell[0][0], cell[1][1], cell[2][2])
cutoff = round(L_min / 2 - 0.5, 1)
print(f"  Cutoff       : {cutoff} Å")

# ── BORN-Mayer-Huggins pair coefficients  (specorder: Li=1, Be=2, F=3, H=4) ──────────
#
#  Pair    A (eV)     ρ (Å)  sigma(Å)     C (eV·Å⁶) D(eV A^8) cutoff   Source
#  Li–Li    0.0       1.0        0.0         
#  Li–Be    0.0       1.0        0.0         Coulomb only
#  Li–F   593.72    0.26310      0.0         Tosi-Fumi 1964
#  Li–H     0.0       1.0        0.0         Coulomb only
#  Be–Be    0.0       1.0        0.0         Coulomb only
#  Be–F  1389.47    0.23604      0.0         Tosi-Fumi extended
#  Be–H     0.0       1.0        0.0         Coulomb only
#  F–F   1127.70    0.27533     14.835       Tosi-Fumi 1964
#  F–H     80.0     0.30000      0.0         soft repulsion for neutral H₂ in ionic melt
#  H–H     50.0     0.30000      0.0         prevents H–H collapse (no Coulomb, q=0)
pair_coeff = [
    '1 1    195.91      1.0      0.0  ',   # Li–Li
    '1 2    151.72      1.0      0.0  ',   # Li–Be
    '1 3  593.72   0.26310    0.0  ',   # Li–F
    '1 4    0.0      1.0      0.0  ',   # Li–H
    '2 2    0.0      1.0      0.0  ',   # Be–Be
    '2 3 1389.47   0.23604    0.0  ',   # Be–F
    '2 4    0.0      1.0      0.0  ',   # Be–H
    '3 3 1127.70   0.27533   14.835',   # F–F
    '3 4   80.0    0.30000    0.0  ',   # F–H  
    '4 4   50.0    0.30000    0.0  ',   # H–H
]
parameters = {
    'units': 'metal',
    'atom_style': 'charge',
    'pair_style': f'born/coul/long {cutoff}',
    'kspace_style': 'ewald 1.0e-5',
    'pair_coeff': pair_coeff,
}
calc = LAMMPS(
    specorder    = ['Li', 'Be', 'F', 'H'],
    atom_style   = 'charge',
    pair_style   = f'born/coul/long {cutoff}',
    kspace_style = 'ewald 1.0e-5',
    pair_coeff   = pair_coeff,
)
flibe_atoms.calc = calc

# ── Energy minimisation to relax inserted atoms ───────────────────────────────
print("\nMinimising energy to relax inserted H₂ positions (fmax=0.05 eV/Å)...")
minimiser = BFGS(flibe_atoms, logfile='LiF_BeF2_H2_minimisation.log')
minimiser.run(fmax=0.05, steps=300)
print(f"  Epot = {flibe_atoms.get_potential_energy()/len(flibe_atoms):.4f} eV/atom")

# ── Reinitialise velocities ───────────────────────────────────────────────────
MaxwellBoltzmannDistribution(flibe_atoms, temperature_K=TEMPERATURE)
Stationary(flibe_atoms)
ZeroRotation(flibe_atoms)

# ── NVT thermostat ────────────────────────────────────────────────────────────
dyn = NoseHooverChainNVT(
    flibe_atoms,
    timestep      = TIMESTEP_FS * units.fs,
    temperature_K = TEMPERATURE,
    tdamp         = TDAMP_FS * units.fs,
    trajectory    = TRAJ_FILE,
    logfile       = LOG_FILE,
)
dyn.attach(MDLogger(dyn, flibe_atoms, "LiF_BeF2_H2_md.log"), interval=LOG_INTERVAL)

# ── Accumulators ──────────────────────────────────────────────────────────────
time_ps, epot_list, ekin_list, temp_list = [], [], [], []
steps_per_block = 500
n_blocks        = N_STEPS // steps_per_block

def collect():
    time_ps.append(dyn.get_time() / (1000 * units.fs))
    epot_list.append(flibe_atoms.get_potential_energy())
    ekin_list.append(flibe_atoms.get_kinetic_energy())
    temp_list.append(flibe_atoms.get_temperature())

dyn.attach(collect, interval=steps_per_block)

# ── Production run ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"NVT Production  T={TEMPERATURE} K  dt={TIMESTEP_FS} fs  "
      f"{N_STEPS} steps  ({N_STEPS*TIMESTEP_FS/1000:.1f} ps)")
print("=" * 60)

for i in range(n_blocks):
    dyn.run(steps_per_block)
    if (i + 1) % 10 == 0:
        print(f"  block {i+1:4d}/{n_blocks}  t={time_ps[-1]:.3f} ps  "
              f"T={temp_list[-1]:.1f} K  "
              f"Etot={(epot_list[-1]+ekin_list[-1])/len(flibe_atoms):.4f} eV/atom")

etot_arr = np.array(epot_list) + np.array(ekin_list)

print("\n" + "=" * 60)
print("Production Summary")
print("=" * 60)
print(f"  T_mean  : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K")
print(f"  E_drift : {(etot_arr[-1]-etot_arr[0])/len(flibe_atoms):.6f} eV/atom")

# ── RDF from trajectory (last 50% of frames) ──────────────────────────────────
print("\nComputing RDF...")
traj        = Trajectory(TRAJ_FILE)
n_traj      = len(traj)
sample_frames = [traj[i] for i in range(n_traj//2, n_traj, max(1, n_traj//200))]

r_max   = cutoff - 0.5
n_bins  = 200
r_edges = np.linspace(0, r_max, n_bins + 1)
r_mid   = 0.5 * (r_edges[:-1] + r_edges[1:])
dr      = r_edges[1] - r_edges[0]

# Pairs to track — H-* pairs added
pairs  = [('Li','F'), ('Be','F'), ('F','F'), ('H','F'), ('H','Li'), ('H','Be'), ('H','H')]
counts = {p: np.zeros(n_bins) for p in pairs}

def _canonical(s1, s2):
    """Return canonical (alphabetically sorted) pair key."""
    return (s1, s2) if (s1, s2) in counts else ((s2, s1) if (s2, s1) in counts else None)

n_frames_used = 0
for frame in sample_frames:
    pos      = frame.get_positions()
    sym      = frame.get_chemical_symbols()
    cd       = np.diag(frame.get_cell())
    n_frames_used += 1
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            key = _canonical(sym[i], sym[j])
            if key is None:
                continue
            d = pos[j] - pos[i]
            d -= cd * np.round(d / cd)
            dist = np.linalg.norm(d)
            if dist < r_max:
                idx = int(dist / dr)
                if idx < n_bins:
                    counts[key][idx] += 1

# Normalise to g(r)
sym_all        = sample_frames[0].get_chemical_symbols()
species_count  = {s: sym_all.count(s) for s in set(sym_all)}
vol            = sample_frames[0].get_volume()
rdf = {}
for (s1, s2), cnt in counts.items():
    n1   = species_count.get(s1, 0)
    n2   = species_count.get(s2, 0)
    rho  = n2 / vol
    shell = (4/3) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
    norm  = n1 * rho * shell * n_frames_used
    rdf[(s1, s2)] = cnt / np.where(norm > 0, norm, 1)

# ── MSD per species ───────────────────────────────────────────────────────────
print("Computing MSD...")
msd = {s: [] for s in ('Li', 'Be', 'F', 'H')}
pos0, idx = None, {}

for frame in traj:
    sym = np.array(frame.get_chemical_symbols())
    pos = frame.get_positions()
    if pos0 is None:
        pos0 = pos.copy()
        for s in msd:
            idx[s] = np.where(sym == s)[0]
    disp = pos - pos0
    for s in msd:
        if len(idx[s]):
            msd[s].append(np.mean(np.sum(disp[idx[s]]**2, axis=1)))
        else:
            msd[s].append(0.0)

msd_time = np.arange(len(msd['Li'])) * TIMESTEP_FS / 1000   # ps

def diffusion_cm2s(msd_arr, time_arr):
    """D from linear fit over last 50% of MSD trace. Returns cm²/s."""
    n     = len(msd_arr)
    slope = np.polyfit(time_arr[n//2:], np.array(msd_arr[n//2:]), 1)[0]  # Å²/ps
    return slope / 6.0 * 1e-4 * 1e12 * 1e-20

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. Energy
ax = axes[0, 0]
n  = len(flibe_atoms)
ax.plot(time_ps, np.array(epot_list)/n, 'b-',  lw=1.2, label='Potential')
ax.plot(time_ps, np.array(ekin_list)/n, 'r-',  lw=1.2, label='Kinetic')
ax.plot(time_ps, etot_arr/n,            'k--', lw=1.2, label='Total')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('Energy/atom (eV)')
ax.set_title('Energy'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 2. Temperature
ax = axes[0, 1]
ax.plot(time_ps, temp_list, 'g-', lw=1.2)
ax.axhline(TEMPERATURE, color='r', ls='--', alpha=0.7, label=f'Target {TEMPERATURE} K')
ax.set_xlabel('Time (ps)'); ax.set_ylabel('T (K)')
ax.set_title('Temperature'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 3. RDF — split into host pairs and H pairs
ax = axes[0, 2]
host_pairs = [('Li','F'), ('Be','F'), ('F','F')]
colors_host = ['steelblue', 'tomato', 'seagreen']
for (s1, s2), c in zip(host_pairs, colors_host):
    ax.plot(r_mid, rdf[(s1,s2)], color=c, lw=1.5, label=f'{s1}–{s2}')
ax.set_xlabel('r (Å)'); ax.set_ylabel('g(r)')
ax.set_title('RDF  (host pairs)'); ax.legend(fontsize=8)
ax.grid(alpha=0.3); ax.set_xlim(0, r_max)

# 4. RDF — H pairs
ax = axes[1, 0]
h_pairs  = [('H','F'), ('H','Li'), ('H','Be'), ('H','H')]
colors_h = ['crimson', 'royalblue', 'darkorange', 'purple']
for (s1, s2), c in zip(h_pairs, colors_h):
    if (s1,s2) in rdf:
        ax.plot(r_mid, rdf[(s1,s2)], color=c, lw=1.5, label=f'{s1}–{s2}')
ax.set_xlabel('r (Å)'); ax.set_ylabel('g(r)')
ax.set_title('RDF  (H pairs)'); ax.legend(fontsize=8)
ax.grid(alpha=0.3); ax.set_xlim(0, r_max)

# 5. MSD per species
ax = axes[1, 1]
colors_msd = {'Li': 'steelblue', 'Be': 'tomato', 'F': 'seagreen', 'H': 'darkorange'}
for s, c in colors_msd.items():
    if any(v > 0 for v in msd[s]):
        ax.plot(msd_time, msd[s], color=c, lw=1.5, label=s)
ax.set_xlabel('Time (ps)'); ax.set_ylabel('MSD (Å²)')
ax.set_title('Mean Square Displacement'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 6. Summary
ax = axes[1, 2]
ax.axis('off')
d_lines = []
for s in ('Li', 'Be', 'F', 'H'):
    if any(v > 0 for v in msd[s]):
        D = diffusion_cm2s(msd[s], msd_time)
        d_lines.append(f"  D({s:<2s}) = {D:.3e} cm²/s")
lines = [
    f"System : LiF·BeF2 + {N_H_PAIRS} H2  ({len(flibe_atoms)} atoms)",
    f"T      : {TEMPERATURE} K",
    f"Steps  : {N_STEPS}  ({N_STEPS*TIMESTEP_FS/1000:.1f} ps)",
    f"dt     : {TIMESTEP_FS} fs",
    "",
    f"T_mean : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K",
    f"E_drift: {(etot_arr[-1]-etot_arr[0])/len(flibe_atoms):.5f} eV/atom",
    "",
    "Diffusion (last 50% MSD linear fit):",
] + d_lines
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va='top', fontsize=9, family='monospace')

plt.suptitle(f'Classical MD  LiF·BeF2+H2  T={TEMPERATURE} K  (LAMMPS/ASE)', fontsize=12)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
print(f"\nPlot saved  → {PLOT_FILE}")
print(f"Trajectory  → {TRAJ_FILE}")
