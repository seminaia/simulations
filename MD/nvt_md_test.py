import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from ase.calculators.lammpsrun import LAMMPS
from ase.visualize import view
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase import Atoms
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md import MDLogger
from ase.optimize import BFGS
from ase.build import bulk, make_supercell
# Set working directory to current script location
if 'ipykernel' in sys.modules:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        print(f"Changed working directory to: {os.getcwd()}")
    except NameError:
        print(f"Working directory: {os.getcwd()}")

# ============================================================
#  CREATE INDIVIDUAL LATTICES
# ============================================================
print("="*60)
print("Creating crystal lattices")
print("="*60)

# LiF - Rocksalt structure
a_lif = 4.03  # Å
lif_unit = bulk('LiF', crystalstructure='rocksalt', a=a_lif, cubic=True)
# Make a supercell
lif_crystal = make_supercell(lif_unit, 3 * np.eye(3, dtype=int))  # 3x3x3
print(f"LiF crystal: {len(lif_crystal)} atoms")
print(f"  Cell: {lif_crystal.cell}")
print(f"  Volume: {lif_crystal.get_volume():.2f} Å³")

# BeF2 - Create a tetragonal structure (simplified)
a_bef2 = 4.77
c_bef2 = 5.18
bef2_cell = [[a_bef2, 0, 0],
             [0, a_bef2, 0],
             [0, 0, c_bef2]]

# Simple tetragonal structure (approximate)
bef2_scaled_positions = [
    (0.0, 0.0, 0.0),    # Be
    (0.5, 0.5, 0.5),    # Be
    (0.25, 0.25, 0.25), # F
    (0.75, 0.75, 0.75), # F
    (0.25, 0.75, 0.0),  # F
    (0.75, 0.25, 0.5),  # F
]
bef2_symbols = ['Be', 'Be', 'F', 'F', 'F', 'F']
#bef2_unit = Atoms(symbols=bef2_symbols,
#                  scaled_positions=bef2_scaled_positions,
#                  cell=bef2_cell,
#                  pbc=True)

bef2_unit = bulk('BeF2', crystalstructure='fluorite', a=a_bef2, c=c_bef2,cubic=True)

# Make a supercell
bef2_crystal = make_supercell(bef2_unit, 3 * np.eye(3, dtype=int))
print(f"\nBeF2 crystal: {len(bef2_crystal)} atoms")
print(f"  Cell: {bef2_crystal.cell}")
print(f"  Volume: {bef2_crystal.get_volume():.2f} Å³")

# ============================================================
#  COMBINE THE TWO LATTICES
# ============================================================
print("\n" + "="*60)
print("Combining lattices")
print("="*60)

# Get the dimensions of each crystal
lif_positions = lif_crystal.get_positions()
lif_cell = lif_crystal.cell

bef2_positions = bef2_crystal.get_positions()
bef2_cell = bef2_crystal.cell

# Calculate offsets to place them side by side
lif_width = np.linalg.norm(lif_cell[0])  # Width in x-direction
bef2_width = np.linalg.norm(bef2_cell[0])

# Offset BeF2 in x-direction by the width of LiF plus some gap
gap = 0.0  # Å gap between crystals
x_offset = lif_width + gap

# Translate BeF2 positions
bef2_positions_translated = bef2_positions + [x_offset, 0, 0]

# Combine atoms
combined_symbols = list(lif_crystal.symbols) + list(bef2_crystal.symbols)
combined_positions = np.vstack([lif_positions, bef2_positions_translated])

# Create combined cell that contains both
combined_cell = [
    [lif_width + gap + bef2_width, 0, 0],
    [0, max(lif_cell[1][1], bef2_cell[1][1]), 0],
    [0, 0, max(lif_cell[2][2], bef2_cell[2][2])]
]

combined_atoms = Atoms(symbols=combined_symbols,
                       positions=combined_positions,
                       cell=combined_cell,
                       pbc=True)

print(f"Combined system: {len(combined_atoms)} atoms")
print(f"  LiF atoms: {len(lif_crystal)}")
print(f"  BeF2 atoms: {len(bef2_crystal)}")
print(f"  Combined cell: {combined_cell}")

# ============================================================
#  VISUALIZE THE COMBINED STRUCTURE
# ============================================================
print("\nOpening visualization window...")
view(combined_atoms)

# ── Assign formal ionic charges ────────────────────────────────────────────────
charge_map = {'Li': 1.0, 'Be': 2.0, 'F': -1.0}
charges    = [charge_map[s] for s in combined_atoms.get_chemical_symbols()]
combined_atoms.set_initial_charges(charges)

# ── LAMMPS calculator — Buckingham + Ewald Coulomb ────────────────────────────
# Pair potential: E_sr = A·exp(−r/ρ) − C/r⁶   (plus long-range Coulomb via Ewald)
# Parameters based on Tosi–Fumi (1964) for LiF, extended to Be–F interactions.
# Species order for LAMMPS types: 1=Li, 2=Be, 3=F
#
#  Pair   A (eV)    ρ (Å)     C (eV·Å⁶)
#  Li-Li   0.0      1.0       0.0        cation-cation: Coulomb only
#  Li-Be   0.0      1.0       0.0
#  Li-F  593.72   0.26310     0.0
#  Be-Be   0.0      1.0       0.0
#  Be-F 1389.47   0.23604     0.0
#  F-F  1127.70   0.27533    14.835
# Get cell dimensions correctly
cell = combined_atoms.get_cell()
row_x, row_y, row_z = cell  # Unpack the three rows

Lx = row_x[0]  # First element of first row
Ly = row_y[1]  # Second element of second row (if needed)
Lz = row_z[2]  # Third element of third row

L_min = min(Lx, Lz)  # Use the smaller of x and z
cutoff = round(L_min / 2 - 0.2, 1)

print(f"Cell dimensions: Lx={Lx:.2f}, Ly={Ly:.2f}, Lz={Lz:.2f}")
print(f"L_min={L_min:.2f}, cutoff={cutoff:.2f}")
print(f"  Cutoff: {cutoff} Å  (L_min/2 = {L_min/2:.2f} Å)")

pair_coeff = [
    '1 1    0.0      1.0      0.0  ',   # Li–Li
    '1 2    0.0      1.0      0.0  ',   # Li–Be
    '1 3  593.72   0.26310    0.0  ',   # Li–F
    '2 2    0.0      1.0      0.0  ',   # Be–Be
    '2 3 1389.47   0.23604    0.0  ',   # Be–F
    '3 3 1127.70   0.27533   14.835',   # F–F
]

calc = LAMMPS(
    specorder=['Li', 'Be', 'F'],
    atom_style='charge',
    pair_style=f'buck/coul/long {cutoff}',
    kspace_style='ewald 1.0e-5',
    pair_coeff=pair_coeff,
)
combined_atoms.calc = calc

# ── Energy minimization — relax strained crystal contacts before MD ───────────
print("\nMinimising energy to relax scaled crystal contacts (fmax=0.01 eV/Å)...")
minimiser = BFGS(combined_atoms, logfile='LiF_BeF2_minimisation.log')
minimiser.run(fmax=0.01, steps=500)
print(f"  Done. Epot = {combined_atoms.get_potential_energy()/len(combined_atoms):.4f} eV/atom")

# ── Initial velocities ────────────────────────────────────────────────────────
timestep_fs = 0.5   # fs
target_temperature = 2000  # K (ionic liquids often simulated at elevated T)
MaxwellBoltzmannDistribution(combined_atoms, temperature_K=target_temperature)
Stationary(combined_atoms)
ZeroRotation(combined_atoms)

# ── NVT equilibration — Nose-Hoover Chain ─────────────────────────────────────
traj_file = 'LiF_BeF2_nvt.traj'
log_file  = 'LiF_BeF2_nvt.log'

dyn = NoseHooverChainNVT(
    combined_atoms,
    timestep_fs * units.fs,
    temperature_K=target_temperature,
    tdamp=100 * timestep_fs * units.fs,   # 100 fs damping (ionic liquids)
    trajectory=traj_file,
    logfile=log_file,
)
dyn.attach(MDLogger(dyn, combined_atoms, 'LiF_BeF2_md.log'), interval=10)


def printenergy(a, step=None):
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    prefix = f'Step {step:6d}: ' if step is not None else ''
    print(
        f'{prefix}'
        f'Epot={epot/len(a):8.4f} eV/atom  '
        f'Ekin={ekin/len(a):7.4f} eV/atom  '
        f'T={temp:7.1f} K  '
        f'Etot={(epot+ekin)/len(a):8.4f} eV/atom'
    )
# ── Run equilibration ──────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print(f'NVT Equilibration: 1:1 LiF·BeF2  at T={target_temperature} K')
print('=' * 60)
printenergy(combined_atoms)

time_ps, epot_list, ekin_list, temp_list, pressure_list = [], [], [], [], []
steps_per_block = 100
n_blocks        = 100
total_steps     = n_blocks * steps_per_block

print(f"\nRunning {total_steps} steps  ({total_steps * timestep_fs / 1000:.1f} ps)...")
print(f"Timestep: {timestep_fs} fs  |  Thermostat τ: 100 fs  |  Cutoff: {cutoff} Å")

for i in range(n_blocks):
    dyn.run(steps_per_block)

    current_time = dyn.get_time() / (1000 * units.fs)
    time_ps.append(current_time)
    epot_list.append(combined_atoms.get_potential_energy())
    ekin_list.append(combined_atoms.get_kinetic_energy())
    temp_list.append(combined_atoms.get_temperature())

    stress   = combined_atoms.get_stress(voigt=False)
    pressure = -np.trace(stress) / 3.0
    pressure_list.append(pressure)

    if (i + 1) % 10 == 0:
        print(f"\n--- Block {i+1}/{n_blocks}  (t={current_time:.3f} ps) ---")
        printenergy(combined_atoms, step=dyn.get_number_of_steps())

etot_arr     = np.array(epot_list) + np.array(ekin_list)
pressure_arr = np.array(pressure_list)

# ── Summary statistics ─────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('Equilibration Summary')
print('=' * 60)
print(f"System   :  LiF·BeF2  ({len(combined_atoms)} atoms, 1:1 molar ratio)")
print(f"T_target : {target_temperature} K")
print(f"T_mean   : {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f} K")
print(f"T_drift  : {np.mean(temp_list) - target_temperature:.1f} K from target")
print(f"E_drift  : {(etot_arr[-1] - etot_arr[0]) / len(combined_atoms):.6f} eV/atom")
print(f"P_mean   : {np.mean(pressure_arr):.4f} ± {np.std(pressure_arr):.4f} eV/Å³")
print(f"Trajectory: {traj_file}")
# ── Plots ──────────────────────────────────────────────────────────────────────
n_atoms = len(combined_atoms)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

ax1.plot(time_ps, np.array(epot_list) / n_atoms, 'b-', label='Potential', lw=1.5)
ax1.plot(time_ps, np.array(ekin_list) / n_atoms, 'r-', label='Kinetic',   lw=1.5)
ax1.plot(time_ps, etot_arr / n_atoms,             'k--', label='Total',    lw=1.5)
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Energy per atom (eV)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title(f'Energy — LiF·BeF2  ({n_atoms} atoms)')

ax2.plot(time_ps, temp_list, 'g-', lw=1.5)
ax2.axhline(y=target_temperature, color='r', ls='--', alpha=0.7,
            label=f'Target {target_temperature} K')
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Temperature (K)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Temperature Evolution')

ax3.hist(temp_list, bins=20, color='green', alpha=0.7, edgecolor='black')
ax3.axvline(x=target_temperature, color='r', ls='--', label='Target')
ax3.axvline(x=np.mean(temp_list), color='b', ls='-',
            label=f'Mean: {np.mean(temp_list):.0f} K')
ax3.set_xlabel('Temperature (K)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.set_title('Temperature Distribution')

ax4.plot(time_ps, pressure_arr, 'b-', lw=1.5)
ax4.set_xlabel('Time (ps)')
ax4.set_ylabel('Pressure (eV/Å³)')
ax4.grid(True, alpha=0.3)
ax4.set_title('Pressure Evolution')

plt.tight_layout()
plot_file = 'LiF_BeF2_nvt_results.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {plot_file}")

# ── Pressure drift analysis ────────────────────────────────────────────────────
last_20_idx = int(0.8 * len(pressure_arr))
z = np.polyfit(time_ps[last_20_idx:], pressure_arr[last_20_idx:], 1)
print(f"\nPressure Analysis:")
print(f"  Mean (full run) : {np.mean(pressure_arr):.6f} ± {np.std(pressure_arr):.6f} eV/Å³")
print(f"  Mean (last 20%) : {np.mean(pressure_arr[last_20_idx:]):.6f} eV/Å³")
print(f"  Drift (last 20%): {z[0]:.6f} eV/Å³/ps")

# ── Save final configuration ───────────────────────────────────────────────────
final_config = 'LiF_BeF2_equilibrated.xyz'
combined_atoms.write(final_config)
print(f"\nEquilibrated configuration saved to: {final_config}")

print("\nFiles in current directory:")
for f in sorted(os.listdir('.')):
    if f.endswith(('.traj', '.cif', '.png', '.log', '.xyz')):
        size_kb = os.path.getsize(f) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")