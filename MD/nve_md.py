#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from ase import units, Atoms
from ase.io import read, write
from ase.md.verlet import VelocityVerlet
from ase.calculators.kim.kim import KIM
from ase.visualize import view
from ase.calculators.lammpsrun import LAMMPS
from gpaw import GPAW, restart
# ============================================================
#  SETTINGS – modify these as needed
# ============================================================
# Input file containing the equilibrated atoms (positions, velocities, cell)
# Supported formats: .traj, .xyz, .pickle (ASE restart), .cif (no velocities)

# NVE simulation parameters
timestep_fs = 1.0                 # timestep in femtoseconds
nve_steps = 500                    # total NVE steps (5 ps at 1 fs)
collect_interval = 10            # collect data every this many steps

# Output
traj_output = 'nve_check.traj'    # optional trajectory file
plot_output = 'nve_energy_drift.png'


atoms.calc = calc

# Optionally view the structure
# view(atoms)

print(f"Loaded {len(atoms)} atoms.")
print(f"Cell volume: {atoms.get_volume():.2f} Å³")
print(f"Initial temperature: {atoms.get_temperature():.1f} K")

# ============================================================
#  SET UP NVE DYNAMICS
# ============================================================
dyn = VelocityVerlet(atoms, timestep_fs * units.fs, trajectory=traj_output)

def print_energy(a, step):
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    print(f"Step {step:6d}  Epot = {epot/len(a):6.3f} eV/atom  "
          f"Ekin = {ekin/len(a):.3f} eV/atom  "
          f"T = {temp:6.1f} K  Etot = {(epot+ekin)/len(a):.3f} eV/atom")

# ============================================================
#  RUN NVE AND COLLECT DATA
# ============================================================
print("\n" + "="*60)
print("Running NVE simulation...")
print("="*60)

time_ps = []
etot_per_atom = []

for step in range(nve_steps):
    dyn.run(1)

    if step % collect_interval == 0:
        t = step * timestep_fs / 1000.0  # ps
        etot = atoms.get_total_energy() / len(atoms)
        time_ps.append(t)
        etot_per_atom.append(etot)

        if step % (collect_interval) == 0:
            print(f"Time = {t:.2f} ps")
            print_energy(atoms, step)
            

# Final energy
print_energy(atoms, nve_steps)

# ============================================================
#  ANALYZE ENERGY DRIFT
# ============================================================
time_ps = np.array(time_ps)
etot_per_atom = np.array(etot_per_atom)

# Linear fit to last part to estimate drift rate
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(time_ps, etot_per_atom)
drift_per_ps = slope                     # eV/atom/ps
drift_per_step = slope * timestep_fs/1000.0   # eV/atom/step

print("\n" + "="*60)
print("NVE Energy Conservation Summary")
print("="*60)
print(f"Simulation length: {time_ps[-1]:.2f} ps ({nve_steps} steps)")
print(f"Total energy drift: {etot_per_atom[-1] - etot_per_atom[0]:.6f} eV/atom")
print(f"Drift rate: {drift_per_ps:.6f} eV/atom/ps  ({drift_per_step:.6f} eV/atom/step)")
print(f"p-value of trend: {p_value:.4f}")

# ============================================================
#  PLOT TOTAL ENERGY VS TIME
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(time_ps, etot_per_atom, 'b-', linewidth=1.5, label='Total energy per atom')
plt.xlabel('Time (ps)')
plt.ylabel('Total energy per atom (eV)')
plt.title('NVE Energy Conservation Check')
plt.grid(alpha=0.3)

# Add linear fit line
fit_line = slope * time_ps + intercept
plt.plot(time_ps, fit_line, 'r--', linewidth=1, label=f'Linear fit (drift = {drift_per_ps:.4f} eV/atom/ps)')
plt.legend()
plt.tight_layout()
plt.savefig(plot_output, dpi=150)
plt.show()

print(f"\nPlot saved to {plot_output}")
if traj_output:
    print(f"Trajectory saved to {traj_output}")