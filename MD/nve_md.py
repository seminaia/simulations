import matplotlib.pyplot as plt
import numpy as np
import os
from ase.calculators.lammpsrun import LAMMPS
from ase.visualize import view
from ase import units
from ase.cluster.cubic import FaceCenteredCubic as ClusterFCC
from ase.io.trajectory import Trajectory
from ase.lattice.cubic import FaceCenteredCubic as LatticeFCC
from ase.lattice.cubic import Diamond as LatticeDiamond
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.optimize import QuasiNewton
from ase.visualize.plot import plot_atoms
from ase.calculators.kim.kim import KIM
# Set working directory to current script location
import sys
if 'ipykernel' in sys.modules:  # Check if running in Interactive Window
    import os
    try:
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        print(f"Changed working directory to: {os.getcwd()}")
    except NameError:   
        # __file__ not defined in interactive mode, use current directory
        print(f"Working directory: {os.getcwd()}")

# Set up initial positions of Si atoms on Diamond crystal lattice
size = 5  # Reduced from 10 to 4 for faster testing
print(f"Creating Diamond lattice with {size}^3 = {size**3} atoms...")
atoms = LatticeDiamond(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    symbol='Si',
    size=(size, size, size),
    pbc=True,
)

# Save the structure
cif_file = 'Si_diamond.cif'
atoms.write(cif_file, format='cif')
print(f"Saved structure to {cif_file}")

calc = KIM("SNAP_ZuoChenLi_2019quadratic_Si__MO_721469752060_000")

# Describe the interatomic interactions with the KIM calculator
atoms.calc = calc

# Set the initial velocities corresponding to T=1000K from Maxwell Boltzmann
# Distribution
MaxwellBoltzmannDistribution(atoms, temperature_K=1000)

# We use Velocity Verlet algorithm to integrate the Newton's equations.
timestep_fs = 5
traj_file = 'Si_md.traj'
dyn = VelocityVerlet(atoms, timestep_fs * units.fs, trajectory=traj_file)

def printenergy(a, step=None):
    """
    Function to print the thermodynamical properties
    """
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    if step is not None:
        print(f'Step {step:4d}: ', end='')
    print(
        f'Energy per atom: Epot = {epot/len(a):6.3f} eV/atom  '
        f'Ekin = {ekin/len(a):.3f} eV/atom '
        f'(T={temp:.1f}K) Etot = {(epot + ekin)/len(a):.3f} eV/atom'
    )

# Optional: View atoms (may open external window)
# Comment this out if you don't want the visualization
print("\nOpening visualization window...")
view(atoms)

# Now run the dynamics
print('\n' + '='*50)
print('Running NVE simulation of Diamond Si')
print('='*50)
printenergy(atoms)

# Initialize lists for energy vs time data
time_ps, epot, ekin, temp = [], [], [], []
mdind = 0
steps_per_block = 10
n_blocks = 20  # Total steps = n_blocks * steps_per_block

for i in range(n_blocks):
    dyn.run(steps_per_block)
    mdind += steps_per_block
    
    # Save data
    time_ps.append(mdind * timestep_fs / 1000.0)  # Convert to ps
    epot.append(atoms.get_potential_energy())
    ekin.append(atoms.get_kinetic_energy())
    temp.append(atoms.get_temperature())
    
    # Print progress
    printenergy(atoms, step=mdind)

# Calculate total energy
etot = np.array(epot) + np.array(ekin)

# Print summary statistics
print('\n' + '='*50)
print('Simulation Summary')
print('='*50)
print(f"Final temperature: {temp[-1]:.1f} K")
print(f"Average temperature: {np.mean(temp):.1f} ± {np.std(temp):.1f} K")
print(f"Energy drift: {(etot[-1] - etot[0])/len(atoms):.6f} eV/atom")
print(f"Trajectory saved to: {traj_file}")

# Plot energies vs time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Energy plot
ax1.plot(time_ps, np.array(epot)/len(atoms), 'b-', label='Potential energy', linewidth=2)
ax1.plot(time_ps, np.array(ekin)/len(atoms), 'r-', label='Kinetic energy', linewidth=2)
ax1.plot(time_ps, np.array(etot)/len(atoms), 'k--', label='Total energy', linewidth=2)
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Energy per atom (eV)')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title(f'Energy Evolution - {len(atoms)} Si atoms')

# Temperature plot
ax2.plot(time_ps, temp, 'g-', linewidth=2)
ax2.axhline(y=300, color='r', linestyle='--', alpha=0.5, label='Target T=300K')
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Temperature (K)')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('Temperature Evolution')

plt.tight_layout()
plt.show()

# Save the plot
plt.savefig('md_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: md_results.png")

# Optional: Check what files were created
print("\nFiles in current directory:")
for f in sorted(os.listdir('.')):
    if f.endswith(('.traj', '.cif', '.png')):
        size_kb = os.path.getsize(f) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")