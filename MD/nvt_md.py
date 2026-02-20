import matplotlib.pyplot as plt
import numpy as np
import os
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.emt import EMT
from ase.visualize import view
from ase import units
from ase.cluster.cubic import FaceCenteredCubic as ClusterFCC
from ase.io.trajectory import Trajectory
from ase.lattice.cubic import FaceCenteredCubic as LatticeFCC
from ase.lattice.cubic import Diamond as LatticeDiamond
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase import Atoms
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.optimize import QuasiNewton
from ase.visualize.plot import plot_atoms
from ase.calculators.idealgas import IdealGas
from ase.calculators.lj import LennardJones
from ase.calculators.kim.kim import KIM
from ase.md import MDLogger  # For better logging
from gpaw import GPAW, PW
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
size = 4  # 4x4x4 = 64 atoms (good for testing)
target_temperature = 3000  # K
print(f"Creating Diamond lattice with {size}^3 = {size**3} atoms...")
atoms = LatticeDiamond(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    symbol='Si',
    size=(size, size, size),
    pbc=True,
)
#atoms = Atoms(['H', 'H'], positions=[[0,0,0], [0,0,0.74]], cell=[4,4,4], pbc=True)# Save the structure
#atoms.calc = GPAW(mode=PW(300), xc='PBE', txt='gpaw_output.txt')
cif_file = 'Si_diamond.cif'
atoms.write(cif_file, format='cif')
print(f"Saved structure to {cif_file}")

# Set up KIM calculator
#calc = KIM("Sim_LAMMPS_IFF_CHARMM_GUI_HeinzLinMishra_2023_Nanomaterials__SM_232384752957_000")
#atoms.calc = EMT()
atoms.calc = GPAW(mode=PW(300), xc='PBE', txt='gpaw_output.txt')
# Set the initial velocities corresponding to target temperature
MaxwellBoltzmannDistribution(atoms, temperature_K=target_temperature)

# Remove any overall momentum or rotation
#Stationary(atoms)  # Zero linear momentum
#ZeroRotation(atoms)  # Zero angular momentum

# NVT Simulation parameters
timestep_fs = 1.0  
friction_coeff = 0.01  # Friction coefficient for Langevin thermostat (in fs^-1)
traj_file = 'H2_nvt_equilibration.traj'
log_file = 'H2_nvt_equilibration.log'

# Use Nose-Hoover Chain dynamics for NVT ensemble
dyn = NoseHooverChainNVT(
    atoms, 
    timestep_fs * units.fs,  # timestep
    temperature_K=target_temperature,  # Target temperature
    tdamp = 1000*timestep_fs * units.fs,  # Thermostat damping time
    trajectory=traj_file,
    logfile=log_file,  # Log file for dynamics info
)

# Add a logger to monitor the simulation
dyn.attach(MDLogger(dyn, atoms, 'nvt_equilibration.log'), interval=10)

def printenergy(a, step=None):
    """
    Function to print the thermodynamical properties
    """
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    if step is not None:
        print(f'Step {step:6.2f} : ', end='')
    print(
        f'Energy per atom: Epot = {epot/len(a):6.3f} eV/atom  '
        f'Ekin = {ekin/len(a):.3f} eV/atom '
        f'(T={temp:.1f}K) Etot = {(epot + ekin)/len(a):.3f} eV/atom'
    )

# Optional: View atoms
print("\nOpening visualization window...")
view(atoms)

# Now run the NVT equilibration
print('\n' + '='*60)
print(f'NVT Equilibration of H2 at T={target_temperature}K')
print('='*60)
printenergy(atoms)

# Initialize lists for data collection
time_ps, epot, ekin, temp, pressure_trace = [], [], [], [], []  # Add pressure_trace
steps_per_block = 100
n_blocks = 100

print(f"\nRunning {n_blocks * steps_per_block} steps of NVT equilibration...")
print(f"Timestep: {timestep_fs} fs, Friction: {friction_coeff} fs^-1")

for i in range(n_blocks):
    dyn.run(steps_per_block)
    
    # Save data
    current_time = dyn.get_time()/(1000 * units.fs)  # Convert to ps
    time_ps.append(current_time)
    epot.append(atoms.get_potential_energy())
    ekin.append(atoms.get_kinetic_energy())
    temp.append(atoms.get_temperature())
    stress = atoms.get_stress(voigt=False)  # 3x3 stress tensor
    pressure = -np.trace(stress) / 3.0  # Hydrostatic pressure
    pressure_trace.append(pressure)
    # ------------------------------------
    # Print progress every block
    if (i+1) % 1 == 0:
        print(f"\n--- Block {i+1}/{n_blocks} (t={current_time:.2f} ps) ---")
        printenergy(atoms, step=dyn.get_number_of_steps())

# Calculate total energy
etot = np.array(epot) + np.array(ekin)

# Print summary statistics
print('\n' + '='*60)
print('Equilibration Summary')
print('='*60)
print(f"Initial temperature: {temp[0]:.1f} K")
print(f"Final temperature: {temp[-1]:.1f} K")
print(f"Average temperature: {np.mean(temp):.1f} ± {np.std(temp):.1f} K")
print(f"Temperature drift from target: {np.mean(temp)-target_temperature:.1f} K")
print(f"Energy drift: {(etot[-1] - etot[0])/len(atoms):.6f} eV/atom")
print(f"Trajectory saved to: {traj_file}")
print(f"Log file: {log_file}")

# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Energy plot
ax1.plot(time_ps, np.array(epot)/len(atoms), 'b-', label='Potential', linewidth=1.5)
ax1.plot(time_ps, np.array(ekin)/len(atoms), 'r-', label='Kinetic', linewidth=1.5)
ax1.plot(time_ps, np.array(etot)/len(atoms), 'k--', label='Total', linewidth=1.5)
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Energy per atom (eV)')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_title(f'Energy Evolution - {len(atoms)} H2 molecules')

# Temperature plot
ax2.plot(time_ps, temp, 'g-', linewidth=1.5)
ax2.axhline(y=target_temperature, color='r', linestyle='--', 
            alpha=0.5, label=f'Target T={target_temperature}K')
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Temperature (K)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_title('Temperature Evolution')

# Temperature histogram
ax3.hist(temp, bins=20, color='green', alpha=0.7, edgecolor='black')
ax3.axvline(x=target_temperature, color='r', linestyle='--', 
            label=f'Target')
ax3.axvline(x=np.mean(temp), color='b', linestyle='-', 
            label=f'Mean: {np.mean(temp):.0f}K')
ax3.set_xlabel('Temperature (K)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.set_title('Temperature Distribution')

# Running average of temperature
window = 10  # Moving average window
temp_running_avg = np.convolve(temp, np.ones(window)/window, mode='valid')
time_running = time_ps[window-1:]
ax4.plot(time_ps, temp, 'g-', alpha=0.3, label='Instantaneous')
#ax4.plot(time_running, temp_running_avg, 'b-', linewidth=2, 
#         label=f'{window}-step running avg')
ax4.axhline(y=target_temperature, color='r', linestyle='--')
ax4.set_xlabel('Time (ps)')
ax4.set_ylabel('Temperature (K)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_title('Temperature with Running Average')

plt.tight_layout()

# Save the plot
plot_file = 'H2_nvt_equilibration_results.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {plot_file}")

# Check what files were created
print("\nFiles in current directory:")
for f in sorted(os.listdir('.')):
    if f.endswith(('.traj', '.cif', '.png', '.log')):
        size_kb = os.path.getsize(f) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")

# Optional: Save final configuration for NVE production run
final_config_file = 'H2_equilibrated.xyz'
atoms.write(final_config_file)
print(f"\nEquilibrated configuration saved to: {final_config_file}")

pressure_trace = np.array(pressure_trace)

# Plot pressure evolution
plt.figure(figsize=(10, 4))
plt.plot(time_ps, pressure_trace, 'b-', linewidth=1.5)
plt.xlabel('Time (ps)')
plt.ylabel('Pressure (eV/Å³)')
plt.title('Pressure Evolution During Equilibration')
plt.grid(True, alpha=0.3)
plt.show()

# Check drift in last 20%
last_20_idx = int(0.8 * len(pressure_trace))
last_20_pressure = pressure_trace[last_20_idx:]
last_20_time = time_ps[last_20_idx:]

# Linear fit to check drift
z = np.polyfit(last_20_time, last_20_pressure, 1)
drift_per_ps = z[0]

print(f"\nPressure Analysis:")
print(f"  Mean pressure (full): {np.mean(pressure_trace):.6f} ± {np.std(pressure_trace):.6f} eV/Å³")
print(f"  Mean pressure (last 20%): {np.mean(last_20_pressure):.6f} eV/Å³")
print(f"  Pressure drift (last 20%): {drift_per_ps:.6f} eV/Å³/ps")
