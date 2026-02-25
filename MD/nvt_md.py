from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import os
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.emt import EMT
from ase.visualize import view
from ase import units
from ase.cluster.cubic import FaceCenteredCubic as ClusterFCC
from ase.io.trajectory import Trajectory
from ase.lattice.cubic import BodyCenteredCubic as LatticeBCC
from ase.lattice.cubic import Diamond as LatticeDiamond
from ase.md.nose_hoover_chain import NoseHooverChainNVT as NHCNVT
from ase.md.langevin import Langevin as LanNVT
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
from gpaw import GPAW, PW, restart
import pickle
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, QuasiNewton

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

conv_frac_positions = [  
    (0.50,  0.50,  0.14),
    (0.00,  0.00,  0.36),
    (0.00,  0.00,  0.64),
    (0.50,  0.50,  0.86),
    (0.00,  0.00,  0.00),
    (0.50,  0.50,  0.50),
    (0.50,  0.00,  0.00),
    (0.00,  0.50,  0.00),
    (0.00,  0.00,  0.18),
    (0.50,  0.50,  0.32),
    (0.50,  0.00,  0.50),
    (0.00,  0.50,  0.50),
    (0.50,  0.50,  0.68),
    (0.00,  0.00,  0.82)]
LNO_symbols = ['La','La','La','La','Ni','Ni','O','O','O','O','O','O','O','O']
a_lno = 3.80
c_lno = 12.45
conv_cell = [
    [a_lno, 0, 0],
    [0, a_lno, 0],
    [0, 0, c_lno]
]

LNO_atoms = Atoms(LNO_symbols, 
                  scaled_positions=conv_frac_positions,
                  cell=conv_cell,
                  pbc=True)
magmoms = [0.6, -0.6, 0.6, -0.6, 2.0, -2.0, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6]
LNO_atoms.set_initial_magnetic_moments(magmoms)
LNO_atoms.write('La2NiO4_conv.cif', format='cif')
view(LNO_atoms, repeat=(2, 2, 1))

bcc_scaled_positions = [
    (0.00, 0.00, 0.00),
    (0.50, 0.50, 0.50)
]
a_w = 3.16
bcc_conv_cell = [
            [a_w, 0, 0],
            [0, a_w, 0],
            [0, 0, a_w]
            ]
w_symbol = ['W', 'W']
w_atoms = Atoms(symbols=w_symbol,
    scaled_positions=bcc_scaled_positions,
    cell=bcc_conv_cell,
    pbc=True,
)
view(w_atoms, repeat=(2, 2, 2))

a_si = 5.43
si_symbols = ['Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si'] 
diamond_scaled_positions = [
       (0.75, 0.75, 0.25),
       (0.00, 0.50, 0.50),
       (0.75, 0.25, 0.75),
       (0.00, 0.00, 0.00),
       (0.25, 0.75, 0.75),
       (0.50, 0.50, 0.00),
       (0.25, 0.25, 0.25),
       (0.50, 0.00, 0.50)]

dia_conv_cell = [
    [a_si, 0, 0],
    [0, a_si, 0],
    [0, 0, a_si]
]
si_atoms = Atoms(symbols=si_symbols,
    scaled_positions=diamond_scaled_positions,
    cell=dia_conv_cell,
    pbc=True,
)
view(si_atoms, repeat=(2, 2, 2))
base_params = {
    "convergence": {"density": 1e-4,
                    "eigenstates": 1e-8,
                    "energy": 1e-6, 
                    "forces": 1e-4},
    "eigensolver": {"name": "cg", 
                    "niter": 5},
    "maxiter": 500,
    "mixer": {"backend": "pulay", 
              "beta": 0.1,
              "method": "fullspin",
              "nmaxold": 5,
              "weight": 100},
    "mode": {"ecut": 520, "name": "pw"},
    "nbands": "nao",
    "symmetry": "off",
    "occupations": {"name": "fermi-dirac",
                    "width": 0.01},
    "txt": None,  # Will be set per material
    "xc": "PBE"
}
w_params = base_params.copy()
w_params["txt"] = "W_rlx.txt"
w_params["occupations"] = {"name": "fermi-dirac", "width": 0.1}
w_params["kpts"] = {"gamma": True, "size": [4, 4, 4]}  
        
w_rlx_atoms = relax(w_atoms, 
                    w_params,
                    fmax=0.01,
                    fixcell=False, 
                    logname='w_opt.log',
                    trajname='w_opt.traj',
                    gpwname='w_rlx.gpw')
if isinstance(w_rlx_atoms, FrechetCellFilter):
    w_rlx_atoms = w_rlx_atoms.atoms

si_params = base_params.copy()
si_params["txt"] = "Si_rlx.txt"
si_params["occupations"] = {"name": "fermi-dirac", "width": 0.01}
si_params["kpts"] = {"gamma": True, "size": [2, 2, 2]}
si_rlx_atoms = relax(si_atoms,
                     si_params,
                     fmax=0.01,
                     fixcell=False,
                     logname='si_opt.log',
                     trajname='si_opt.traj',
                     gpwname='si_rlx.gpw')
if isinstance(si_rlx_atoms, FrechetCellFilter):
    si_rlx_atoms = si_rlx_atoms.atoms

#lno_rlx_atoms = relax(LNO_atoms, 
#                           calculator_params,
#                           fmax=0.01,
#                           fixcell=False, 
#                           logname='lno_opt.log',
#                           trajname='lno_opt.traj',
#                           gpwname='lno_rlx.gpw')
#view(lno_rlx_atoms, repeat=(2, 2, 2))
#if isinstance(lno_rlx_atoms, FrechetCellFilter):
#    lno_rlx_atoms = lno_rlx_atoms.atoms
print("Calculating potential energy...")
print(f"Potential energy of W: {w_rlx_atoms.get_potential_energy():.3f} eV")
print(f"Potential energy of Si: {si_rlx_atoms.get_potential_energy():.3f} eV")
#print(f"Potential energy of La2NiO4: {lno_rlx_atoms.get_potential_energy():.3f} eV")
LiF_atoms = Atoms('LiF', positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)
LiF_atoms.calc = GPAW(**base_params)
LiF_atoms.get_potential_energy()  
BeF2_atoms = Atoms('BeF2', positions=[[0, 0, 0], [1.5, 1.5, 1.5], [3, 3, 3]], cell=[6, 6, 6], pbc=True)
BeF2_atoms.calc = GPAW(**base_params)
BeF2_atoms.get_potential_energy()

# Re-initialize velocities
target_temperature = 2000
MaxwellBoltzmannDistribution(w_rlx_atoms, temperature_K=target_temperature)
Stationary(w_rlx_atoms)
ZeroRotation(w_rlx_atoms)

MaxwellBoltzmannDistribution(si_rlx_atoms, temperature_K=target_temperature)
Stationary(si_rlx_atoms)
ZeroRotation(si_rlx_atoms)

# Define restart file name
w_restart_file = 'W_nvt_restart.pickle'
   
# NVT Simulation parameters
timestep_fs = 1.0  
friction_coeff = 0.01  # Friction coefficient for Langevin thermostat (in fs^-1)
w_traj_file = 'W_nvt_equilibration.traj'
w_log_file = 'W_nvt_equilibration.log'

# Use Nose-Hoover Chain dynamics for NVT ensemble
w_dyn = NHCNVT(
    w_rlx_atoms, 
    timestep_fs * units.fs,  # timestep
    temperature_K=target_temperature,  # Target temperature
    tdamp = 100*timestep_fs * units.fs,  # Thermostat damping time
    trajectory=w_traj_file,
    logfile=w_log_file,  # Log file for dynamics info
)
w_dyn.attach(MDLogger(w_dyn, w_rlx_atoms, w_log_file), interval=10)

si_dyn = NHCNVT(
    si_rlx_atoms, 
    timestep_fs * units.fs,  # timestep
    temperature_K=target_temperature,  # Target temperature
    tdamp = 100*timestep_fs * units.fs,  # Thermostat damping time
    trajectory='Si_nvt_equilibration.traj',
    logfile='Si_nvt_equilibration.log',  # Log file for dynamics info
)
si_dyn.attach(MDLogger(si_dyn, si_rlx_atoms, 'Si_nvt_equibration.log'), interval=10)

def printenergy(a, step=None):
    """
    Function to print the thermodynamical properties
    """
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    
    stress = a.get_stress(voigt=False)  # 3x3 stress tensor
    pressure = -np.trace(stress) / 3.0  # Hydrostatic pressure

    if step is not None:
        print(f'Step {step:6.2f} : ', end='')
    print(
        f'Energy per atom: Epot = {epot/len(a):6.3f} eV/atom  '
        f'Ekin = {ekin/len(a):.3f} eV/atom '
        f'(T={temp:.1f}K) Etot = {(epot + ekin)/len(a):.3f} eV/atom'
        f' Pressure = {pressure:.3f} eV/Å³'
    )

# Now run the NVT equilibration
print('\n' + '='*60)
print(f'NVT Equilibration of W at T={target_temperature}K')
print('='*60)
printenergy(w_rlx_atoms)

# Initialize lists for data collection
w_time_ps, w_epot, w_ekin, w_temp, w_pressure_trace = [], [], [], [], []  # Add pressure_trace
si_time_ps, si_epot, si_ekin, si_temp, si_pressure_trace = [], [], [], [], []  # Add pressure_trace
steps_per_block = 100
n_blocks = 100

print(f"\nRunning {n_blocks * steps_per_block} steps of NVT equilibration...")
print(f"Timestep: {timestep_fs} fs, Friction: {friction_coeff} fs^-1")

for i in range(n_blocks):
    w_dyn.run(steps_per_block)
    
    # Save data
    current_time = w_dyn.get_time()/(1000 * units.fs)  # Convert to ps
    w_time_ps.append(current_time)
    w_epot.append(w_rlx_atoms.get_potential_energy())
    w_ekin.append(w_rlx_atoms.get_kinetic_energy())
    w_temp.append(w_rlx_atoms.get_temperature())
    stress = w_rlx_atoms.get_stress(voigt=False)  # 3x3 stress tensor
    pressure = -np.trace(stress) / 3.0  # Hydrostatic pressure
    w_pressure_trace.append(pressure)
        # Print progress every block
    if (i+1) % 1 == 0:
        print(f"\n--- Block {i+1}/{n_blocks} (t={current_time:.2f} ps) ---")
        printenergy(w_rlx_atoms, step=w_dyn.get_number_of_steps())

    si_dyn.run(steps_per_block)
    si_time_ps.append(si_dyn.get_time()/(1000 * units.fs))  # Convert to ps
    si_epot.append(si_rlx_atoms.get_potential_energy())
    si_ekin.append(si_rlx_atoms.get_kinetic_energy())
    si_temp.append(si_rlx_atoms.get_temperature())
    stress_si = si_rlx_atoms.get_stress(voigt=False)  # 3x3 stress tensor
    pressure_si = -np.trace(stress_si) / 3.0  # Hydrostatic pressure
    si_pressure_trace.append(pressure_si)    
    # ------------------------------------
    if (i+1) % 1 == 0:
        print(f"\n--- Block {i+1}/{n_blocks} (t={current_time:.2f} ps) ---")
        printenergy(si_rlx_atoms, step=si_dyn.get_number_of_steps())
        
# Calculate total energy
w_etot = np.array(w_epot) + np.array(w_ekin)
si_etot = np.array(si_epot) + np.array(si_ekin)
# Save restart file at the end
W_restart_data = {
    'positions': w_rlx_atoms.get_positions(),
    'velocities': w_rlx_atoms.get_velocities(),
    'cell': w_rlx_atoms.get_cell(),
    'step': w_dyn.get_number_of_steps(),
    'time': w_dyn.get_time() / units.fs,  # time in fs
}
with open(w_restart_file, 'wb') as f:
    pickle.dump(W_restart_data, f)
print(f"Restart file saved: {w_restart_file}")

Si_restart_data = {
    'positions': si_rlx_atoms.get_positions(),
    'velocities': si_rlx_atoms.get_velocities(),
    'cell': si_rlx_atoms.get_cell(),
    'step': si_dyn.get_number_of_steps(),
    'time': si_dyn.get_time() / units.fs,  # time in fs
}
with open('Si_nvt_restart.pickle', 'wb') as f:
    pickle.dump(Si_restart_data, f)
print(f"Restart file saved: Si_nvt_restart.pickle")

# Print summary statistics
print('\n' + '='*60)
print('Equilibration Summary')
print('='*60)
print(f"Initial temperature: {w_temp[0]:.1f} K")
print(f"Final temperature: {w_temp[-1]:.1f} K")
print(f"Average temperature: {np.mean(w_temp):.1f} ± {np.std(w_temp):.1f} K")
print(f"Temperature drift from target: {np.mean(w_temp)-target_temperature:.1f} K")
print(f"Energy drift: {(w_etot[-1] - w_etot[0])/len(w_rlx_atoms):.6f} eV/atom")
print(f"Trajectory saved to: {w_traj_file}")
print(f"Log file: {w_log_file}")

# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Energy plot
ax1.plot(w_time_ps, np.array(w_epot)/len(w_rlx_atoms), 'b-', label='W Potential', linewidth=1.5)
ax1.plot(w_time_ps, np.array(w_ekin)/len(w_rlx_atoms), 'r-', label='W Kinetic', linewidth=1.5)
ax1.plot(w_time_ps, np.array(w_etot)/len(w_rlx_atoms), 'k--', label='W Total', linewidth=1.5)
ax1.plot(si_time_ps, np.array(si_epot)/len(si_rlx_atoms), 'c-', label='Si Potential', linewidth=1.5)
ax1.plot(si_time_ps, np.array(si_ekin)/len(si_rlx_atoms), 'm-', label='Si Kinetic', linewidth=1.5)
ax1.plot(si_time_ps, np.array(si_etot)/len(si_rlx_atoms), 'y--', label='Si Total', linewidth=1.5)
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Energy per atom (eV)')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_title(f'Energy Evolution - {len(w_rlx_atoms)} W atoms, {len(si_rlx_atoms)} Si atoms')

# Temperature plot
ax2.plot(w_time_ps, np.array(w_temp), 'g-', linewidth=1.5)
ax2.plot(si_time_ps, np.array(si_temp), 'orange', linewidth=1.5)
ax2.axhline(y=target_temperature, color='r', linestyle='--', 
            alpha=0.5, label=f'Target T={target_temperature}K')
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Temperature (K)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_title('Temperature Evolution')

# Temperature histogram
ax3.hist(w_temp, bins=20, color='green', alpha=0.7, edgecolor='black')
ax3.hist(si_temp, bins=20, color='orange', alpha=0.7, edgecolor='black')
ax3.axvline(x=target_temperature, color='r', linestyle='--', 
            label=f'Target')
ax3.axvline(x=np.mean(w_temp), color='b', linestyle='-', 
            label=f'Mean: {np.mean(w_temp):.0f}K')
ax3.axvline(x=np.mean(si_temp), color='orange', linestyle='-', 
            label=f'Mean: {np.mean(si_temp):.0f}K')
ax3.set_xlabel('Temperature (K)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.set_title('Temperature Distribution')

# Running average of temperature
window = 10  # Moving average window
temp_running_avg = np.convolve(w_temp, np.ones(window)/window, mode='valid')
time_running = w_time_ps[window-1:]
ax4.plot(w_time_ps, w_temp, 'g-', alpha=0.3, label='Instantaneous')
ax4.plot(time_running, temp_running_avg, 'b-', linewidth=2, 
         label=f'{window}-step running avg')
ax4.axhline(y=target_temperature, color='r', linestyle='--')
ax4.set_xlabel('Time (ps)')
ax4.set_ylabel('Temperature (K)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_title('Temperature with Running Average')

plt.tight_layout()

# Save the plot
plot_file = 'Si_W_nvt_equilibration_results.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {plot_file}")

W_final_config_file = 'W_equilibrated.xyz'
w_rlx_atoms.write(W_final_config_file)

Si_final_config_file = 'Si_equilibrated.xyz'
si_rlx_atoms.write(Si_final_config_file)

print(f"\nEquilibrated configuration saved to: {W_final_config_file} and {Si_final_config_file}")
w_pressure_trace = np.array(w_pressure_trace)
si_pressure_trace = np.array(si_pressure_trace)

# Plot pressure evolution
plt.figure(figsize=(10, 4))
plt.plot(w_time_ps, w_pressure_trace, 'b-', linewidth=1.5, label='W')
plt.plot(si_time_ps, si_pressure_trace, 'orange', linewidth=1.5, label='Si')
plt.xlabel('Time (ps)')
plt.ylabel('Pressure (eV/Å³)')
plt.title('Pressure Evolution During Equilibration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check drift in last 20%
W_last_20_idx = int(0.8 * len(w_pressure_trace))
W_last_20_pressure = w_pressure_trace[W_last_20_idx:]
W_last_20_time = w_time_ps[W_last_20_idx:]

Si_last_20_idx = int(0.8 * len(si_pressure_trace))
Si_last_20_pressure = si_pressure_trace[Si_last_20_idx:]
Si_last_20_time = si_time_ps[Si_last_20_idx:]

# Linear fit to check drift
z_W = np.polyfit(W_last_20_time, W_last_20_pressure, 1)
w_drift_per_ps = z_W[0]

z_si = np.polyfit(Si_last_20_time, Si_last_20_pressure, 1)
si_drift_per_ps = z_si[0]

print(f"\nPressure Analysis:")
print(f"  Mean pressure (full): {np.mean(w_pressure_trace):.6f} ± {np.std(w_pressure_trace):.6f} eV/Å³")
print(f"  Mean pressure (last 20%): {np.mean(W_last_20_pressure):.6f} eV/Å³")
print(f"  Pressure drift (last 20%): {w_drift_per_ps:.6f} eV/Å³/ps")

print(f"  Mean pressure (full): {np.mean(si_pressure_trace):.6f} ± {np.std(si_pressure_trace):.6f} eV/Å³")
print(f"  Mean pressure (last 20%): {np.mean(Si_last_20_pressure):.6f} eV/Å³")
print(f"  Pressure drift (last 20%): {si_drift_per_ps:.6f} eV/Å³/ps")
