from ase.visualize import view
from ase import units
from ase.cluster.cubic import FaceCenteredCubic as ClusterFCC
from ase.io.trajectory import Trajectory
from ase.lattice.cubic import BodyCenteredCubic as LatticeBCC
from ase.lattice.cubic import Diamond as LatticeDiamond
from ase.build import bulk, make_supercell
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
from ase.calculators.emt import EMT
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional, Union

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
    # Create optimizer
    opt = BFGS(opt_atoms, 
               logfile=logname,
               trajectory=trajname)
        
    # Run relaxation
    print(f"Starting relaxation with fmax={fmax} eV/Å")
    opt.run(fmax=fmax, steps=500)

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
    
    # Get final forces for reporting
    if hasattr(opt_atoms, 'get_forces'):
        forces = opt_atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        print(f"Relaxation completed. Maximum force: {max_force:.6f} eV/Å")
    
    return opt_atoms

# ============================================================
#  BASE CALCULATOR PARAMETERS
# ============================================================
base_params = {
    "convergence": {"density": 1e-4,
                    "eigenstates": 1e-8,
                    "energy": 1e-6, 
                    "forces": 1e-4},
    "eigensolver": {"name": "dav", 
                    "niter": 5},
    "maxiter": 500,
    "mixer": {"backend": "pulay", 
              "beta": 0.1,
              "method": "fullspin",
              "nmaxold": 5,
              "weight": 100},
    "mode": {"ecut": 520, "name": "pw"},
    "nbands": "nao",
#    "symmetry": "off",
    "occupations": {"name": "fermi-dirac",
                    "width": 0.01},
    "txt": None,  # Will be set per material
    "xc": 'PBE',
}

# ============================================================
#  CREATE REALISTIC CRYSTAL STRUCTURES
# ============================================================
print("="*60)
print("Creating crystal structures")
print("="*60)

# LiF - Rocksalt structure (FCC with two atoms basis)
# Experimental lattice constant: 4.03 Å
a_lif = 4.03
LiF_atoms = bulk('LiF', crystalstructure='rocksalt', a=a_lif, cubic=True)
LiF_params = base_params.copy()
LiF_params["txt"] = "LiF_rlx.txt"
# LiF is an insulator, so moderate k-points are fine
LiF_params["kpts"] = {"gamma": True, "size": [2, 2, 2]}

LiF_rlx_atoms = relax(LiF_atoms,
                      LiF_params,
                      fmax=0.01,
                      fixcell=False,
                      logname='LiF_relax.log',
                      trajname='LiF_relax.traj',
                      gpwname='LiF_rlx.gpw')
print(f"LiF: {len(LiF_rlx_atoms)} atoms, cell volume: {LiF_rlx_atoms.get_volume():.2f} Å³")
# BeF2 - Using beta-quartz-like structure (hexagonal)
# Lattice constants from literature (approximate)
a_bef2 = 4.77
c_bef2 = 5.18
bef2_cell = [[a_bef2, 0, 0],
             [-a_bef2/2, a_bef2*np.sqrt(3)/2, 0],
             [0, 0, c_bef2]]

bef2_scaled_positions = [
    (0.0, 0.0, 0.0),        # Be1
    (1/3, 2/3, 0.5),        # Be2
    (0.2, 0.4, 0.25),       # F1
    (0.8, 0.6, 0.75),       # F2
    (0.4, 0.2, 0.75),       # F3
    (0.6, 0.8, 0.25),       # F4
]
bef2_symbols = ['Be', 'Be', 'F', 'F', 'F', 'F']

BeF2_atoms = Atoms(symbols=bef2_symbols,
                   scaled_positions=bef2_scaled_positions,
                   cell=bef2_cell,
                   pbc=True)
BeF2_params = base_params.copy()
BeF2_params["txt"] = "BeF2_rlx.txt"
BeF2_params["kpts"] = {"gamma": True, "size": [1, 1, 1]}  # Hexagonal cell
BeF2_rlx_atoms = relax(BeF2_atoms,
                       BeF2_params,
                       fmax=0.01,
                       fixcell=False,
                       logname='BeF2_relax.log',
                       trajname='BeF2_relax.traj',
                       gpwname='BeF2_rlx.gpw')

print(f"BeF2: {len(BeF2_rlx_atoms)} atoms, cell volume: {BeF2_rlx_atoms.get_volume():.2f} Å³")

# Print relaxed energies
print("\n" + "="*60)
print("Relaxed energies")
print("="*60)
print(f"LiF relaxed energy: {LiF_rlx_atoms.get_potential_energy():.3f} eV")
print(f"LiF energy per atom: {LiF_rlx_atoms.get_potential_energy()/len(LiF_rlx_atoms):.3f} eV/atom")
print(f"BeF2 relaxed energy: {BeF2_rlx_atoms.get_potential_energy():.3f} eV")
print(f"BeF2 energy per atom: {BeF2_rlx_atoms.get_potential_energy()/len(BeF2_rlx_atoms):.3f} eV/atom")

# ============================================================
#  NVT EQUILIBRATION
# ============================================================
# Set appropriate temperatures (below melting points)
# LiF melting point: ~1121 K
# BeF2 melting point: ~825 K
lif_target_temp = 2000   # Below LiF melting point
bef2_target_temp = 2000  # Below BeF2 melting point

print("\n" + "="*60)
print(f"Initializing NVT equilibration")
print(f"LiF target: {lif_target_temp}K, BeF2 target: {bef2_target_temp}K")
print("="*60)

# Re-initialize velocities
MaxwellBoltzmannDistribution(LiF_rlx_atoms, temperature_K=lif_target_temp)
Stationary(LiF_rlx_atoms)
ZeroRotation(LiF_rlx_atoms)

MaxwellBoltzmannDistribution(BeF2_rlx_atoms, temperature_K=bef2_target_temp)
Stationary(BeF2_rlx_atoms)
ZeroRotation(BeF2_rlx_atoms)

# NVT Simulation parameters
timestep_fs = 1.0  
friction_coeff = 0.01  # Friction coefficient for Langevin thermostat
lif_traj_file = 'LiF_nvt_equilibration.traj'
lif_log_file = 'LiF_nvt_equilibration.log'
lif_restart_file = 'LiF_nvt_restart.pickle'

bef2_traj_file = 'BeF2_nvt_equilibration.traj'
bef2_log_file = 'BeF2_nvt_equilibration.log'
bef2_restart_file = 'BeF2_nvt_restart.pickle'

# Use Nose-Hoover Chain dynamics for NVT ensemble
lif_dyn = NHCNVT(
    LiF_rlx_atoms, 
    timestep_fs * units.fs,
    temperature_K=lif_target_temp,
    tdamp=100 * timestep_fs * units.fs,  # 100 fs damping
    trajectory=lif_traj_file,
    logfile=lif_log_file,
)
lif_dyn.attach(MDLogger(lif_dyn, LiF_rlx_atoms, lif_log_file), interval=10)

bef2_dyn = NHCNVT(
    BeF2_rlx_atoms, 
    timestep_fs * units.fs,
    temperature_K=bef2_target_temp,
    tdamp=100 * timestep_fs * units.fs,
    trajectory=bef2_traj_file,
    logfile=bef2_log_file,
)
bef2_dyn.attach(MDLogger(bef2_dyn, BeF2_rlx_atoms, bef2_log_file), interval=10)

def printenergy(a, step=None, label=""):
    """
    Function to print the thermodynamical properties
    """
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    
    try:
        stress = a.get_stress(voigt=False)
        pressure = -np.trace(stress) / 3.0
    except:
        pressure = 0.0
    
    if step is not None:
        print(f'{label} Step {step:6d}: ', end='')
    print(
        f'Epot = {epot/len(a):6.3f} eV/atom  '
        f'Ekin = {ekin/len(a):.3f} eV/atom '
        f'T={temp:.1f}K  P={pressure:.3f} eV/Å³'
    )

print("\nStarting NVT equilibration...")
printenergy(LiF_rlx_atoms, label="LiF")
printenergy(BeF2_rlx_atoms, label="BeF2")

# Initialize lists for data collection
lif_time_ps, lif_epot, lif_ekin, lif_temp, lif_pressure = [], [], [], [], []
bef2_time_ps, bef2_epot, bef2_ekin, bef2_temp, bef2_pressure = [], [], [], [], []

steps_per_block = 100
n_blocks = 100 

print(f"\nRunning {n_blocks * steps_per_block} steps of NVT equilibration...")
print(f"Timestep: {timestep_fs} fs")

for i in range(n_blocks):
    # Run LiF dynamics
    lif_dyn.run(steps_per_block)
    current_time = lif_dyn.get_time()
    lif_time_ps.append(current_time)
    lif_epot.append(LiF_rlx_atoms.get_potential_energy())
    lif_ekin.append(LiF_rlx_atoms.get_kinetic_energy())
    lif_temp.append(LiF_rlx_atoms.get_temperature())
    try:
        stress = LiF_rlx_atoms.get_stress(voigt=False)
        lif_pressure.append(-np.trace(stress) / 3.0)
    except:
        lif_pressure.append(0.0)
    
    # Run BeF2 dynamics
    bef2_dyn.run(steps_per_block)
    bef2_time_ps.append(bef2_dyn.get_time())
    bef2_epot.append(BeF2_rlx_atoms.get_potential_energy())
    bef2_ekin.append(BeF2_rlx_atoms.get_kinetic_energy())
    bef2_temp.append(BeF2_rlx_atoms.get_temperature())
    try:
        stress = BeF2_rlx_atoms.get_stress(voigt=False)
        bef2_pressure.append(-np.trace(stress) / 3.0)
    except:
        bef2_pressure.append(0.0)
    
    # Print progress every 10 blocks
    if (i+1) % 10 == 0:
        print(f"\n--- Block {i+1}/{n_blocks} (t={current_time:.2f} ps) ---")
        printenergy(LiF_rlx_atoms, step=lif_dyn.get_number_of_steps(), label="LiF")
        printenergy(BeF2_rlx_atoms, step=bef2_dyn.get_number_of_steps(), label="BeF2")

# Calculate total energies
lif_etot = np.array(lif_epot) + np.array(lif_ekin)
bef2_etot = np.array(bef2_epot) + np.array(bef2_ekin)

# ============================================================
#  SAVE RESTART FILES
# ============================================================
# LiF restart
LiF_restart_data = {
    'positions': LiF_rlx_atoms.get_positions(),
    'velocities': LiF_rlx_atoms.get_velocities(),
    'cell': LiF_rlx_atoms.get_cell(),
    'step': lif_dyn.get_number_of_steps(),
    'time': lif_dyn.get_time() / units.fs,
    'temperature': lif_target_temp,
}
with open(lif_restart_file, 'wb') as f:
    pickle.dump(LiF_restart_data, f)
print(f"\nRestart file saved: {lif_restart_file}")

# BeF2 restart
BeF2_restart_data = {
    'positions': BeF2_rlx_atoms.get_positions(),
    'velocities': BeF2_rlx_atoms.get_velocities(),
    'cell': BeF2_rlx_atoms.get_cell(),
    'step': bef2_dyn.get_number_of_steps(),
    'time': bef2_dyn.get_time() / units.fs,
    'temperature': bef2_target_temp,
}
with open(bef2_restart_file, 'wb') as f:
    pickle.dump(BeF2_restart_data, f)
print(f"Restart file saved: {bef2_restart_file}")

# ============================================================
#  SUMMARY STATISTICS
# ============================================================
print('\n' + '='*60)
print('Equilibration Summary')
print('='*60)

print(f"\nLiF (Target T={lif_target_temp}K):")
print(f"  Initial temperature: {lif_temp[0]:.1f} K")
print(f"  Final temperature: {lif_temp[-1]:.1f} K")
print(f"  Average temperature: {np.mean(lif_temp):.1f} ± {np.std(lif_temp):.1f} K")
print(f"  Temperature drift: {np.mean(lif_temp)-lif_target_temp:.1f} K")
print(f"  Energy drift: {(lif_etot[-1] - lif_etot[0])/len(LiF_rlx_atoms):.6f} eV/atom")

print(f"\nBeF2 (Target T={bef2_target_temp}K):")
print(f"  Initial temperature: {bef2_temp[0]:.1f} K")
print(f"  Final temperature: {bef2_temp[-1]:.1f} K")
print(f"  Average temperature: {np.mean(bef2_temp):.1f} ± {np.std(bef2_temp):.1f} K")
print(f"  Temperature drift: {np.mean(bef2_temp)-bef2_target_temp:.1f} K")
print(f"  Energy drift: {(bef2_etot[-1] - bef2_etot[0])/len(BeF2_rlx_atoms):.6f} eV/atom")

# ============================================================
#  PLOTTING
# ============================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Energy plot
ax1.plot(lif_time_ps, np.array(lif_epot)/len(LiF_rlx_atoms), 'b-', label='LiF Potential', alpha=0.7)
ax1.plot(lif_time_ps, np.array(lif_ekin)/len(LiF_rlx_atoms), 'r-', label='LiF Kinetic', alpha=0.7)
ax1.plot(lif_time_ps, np.array(lif_etot)/len(LiF_rlx_atoms), 'k--', label='LiF Total', alpha=0.7)
ax1.plot(bef2_time_ps, np.array(bef2_epot)/len(BeF2_rlx_atoms), 'c-', label='BeF2 Potential', alpha=0.7)
ax1.plot(bef2_time_ps, np.array(bef2_ekin)/len(BeF2_rlx_atoms), 'm-', label='BeF2 Kinetic', alpha=0.7)
ax1.plot(bef2_time_ps, np.array(bef2_etot)/len(BeF2_rlx_atoms), 'y--', label='BeF2 Total', alpha=0.7)
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Energy per atom (eV)')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_title('Energy Evolution')

# Temperature plot
ax2.plot(lif_time_ps, lif_temp, 'g-', label='LiF', linewidth=1.5)
ax2.plot(bef2_time_ps, bef2_temp, 'orange', label='BeF2', linewidth=1.5)
ax2.axhline(y=lif_target_temp, color='g', linestyle='--', alpha=0.5, label=f'LiF Target')
ax2.axhline(y=bef2_target_temp, color='orange', linestyle='--', alpha=0.5, label=f'BeF2 Target')
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Temperature (K)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_title('Temperature Evolution')

# Pressure plot
ax3.plot(lif_time_ps, lif_pressure, 'b-', label='LiF', linewidth=1.5)
ax3.plot(bef2_time_ps, bef2_pressure, 'orange', label='BeF2', linewidth=1.5)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.set_xlabel('Time (ps)')
ax3.set_ylabel('Pressure (eV/Å³)')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_title('Pressure Evolution')

# Temperature histogram
ax4.hist(lif_temp, bins=20, color='green', alpha=0.7, edgecolor='black', label='LiF')
ax4.hist(bef2_temp, bins=20, color='orange', alpha=0.7, edgecolor='black', label='BeF2')
ax4.axvline(x=lif_target_temp, color='g', linestyle='--', label=f'LiF Target')
ax4.axvline(x=bef2_target_temp, color='orange', linestyle='--', label=f'BeF2 Target')
ax4.set_xlabel('Temperature (K)')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.set_title('Temperature Distribution')

plt.tight_layout()

# Save the plot
plot_file = 'LiF_BeF2_nvt_equilibration.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {plot_file}")
plt.show()

# ============================================================
#  PRESSURE DRIFT ANALYSIS
# ============================================================
lif_pressure = np.array(lif_pressure)
bef2_pressure = np.array(bef2_pressure)

# Check drift in last 20%
lif_last_idx = int(0.8 * len(lif_pressure))
lif_last_pressure = lif_pressure[lif_last_idx:]
lif_last_time = lif_time_ps[lif_last_idx:]

bef2_last_idx = int(0.8 * len(bef2_pressure))
bef2_last_pressure = bef2_pressure[bef2_last_idx:]
bef2_last_time = bef2_time_ps[bef2_last_idx:]

# Linear fit to check drift
z_lif = np.polyfit(lif_last_time, lif_last_pressure, 1)
lif_drift = z_lif[0]

z_bef2 = np.polyfit(bef2_last_time, bef2_last_pressure, 1)
bef2_drift = z_bef2[0]

print('\n' + '='*60)
print('Pressure Analysis')
print('='*60)
print(f"\nLiF Pressure:")
print(f"  Mean (full): {np.mean(lif_pressure):.6f} ± {np.std(lif_pressure):.6f} eV/Å³")
print(f"  Mean (last 20%): {np.mean(lif_last_pressure):.6f} eV/Å³")
print(f"  Drift (last 20%): {lif_drift:.6f} eV/Å³/ps")

print(f"\nBeF2 Pressure:")
print(f"  Mean (full): {np.mean(bef2_pressure):.6f} ± {np.std(bef2_pressure):.6f} eV/Å³")
print(f"  Mean (last 20%): {np.mean(bef2_last_pressure):.6f} eV/Å³")
print(f"  Drift (last 20%): {bef2_drift:.6f} eV/Å³/ps")

# ============================================================
#  SAVE FINAL CONFIGURATIONS
# ============================================================
LiF_rlx_atoms.write('LiF_equilibrated.xyz')
BeF2_rlx_atoms.write('BeF2_equilibrated.xyz')
print(f"\nEquilibrated configurations saved to:")
print(f"  LiF: LiF_equilibrated.xyz")
print(f"  BeF2: BeF2_equilibrated.xyz")

# Optional: View the structures
view(LiF_rlx_atoms)
view(BeF2_rlx_atoms)