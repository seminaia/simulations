
import re
from ase.build import bulk
import ase.build
import ase.lattice
from shakenbreak.input import Distortions
from ase.spacegroup import crystal
from ase.optimize import BFGS, QuasiNewton
from ase.visualize import view
from ase.vibrations import Vibrations
from ase.filters import FrechetCellFilter
from ase.transport.calculators import TransportCalculator
from gpaw import GPAW, PW, restart, Mixer, MixerSum, MixerDif
from ase.eos import calculate_eos
from matplotlib.pylab import eig
import numpy as np
#from atomistics.calculators.ase import evaluate_with_ase
#from atomistics.workflows import ElasticMatrixWorkflow
#from atomistics.workflows import EnergyVolumeCurveWorkflow
#import atomistics
from ase import Atoms
import os
from ase.io import Trajectory, read
import matplotlib.pyplot as plt
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
    
    # Get final forces for reporting
    if hasattr(opt_atoms, 'get_forces'):
        forces = opt_atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        print(f"Relaxation completed. Maximum force: {max_force:.6f} eV/Å")
    return opt_atoms
# Structure setup
a0 = 3.80
c0 = 12.45
conv_cell = [
    [a0, 0, 0],
    [0, a0, 0],
    [0, 0, c0]
]
prim_cell = [
    [a0, 0, 0],
    [0, a0, 0],
    [-a0, -a0, c0]
]
symbols = ('La','La','La','La','Ni','Ni','O','O','O','O','O','O','O','O')
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

prim_frac_positions = [
    (0.36, 0.36, 0.36),
    (0.36, 0.36, 0.86),
    (0.64, 0.64, 0.14),
    (0.64, 0.64, 0.64),
    (0.00, 0.00, 0.00),
    (0.00, 0.00, 0.50),
    (0.18, 0.18, 0.18),
    (0.18, 0.18, 0.68),
    (0.82, 0.82, 0.32),
    (0.82, 0.82, 0.82),
    (0.50, 0.00, 0.00),
    (0.50, 0.00, 0.50),
    (0.00, 0.50, 0.00),
    (0.00, 0.50, 0.50)]
magmoms = [0.6, -0.6, 0.6, -0.6, 2.0, -2.0, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6]

conv_atoms = Atoms(symbols=symbols,
              scaled_positions=conv_frac_positions,
              cell=conv_cell,
              pbc=True)
conv_atoms.set_initial_magnetic_moments(magmoms)
conv_atoms.write('La2NiO4_conv.cif', format='cif')
view(conv_atoms, repeat=(2, 2, 1))

prim_atoms = Atoms(symbols=symbols,
                scaled_positions=prim_frac_positions,
                cell=prim_cell,
                pbc=True)
prim_atoms.set_initial_magnetic_moments(magmoms)
prim_atoms.write('La2NiO4_prim.cif', format='cif')
#view(prim_atoms, repeat=(2, 2, 1))

calculator_params = {
    "convergence": {"density": 1e-4,
                    "eigenstates": 1e-8,
                    "energy": 1e-6,
                    "forces": 1e-4},
    "eigensolver": {"name": "cg",
                    "niter":5},
    "kpts": {"gamma": True,
            "size": [2, 2, 1]},
    "maxiter": 500,
    "mixer":{"backend": "pulay",
            "beta": 0.1,
            "method": "fullspin",
            "nmaxold": 5,
            "weight":100},
    "mode": {"ecut": 520,
            "name": "pw"},
    "nbands":"nao",
    "occupations": {"name": "fermi-dirac",
                    "width": 0.01},
    "setups": {"Ni": ':d, 6.2'},
    "symmetry":"off",
    "txt": "rlx.txt",
    "xc": "PBE"
}

# Run relaxation
conv_relaxed_atoms = relax(conv_atoms, 
                           calculator_params,
                      fmax=0.01,
                      fixcell=False, 
                      logname='La2NiO4_conv_opt.log',
                      trajname='La2NiO4_conv_opt.traj',
                      gpwname='La2NiO4_conv_rlx.gpw')
#view(conv_relaxed_atoms, repeat=(2, 2, 1))
if isinstance(conv_relaxed_atoms, FrechetCellFilter):
    conv_relaxed_atoms = conv_relaxed_atoms.atoms

# Get results
E_conv = conv_relaxed_atoms.get_potential_energy()
conv_pos = conv_relaxed_atoms.get_positions()
conv_relax_cell = conv_relaxed_atoms.get_cell()
#conv_relaxed_atoms.write('La2NiO4_relaxed_conv.cif', format='cif')
print("Relaxed cell:")
print(conv_relax_cell)
print("Potential energy (eV):", E_conv)
#print(conv_scaled_pos)
print("All positions (Angstrom):")
print(conv_pos)
#prim_relaxed_atoms = relax(prim_atoms, 
#                           calculator_params,
#                           fmax =0.01,
#                           fixcell=False,
#                           logname='La2NiO4_prim_opt.log',
#                           trajname='La2NiO4_prim_opt.traj',
#                           gpwname='La2NiO4_prim_rlx.gpw')
#prim_relaxed_atoms.set_cell(conv_relaxed_atoms.get_cell(), scale_atoms=True)
#
#E_prim = prim_relaxed_atoms.get_potential_energy()
#prim_relaxed_atoms.write('La2NiO4_relaxed_prim.cif', format='cif')
#prim_scaled_positions = prim_relaxed_atoms.get_scaled_positions()
#prim_positions = prim_relaxed_atoms.get_positions()
#print("Potential energy (eV):", E_prim)
#print("All scaled positions (fractional):")
#print(prim_scaled_positions)
#print("All positions (Angstrom):")
#print(prim_positions)
# Lattice constant scan
print("\nStarting lattice constant scan...")
eps = 0.03
# Reopen or create new trajectory for scan results
scan_traj = Trajectory('La2NiO4_conv_scan.traj', 'w')
for a in a0 * np.linspace(1-eps, 1+eps, 3):
    for c in c0 * np.linspace(1-3*eps, 1+3*eps, 5):
        scan_atoms = conv_relaxed_atoms.copy()
        scan_atoms.set_cell([[a, 0, 0], [0, a, 0], [0, 0, c]], scale_atoms=True)
        #magmoms = conv_relaxed_atoms.get_magnetic_moments()
        scan_params = dict(calculator_params)  
        scan_params["txt"] = f"lat_scan.txt"
        scan_atoms.calc = GPAW(**scan_params)
        # Calculate energy
        E = scan_atoms.get_potential_energy()
        print(f"a={a:.3f} c={c:.3f} c/a = {c/a:.3f} E={E:.6f} eV")
        scan_atoms.write(f'La2NiO4_conv_scan_a{a:.3f}_c{c:.3f}.cif', format='cif')
        scan_traj.write(scan_atoms)

scan_traj.close()

# Fit EOS
print("\nFitting equation of state...")
a_list, c_list, energies = [], [], []
configs = read('La2NiO4_conv_scan.traj',index=':')  # Read all configurations from scan trajectory
for config in configs:
    cell = config.get_cell()
    a_list.append(cell[0, 0])
    c_list.append(cell[2, 2])
    energies.append(config.get_potential_energy())

a = np.array(a_list)
c = np.array(c_list)
energies = np.array(energies)

# Dimensionless strains about (a0, c0)
x_all = (a - a0) / a0
y_all = (c - c0) / c0

# ---- window near the minimum BEFORE fitting ----
Emin = energies.min()
mask = energies < Emin + 0.05   # 50 meV window (tune if needed)

x = x_all[mask]
y = y_all[mask]
Efit = energies[mask]

# Design matrix for quadratic surface in (x, y)
X = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2])

# Least-squares fit: E(x,y) = E0 + A x + B y + C x^2 + D x y + F y^2
p = np.linalg.lstsq(X, Efit, rcond=None)[0]
E0, A, B, C, D, F = p

# Hessian and gradient in (x,y)
H = np.array([[2*C, D],
              [D, 2*F]])
g = np.array([A, B])

# Check convexity (minimum)
eigvals = np.linalg.eigvals(H)
assert np.all(eigvals > 0), f"Fit not convex (eigvals={eigvals}); reduce eps or window tighter."

# Solve for minimum in strain space
x_eq, y_eq = -np.linalg.solve(H, g)

a_eq = a0 * (1 + x_eq)
c_eq = c0 * (1 + y_eq)
Veq = a_eq**2 * c_eq

# Evaluate energy at minimum USING x_eq, y_eq (not a_eq, c_eq)
Eeq = E0 + A*x_eq + B*y_eq + C*x_eq**2 + D*x_eq*y_eq + F*y_eq**2

print(f"Equilibrium a={a_eq:.6f} Å")
print(f"Equilibrium c={c_eq:.6f} Å")
print(f"Equilibrium volume={Veq:.6f} Å^3")
print(f"Minimum energy={Eeq:.10f} eV")

# Plot the sampled points (in strain space)
plt.figure()
plt.scatter(x, y, c=Efit)
plt.xlabel('(a - a0)/a0')
plt.ylabel('(c - c0)/c0')
plt.colorbar(label='Energy (eV)')
plt.title('Lattice Constant Scan Energy (windowed)')
plt.grid(True)
plt.show()
