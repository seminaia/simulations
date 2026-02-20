from lammps import lammps
import numpy as np

# Create LAMMPS instance
L = lammps()

# Use metal units for aluminum (Angstroms, eV, etc.)
L.command("units metal")
L.command("atom_style atomic")
L.command("atom_modify map array sort 0 0")

# Create aluminum block dimensions (e.g., 10x10x10 unit cells)
# Aluminum FCC lattice constant = 4.05 Å
lattice_constant = 4.05  # Ångstroms
nx, ny, nz = 5, 5, 5  # Number of unit cells in each direction

# Calculate box dimensions
xlo, xhi = 0, nx * lattice_constant
ylo, yhi = 0, ny * lattice_constant
zlo, zhi = 0, nz * lattice_constant

print(f"Creating aluminum block:")
print(f"  Lattice constant: {lattice_constant} Å")
print(f"  Dimensions: {xhi:.1f} x {yhi:.1f} x {zhi:.1f} Å³")
print(f"  Unit cells: {nx} x {ny} x {nz}")

# Create simulation box
L.command(f"region box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}")
L.command("create_box 1 box")

# Set up FCC lattice for aluminum
# Aluminum has FCC structure with 4 atoms per unit cell
L.command(f"lattice fcc {lattice_constant}")
L.command("region myblock block 0 {} 0 {} 0 {}".format(nx, ny, nz))
L.command("create_atoms 1 region myblock")

# Set aluminum mass (in g/mol, but LAMMPS metal units use g/mol)
L.command("mass 1 26.98")  # Aluminum atomic mass

# melt example
L.command("pair_style lj/cut 2.5")
L.command("pair_coeff 1 1 1.0 1.0 2.5")
L.command("neighbor 0.3 bin")
L.command("neigh_modify every 20 delay 0 check no")
L.command("fix 1 all nve")
L.command("thermo 50")
L.command("run 250")# Compute initial temperature and energy
L.command("compute temp all temp")
L.command("compute pe all pe")
L.command("compute ke all ke")

# Set initial velocities (corresponding to 300K)
L.command("velocity all create 300.0 4928459 rot yes dist gaussian")

# Define thermodynamic output
L.command("thermo 100")
L.command("thermo_style custom step temp pe ke etotal press")

# Minimize energy to relax the structure
print("\nPerforming energy minimization...")
L.command("min_style cg")
L.command("minimize 1.0e-6 1.0e-8 1000 10000")

# Run a short equilibration at 300K
print("\nEquilibrating at 300K...")
L.command("fix 1 all nve")
L.command("fix 2 all nvt temp 300.0 300.0 0.1")
L.command("run 1000")

# Switch to NVE for production run
print("\nRunning production simulation...")
L.command("unfix 1")
L.command("unfix 2")
L.command("fix 3 all nve")
L.command("run 5000")

# Get final atom count
nlocal = L.extract_setting("nlocal")
print(f"\nSimulation complete!")
print(f"Total atoms: {nlocal}")

# Extract and display final temperature and energy
temp = L.extract_compute("temp", 0, 0)
pe = L.extract_compute("pe", 0, 0)
ke = L.extract_compute("ke", 0, 0)
print(f"Final temperature: {temp:.2f} K")
print(f"Final potential energy: {pe:.4f} eV")
print(f"Final kinetic energy: {ke:.4f} eV")
print(f"Total energy: {pe+ke:.4f} eV")

# Optional: Get coordinates of last atom as example
if nlocal > 0:
    x = L.numpy.extract_atom("x")
    print(f"\nExample - Last atom position: ({x[-1][0]:.3f}, {x[-1][1]:.3f}, {x[-1][2]:.3f}) Å")

L.close()