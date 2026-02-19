from ase.build import molecule
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from gpaw import GPAW, PW, Mixer

# Create O₂ molecule with proper vacuum
atoms = molecule('O2')
atoms.center(vacuum=7.0)  # Increased vacuum for better isolation

# GPAW calculator with convergence settings
calc = GPAW(
    mode=PW(400),          # Increased cutoff for better accuracy
    xc='PBE',
    mixer=Mixer(beta=0.05, nmaxold=5),  # Stabilize SCF
    kpts=(1, 1, 1),        # Gamma-point only for molecule
    occupations={'name': 'fermi-dirac', 'width': 0.05},  # Smearing
    spinpol=True,           # Important for O₂ triplet state
    symmetry='off',         # Disable symmetry for vibrations
    txt='pbe_optimized.txt'
)
atoms.calc = calc

# Geometry optimization with tighter forces
dyn = QuasiNewton(atoms, trajectory='O2_opt.traj')
dyn.run(fmax=0.001)  # Tighter convergence

# Verify electronic structure
pe = atoms.get_potential_energy()
magmom = atoms.get_magnetic_moment()
print(f"Final magnetic moment: {magmom:.2f} μB")

# Vibrational analysis (only non-zero modes)
vib = Vibrations(atoms, name='vib_O2')
vib.run()
vib_energies = vib.get_energies()

# Thermodynamic properties with correct parameters
thermo = IdealGasThermo(
    vib_energies=vib_energies,
    potentialenergy=pe,
    geometry='linear',
    atoms=atoms,
    symmetrynumber=1,
    spin=1,        # Triplet state (S=1 → 2S+1=3)
    ignore_imag_modes=True  # O₂ should have 1 imaginary mode (rotation)
)
# Temperature and pressure setup
T = 298.15  # Standard temperature (25°C)
pressure = 101325  # 1 atm in Pa

# Calculate properties with proper unit conversion
S = thermo.get_entropy(T, pressure) * 96485.3329  # J/(mol·K)
H = thermo.get_enthalpy(T) * 96485.3329           # J/mol
G = thermo.get_gibbs_energy(T, pressure) * 96485.3329  # J/mol

print(f"\nResults for O₂ at {T} K:")
print(f"Entropy: {S:.2f} J/(mol·K)")
print(f"Enthalpy: {H:.2f} J/mol")
print(f"Gibbs Free Energy: {G:.2f} J/mol")