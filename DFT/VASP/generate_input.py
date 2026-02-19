from pymatgen.core.structure import Structure
from pymatgen.analysis.elasticity.strain import DeformedStructureSet
from pymatgen.io.vasp import Incar, Poscar, Kpoints, Potcar, VaspInput
import os

# ---------------------------
# Configuration (Edit these!)
# ---------------------------
MAIN_DIR = "elastic_inputs"  # All files will go here
ENCUT = 600                  # Adjust based on your POTCAR ENMAX
# ---------------------------

def generate_static_input(structure, output_dir):
    incar = Incar({
        "PREC": "Accurate",
        "EDIFF": 1e-6,
        "ENCUT": ENCUT,
        "METAGGA": "R2SCAN",
        "LASPH": True,
        "ISMEAR": 0,
        "SIGMA": 0.1,
        "KPAR" : 6,
        "NCORE" : 12,
        "ALGO": "All",
        "LREAL": False,
        "ADDGRID": True,
        "IBRION": -1,
        "NSW": 0,
        "ISIF": 3,
        "LWAVE": False,
        "LCHARG": False,
    })
    
    poscar = Poscar(structure)
    kpoints = Kpoints.automatic_gamma_density(structure,kppa = 5000)
    potcar = Potcar(symbols=[s.split("_")[0] for s in structure.symbol])
    
    # Write all input files to the target directory
    VaspInput(incar, kpoints, poscar, potcar).write_input(output_dir)

# Setup main directory
os.makedirs(MAIN_DIR, exist_ok=True)

# Load structure and generate deformations
structure = Structure.from_file("La2NiO4_I4_mmm.cif")
dss = DeformedStructureSet(structure, symmetry=True)

# Generate input files for each deformation
for i, deformation in enumerate(dss.deformations):
    deform_dir = os.path.join(MAIN_DIR, f"deformation_{i:02d}")
    os.makedirs(deform_dir, exist_ok=True)
    deformed_structure = deformation.apply_to_structure(structure)
    generate_static_input(deformed_structure, deform_dir)

print(f"Generated {len(dss.deformations)} deformation directories in '{MAIN_DIR}'.")