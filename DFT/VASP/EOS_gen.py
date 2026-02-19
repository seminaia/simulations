import os
from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MPScanRelaxSet
from pymatgen.core import Lattice

def create_master_submission_script(script_path, subfolders_name_relax, subfolders_name_static):
    """
    Create a master submission script that loops over subfolders and submits jobs.
    """
    script_content = f"""#!/bin/bash

for dir in ./{subfolders_name_relax}/scale_*; do
    folder_name=$(basename "$(dirname "$dir")")
    mag_type=$(basename "$dir")
    
    # Copy update_mag.py if needed
    cp update_mag.py "$dir"
    
    # Navigate to the directory
    cd "$dir"
    
    # If a relax script is found, prepare and submit it
    if [ -f submit_job_relax.sh ]; then
        cp INCAR_RELAX_ISIF_2 INCAR

        sbatch submit_job_relax.sh
    fi

    cd ../../../  # Go back three levels to the base directory
done

for dir in {subfolders_name_static}/lattice_*/{{FM,AFM,PM,NM}}; do
    folder_name=$(basename "$(dirname "$dir")")
    mag_type=$(basename "$dir")
    cp update_mag.py "$dir"
    # Navigate to the directory
    cd "$dir"
    
    # If a static script is found, prepare and submit it
    if [ -f submit_job_static.sh ]; then
        cp INCAR_STATIC_ISIF_4 INCAR
        
        # Copy relaxed OUTCAR and CONTCAR from corresponding relax folder
        cp ../../../eos_relax/"$folder_name"/"$mag_type"/OUTCAR ./
        cp ../../../eos_relax/"$folder_name"/"$mag_type"/CONTCAR ./POSCAR
        
        python update_mag.py     
        sbatch  submit_job_static.sh
    fi

    cd ../../../
done
"""
    write_file_safe(script_path, script_content) # type: ignore
    
# Configuration
API_KEY = "n8aGzYQW2CXYIsrNvNzUeMsY37hJBtV0"
MATERIAL_FORMULA = "La2NiO4"
TARGET_SPACE_GROUP = "I4/mmm"
SCALING_FACTORS = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]
AFM_ELEMENTS = {"Ni"}
MAIN_DIR = f"./{MATERIAL_FORMULA}_EOS"

# Element properties
ELEMENT_PROPERTIES = {
    'La': {'charge': +3, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
    'Ni': {'charge': +2, 'magmom': 2.0, 'LDAUL': 2, 'LDAUU': 6.2, 'LDAUJ': 0.0},
    'O':  {'charge': -2, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
}
POTCAR_SETTINGS = {"La": "La", "Ni": "Ni_pv", "O": "O"}

# INCAR settings for R2SCAN+U+RVV10
INCAR_SETTINGS = {
    "SYSTEM": MATERIAL_FORMULA,
    "ALGO": "All",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "ICHARG": 2,
    "ISIF" : 2,
    "ISMEAR": 0,
    "ISPIN": 2,
    "ISYM": 0,
    "SIGMA": 0.1,
    "NELM": 200,
    "KPAR": 6,
    "NCORE": 10,
    "LCHARG": True,
    "LMAXMIX": 6,
    "LORBIT": 11,
    "LREAL": False,
    "LWAVE": True,
    "NSW": 100,
    "PREC": "Accurate",
    "KSPACING": 0.2,
    "LAECHG": True,
    "LASPH": True,
    "LMIXTAU": True,
    "LVTOT": True,
    "LDAU": True,
    "LDAUTYPE": 2,
    "LDAUL": {el: ELEMENT_PROPERTIES[el]["LDAUL"] for el in ELEMENT_PROPERTIES},
    "LDAUU": {el: ELEMENT_PROPERTIES[el]["LDAUU"] for el in ELEMENT_PROPERTIES},
    "LDAUJ": {el: ELEMENT_PROPERTIES[el]["LDAUJ"] for el in ELEMENT_PROPERTIES},
    "LUSE_VDW": True,
    "BPARAM": 11.95,
    "CPARAM": 0.0093,
}

def assign_afm_moments(structure):
    """Assign alternating magnetic moments to AFM elements"""
    magmom = []
    counters = {el: 0 for el in AFM_ELEMENTS}
    
    for site in structure:
        species = site.species_string
        if species in AFM_ELEMENTS:
            base = ELEMENT_PROPERTIES[species]["magmom"]
            moment = base if (counters[species] % 2 == 0) else -base
            counters[species] += 1
        else:
            moment = ELEMENT_PROPERTIES.get(species, {}).get("magmom", 0.0)
        magmom.append(moment)
        
    return magmom

def create_job_script(directory: str, scale: float, email: str):
    """Create Slurm job script for EOS calculations"""
    script = f"""#!/bin/bash
#SBATCH --job-name=EOS_{scale:.2f}
#SBATCH -p spr
#SBATCH -o job.out
#SBATCH -e job.err
#SBATCH -N 6
#SBATCH -n 96
#SBATCH -t 48:00:00

module load vasp/6.4.2
ibrun --mpi=pmi2 vasp_std > vasp.out
"""
    with open(os.path.join(directory, "submit.sh"), "w") as f:
        f.write(script)
    os.chmod(os.path.join(directory, "submit.sh"), 0o755)

def main():
    # Create main directory
    os.makedirs(MAIN_DIR, exist_ok=True)

    # Get structure from Materials Project if CONTCAR not found
    
    with MPRester(API_KEY) as mpr:
        entries = mpr.get_entries(
            MATERIAL_FORMULA,
            conventional_unit_cell=True,
            inc_structure=True
        )
        structure = next(
            e.structure for e in entries
            if SpacegroupAnalyzer(e.structure).get_space_group_symbol() == TARGET_SPACE_GROUP
        )

    # Assign magnetic moments
    magmom = assign_afm_moments(structure)
    structure.add_site_property("magmom", magmom)
    
    original_structure = structure.copy()
    original_structure.add_site_property("magmom", magmom)

    for scale in SCALING_FACTORS:
        scale_dir = os.path.join(MAIN_DIR, f"scale_{scale:.2f}")
        os.makedirs(scale_dir, exist_ok=True)

        # Write VASP inputs with ORIGINAL structure
        vasp_set = MPScanRelaxSet(
            original_structure,
            user_incar_settings=INCAR_SETTINGS,
            user_potcar_settings=POTCAR_SETTINGS,
            vdw="rvv10",
        )
        vasp_set.write_input(scale_dir)

        # Override POSCAR with explicit scaling factor
        poscar_path = os.path.join(scale_dir, "POSCAR")
        with open(poscar_path, "r") as f:
            lines = f.readlines()
        
        # Modify the scaling factor line (second line)
        lines[1] = f"{scale:.8f}\n"  # <-- Actual scaling factor here
        
        with open(poscar_path, "w") as f:
            f.writelines(lines)

        # Create job script
        create_job_script(scale_dir, scale, "ssem@wpi.edu")


if __name__ == "__main__":
    main()