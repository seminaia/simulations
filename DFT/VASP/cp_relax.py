import os
from monty.serialization import dumpfn, loadfn
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar, Kpoints, Incar, Potcar, VaspInput

# Define elements and k-points grid
elements = ["La", "Ni", "O"]
material = "La2NiO4"
converged = {
    "La15(NiO4)8_EaH_0.0492": [2, 2, 3],
    "La2NiO4_EaH_0.0469": [3, 3, 1],
    "La2NiO4_EaH_0.0488": [3, 3, 2],
    "La2O3_EaH_0": [3, 3, 3],
    "La2O3_EaH_0.0244": [4, 4, 2],
    "La2O3_EaH_0.0296": [2, 2, 1],
    "La3Ni2O7_EaH_0.0234": [3, 3, 5],
    "La_EaH_0": [9, 9, 2],
    "La_EaH_0.0007": [13, 13, 13],
    "LaNiO3_EaH_0": [5, 5, 5],
    "LaNiO3_EaH_0.0199": [4, 3, 4],
    "LaNiO3_EaH_0.0352": [7, 7, 7],
    "Ni15O16_EaH_0.0268": [5, 5, 5],
    "Ni_EaH_0": [16, 16, 16],
    "Ni_EaH_0.0258": [17, 17, 10],
    "NiO_EaH_0": [7, 7, 4],
    "O2_EaH_0": [12, 12, 18]
}

# Create a new dictionary with separated keys
separate = {}

for key, value in converged.items():
    compound, eah = key.split("_EaH_")
    eah_value = float(eah)  # Convert EaH value to a float
    separate[(compound, eah_value)] = value

potcar_yaml = "/home/soki/doped/doped/VASP_sets/PotcarSet.yaml"

# Ensure YAML file exists
if not os.path.exists(potcar_yaml):
    raise FileNotFoundError(f"YAML file not found at {potcar_yaml}")

# Initialize MPRester
with MPRester() as mpr:
    for (compound, eah), kpts in separate.items():
        try:
            name = f"{compound}_EaH_{eah:}"
            print(f"Processing {name}...")

            entries = mpr.get_entries(compound, inc_structure=True, sort_by_e_above_hull=True)
            if not entries:
                raise ValueError(f"No entries found for compound: {compound}")

                # Log all entries for debugging            
            for entry in entries:
                # Filter entries based on the specified EaH
                TOLERANCE = 1e-3
            
                entry = next((entry for entry in entries if abs(entry.data.get('e_above_hull', 0) - eah) < TOLERANCE), None)
                if not entry:
                    raise ValueError(f"No entry with EaH={eah:.4f} found for compound {compound}")
                    # Log selected entry for debugging
                
                structure = entry.structure
            
            print(f"Selected Structure: {entry.composition}, e_above_hull: {entry.data.get('e_above_hull', 0):.4f}")

            # Create directory path
            path = os.path.join("competing_phases", name, "vasp_std")
            os.makedirs(path, exist_ok=True)

            # Write POSCAR file
            poscar = Poscar(structure)
            poscar.write_file(os.path.join(path, "POSCAR"))

            # Write KPOINTS file
            kpoints = Kpoints.gamma_automatic(kpts=kpts)
            kpoints.write_file(os.path.join(path, "KPOINTS"))

            # Write INCAR file
            incar_path = "INCAR"

            if not os.path.exists(incar_path):
                raise FileNotFoundError(f"INCAR template not found at {incar_path}")
            incar = Incar.from_file(incar_path)
            incar.write_file(os.path.join(path, "INCAR"))

            # Load pseudopotentials from YAML file
            potcar_dict = loadfn(potcar_yaml)
            if "POTCAR" not in potcar_dict:
                raise KeyError("Invalid YAML format. 'POTCAR' key not found.")

            # Retrieve POTCAR names for each element
            potcar_names = []
            for el in structure.composition.elements:
                try:
                    potcar_name = potcar_dict["POTCAR"][str(el)]
                    potcar_names.append(potcar_name)
                except KeyError:
                    raise KeyError(f"Pseudopotential for element '{el}' not found in POTCAR dictionary")

            print(f"Pseudopotentials for {name}: {potcar_names}")

            # Write POTCAR file
            potcar = Potcar(potcar_names)
            potcar.write_file(os.path.join(path, "POTCAR"))

            # Write VASP input files
            vasp_input = VaspInput(incar, kpoints, poscar, potcar)
            vasp_input.write_input(path)

            print(f"Processed {name} successfully.")

        except Exception as e:
            print(f"An error occurred for {name}: {e}")
