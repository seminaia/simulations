import os
import logging
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Incar
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_elements(structure):
    """
    Return a list of unique elements (species) from a pymatgen Structure.
    """
    unique_elements = OrderedDict()
    for site in structure:
        unique_elements[site.species_string] = None
    return list(unique_elements.keys())

def load_structure(file_path: str) -> Structure:
    """Load crystal structure from file with error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    return Structure.from_file(file_path)

# Constants
MATERIAL = "La2NiO4"
VASPRUN_FILE = f"./{MATERIAL}_bulk/PBE_DOS/DOS_8/vasprun.xml"
INCAR_FILE = f"./{MATERIAL}_bulk/PBE_DOS/DOS_8/INCAR"
OUTPUT_DIR = f"./{MATERIAL}_bulk/Static_Dielectric"
contcar_path = "./CONTCAR"
structure = load_structure(contcar_path)

elements = extract_elements(structure)
# ------------------- Define Element Properties & Magnetism ---------------
element_properties = {
    'La': {'charge': +3, 'magmom': 2.0, 'LDAUL':  -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
    'Ni': {'charge': +2, 'magmom': 2.0, 'LDAUL':  2, 'LDAUU':  8.0, 'LDAUJ': 0.0},
    'O':  {'charge': -2, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU':  0.0, 'LDAUJ': 0.0},
}
# Settings
POTCAR_OVERRIDES = {"La": "La", "Ni": "Ni_pv", "O": "O"}
STATIC_INCAR_SETTINGS = {
    "ALGO": "All",
    "EDIFF": 1e-7,
    "EDIFFG":-0.001,
    "ENCUT": 520,
    "ISIF": 2,
    "IBRION": 8,
    "ISMEAR": 0,
    "SIGMA": 0.01,
    "PREC": "Accurate",
    "LASPH": True,
    "LCALCEPS":False,
    "LPEAD": True,
    "LREAL": False,
    "LWAVE": False,
    "LEPSILON": True,
    "KPAR": 6,
    "NCORE": 1,
    "POTIM":0.015
}

r2scan_u = {
    "METAGGA": "R2SCAN",
    "ADDGRID": True,
}
u = {
    "LDAU": True,
    "LDAUTYPE": 2,
    "LDAUL": {el: element_properties[el]["LDAUL"] for el in elements},
    "LDAUU": {el: element_properties[el]["LDAUU"] for el in elements},
    "LDAUJ": {el: element_properties[el]["LDAUJ"] for el in elements}
    }
r2scan_u_settings = {**STATIC_INCAR_SETTINGS, **r2scan_u}
static_u_settings = {**STATIC_INCAR_SETTINGS, **u}
def validate_vasprun_file(file_path: str) -> Structure:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Vasprun file '{file_path}' not found.")
    return Vasprun(file_path, parse_dos=False, parse_eigen=False).final_structure

def parse_magnetic_moments(incar_path: str, num_sites: int) -> list:
    try:
        incar = Incar.from_file(incar_path)
    except Exception as e:
        logging.error(f"Failed to parse INCAR: {e}")
        return [0.0] * num_sites

    magmom = incar.get("MAGMOM", [0.0] * num_sites)
    
    if isinstance(magmom, str):
        magmom_list = []
        for part in magmom.strip().split():
            if "*" in part:
                count, val = part.split("*")
                magmom_list.extend([float(val)] * int(count))
            else:
                magmom_list.append(float(part))
        return magmom_list
    elif isinstance(magmom, list):
        return magmom
    else:
        logging.warning(f"Invalid MAGMOM type: {type(magmom)}. Using defaults.")
        return [0.0] * num_sites

def main():
    try:
        #structure = validate_vasprun_file(VASPRUN_FILE)
        structure.add_site_property("magmom", parse_magnetic_moments(INCAR_FILE, len(structure.sites)))
        structure.add_oxidation_state_by_element({el: element_properties[el]['charge'] for el in elements})
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        vasp_set = MPStaticSet(
            structure,
            user_incar_settings=static_u_settings,
            user_potcar_settings=POTCAR_OVERRIDES,
            user_potcar_functional="PBE_54",
            force_gamma=True,
            lepsilon=True,
        )
        vasp_set.write_input(OUTPUT_DIR)
        logging.info(f"Input files generated in '{OUTPUT_DIR}'.")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()