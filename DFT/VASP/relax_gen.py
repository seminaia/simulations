import os
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPScanStaticSet, MPRelaxSet, MPScanRelaxSet, MPStaticSet
from ase.io import write
import crystal_toolkit.components as ctc
from crystal_toolkit.settings import SETTINGS

def format_magmom_for_incar(magmom_list: list) -> str:
    """Convert magnetic moments list to INCAR format string."""
    return " ".join(map(str, magmom_list))

def add_magmom_to_incar(incar_path: str, magmom_incar: str) -> None:
    """Add/update MAGMOM tag in INCAR file."""
    os.makedirs(os.path.dirname(incar_path), exist_ok=True)
    # Preserve existing INCAR content except MAGMOM
    incar_lines = []
    if os.path.exists(incar_path):
        with open(incar_path, "r") as f:
            incar_lines = [line for line in f if not line.startswith("MAGMOM")]
    # Write updated INCAR with proper line break
    with open(incar_path, "w") as f:
        f.writelines(incar_lines)
        f.write(f"MAGMOM = {magmom_incar}\n")  # Fixed syntax error

def extract_elements(structure):
    """Return a list of unique elements from a pymatgen Structure."""
    return list({site.species_string for site in structure})

def load_structure(file_path: str) -> Structure:
    """Load crystal structure with error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    return Structure.from_file(file_path)

def assign_magnetic_moments(structure: Structure, element_props: dict, afm_species: set) -> list:
    """Assign magnetic moments with alternating signs for AFM species."""
    magmom = []
    afm_counters = {el: 0 for el in afm_species}
    
    for site in structure:
        species = site.species_string
        if species in element_props:
            base = element_props[species]["magmom"]
            if species in afm_species:
                moment = base if (afm_counters[species] % 2 == 0) else -base
                afm_counters[species] += 1
            else:
                moment = base
        else:
            moment = 0.0
        magmom.append(moment)
    return magmom

def assign_charges(structure: Structure, element_props: dict) -> list:
    """Assign formal charges to sites."""
    return [element_props.get(site.species_string, {}).get("charge", 0.0) 
            for site in structure]

# Modified main function with dual functional workflow
def main():
    try:
        # Configuration parameters
        API_KEY = "zBYiak7h6ies4ziNlAWqzXXHhc7rMBDB"
        MATERIAL_FORMULA = "La2NiO4"
        TARGET_SPACE_GROUP = "Cmc2_1"  # Example space group
        CONTCAR_PATH = "./POSCAR"
        MAIN_DIR = f"./{MATERIAL_FORMULA}_{TARGET_SPACE_GROUP.replace('/','_')}"
        SUPERCELL_SIZE = [[ 2,  0,  0],
                          [0,  2,  0],
                          [ 0, 0,  1]]

        # Functional-specific directories
        relax_dir =  f"{MATERIAL_FORMULA}_bulk/Supercell_relax"
        static_dir = f"{MATERIAL_FORMULA}_bulk/Supercell_static"
        os.makedirs(relax_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)

        # Load structure (same initial structure for both)
        try:
            structure = load_structure(CONTCAR_PATH)
            print("Loaded structure from CONTCAR")
        except FileNotFoundError:
            print("CONTCAR not found, fetching from Materials Project")
            with MPRester(API_KEY) as mpr:
                entries = mpr.get_entries_in_chemsys(
                    MATERIAL_FORMULA,
                    conventional_unit_cell=True,
                    inc_structure=True
                )
                filtered = [
                    e for e in entries
                    if SpacegroupAnalyzer(e.structure).get_space_group_symbol() == TARGET_SPACE_GROUP
                ]
                if not filtered:
                    raise ValueError(f"No entries found with space group {TARGET_SPACE_GROUP}")
                structure = filtered[0].structure
        structure.get_primitive_structure(use_site_props=True)
        
        elements = extract_elements(structure)

        
        # Now elements is defined and can be used below
        sga = SpacegroupAnalyzer(structure)
        print(f"Final space group: {sga.get_space_group_symbol()}")

        ELEMENT_PROPERTIES = {
            'La': {'charge': +3, 'magmom': 2.0, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
            'Ni': {'charge': +2, 'magmom': 2.0, 'LDAUL': 2, 'LDAUU': 6.2, 'LDAUJ': 0.0},
            'O':  {'charge': -2, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
        }
        AFM_SPECIES = {"Nd", "Ni"}
        POTCAR_SETTINGS = {"La": "La", "Ni": "Ni_pv", "O": "O"}

        print("\n=== Generating Inputs ===")
        incar = {
            "LASPH": True,
            "ALGO": "All",
            "EDIFFG": -0.01,
            "KPAR": 6,
            "LDAU": True,
            "LDAUTYPE": 2,
            "LDAUL": {el: ELEMENT_PROPERTIES[el]["LDAUL"] for el in elements},
            "LDAUU": {el: ELEMENT_PROPERTIES[el]["LDAUU"] for el in elements},
            "LDAUJ": {el: ELEMENT_PROPERTIES[el]["LDAUJ"] for el in elements},
            "SYMPREC":1e-4
        }
        
        relax_incar = incar.copy()
        relax_incar.update({
            "EDIFF": 1e-5,
            "NSW": 200,
            
        })
        
        static_incar = incar.copy()
        static_incar.update({
            'ALGO': 'Normal',
            "EDIFF": 1e-7,
            "NSW": 0,
            "KGAMMA": True
        })
        r2scan_structure = structure.make_supercell(SUPERCELL_SIZE)
        

        relax_set = MPRelaxSet(
            r2scan_structure,
            user_incar_settings=relax_incar,
            user_potcar_functional="PBE_54",
            user_kpoints_settings={"grid_density": 1000},
            user_potcar_settings=POTCAR_SETTINGS,
            international_monoclinic=False,
        )
        relax_set.write_input(relax_dir)

        static_set = MPStaticSet(
            r2scan_structure,
            user_incar_settings=static_incar,
            user_potcar_functional="PBE_54",
            user_kpoints_settings={"grid_density": 1000},
            user_potcar_settings=POTCAR_SETTINGS,
            international_monoclinic=False,
        )
        static_set.write_input(static_dir)
        
        # Add magnetic moments for R2SCAN
        r2scan_magmom = assign_magnetic_moments(r2scan_structure, ELEMENT_PROPERTIES, AFM_SPECIES)
        add_magmom_to_incar(f"{relax_dir}/INCAR", format_magmom_for_incar(r2scan_magmom)),
        add_magmom_to_incar(f"{static_dir}/INCAR", format_magmom_for_incar(r2scan_magmom))
        r2scan_structure.add_oxidation_state_by_element({el: ELEMENT_PROPERTIES[el]['charge'] for el in elements})
        r2scan_structure.add_site_property("magmom", r2scan_magmom)
        print(f"Inputs written to {relax_dir}")

        # Write visualization files for final structure
        cif_path = f"{relax_dir}/{MATERIAL_FORMULA}_{TARGET_SPACE_GROUP.replace('/','_')}.cif"
        CifWriter(r2scan_structure,write_site_properties=True).write_file(cif_path)
        print(f"Final CIF written to {cif_path}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()