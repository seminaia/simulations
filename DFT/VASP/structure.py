import os
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPScanRelaxSet, MPRelaxSet
from pymatgen.io.vasp.inputs import Kpoints
from ase.visualize import view
from ase.io import write
import matplotlib.pyplot as plt
from collections import OrderedDict

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
    
    # Write updated INCAR
    with open(incar_path, "w") as f:
        f.writelines(incar_lines)
        f.write(f"MAGMOM = {magmom_incar}\n")


    
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

def assign_magnetic_moments(structure: Structure, element_properties: dict, afm_species: set) -> list:
    """Assign magnetic moments with alternating signs for AFM species."""
    magmom_list = []
    afm_counter = {element: 0 for element in afm_species}
    
    for site in structure:
        species = site.species_string
        if species in element_properties:
            base_moment = element_properties[species]["magmom"]
            if species in afm_species:
                moment = base_moment if (afm_counter[species] % 2 == 0) else -base_moment
                afm_counter[species] += 1
            else:
                moment = base_moment
        else:
            moment = 0.0
        magmom_list.append(moment)
    return magmom_list

def assign_charges(structure: Structure, element_properties: dict) -> list:
    """Assign formal charges to structure sites."""
    return [element_properties[site.species_string]["charge"] 
            if site.species_string in element_properties else 0.0 
            for site in structure]

def main():
    try:
        # Configuration
        api_key = "n8aGzYQW2CXYIsrNvNzUeMsY37hJBtV0"
        material_formula = "La2NiO4"
        target_space_group = "I4/mmm"  
        crystal_system = "tetragonal"  
        contcar_path = f"./POSCAR"
        main_dir = f"./{material_formula}_{target_space_group.replace('/','_')}"
        bulk_dir = f"{main_dir}/{material_formula}_bulk/PBE_relax"
        incar_path =f"{bulk_dir}/INCAR"
        os.makedirs(bulk_dir, exist_ok=True)
        os.makedirs(main_dir, exist_ok=True)

        # Attempt to load local CONTCAR first
        try:
            structure = load_structure(contcar_path)
            print("Loaded structure from CONTCAR")
        except FileNotFoundError:
            print("CONTCAR not found, using Materials Project structure")
            # Initialize Materials Project connection
            mpr = MPRester(api_key)
            entries = mpr.get_entries(
                material_formula)
            
            print(f"All Entries:{[(SpacegroupAnalyzer(e.structure).get_crystal_system(), e.structure.get_space_group_info()) for e in entries]}")
            filtered_entries = [
                e for e in entries
                if SpacegroupAnalyzer(e.structure).get_crystal_system() == crystal_system  and SpacegroupAnalyzer(e.structure).get_space_group_symbol() == target_space_group
            ]
            if not filtered_entries:
                raise ValueError(f"No entries found with space group {target_space_group}")
            structure = filtered_entries[0].structure  # Use get_structure()
        structure.make_supercell((2,2,2))  # Adjust supercell size as needed
        elements = extract_elements(structure)

        # ------------------- Define Element Properties & Magnetism ---------------
        element_properties = {
            'La': {'charge': +3, 'magmom': 0.6, 'LDAUL':  -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
            'Ni': {'charge': +2, 'magmom': 2.0, 'LDAUL':  2, 'LDAUU': 6.2, 'LDAUJ': 0.0},
            'O':  {'charge': -2, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
        }
        AFM_species = {"Pr","Ni"}  # Corrected AFM species
        
        user_potcar_settings = {
            "La": "La",
            "Ni": "Ni_pv",
            "O": "O"
        }


        # ------------------- INCAR Settings -------------------
        base_incar_settings = {
            "SYSTEM": f"{material_formula}",
            "ALGO": "Normal",
            "EDIFF": 1e-6,
            "EDIFFG": -0.001,
            "IBRION": 2,
            "ICHARG": 2,
            "ISMEAR": 0,
            "ISPIN": 2,
            "SIGMA": 0.01,
            "NELM": 200,
            "NELMIN": 4,
            "KPAR": 6,
            "NCORE": 12,
            "LCHARG": False,
            "LMAXMIX": 6,
            "LORBIT": 11,
            "LREAL": "Auto",
            "LWAVE": False,
            "NSW": 100,
            "PREC": "Accurate",
            "SYMPREC": "1e-5",
        }

        r2scan = {
        "KSPACING":0.2,
        "LAECHG": True,
        "LASPH": True,
        "LMIXTAU" : True,
        "LVTOT" : True,
        }
        
        u = {
        "LDAU": True,
        "LDAUPRINT": 1,
        "LDAUTYPE": 2,
        "LDAUL": {el: element_properties[el]["LDAUL"] for el in elements},
        "LDAUU": {el: element_properties[el]["LDAUU"] for el in elements},
        "LDAUJ": {el: element_properties[el]["LDAUJ"] for el in elements}
        }   


        r2scan_settings = {**base_incar_settings, **r2scan}
        r2scan_u_settings = {**base_incar_settings, **r2scan, **u}
        pbe_u_settings = {**base_incar_settings, **u}
        # Structure analysis
        analyzer = SpacegroupAnalyzer(structure)
        space_group = analyzer.get_space_group_symbol()
        crystal_system = analyzer.get_crystal_system()
        print(f"\nMaterial: {material_formula}")
        print(f"Space group: {space_group}")
        print(f"Crystal system: {crystal_system}")


        
        # Generate VASP inputs
#        vasp_input = MPScanRelaxSet(
#            structure,
#            user_incar_settings=r2scan_u_settings,
#            auto_kspacing=True,
#            user_potcar_settings=user_potcar_settings,
#            vdw = 'rvv10',
#        )
        vasp_input =  MPRelaxSet(structure,
            user_incar_settings=pbe_u_settings,
            user_potcar_functional="PBE_54",
            user_potcar_settings=user_potcar_settings,
            user_kpoints_settings={"grid_density": 1000},
        )
        vasp_input.write_input(bulk_dir)
        print(f"VASP inputs generated in '{bulk_dir}' directory")
        magmom_list = assign_magnetic_moments(structure, element_properties, AFM_species)
        charge_list = assign_charges(structure, element_properties)
        
        # Add properties to structure
        structure.add_site_property("magmom", magmom_list)
        structure.add_site_property("charge", charge_list)
        total_charge = sum(charge_list)
        total_magmom = sum(magmom_list)

        print(f"Total charge of the structure: {total_charge}")
        print(f"Total Magnetic Moment of the structure: {total_magmom}")
        
        magmom_incar = format_magmom_for_incar(magmom_list)
        add_magmom_to_incar(incar_path, magmom_incar)

        # Write CIF file
        cif_filename = f"{material_formula}_{space_group.replace('/', '_')}.cif"
        cif_filepath = os.path.join(bulk_dir, cif_filename)
        CifWriter(structure, write_site_properties=True).write_file(cif_filepath)
        print(f"Structure written to {cif_filepath}")


        # Visualization
        atoms = AseAtomsAdaptor.get_atoms(structure)
        vis_image_path = os.path.join(bulk_dir, f"{material_formula}_{space_group.replace('/','_')}.png")
        write(vis_image_path, atoms, rotation='-10x,20y', scale=150)
        print(f"Visualization image saved to {vis_image_path}")

    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        raise

if __name__ == "__main__":
    main()