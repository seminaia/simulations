import os
from monty.serialization import loadfn
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.sets import MPScanStaticSet, MPScanRelaxSet,MPRelaxSet,MPStaticSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from mp_api.client import MPRester

from collections import OrderedDict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_elements(structure):
    """Extract unique elements from a structure."""
    unique_elements = OrderedDict()
    for site in structure:
        unique_elements[site.species_string] = None
    return list(unique_elements.keys())

def assign_charges(structure: Structure, element_properties: dict) -> list:
    """Assign formal charges to structure sites."""
    return [element_properties[site.species_string]["charge"] 
            if site.species_string in element_properties else 0.0 
            for site in structure]
    
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
def fetch_structure_from_materials_project(material_id: str, api_key: str, target_space_group: str = None, formula: str = None) -> Structure:
    """
    Fetch structure from Materials Project using summary.search method.
    
    Args:
        material_id (str): The Materials Project ID of the material.
        api_key (str): Your MP API key.
        target_space_group (str, optional): Filter by space group.
        formula (str, optional): Filter by chemical formula.
        
    Returns:
        Structure: The fetched structure.
    """
    try:
        with MPRester(api_key) as mpr:
            # Build search query
            search_params = {"material_ids": [material_id]}
            
            if target_space_group:
                search_params["spacegroup_symbol"] = target_space_group
            if formula:
                search_params["formula"] = formula
            
            # Fetch document - summary.search returns a list of documents
            docs = mpr.summary.search(**search_params)
            
            if not docs:
                raise ValueError(f"No entries found for material ID: {material_id}")
            
            # Get first document
            doc = docs[0]
            
            # Extract structure from document
            # The structure attribute might be a dict or Structure object
            structure_data = doc.structure
            
            # Convert to Structure object if needed
            if isinstance(structure_data, dict):
                structure = Structure.from_dict(structure_data)
            elif hasattr(structure_data, 'lattice'):
                structure = structure_data
            else:
                # Try to handle as list or other format
                logging.warning(f"Unexpected structure data type: {type(structure_data)}")
                structure = Structure.from_dict(structure_data)
            
            logging.info(f"Successfully fetched structure for {material_id}")
            logging.info(f"Formula: {structure.formula}")
            logging.info(f"Number of sites: {len(structure)}")
            
            return structure
            
    except Exception as e:
        logging.error(f"Failed to fetch structure for {material_id}: {e}")
        raise
    
def generate_vasp_input_files(structure: Structure, material_id: str, relax_dir: str, static_dir: str,supercell_matrix=None):
    """Generate VASP input files for relaxation, static, and HSE calculations."""
    element_properties = {
        'La': {'charge': +3, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
        'Ni': {'charge': +2, 'magmom': 2.0, 'LDAUL': 2, 'LDAUU': 4.0, 'LDAUJ': 0.0},
        'O': {'charge': -2, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
    }
    AFM_species = {'Nd', 'Pb'}
      
    POTCAR_OVERRIDES = {
        "La": "La",
        "Ni": "Ni_pv",
        "O": "O"
    }
    
    # Base INCAR settings
    base_incar_settings = {
        "ALGO": "Normal",
        "ISIF" : 3,
        "ISMEAR": 0,
        "KPAR" : 6,
        "LASPH": True,
        "LCHARG": True,
        "LMAXMIX": 6,
        "LORBIT": 11,
        "LVHAR": False,
        "LWAVE": False,
        "PRECFOCK": "Fast",
        "NUPDOWN": 0,
        "SIGMA": 0.05,
    }
    
    # Relaxation-specific INCAR settings
    relax_incar_settings = base_incar_settings.copy()
    relax_incar_settings.update({
        "IBRION": 2,
        "EDIFF": 1E-7,
        "LDAU": False,
        #"LDAUTYPE": 2,
        #"LDAUU": {el: element_properties[el]["LDAUU"] for el in extract_elements(structure)},
        #"LDAUL": {el: element_properties[el]["LDAUL"] for el in extract_elements(structure)},
        #"LDAUJ": {el: element_properties[el]["LDAUJ"] for el in extract_elements(structure)},
    })
    
    # Static-specific INCAR settings
    static_incar_settings = base_incar_settings.copy()
    static_incar_settings.update({
        "EDIFF": 1E-8,
        "IBRION": -1,
        "ICHARG": 2,
        "ISIF": 2,
        "ISMEAR" : -5,
        "LDAU": True,
        "LDAUTYPE": 2,
        "LDAUU": {el: element_properties[el]["LDAUU"] for el in extract_elements(structure)},
        "LDAUL": {el: element_properties[el]["LDAUL"] for el in extract_elements(structure)},
        "LDAUJ": {el: element_properties[el]["LDAUJ"] for el in extract_elements(structure)},
        "LREAL": False,
        "NEDOS": 3000,
    })
    
    # Assign magnetic moments to the structure
    bulk_magmom = assign_magnetic_moments(structure, element_properties, AFM_species)
    bulk_charge = assign_charges(structure, element_properties)
    structure.add_site_property("magmom", bulk_magmom)
    structure.add_site_property("charge",bulk_charge)
    if supercell_matrix is not None:
        structure.make_supercell(supercell_matrix)  # Use the provided supercell matrix
    
    # Initialize sets and write input files
    relax_set = MPRelaxSet(structure, 
                        user_incar_settings=relax_incar_settings,
                        user_potcar_settings=POTCAR_OVERRIDES,
                        user_potcar_functional='PBE',
                        user_kpoints_settings={'reciprocal_density': 100},
                        international_monoclinic=False)
    relax_set.write_input(relax_dir)
    static_set = MPStaticSet(structure,
                             user_incar_settings=static_incar_settings,
                             user_potcar_settings=POTCAR_OVERRIDES,
                             user_potcar_functional='PBE',
                             user_kpoints_settings={'reciprocal_density': 300})
    static_set.write_input(static_dir)

    logging.info(f"VASP input files written to {relax_dir}")
    logging.info(f"VASP input files written to {static_dir}")

if __name__ == "__main__":
    material_id = 'mp-21326'
    material_name = 'La2NiO4'
    main_dir = f"{material_name}_bulk"
    target_space_group = "I4/mmm"  
    prim_relax_dir = "prim_relax"
    prim_static_dir = "prim_static"
    conv_relax_dir = "conv_relax"
    conv_static_dir = "conv_static"
    api_key ="n8aGzYQW2CXYIsrNvNzUeMsY37hJBtV0"
    # Fetch structure and generate VASP input files
    try:
                # Load structure with fallback
        prim_structure = fetch_structure_from_materials_project(material_id,
                                                           api_key,
                                                           target_space_group=target_space_group,
                                                           formula=material_name)
        conv_structure = prim_structure.to_conventional()
        prim_sg = SpacegroupAnalyzer(prim_structure).get_space_group_symbol()
        conv_sg = SpacegroupAnalyzer(conv_structure).get_space_group_symbol()
        
        print(f"Fetched structure for {material_id} successfully.")
        print(f"Primitive Cell: {prim_structure}")
        print(f"Primitive Cell space group: {prim_sg}")
        print(f"Conventional Cell: {conv_structure}")
        print(f"Conventional Cell space group: {conv_sg}")
        Poscar(prim_structure).write_file("prim_struc_POSCAR")
        Poscar(conv_structure).write_file("conv_struc_POSCAR")
        
        # Create subdirectories inside main_dir
        prim_relax_path = os.path.join(main_dir, prim_relax_dir)
        prim_static_path = os.path.join(main_dir, prim_static_dir)
        conv_relax_path = os.path.join(main_dir, conv_relax_dir)
        conv_static_path = os.path.join(main_dir, conv_static_dir)
        os.makedirs(prim_relax_path, exist_ok=True)
        os.makedirs(prim_static_path, exist_ok=True)
        os.makedirs(conv_relax_path, exist_ok=True)
        os.makedirs(conv_static_path, exist_ok=True)
        generate_vasp_input_files(prim_structure, material_id, prim_relax_path, prim_static_path, supercell_matrix=[1,1,2])
        generate_vasp_input_files(conv_structure, material_id, conv_relax_path, conv_static_path)
    except Exception as e:
        logging.error(f"An error occurred: {e}")