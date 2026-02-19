import os
import json
import logging
import numpy as np
from pymatgen.io.vasp.inputs import Poscar, Incar, Kpoints
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.core.structure import Structure

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_serializable(data):
    """
    Recursively convert NumPy arrays to lists in a nested dictionary or list.
    Args:
        data: The input data (dict, list, or other).
    Returns:
        The data with all NumPy arrays converted to lists.
    """
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif hasattr(data, "tolist"):  # Check if the object has a .tolist() method (e.g., NumPy arrays)
        return data.tolist()
    else:
        return data

def magmom_incar(incar_path):
    """
    Parse the INCAR file and extract the MAGMOM tag.
    Args:
        incar_path (str): Path to the INCAR file.
    Returns:
        list: A list of magnetic moments for each atom.
    """
    if not os.path.exists(incar_path):
        raise FileNotFoundError(f"INCAR file not found: {incar_path}")
    
    incar = Incar.from_file(incar_path)
    magmom_value = incar.get("MAGMOM", None)
    if magmom_value is None:
        raise ValueError("MAGMOM tag not found in the INCAR file.")
    
    # Handle both string and list cases for MAGMOM
    magmom_list = []
    if isinstance(magmom_value, str):  # If MAGMOM is a string
        for part in magmom_value.split():
            if "*" in part:
                count, value = part.split("*")
                magmom_list.extend([float(value)] * int(count))
            else:
                magmom_list.append(float(part))
    elif isinstance(magmom_value, list):  # If MAGMOM is already a list
        magmom_list = magmom_value
    else:
        raise TypeError("Unexpected type for MAGMOM in INCAR file. Expected string or list.")
    
    return magmom_list

def parse_poscar(poscar_path, magmom_list):
    """
    Parse a POSCAR file and extract relevant information including k-grid, k-points, and k-path.
    Args:
        poscar_path (str): Path to the POSCAR file.
        magmom_list (list): List of magnetic moments for each atom.
    Returns:
        dict: Extracted data including elements, lattice_param, cell, atoms, kgrid, kpoints, and kpath.
    """
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"POSCAR file not found: {poscar_path}")
    
    # Read the POSCAR file using pymatgen
    poscar = Poscar.from_file(poscar_path)
    structure = poscar.structure
    
    # Ensure magmom_list has the correct length
    if len(magmom_list) != len(structure):
        raise ValueError(f"MAGMOM list length ({len(magmom_list)}) does not match number of atoms ({len(structure)}).")
    
    # Add magnetic moments as a site property
    structure.add_site_property("magmom", magmom_list)
    
    # Extract lattice parameter (scaling factor) from the second line of the POSCAR
    with open(poscar_path, "r") as f:
        lines = f.readlines()
    lattice_param = float(lines[1].strip())  # Read the second line as the lattice scaling factor
    
    # Extract cell vectors
    cell = [[round(val, 10) if abs(val) < 1e-8 else val for val in row] for row in structure.lattice.matrix.tolist()]
    
    # Extract elements in the order they appear in the POSCAR
    elements_order = []
    element_counts = poscar.natoms  # Number of atoms per species
    element_symbols = poscar.site_symbols  # Element symbols in order
    
       # Parse atomic positions
    atoms = []
    index = 0
    for symbol, count in zip(element_symbols, element_counts):
        for _ in range(count):
            site = structure[index]
            element = str(site.specie).strip("/")
            coords = site.frac_coords.tolist()  # Fractional coordinates
            
            # Format magnetic moment as 1-element array
            magmom = site.properties.get("magmom", 0.0)
            formatted_magmom = [float(magmom)]  # Wrap in list
            
            atoms.append([element, coords, formatted_magmom])
            index += 1
    
    # Generate k-grid using Kpoints.automatic_density
    kgrid_density = 1000  # Desired k-point density per reciprocal Angstrom
    kgrid_pbe = Kpoints.automatic_density(structure, kgrid_density).kpts[0]  
    kgrid_hse = Kpoints.automatic_density(structure, kgrid_density/2).kpts[0] 
    

    # Generate k-path using symmetry analysis
    kpath_analyzer = HighSymmKpath(structure)
    kpath_segments = kpath_analyzer.kpath["path"]
    raw_custom_kpoints = kpath_analyzer.kpath["kpoints"]

    # Symbol conversion mapping (LaTeX → simplified)
    SYMBOL_CONVERSION = {
        "\\Gamma": "G",
        "\\Sigma": "S",
        "\\Delta": "D",
        "\\Lambda": "L",
        "\\mathrm{X}": "X",
        "\\mathrm{Y}": "Y",
        "\\mathrm{Z}": "Z"
    }

    def convert_label(label):
        """Convert LaTeX k-point labels to simplified format"""
        # Replace known LaTeX symbols
        for tex, abbr in SYMBOL_CONVERSION.items():
            label = label.replace(tex, abbr)
        
        # Handle subscripted labels (e.g., Σ₁ → S1)
        if "_" in label:
            parts = label.split("_")
            base = parts[0]
            subscript = "".join(parts[1:])
            base = SYMBOL_CONVERSION.get(base, base[0])
            return f"{base}{subscript}"
        
        # Default to first character if unknown
        return SYMBOL_CONVERSION.get(label, label[0])

    # Process k-path segments
    converted_points = []
    for segment in kpath_segments:
        for point in segment:
            converted = convert_label(point)
            converted_points.append(converted)

    # Create final space-separated string
    kpath_str = " ".join(converted_points)

    # Process custom kpoints dictionary
    custom_kpoints = {}
    for label, coords in raw_custom_kpoints.items():
        custom_kpoints[convert_label(label)] = coords

    return {
        "elements": element_symbols,
        "lattice_param": lattice_param,
        "cell": cell,
        "atoms": atoms,
        "kgrid_hse": kgrid_hse,
        "kgrid_pbe": kgrid_pbe,
        "custom_kpoints": custom_kpoints,
        "kpath": kpath_str
    }

def generate_json(material_data):
    """
    Generate a JSON input file for the Bayesian optimization package.
    Args:
        material_data (dict): A dictionary containing material-specific data.
    Returns:
        dict: The generated JSON structure.
    """
    elements = material_data["elements"]
    lattice_param = material_data["lattice_param"]
    cell = material_data["cell"]
    atoms = material_data["atoms"]
    kgrid_hse = material_data.get("kgrid_hse", {})
    kgrid_pbe = material_data.get("kgrid_pbe", {})
    custom_kpoints = material_data.get("custom_kpoints", {})
    ldau_luj = material_data.get("ldau_luj", {})
    
    # Construct the JSON structure
    json_data = {
        "vasp_env": {
            "vasp_run_command": "srun --mpi=pmi2 vasp > vasp.out",
            "out_file_name": "vasp.out",
            "vasp_pp_path": "/nfs/home/5/zhongy/PBE_pmg",
            "dry_run": False,
            "dftu_only": False,
            "get_optimal_band": True
        },
        "bo": {
            "resume_checkpoint": False,
            "baseline": "hse",
            "which_u": [1, 1, 0],
            "br": [5, 5],
            "kappa": 5,
            "alpha_gap": 0.25,
            "alpha_band": 0.75,
            "alpha_mag": 0.0,
            "mag_axis": "all",
            "threshold": 0.0001,
            "urange": [0.0, 20.0],
            "elements": elements,
            "iteration": 50,
            "report_optimum_interval": 10,
            "threshold_opt_u": 0.0,
            "print_magmom": False
        },
        "structure_info": {
            "lattice_param": lattice_param,
            "cell": cell,
            "atoms": atoms,
            "kgrid_hse": kgrid_hse,
            "kgrid_pbe": kgrid_pbe,
            "num_kpts": 25,
            "kpath": material_data["kpath"],
            "custom_kpoints": custom_kpoints,
            "custom_POTCAR_path": None
        },
        "general_flags": {
            "encut": 680,
            "sigma": 0.05,
            "ediff": 1e-06,
            "prec": "N",
            "algo": "D",
            "saxis": [0, 0, 1],
            "metagga": True,
            "lasph": True,
            "nbands": 320,
            "kpar": 8,
            "ncore": 16,
            "bmix": 1,
            "amin": 0.01,
            "lorbit": 11,
            "lmaxmix": 6,
            "nelm": 200
        },
        "scf": {
            "icharg": 2,
            "istart": 0,
            "xc": "r2scan"
        },
        "band": {
            "icharg": 11,
            "lcharg": False,
            "lwave": False
        },
        "r2scan": {
            "metagga": True,
            "lasph": True,
            "xc": "r2scan",
            "ldau": True,
            "ldau_luj": ldau_luj
        },
        "hse": {
            "icharg": 2,
            "xc": "hse06",
            "aexx": 0.25,
            "time": 0.4,
            "ldiag": True,
            "lhfcalc": True,
            "precfock": "Fast"
        }
    }
    return json_data

if __name__ == "__main__":
    try:
        # Paths to the POSCAR and INCAR files
        poscar_path = "CONTCAR"
        incar_path = "Nd2NiO4_bulk/Bulk_HSE/INCAR"
        # Parse the INCAR file to get magnetic moments
        magmom_list = magmom_incar(incar_path)
        logging.info("INCAR file parsed successfully.")
        
        # Parse the POSCAR file
        poscar_data = parse_poscar(poscar_path, magmom_list)
        logging.info("POSCAR file parsed successfully.")
        # Update material_data with POSCAR information
        material_data = {
            "ldau_luj": {
                "Nd": {"L": -1, "U": 0.0, "J": 0.0},
                "Ni": {"L": 2, "U": 6.0, "J": 0.0},
                "O": {"L": -1, "U": 0.0, "J": 0.0}
            }
        }
        material_data.update(poscar_data)
        
        # Generate the JSON
        json_data = generate_json(material_data)
        logging.info("JSON data generated successfully.")
        
        # Convert NumPy arrays to lists for JSON serialization
        json_data = convert_to_serializable(json_data)
        
        # Save to a file
        with open("input.json", "w") as f:
            json.dump(json_data, f, indent=4)
        logging.info("JSON file generated successfully!")
    
    except FileNotFoundError as fnf_error:
        logging.error(fnf_error)
    except ValueError as ve:
        logging.error(ve)
    except TypeError as te:
        logging.error(te)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")