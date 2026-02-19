import os
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MPScanStaticSet
from pymatgen.io.cif import CifWriter
from pymatgen.core import Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

def load_structure(file_path: str) -> Structure:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    return Structure.from_file(file_path)

def assign_magnetic_moments(structure: Structure, element_properties: dict) -> list:
    return [element_properties.get(site.species_string, {}).get("magmom", 0.0)
            for site in structure]

def assign_charges(structure: Structure, element_properties: dict) -> list:
    return [element_properties.get(site.species_string, {}).get("charge", 0.0)
            for site in structure]

def main():
    # --- Configuration ---
    api_key          = "n8aGzYQW2CXYIsrNvNzUeMsY37hJBtV0"
    material_formula = "Nb"
    target_sg        = "Im-3m"
    contcar_path     = "CONTCAR"
    main_dir         = f"./PhDOS_{material_formula}"
    os.makedirs(main_dir, exist_ok=True)

    # --- Load or fetch structure ---
    try:
        structure = load_structure(contcar_path)
        print("Loaded structure from CONTCAR")
    except FileNotFoundError:
        from pymatgen.ext.matproj import MPRester
        mpr = MPRester(api_key)
        entries = mpr.get_entries(material_formula,
                                  additional_criteria={"energy_above_hull": 0.0},
                                  conventional_unit_cell=True)
        structure = next(e.structure for e in entries
                         if SpacegroupAnalyzer(e.structure).get_space_group_symbol() == target_sg)
        print("Fetched structure from MP")

    analyzer = SpacegroupAnalyzer(structure)
    print(f"Material: {material_formula}, SG: {analyzer.get_space_group_symbol()}")

    # --- Assign fixed properties ---
    element_props = {"Nb": {"charge": 0.0, "magmom": 0.0}}
    magmoms = assign_magnetic_moments(structure, element_props)
    charges = assign_charges(structure, element_props)
    structure.add_site_property("magmom", magmoms)
    structure.add_site_property("charge", charges)

    # --- Phonopy setup ---
    phonopy_atoms = PhonopyAtoms(
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
        numbers=[site.specie.Z for site in structure]
    )
    scaling = [[2,0,0],[0,2,0],[0,0,2]]
    phonon = Phonopy(
        phonopy_atoms,
        supercell_matrix=scaling,
        primitive_matrix='auto',
        is_symmetry=True   # keep symmetry reduction
    )

    # Generate ± displacements for the irreducible set
    phonon.generate_displacements(distance=0.01, is_plusminus=True)
    scells = phonon.supercells_with_displacements
    print(f"Writing out {len(scells)} displacement cases (± each irreducible).")

    # --- Write VASP inputs for each displaced supercell ---
    base_incar = {
        "EDIFF": 1e-7, "EDIFFG": -0.001, "IBRION": 6, "ICHARG": 2,
        "ISMEAR": 1, "ISPIN": 1, "ISYM": 0, "KPAR": 6,
        "LCHARG": True, "LCALCEPS": True, "LEPSILON": False,
        "LMAXMIX": 4, "LORBIT": 11, "LPEAD": True,
        "LREAL": "Auto", "LWAVE": False, "NSW": 1, "PREC": "Accurate",
        "POTIM": 0.015, "SIGMA": 0.02, "KSPACING": 0.15,
        "LAECHG": True, "LASPH": True, "LMIXTAU": True, "LVTOT": True
    }
    user_potcar = {"Nb": "Nb_pv"}

    for idx, sc in enumerate(scells):
        disp_struct = Structure(
            lattice=sc.cell,
            species=[Element.from_Z(z).symbol for z in sc.numbers],
            coords=sc.scaled_positions,
            coords_are_cartesian=False
        )
        disp_dir = os.path.join(main_dir, f"disp_{idx:02d}")
        os.makedirs(disp_dir, exist_ok=True)
        disp_struct.to(filename=os.path.join(disp_dir, "POSCAR"))
        vasp_set = MPScanStaticSet(
            disp_struct,
            user_incar_settings=base_incar,
            user_potcar_settings=user_potcar,
            auto_metal_kpoints=True
        )
        vasp_set.write_input(disp_dir)
    CifWriter(structure, write_site_properties=True).write_file(f"{material_formula}.cif")
    write(f"{material_formula}.png", AseAtomsAdaptor.get_atoms(structure))

if __name__ == "__main__":
    main()
