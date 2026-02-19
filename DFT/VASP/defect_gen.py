from operator import ge
import os
from collections import OrderedDict
from turtle import write
from doped.generation import DefectsGenerator
from doped.vasp import DefectsSet, DefectRelaxSet
from pymatgen.io.vasp.sets import MPScanStaticSet, MPStaticSet
from pymatgen.core.structure import Structure
from monty.serialization import dumpfn
from pymatgen.io.cif import CifWriter
from shakenbreak.input import Distortions
from pymatgen.analysis.defects.core import DefectType
from sympy import use
import shutil
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure, write_supercells_with_displacements
from phonopy.structure.cells import get_supercell

import numpy as np
def load_structure(file_path: str) -> Structure:
    """Load crystal structure from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    return Structure.from_file(file_path)

def extract_elements(structure: Structure) -> list:
    """Extract unique elements from structure while preserving order."""
    return list(OrderedDict((site.species_string, None) for site in structure).keys())

def assign_magnetic_moments(structure: Structure, element_properties: dict, afm_species: set) -> list:
    """Assign magnetic moments with alternating signs for AFM species."""
    magmom_list = []
    afm_counter = {element: 0 for element in afm_species}
    
    for site in structure:
        element = site.species.elements[0].symbol
        if element in element_properties:
            base_moment = element_properties[element]["magmom"]
            if element in afm_species:
                moment = base_moment if (afm_counter[element] % 2 == 0) else -base_moment
                afm_counter[element] += 1
            else:
                moment = base_moment
        else:
            moment = 0.0
        magmom_list.append(moment)
    return magmom_list


def format_magmom_for_incar(magmom_list: list) -> str:
    """Convert magnetic moments list to INCAR format string."""
    return " ".join(map(str, magmom_list))

def add_magmom_to_incar(incar_dir: str, magmom_incar: str) -> None:
    """Add/update MAGMOM tag in INCAR file."""
    os.makedirs(incar_dir, exist_ok=True)
    incar_path = os.path.join(incar_dir, "INCAR")
    
    # Preserve existing INCAR content except MAGMOM
    incar_lines = []
    if os.path.exists(incar_path):
        with open(incar_path, "r") as f:
            incar_lines = [line for line in f if not line.startswith("MAGMOM")]
    
    # Write updated INCAR
    with open(incar_path, "w") as f:
        f.writelines(incar_lines)
        f.write(f"MAGMOM = {magmom_incar}\n")

CONFIG = {
    "material": "La2NiO4",
    "elements": ["La", "Ni", "O"],
    "defect_categories": ['Interstitial', 'Vacancy'],
    "defect_dir": ['Interstitial', 'Vacancy', 'Combined'],
    "e_above_hull": 0.09,
    "processes": 4,
}
def main():
    # ========== Configuration ==========
    structure_file = 'POSCAR'
    main_dir = f"./{CONFIG['material']}_defects3"
    output_file = f"{main_dir}/{CONFIG['material']}_dict.json"
    os.makedirs(main_dir, exist_ok=True)
    
    # ========== Initial Setup ==========
    primitive = load_structure(structure_file)
    elements = extract_elements(primitive)
    
    U_SCAN = [4, 5, 6, 7, 8]
    U_ROOT = os.path.join(main_dir, "U")

    U_TARGETS = {
        "bulk": None,
        "O_i_C2v_0": "Interstitial",
        "O_i_C2v_-1": "Interstitial",
        "O_i_C2v_-2": "Interstitial",
    }

    VIB_TAGS = {
        "IBRION": 5,
        "NSW": 1,
        "ISIF": 3,
        "POTIM": 0.015,
        "EDIFF": 1e-8,
        "LCHARG": False,
        "LEPSILON": True,
        "LOPTICS": True,
        "LWAVE": False,
        "ICHARG": 2,
    }

    element_properties = {
        'La': {'charge': +3, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
        'Ni': {'charge': +2, 'magmom': 2.0, 'LDAUL': 2, 'LDAUU': 7.0, 'LDAUJ': 0.0},
        'O': {'charge': -2, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
        'Cu': {'charge': +2, 'magmom': 2},
        'Sr': {'charge': +2, 'magmom': 2},
        'Mn': {'charge': +3, 'magmom': 2},
        'Co': {'charge': +3, 'magmom': 2}
    }
    
    afm_species = {"Pb", 'Cu', "Ni", "Nd", 'Pr'}
    
    user_potcar_settings = {
        "La": "La",
        "Ni": "Ni_pv",
        "O": "O"
    }
    
    # ========== VASP Settings ==========
    user_incar_settings = {
        "SYSTEM": f"{CONFIG['material']}",
        "ALGO": "All",
        "EDIFF": 1e-7,
        "EDIFFG": -0.01,
        "ENCUT": 520,
        "IBRION": 1,
        "ICHARG": 2,
        "ISIF": 2,
        "ISMEAR": 0,
        "KPAR" : 5,
        "NCORE": 2,
        "NELM": 300,
        "LCHARG": True,
        "LHFCALC": False,
        "LMAXMIX": 6,
        "LORBIT": 11,
        "LREAL": False,
        "LWAVE": False,
        "NSW": 200,
        "PREC": "Accurate",
        "SIGMA": 0.001
    }
    
    u_settings = {
        "LDAU": True,
        "LDAUPRINT": 1,
        "LDAUTYPE": 2,
        "LDAUL": {el: element_properties[el]["LDAUL"] for el in elements if el in element_properties},
        "LDAUU": {el: element_properties[el]["LDAUU"] for el in elements if el in element_properties},
        "LDAUJ": {el: element_properties[el]["LDAUJ"] for el in elements if el in element_properties},}
    
    pbe = {
        "GGA": 'Pe',
        "LASPH": True,
        "SYMPREC": 1e-5,
    }
    pbe_settings = {**user_incar_settings, **pbe}
    pbe_u_settings = {**pbe_settings, **u_settings}
    vib_settings = {**pbe_settings, **VIB_TAGS}
    
    r2scan = {
        "ENAUG": 1040,
        "KGAMMA": True,
        "LAECHG": True,
        "LASPH": True,
        "LDAU": False,
        "LMIXTAU": True,
        "LVTOT": True,
        'METAGGA': 'R2SCAN',
        "SYMPREC": 1e-4,
    }
    r2scan_settings = {**user_incar_settings, **r2scan}
    r2scan_u_settings = {**r2scan_settings, **u_settings}
    
    rvv10 = {
        'LUSE_VDW': True,
        'BPARAM': 11.95,
        'CPARAM': 0.0093,
    }
    r2scan_vvv10_settings = {**r2scan_settings, **rvv10}
    pbe_vvv10_settings = {**pbe_settings, **rvv10}
    r2scan_u_rvv10_settings = {**r2scan_u_settings, **rvv10}
    pbe_u_rvv10_settings = {**pbe_u_settings, **rvv10}
    
    # ========== Defect Generation ==========
    extrinsic = {"P": ['Ca', 'Sr'], "Pb": ['Cu', 'Mn']}
    substitution_elements = ['La', 'Ni', 'Ca', 'Sr', 'Co', 'Mn']
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    
    supercell = primitive * supercell_matrix
    defect_gen = DefectsGenerator(
        supercell,
        extrinsic=extrinsic,
        interstitial_gen_kwargs=True,
        interstitial_elements=['O'],
        vacancy_gen_kwargs=True,
        vacancy_elements=['La', 'Ni', 'O'],
        substitution_gen_kwargs=False,
        substitution_elements=[],
        generate_supercell=False,
        supercell_gen_kwargs={'force_diagonal': True},
    )
    defect_gen.to_json(filename=f"{main_dir.split('/')[-1]}_defect_generator.json")
    print(f"Bulk supercell matrix: {defect_gen.supercell_matrix}")
    print(f"Bulk Supercell matrix multiplied: {supercell_matrix}")
    
    dumpfn(defect_gen.defect_entries, output_file)
    
    # Defect type mapping
    defect_type_map = {
        DefectType.Vacancy.value: "Vacancy",
        DefectType.Interstitial.value: "Interstitial",
        DefectType.Substitution.value: "Substitution"  # In case substitutions are enabled later
    }
    
    # Create bulk directory once (shared)
    bulk_path = os.path.join(main_dir, "bulk", "vasp_gam")
    os.makedirs(bulk_path, exist_ok=True)
    
    # ========== Generate Bulk Inputs (once) ==========
    bulk_supercell = supercell  # Use the same supercell as base for bulk
    bulk_magmom = assign_magnetic_moments(bulk_supercell, element_properties, afm_species)
    bulk_magmom_incar = format_magmom_for_incar(bulk_magmom)
    bulk_supercell.add_site_property("magmom", bulk_magmom)
    bulk_supercell.add_oxidation_state_by_element({el: element_properties[el]['charge'] for el in elements if el in element_properties})
    
    cif_path = os.path.join(bulk_path, f"{CONFIG['material']}_bulk.cif")
    CifWriter(bulk_supercell, write_site_properties=True).write_file(cif_path)

    vasp_input = MPStaticSet(
        bulk_supercell,
        user_incar_settings=pbe_u_settings,
        user_potcar_functional="PBE",
        user_potcar_settings=user_potcar_settings,
        international_monoclinic=False,
        force_gamma=True,
        auto_kspacing=False,
    )
    vasp_input.write_input(bulk_path)
    add_magmom_to_incar(bulk_path, bulk_magmom_incar)
    # ========== Process Defects ==========
    for defect_entry in defect_gen.defect_entries.values():
        defect_name = defect_entry.name
        defect_type_value = defect_entry.defect.defect_type.value
        defect_type = defect_type_map.get(defect_type_value, "Unknown")

        # Let ShakeNBreak write directly into the type folder
        type_dir = os.path.join(main_dir, defect_type)
        defect_dir = os.path.join(type_dir, defect_name)
        os.makedirs(defect_dir, exist_ok=True)

        DefectRelaxSet(defect_entry=defect_entry,
                       user_incar_settings=pbe_u_settings,
                       user_potcar_functional="PBE",
                       user_potcar_settings=user_potcar_settings,
                       ).write_gam(defect_dir)

        defect_supercell = defect_entry.defect_supercell
        defect_magmom = assign_magnetic_moments(defect_supercell, element_properties, afm_species)
        defect_magmom_incar = format_magmom_for_incar(defect_magmom)
        add_magmom_to_incar(os.path.join(defect_dir, "vasp_gam"), defect_magmom_incar)

        defect_supercell.add_site_property("magmom", defect_magmom)
        defect_elements = list(OrderedDict.fromkeys([site.species.elements[0].symbol for site in defect_supercell]))
        defect_supercell.add_oxidation_state_by_element(
            {el: element_properties[el]['charge'] for el in defect_elements if el in element_properties}
        )
        defect_cif_path = os.path.join(defect_dir, f"{defect_name}.cif")
        CifWriter(defect_supercell, write_site_properties=True).write_file(defect_cif_path)
        
        defect_dict, defect_metadata = Distortions.apply_distortions(defect_entry)
        
        
        os.makedirs(defect_dir, exist_ok=True)
        
        distortion_subdirs = [d for d in os.listdir(defect_dir) if os.path.isdir(os.path.join(defect_dir, d))]
        for subdir in distortion_subdirs:
            add_magmom_to_incar(os.path.join(defect_dir, subdir), defect_magmom_incar)
        print("VASP files generated successfully.")
    # ==========================================================
    # ========== POST-PROCESS: U TESTS =========================
    # ==========================================================

    os.makedirs(U_ROOT, exist_ok=True)

    def clone_and_modify(calc_src, calc_dst, incar_updates):
        os.makedirs(calc_dst, exist_ok=True)

        for f in ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]:
            shutil.copy(os.path.join(calc_src, f), calc_dst)

        incar_path = os.path.join(calc_dst, "INCAR")

        with open(incar_path, "r") as f:
            lines = f.readlines()

        updated = {k.upper(): False for k in incar_updates}
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue

            key = stripped.split("=", 1)[0].strip().upper()
            if key in updated:
                new_lines.append(f"{key} = {incar_updates[key]}\n")
                updated[key] = True
            else:
                new_lines.append(line)

        # append only missing keys
        for k, done in updated.items():
            if not done:
                new_lines.append(f"{k} = {incar_updates[k]}\n")

        with open(incar_path, "w") as f:
            f.writelines(new_lines)

    # ---------- BULK ----------    
    bulk_src = os.path.join(main_dir, "bulk", "vasp_gam")

    for U in U_SCAN:
        dst = os.path.join(U_ROOT, "bulk", f"U_{U}")
        clone_and_modify(
            bulk_src,
            dst,
            {"LDAUU": f"0 {U} 0"}  # Ni-only
        )

    # ---------- DEFECTS ----------
    for defect_name, defect_type in U_TARGETS.items():
        if defect_name == "bulk":
            continue

        src = os.path.join(main_dir, defect_type, defect_name, "vasp_gam")
        for U in U_SCAN:
            dst = os.path.join(U_ROOT, defect_name, f"U_{U:.1f}")
            clone_and_modify(
                src,
                dst,
                {"LDAUU": f"0 {U} 0"}
            )
    
    vib_dir = os.path.join(main_dir, "vib")
    os.makedirs(vib_dir, exist_ok=True)
    os.makedirs(os.path.join(vib_dir,f"fd"), exist_ok=True)
    unitcell, optional_struct_info = read_crystal_structure(structure_file,interface_mode="vasp")
    phonon = Phonopy(unitcell, supercell_matrix,calculator='vasp',set_factor_by_calculator=True)
    phonon.generate_displacements(distance=0.01)
    supercell_disps = phonon.supercells_with_displacements
    sposcar = get_supercell(unitcell, supercell_matrix)
    write_crystal_structure("SPOSCAR",
                        sposcar,
                        interface_mode="vasp",
                        optional_structure_info=optional_struct_info)
    

    for i, sc in enumerate(supercell_disps):
        os.makedirs(os.path.join(vib_dir,f"disp-{i}"), exist_ok=True)        
        write_crystal_structure(os.path.join(vib_dir,f"disp-{i}" ,f"POSCAR"),
                                sc,
                                interface_mode="vasp",
                                optional_structure_info=optional_struct_info)

           
    phonon.save(os.path.join(vib_dir, "phonopy_disp.yaml"))

if __name__ == "__main__":
    main()