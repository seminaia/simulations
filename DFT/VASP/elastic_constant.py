from pymatgen.core.structure import Structure
from pymatgen.analysis.elasticity import DeformedStructureSet
#from jarvis.analysis.elastic.tensor import ElasticTensor
from jarvis.io.vasp.outputs import Vasprun
from pymatgen.analysis.elasticity.elastic import ElasticTensor 
import os
import numpy as np
import json
MAIN_DIR = "elastic_inputs"
STRUCTURE_FILE = "La2NiO4_I4_mmm.cif"

def main():
    # Load original structure and generate deformation set (must match your calculations)
    pmg_struct = Structure.from_file(STRUCTURE_FILE)
    dss = DeformedStructureSet(pmg_struct, symmetry=True)
    
    # Collect stress-strain pairs
    stress_strain = []
    missing = []
    
    # Sort directories by deformation index
    deform_dirs = sorted([d for d in os.listdir(MAIN_DIR) if d.startswith("deformation_")],
                        key=lambda x: int(x.split("_")[-1]))
    
    for i, deform_dir in enumerate(deform_dirs):
        dir_path = os.path.join(MAIN_DIR, deform_dir)
        vasprun_file = os.path.join(dir_path, "vasprun.xml")
        
        if not os.path.exists(vasprun_file):
            missing.append(deform_dir)
            continue
            
        try:
            # Get stress using JARVIS parser
            vr = Vasprun(vasprun_file)
            stress = vr.all_stresses  # 3x3 stress tensor
            
            # Get corresponding strain from DeformedStructureSet
            strain = dss.deformations[i].green_lagrange_strain
            strain_voigt = strain.voigt
            
            stress_strain.append((strain_voigt, stress))
            print(stress_strain,strain_voigt,stress)
            
        except Exception as e:
            print(f"Error in {deform_dir}: {str(e)}")
            missing.append(deform_dir)
    
    if missing:
        print(f"Missing {len(missing)} deformations: {missing}")
    
    if not stress_strain:
        raise ValueError("No valid stress-strain data found!")
    
    # Convert to JARVIS format
    strains, stresses = zip(*stress_strain)
    
    # Calculate elastic tensor
    et = ElasticTensor.from_independent_strains(strains=strains, stresses=stresses)
    
    print("Compliance Tensor (GPa):\n", et.compliance_tensor)
    print("\nBulk Modulus (Voigt):", et.k_voigt, "GPa")
    print("Shear Modulus (Voigt):", et.g_voigt, "GPa")
    et.to_json("elastic_tensor.json")

if __name__ == "__main__":
    main()