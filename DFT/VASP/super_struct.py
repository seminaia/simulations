from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
import numpy as np

def create_proper_supercell(input_file, scaling_matrix, output_file="supercell.cif"):
    """
    Create a supercell while maintaining correct symmetry relationships
    
    Args:
        input_file: Input structure file (CIF, POSCAR, etc.)
        scaling_matrix: Diagonal scaling factors (e.g., [2,2,2])
        output_file: Output CIF file name
    """
    # Load structure and create supercell
    scaling = np.diag(scaling_matrix)
    struct = Structure.from_file(input_file).make_supercell(scaling_matrix)

    # Write CIF with automatic symmetry detection
    writer = CifWriter(struct, write_site_properties=True)
    writer.write_file(output_file)
    poscar = struct.to(filename="POSCAR_supercell",fmt="poscar")
    print(f"Created supercell with {len(struct)} atoms")

# Example usage
create_proper_supercell("CONTCAR", [4,4,4], "Al_supercell.cif")
