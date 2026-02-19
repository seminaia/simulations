import numpy as np
import os

def read_poscar(filename):
    """
    Reads a POSCAR file and returns the lattice vectors and atomic positions.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    lattice = np.array([
        [float(x) for x in lines[2].split()],
        [float(x) for x in lines[3].split()],
        [float(x) for x in lines[4].split()]]) * scale
    atom_types = lines[5].split()
    atom_counts = [int(x) for x in lines[6].split()]
    total_atoms = sum(atom_counts)
    coord_type = lines[7].strip()[0]
    positions = []
    for i in range(8, 8 + total_atoms):
        positions.append([float(x) for x in lines[i].split()[:3]])
    return lattice, np.array(positions), coord_type, atom_types, atom_counts

def write_poscar(filename, lattice, positions, atom_types, atom_counts, coord_type='Cartesian'):
    """
    Writes a POSCAR file with the given lattice vectors and atomic positions.
    """
    with open(filename, 'w') as f:
        f.write("Symmetry Broken Al SCC\n")
        f.write("1.0\n")
        for vec in lattice:
            f.write(f" {vec[0]:20.16f} {vec[1]:20.16f} {vec[2]:20.16f}\n")
        f.write(" ".join(atom_types) + "\n")
        f.write(" ".join(map(str, atom_counts)) + "\n")
        f.write(coord_type + "\n")
        for pos in positions:
            f.write(f" {pos[0]:20.16f} {pos[1]:20.16f} {pos[2]:20.16f}\n")

def apply_symmetry_breaking(positions, lattice, d_int=0.5):
    """
    Apply symmetry-breaking distortion to every other (001) layer with CYCLIC PATTERN:
    Layer 0: (+d_int, 0, 0)
    Layer 2: (0, +d_int, 0)
    Layer 4: (+d_int, +d_int, 0)
    Layer 6: (+d_int, 0, 0) [repeats pattern]
    """
    positions_cart = positions.copy()
    inv_lattice = np.linalg.inv(lattice)
    frac_positions = positions_cart @ inv_lattice.T
    
    # Group atoms by fractional z-coordinate
    z_coords = np.unique(np.round(frac_positions[:, 2], decimals=4))
    layers = []
    for z in z_coords:
        mask = np.abs(frac_positions[:, 2] - z) < 1e-4
        layers.append(np.where(mask)[0])
    
    # Apply patterned displacement to every other layer
    for j, layer in enumerate(layers):
        if j % 2 == 0:  # Apply to even-indexed layers (0, 2, 4, ...)
            pattern_index = (j // 2) % 3
            if pattern_index == 0:
                disp = np.array([d_int, 0, 0])
            elif pattern_index == 1:
                disp = np.array([0, d_int, 0])
            else:  # pattern_index == 2
                disp = np.array([d_int, d_int, 0])
                
            for idx in layer:
                positions_cart[idx] += disp
                
    return positions_cart

if __name__ == "__main__":
    ncc_file = "Al_NCC/Bulk_static/POSCAR"
    lattice, positions, coord_type, atom_types, atom_counts = read_poscar(ncc_file)
    
    # Convert to Cartesian if needed
    if coord_type.lower()[0] == 'd':  # Direct/Fractional coordinates
        positions_cart = positions @ lattice
    else:
        positions_cart = positions.copy()
    
    # Apply symmetry breaking (using paper's cyclic pattern)
    scc_positions_cart = apply_symmetry_breaking(positions_cart, lattice, d_int=0.5)
    
    # Convert back to original coordinate system
    if coord_type.lower()[0] == 'd':
        inv_lattice = np.linalg.inv(lattice)
        scc_positions = scc_positions_cart @ inv_lattice.T
    else:
        scc_positions = scc_positions_cart
    
    write_poscar("POSCAR_SCC", lattice, scc_positions, atom_types, atom_counts, coord_type=coord_type)
    print("Symmetry breaking applied (cyclic pattern) and POSCAR_SCC written.")