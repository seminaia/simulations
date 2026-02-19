import json

def json_to_poscar(json_file, poscar_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    crystal_structure = data['crystal_structure']
    lattice = crystal_structure['lattice']
    atomic_positions = crystal_structure['atomic_positions']

    # Extract lattice parameters
    a = float(lattice['a'].split()[0])
    b = float(lattice['b'].split()[0])
    c = float(lattice['c'].split()[0])

    # Extract atomic positions and elements
    elements = []
    num_atoms = []
    positions = []
    for atom in atomic_positions:
        element = atom['element']
        if element not in elements:
            elements.append(element)
            num_atoms.append(1)
        else:
            num_atoms[elements.index(element)] += 1
        
        # Convert fractional positions to numerical values
        x = eval(atom['x'])
        y = eval(atom['y'])
        z = eval(atom['z'])
        positions.append([x, y, z])

    with open(poscar_file, 'w') as f:
        f.write(f"Generated POSCAR\n")
        f.write(f"1.0\n")
        f.write(f"{a:.6f} 0.000000 0.000000\n")
        f.write(f"0.000000 {b:.6f} 0.000000\n")
        f.write(f"0.000000 0.000000 {c:.6f}\n")
        f.write(" ".join(elements) + "\n")
        f.write(" ".join(map(str, num_atoms)) + "\n")
        f.write("Direct\n")
        for pos in positions:
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

# Example usage
json_to_poscar('mp-17554.json', 'POSCAR')
