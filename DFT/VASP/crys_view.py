import crystal_toolkit.components as ctc
import dash
from dash import Dash, html, dcc, callback, Output, Input
import os
from crystal_toolkit.settings import SETTINGS
from crystal_toolkit.core.plugin import CrystalToolkitPlugin
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.core.lattice import Lattice

def load_structure(file_path: str) -> Structure:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    return Structure.from_file(file_path)

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

# Element-specific properties
ELEMENT_PROPERTIES = {
    'La': {'charge': +3, 'magmom': 2.0, 'LDAUL': -1, 'LDAUU': 6.0, 'LDAUJ': 0.0},
    'Ni': {'charge': +2, 'magmom': 2.0, 'LDAUL': 2, 'LDAUU': 6.2, 'LDAUJ': 0.0},
    'O':  {'charge': -2, 'magmom': 0.6, 'LDAUL': -1, 'LDAUU': 0.0, 'LDAUJ': 0.0},
}
material = "La2NiO4"
afm_species = {"Ni"}
original_structure = load_structure('POSCAR')
supercell_structure = original_structure.copy() 
supercell_structure.make_supercell([1, 1, 1])  # Actual supercell scaling
magmom = assign_magnetic_moments(supercell_structure, ELEMENT_PROPERTIES, afm_species)
charge = assign_charges(supercell_structure, ELEMENT_PROPERTIES)
supercell_structure.add_site_property("charge", charge)
supercell_structure.add_site_property("magmom", magmom)

# Configure component to show site properties
structure_component = ctc.StructureMoleculeComponent(
    supercell_structure,
    id='LNO-supercell',
    show_settings=SETTINGS,
    show_image_button=True,
    unit_cell_choice="primitive",
    scene_settings={"defaultZoom": 0.8},
    color_scheme="VESTA",  # Uses oxidation states for coloring
    show_legend=True,
)

my_layout = html.Div(
    [
        html.H1("StructureMoleculeComponent"),
        html.H2("Standard Layout"),
        structure_component.layout(),
        html.H2("Optional Title Layout"),
        structure_component.title_layout(),
    ],
    style=dict(
        margin="2em auto", display="grid", placeContent="center", placeItems="center"
    ),
)
    
app = Dash(__name__, plugins=[CrystalToolkitPlugin(my_layout)])

if __name__ == "__main__":
 # ====== Write CIF File ======
    supercell_cif_path = f"{material}.cif"
    CifWriter(
        supercell_structure,
        write_site_properties=True
    ).write_file(supercell_cif_path)
    app.run(debug=True, port=8050)
   