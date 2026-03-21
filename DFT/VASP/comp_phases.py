from cmath import phase
import os
from turtle import up
from pymatgen.ext.matproj import MPRester
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, Element
import doped
from doped.chemical_potentials import CompetingPhases, CompetingPhasesAnalyzer
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
import matplotlib.pyplot as plt
from pymatgen.io.vasp.sets import MPScanRelaxSet, MPRelaxSet, MP24RelaxSet
from scipy.__config__ import show
from monty.serialization import dumpfn, loadfn
import logging
from pymatgen.core import Composition, Structure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("defect_analysis.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
plt.rcdefaults()
plt.style.use(f"{doped.__path__[0]}/utils/doped.mplstyle")

# Setup environment
plt.style.use(f"{doped.__path__[0]}/utils/doped.mplstyle")
elements = ["La", "Ni", "O"]
material = "La2NiO4"
target_space_group = "I4/mmm"  # Corrected space group for La₂NiO₄
output_dir = "CompetingPhases4"
api_key = "zBYiak7h6ies4ziNlAWqzXXHhc7rMBDB"
os.makedirs(output_dir, exist_ok=True)

# User-defined POTCAR and INCAR settings
user_potcar_settings = {
    "La": "La",
    "Ni": "Ni_pv",
    "O": "O",
}
user_incar_settings = {"ISMEAR": 0, 
                       "KPAR": 6,
                       "NCORE": 8}

# Initialize CompetingPhases with correct parameters
try:
    #mpr = MPRester(api_key)
    #entries = mpr.get_entries_in_chemsys(elements, compatible_only=True)
    # Load structure from CIF file
    #structure = Structure.from_file(f"{material}_{target_space_group.replace('/', '_')}.cif")
    #structure = Structure.from_file('CONTCAR')
    # Initialize CompetingPhases for La₂NiO₄
    cp = CompetingPhases(
        material,
        energy_above_hull=0.1,  # Increased to include metastable phases
        api_key=api_key,
        full_phase_diagram=True,
    )
    entries = cp.entries
    
    cpa = CompetingPhasesAnalyzer(material,
                                  entries)
    cpa.calculate_chempots()
    # Combine entries from both sources
    entries = cpa.entries
    
    # Save formation energy data
    cp_form_energies_csv = os.path.join(output_dir, f"{material}_formation_energies.csv")
    cp_chempots_json = os.path.join(output_dir, "chemical_potentials.json")

    formation_energy_df = cpa.get_formation_energy_df(prune_polymorphs=False, include_dft_energies=True)
    formation_energy_df.to_csv(cp_form_energies_csv)

    # Get and save chemical potentials
    chempots = cpa.chempots
    
    logger.info(f"Chemical potentials: {chempots}")
    dumpfn(chempots, cp_chempots_json)

except Exception as e:
    print(f"Error initializing CompetingPhases: {e}")

# Plot phase diagram with proper entries
if entries:
    try:
        pd = PhaseDiagram(entries)
        plotter = PDPlotter(pd,
                            backend="matplotlib",
                            show_unstable=0)  # Show phases up to 0.5 eV/atom above hull
        plt.figure(figsize=(10, 8))
        plot = plotter.get_plot()
        for text in plot.texts:
            if "La$_{2}$NiO$_{4}$" in text.get_text():  # Handle subscript formatting
                # Get current position
                x, y = text.get_position()
                # Move down by 50 energy units (adjust as needed)
                text.set_position((x+80, y - 30))
            if "La$_{3}$Ni$_{2}$O$_{7}$" in text.get_text():
                # Get current position
                x, y = text.get_position()
                # Move down by 50 energy units (adjust as needed)
                text.set_position((x+55, y))
            if "La$_{7}$Ni$_{16}$" in text.get_text():
                # Get current position
                x, y = text.get_position()
                # Move down by 50 energy units (adjust as needed)
                text.set_position((x-65, y))
            if "LaNi" in text.get_text():
                # Get current position
                x, y = text.get_position()
                # Move down by 50 energy units (adjust as needed)
                text.set_position((x-20, y))
            if "LaNi$_{5}$" in text.get_text():
                # Get current position
                x, y = text.get_position()
                # Move down by 50 energy units (adjust as needed)
                text.set_position((x-20, y))
            if "La$_{2}$Ni$_{7}$" in text.get_text():
                # Get current position
                x, y = text.get_position()
                # Move down by 50 energy units (adjust as needed)
                text.set_position((x-20, y+35))
        plt.title(f"{material} Phase Diagram")
        plt.savefig(os.path.join(output_dir, f"{material}_phase_diagram.png"), dpi=300)
        plt.close()
                # Create chemical potential diagram
        formulas_to_draw = ["LaNiO3", "La2O3", "NiO"]

        cpd = ChemicalPotentialDiagram(entries, 
                                       #limits={Element('La'):[-10,0], Element('Ni'):[-8,0], Element('O'):[-10,0]},
                                       default_min_limit=-100
                                       )
        
        # Get formulas to draw - include both required phases
        # Create and save chemical potential diagram
        cp_plot_file = os.path.join(output_dir, f"{material}_chemical_potentials.png")
        fig = cpd.get_plot(elements=elements, formulas_to_draw=formulas_to_draw,formula_colors=['red', 'blue','green','orange'],draw_formula_lines=True,draw_formula_meshes=True)
        fig.write_image(cp_plot_file, format="png", scale=2)
        
        plt.title(f"Chemical Potential Diagram for {material}")

        logger.info(f"Chemical potential diagram saved to {cp_plot_file}")
    except Exception as e:
        logger.error(f"Error plotting phase diagram: {e}")
else:
    logger.warning("No entries available for phase diagram plotting")

# Define Hubbard U parameters (now with proper element keys and structure)
hubbard_u_settings = {
    'La': {'L': -1, 'U':0.0, 'J': 0}, 
    'Ni': {'L': 2, 'U': 7.0, 'J': 0},  
    'O': {'L': -1, 'U': 0.0, 'J': 0}  
}

rvv10= {"BPARAM":11.95,
        "CPARAM":0.0093}

#user_incar_settings.update(rvv10)# Modified VASP setup section
if entries:
    try:

        for entry in entries:
            naming = entry.data['doped_name'].replace("/", "_")
            phase_dir = os.path.join(output_dir, f"{naming}")
            os.makedirs(phase_dir, exist_ok=True)
            
            print(f"Setting up VASP inputs in {phase_dir}...")
            # Get ordered elements from structure (POTCAR order)
            structure = entry.structure
            elements_in_structure = [s.symbol for s in structure.types_of_species]  # Unique elements in structure order
            
            # Check if any element requires Hubbard U corrections
            requires_ldau = any(el in hubbard_u_settings and hubbard_u_settings[el]['U'] > 0 for el in elements_in_structure)
            
            # Prepare LDAU parameters
            ldau_params = {
                'LDAUL': {el: hubbard_u_settings[el]['L'] for el in elements_in_structure},
                'LDAUU': {el: hubbard_u_settings[el]['U'] for el in elements_in_structure},
                'LDAUJ': {el: hubbard_u_settings[el]['J'] for el in elements_in_structure}
            }
            
            # Update INCAR settings
            updated_incar = user_incar_settings.copy()
            if requires_ldau:
                updated_incar.update({
                    'IBRION': 2,
                    'LDAU': True,
                    'LDAUTYPE': 2,
                    **ldau_params  # Unpack dictionaries
                })
            else:
                updated_incar['LDAU'] = False
            
            # Write VASP input files
            #MPScanRelaxSet(
            #    structure,
            #    user_potcar_settings=user_potcar_settings,
            #    user_incar_settings=updated_incar,
            #    international_monoclinic=False,
            #    auto_kspacing=True
            #).write_input(phase_dir)
            
            MPRelaxSet(
                structure,
                user_potcar_settings=user_potcar_settings,
                user_incar_settings=updated_incar,
                auto_kspacing=True
            ).write_input(phase_dir)
            #MP24RelaxSet(
            #    structure,
            #    user_potcar_settings=user_potcar_settings,
            #    user_incar_settings=updated_incar,
            #    international_monoclinic=False,
            #    auto_kspacing=False,
            #    user_potcar_functional="PBE_54",
            #    xc_functional="PBE"
            #).write_input(phase_dir)
            
    except Exception as e:
        logger.error(f"VASP setup failed: {str(e)}")
else:
    logger.warning("No entries to process")

