from pathlib import Path

import matplotlib.pyplot as plt
from pycalphad import Database, ternplot, binplot, Workspace
from pycalphad.core.utils import filter_phases
from pycalphad.mapping import BinaryStrategy, plot_binary
import pycalphad.variables as v
import numpy as np
import pandas as pd
from pycalphad.property_framework.metaproperties import IsolatedPhase
# Load database

f1 = Path(__file__).resolve().with_name("mmc1.TDB")
with f1.open('r', encoding='latin-1') as file:
    content = file.read()
dbf = Database(content)
tern_comp = ['CU', 'SN','C','O','VA']
phase_keys = list(dbf.phases.keys())  # Get all phase keys from the database

print("All phase keys from the database:", phase_keys)
filtered_phases_tern = [
    phase for phase in filter_phases(dbf, tern_comp, phase_keys)
    if phase != 'BCC_B2'
]  # BCC_B2's legacy DIS_PART BCC_A2 model is incompatible with pycalphad
print("Filtered phases for the ternary plot:", list(filtered_phases_tern))
conds_tern = {
    v.T: (800, 1800,100),  # vary from 1600 to 2000 step 10
    v.P: 101325,
    v.N: 1,
    v.X('O'): 0.01,
    v.W('SN'): 0.3,
    v.X('C'): (0, 0.1, 0.01),  
}    
print(f"Phases considered in the ternary plot: {filtered_phases_tern}")
print(f"Components considered in the ternary plot: {tern_comp}")
print(f"Number of phases in the ternary plot: {len(filtered_phases_tern)}")

dof_tern = len(tern_comp) - len(filtered_phases_tern) 
print(f"Degrees of freedom for the ternary plot: {dof_tern}")
print(f"Degrees of freedom for the binary plot: {dof_tern}")
bin_strat = BinaryStrategy(dbf, tern_comp, filtered_phases_tern, conds_tern)
bin_strat.do_map()
ax = plot_binary(bin_strat, x=v.X('C'), y=v.T, tielines=False)
fig = ax.figure
ax.set_title("Binary Phase Diagram for CU-SN-C System")
ax.set_xlabel("C Mole Fraction")
ax.set_ylabel("Temperature (K)")
plt.show()