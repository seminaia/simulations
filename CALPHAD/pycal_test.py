import matplotlib.pyplot as plt
from pycalphad import Database, ternplot, binplot, Workspace
from pycalphad.core.utils import filter_phases
from pycalphad.mapping import BinaryStrategy, plot_binary
import pycalphad.variables as v
import numpy as np
import pandas as pd
from pycalphad.property_framework.metaproperties import IsolatedPhase
# Load database

f1 = "Fe-C.TDB"
f2 = "mmc1.TDB"
f3 = "mf-steel-3g.tdb"
with open(f1, 'r',encoding='latin-1') as file:
    content = file.read()
dbf = Database(content)
tern_comp = ['FE', 'C', 'VA']
phase_keys = list(dbf.phases.keys())  # Get all phase keys from the database

print("All phase keys from the database:", phase_keys)
filtered_phases_tern = filter_phases(dbf, tern_comp, phase_keys)  # Filter phases based on components
print("Filtered phases for the ternary plot:", list(filtered_phases_tern))
conds_tern = {
    v.T: (800, 1800,10),  # vary from 1600 to 2000 step 10
    v.P: 101325,
    v.N: 1,
    v.W('C'): (0, 0.2, 0.01),  # vary from 0 to 0.08 step 0.001
}    
print(f"Phases considered in the binary plot: {filtered_phases_tern}")
print(f"Components considered in the binary plot: {tern_comp}")
print(f"Number of phases in the binary plot: {len(filtered_phases_tern)}")

dof_tern = len(tern_comp) - len(filtered_phases_tern) 
print(f"Degrees of freedom for the binary plot: {dof_tern}")
bin_strat= BinaryStrategy(dbf, tern_comp, phase_keys, conds_tern)
bin_strat.do_map()
ax = plot_binary(bin_strat,x=v.W('C'),y=v.T)
fig = ax.figure
ax.set_title("Binary Phase Diagram for FE-C System")
ax.set_xlabel("C Mole Fraction")
ax.set_ylabel("Temperature (K)")
plt.show()