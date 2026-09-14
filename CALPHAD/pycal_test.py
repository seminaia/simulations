import matplotlib.pyplot as plt
from pycalphad import Database, ternplot, binplot, Workspace
from pycalphad.core.utils import filter_phases
from pycalphad.mapping import BinaryStrategy, plot_binary
import pycalphad.variables as v
import numpy as np
import pandas as pd
from pycalphad.property_framework.metaproperties import IsolatedPhase
# Load database

f = "mf-steel-3g.tdb"
with open(f, 'r',encoding='latin-1') as file:
    content = file.read()
dbf = Database(content)
tern_comp = ['FE', 'C', 'VA']
phase_keys = list(dbf.phases.keys())  # Get all phase keys from the database

print("All phase keys from the database:", phase_keys)
filtered_phases_tern = filter_phases(dbf, tern_comp, phase_keys)  # Filter phases based on components
print("Filtered phases for the ternary plot:", list(filtered_phases_tern))
conds_tern = {
    v.T: (800, 1800,100),  # vary from 1600 to 2000 step 10
    v.P: 101325,
    v.N: 1,
    v.X('C'): (0, 0.8, 0.1),  # vary from 0 to 0.08 step 0.001
}    
print(f"Phases considered in the ternary plot: {filtered_phases_tern}")
print(f"Components considered in the ternary plot: {tern_comp}")
print(f"Number of phases in the ternary plot: {len(filtered_phases_tern)}")

dof_tern = len(tern_comp) - len(filtered_phases_tern) + 2
print(f"Degrees of freedom for the ternary plot: {dof_tern}")
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
bin_strat= BinaryStrategy(dbf, tern_comp, phase_keys, conds_tern)
bin_strat.generate_automatic_starting_points()
bin_strat.do_map()
plot = plot_binary(bin_strat,x=v.X('C'),y=v.T)
plt.show()