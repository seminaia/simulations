import matplotlib.pyplot as plt
from pycalphad import Database, ternplot, binplot, Workspace
from pycalphad.core.utils import filter_phases
from pycalphad.mapping import plot_binary
import pycalphad.variables as v
import numpy as np
import pandas as pd
from pycalphad.property_framework.metaproperties import IsolatedPhase
# Load database
f = "SGTE-unary1991-2010.TDB"
with open(f, 'r',encoding='latin-1') as file:
    content = file.read()
dbf = Database(content)
tern_comp = ['FE', 'C', 'VA']
phase_keys = list(dbf.phases.keys())  # Get all phase keys from the database
filtered_phases_tern = filter_phases(dbf, tern_comp, phase_keys)  # Filter phases based on components
conds_tern = {
    v.T: (1000, 2000,100),  # vary from 1000 to 2000 step 100
    v.P: 101325,
    v.N: 1,
    v.X('C'): (0, 0.08, 0.01),  # vary from 0 to 0.08 step 0.01
}    

print(f"Phases considered in the ternary plot: {filtered_phases_tern}")
print(f"Components considered in the ternary plot: {tern_comp}")
print(f"Number of phases in the ternary plot: {len(filtered_phases_tern)}")

dof_tern = len(tern_comp) - len(filtered_phases_tern) + 2
print(f"Degrees of freedom for the ternary plot: {dof_tern}")

binplot(dbf, tern_comp,phase_keys, conds_tern)
