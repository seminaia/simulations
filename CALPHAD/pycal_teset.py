import matplotlib.pyplot as plt
import numpy as np
from pycalphad import Database, equilibrium, binplot
import pycalphad.variables as v

dbf = Database('PrecHiMn-04.tdb')

my_phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'GAS','HCP_A3','FE4N_L1']
my_components = ['FE','CR', 'N', 'C']
dof = len(my_components)-len(my_phases) + 2
print(f"Degrees of freedom: {dof}")
fig = plt.figure()
axes = fig.gca()

binplot(dbf, my_components, my_phases, {v.X('C'):(0,1,0.01), v.N:1, 
                                       v.T:(300,2000,10), v.P:101325})

plt.show()