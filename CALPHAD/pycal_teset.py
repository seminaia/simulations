import matplotlib.pyplot as plt
from pycalphad import Database, binplot, equilibrium
import pycalphad.variables as v
f='COST507-modified.tdb'

dbf = Database(f)
elements = dbf.elements
elements.discard('/-')
components = ['AL', 'MG','VA']
phases = list(dbf.phases.keys())
print("Elements:", elements)
print("Number of elements:", len(elements))
print("Phases:", phases)
print("Number of phases:", len(phases))
dof = len(components) - len(phases) + 2
print("Degrees of freedom:", dof)
binplot(
    dbf,
    components,
    phases,
    conditions={
        v.X('MG'):(0,1,0.01),
        #v.W('Mo'):0.157,
        #v.W('Cr'):0.0711,
        #v.W('Fe'):0.0371,
        #v.W('Mn'):0.0056,
        #v.W('Si'):0.0021,
        #v.W('C') : 0.00058,
        #v.W('W'): 0.0003,
        #v.W('Cu'): 0.00014,
        #v.W('B'): 0.000045,
        #v.W('Al'): 0.0011,
        #v.W('Co'): 0.0001,
        #v.W('H'):(0, 1, 0.01),       
        v.T: (300, 1000, 10),
        v.P: 101325,
        v.N: 1
    }
)
plt.show()
