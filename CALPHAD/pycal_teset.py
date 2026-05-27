import matplotlib.pyplot as plt
from pycalphad import Database, binplot, equilibrium
import pycalphad.variables as v
f="mmc1.TDB"
dbf = Database(f)
elements = dbf.elements
elements.discard('/-')
components = ['FE', 'MN','Sn','C' ,'VA']
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
        v.X('FE'):(0,1,0.1),
        v.X('MN'):(0,1,0.1),
        v.X('Sn'):(0,1,0.1),
        v.X('C'):(0,1,0.1),
        v.T: (300, 1000, 10),
        v.P: 101325,
        v.N: 1
    }
)
plt.show()
