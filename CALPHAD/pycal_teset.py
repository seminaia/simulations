import matplotlib.pyplot as plt
from pycalphad import Database, binplot, equilibrium
import pycalphad.variables as v

dbf = Database('Fe-C.TDB')
components = ['FE','C','VA']
phases = list(dbf.phases.keys())
fig, ax = plt.subplots()
print("Components:", components)
print("Number of components:", len(components))
print("Phases:", phases)
print("Number of phases:", len(phases))
dof = len(components) - len(phases) + 2
print("Degrees of freedom:", dof)

binplot(
    dbf,
    components,
    phases,
    {
        v.X('C'): (0, 0.6, 0.01),
        v.T: (300, 1600, 10),
        v.P: 101325,
        v.N: 1
    }
)
plt.show()
