import matplotlib.pyplot as plt
from pycalphad import Database, binplot, equilibrium
import pycalphad.variables as v

dbf = Database('cost507.tdb')
components = ['FE', 'C', 'N', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3', 'N2GAS','CR']
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
        v.X('C'): (0, 0.1, 0.01),
        v.X('N'): 0.005,
        v.T: (300, 1000, 10),
        v.P: 101325,
        v.N: 1
    }
)
plt.show()
