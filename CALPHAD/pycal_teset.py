import matplotlib.pyplot as plt
from pycalphad import Database, binplot
import pycalphad.variables as v

dbf = Database('PrecHiMn-04.tdb')

components = ['FE', 'CR', 'C', 'N', 'VA']
phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3', 'FE4N_L1']

fig, ax = plt.subplots()

binplot(
    dbf,
    components,
    phases,
    {
        v.X('CR'): (0, 0.5, 0.01),
        v.X('C'): 0.01,
        v.X('N'): 0.01,
        v.T: (300, 2000, 10),
        v.P: 101325,
        v.N: 1
    },
    ax=ax
)

plt.show()