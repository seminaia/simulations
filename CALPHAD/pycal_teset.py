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
Traceback (most recent call last):
  File "/home/soki/simulations/CALPHAD/pycal_teset.py", line 5, in <module>
    dbf = Database('cost507.tdb')
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/io/database.py", line 115, in __new__
    return cls.from_file(fname, fmt=fmt)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/io/database.py", line 224, in from_file
    format_registry[fmt.lower()].read(dbf, fd)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/io/tdb.py", line 985, in read_tdb
    raise e
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/io/tdb.py", line 968, in read_tdb
    tokens = grammar.parseString(command)
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pyparsing/util.py", line 466, in _inner
    return fn(self, *args, **kwargs)
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pyparsing/core.py", line 1346, in parse_string
    raise exc.with_traceback(None)
pyparsing.exceptions.ParseException: Invalid TDB syntax.
 ELEMENT  Y    HCP_A3              8.89059+01  5.9664E+03  4.4434E+01 
                                          ^, found '+'  (at char 2013), (line:138, col:42)