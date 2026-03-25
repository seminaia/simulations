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

/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/io/tdb.py:287: UserWarning: Type definitions using IF/THEN logic is not supported
  warnings.warn("Type definitions using IF/THEN logic is not supported")
Traceback (most recent call last):
  File "/home/soki/simulations/CALPHAD/pycal_teset.py", line 12, in <module>
    binplot(
    ~~~~~~~^
        dbf,
        ^^^^
    ...<10 lines>...
        ax=ax
        ^^^^^
    )
    ^
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/mapping/compat_api.py", line 53, in binplot
    strategy.do_map()
    ~~~~~~~~~~~~~~~^^
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/mapping/strategy/strategy_base.py", line 268, in do_map
    self.generate_automatic_starting_points()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/mapping/strategy/binary_strategy.py", line 97, in generate_automatic_starting_points
    step.do_map()
    ~~~~~~~~~~~^^
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/mapping/strategy/strategy_base.py", line 268, in do_map
    self.generate_automatic_starting_points()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/mapping/strategy/step_strategy.py", line 27, in generate_automatic_starting_points
    self.add_nodes_from_conditions(mid_conds, None, True)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/mapping/strategy/strategy_base.py", line 176, in add_nodes_from_conditions
    point = point_from_equilibrium(self.dbf, self.components, self.phases, conditions, models=self.models, phase_record_factory=self.phase_records)
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/mapping/starting_points.py", line 33, in point_from_equilibrium
    chemical_potentials = np.squeeze(wks.eq.MU)
                                     ^^^^^^
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/core/workspace.py", line 282, in __get__
    default_value = obj.recompute()
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/core/workspace.py", line 362, in recompute
    properties = starting_point(unitless_conds, state_variables, self.phase_record_factory, grid)
  File "/home/soki/miniconda3/envs/myCALPHADenv/lib/python3.13/site-packages/pycalphad/core/starting_point.py", line 84, in starting_point
    raise ValueError('Number of degrees of freedom is not zero')
ValueError: Number of degrees of freedom is not zero