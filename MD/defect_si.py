from ase.io import read
atoms = read('Si_equilibrated.xyz')  # or your saved file

import numpy as np
n_dopants = 2  # number of phosphorus atoms
indices = np.random.choice(len(atoms), size=n_dopants, replace=False)
for idx in indices:
    atoms[idx].symbol = 'P'
print(f"Replaced {n_dopants} Si atoms with P at indices {indices}")

from ase.calculators.kim.kim import KIM
calc = KIM("model_name_that_includes_Si_and_P")
atoms.calc = calc