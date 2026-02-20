from gpaw import GPAW
from ase import Atoms

def calc_gpaw(calculator_params):
    """Calculate the potential energy using GPAW."""
    calc = GPAW(**calculator_params)
    calc.calculate()
    return calc

def potentials(epsilon, sigma, r, derivative=False):
    if derivative:
        return 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r
    else:
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    
