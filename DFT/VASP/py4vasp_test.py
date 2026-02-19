import py4vasp
from py4vasp import Calculation
import py4vasp.calculation
calc_path = "La2NiO4_bulk/PBE_DOS/vasprun.xml.gz"
calc = py4vasp.Calculation.from_path(calc_path)
dos = calc.dos.plot()
