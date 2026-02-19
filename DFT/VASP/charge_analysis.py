from pymatgen.io.vasp import Chgcar

Chg = Chgcar.from_file("CHGCAR").data

print(Chg)