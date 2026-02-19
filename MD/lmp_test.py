from lammps import lammps

# melt example
L = lammps()
L.command("region box block 0 10 0 5 -0.5 0.5")
