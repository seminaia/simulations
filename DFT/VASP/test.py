from mpi4py import MPI
from lammps import lammps

lmp = lammps()
lmp.file("in.lj")
comm = MPI.COMM_WORLD
print("Proc %dd out of %dd procs" % (comm.Get_rank(),comm.Get_size()))
