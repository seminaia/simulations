from jarvis.tasks.vasp.vasp import VaspJob, write_vaspjob
from jarvis.io.vasp.inputs import Potcar, Incar, Poscar
from jarvis.db.jsonutils import dumpjson
from jarvis.core.atoms import Atoms
from jarvis.core.kpoints import Kpoints3D
from jarvis.tasks.queue_jobs import Queue
import os

# Load/build crystal structure
# mat = Poscar.from_file('POSCAR')
coords = [[0, 0, 0], [0.25, 0.25, 0.25]]
elements = ["Si", "Si"]
box = [[2.715, 2.715, 0], [0, 2.715, 2.715], [2.715, 0, 2.715]]
atoms = Atoms(lattice_mat=box, coords=coords, elements=elements)
mat = Poscar(atoms)
mat.comment = "Silicon"

# Build INCAR file
data = dict(
    PREC="Accurate",
    ISMEAR=0,
    SIGMA=0.01,
    IBRION=2,
    ISIF=3,
    GGA="BO",
    PARAM1=0.1833333333,
    PARAM2=0.2200000000,
    LUSE_VDW=".TRUE.",
    AGGAC=0.0000,
    EDIFF="1E-7",
    EDIFFG="-1E-3",
    NELM=400,
    ISPIN=2,
    LCHARG=".FALSE.",
    LVTOT=".FALSE.",
    LVHAR=".FALSE.",
    LWAVE=".FALSE.",        AFM_species = {"La","Ni"}  # Corrected AFM species

)
inc = Incar(data)
# Build POTCAR info
# export VASP_PSP_DIR = 'PATH_TO_YOUR_PSP'
pot = Potcar(elements=mat.atoms.elements)

# Build Kpoints info
kp = Kpoints3D().automatic_length_mesh(
    lattice_mat=mat.atoms.lattice_mat, length=20
)

vasp_cmd = "/users/knc6/VASP/vasp54/src/vasp.5.4.1DobbySOC2/bin/vasp_std"
copy_files = ["/users/knc6/bin/vdw_kernel.bindat"]
jobname = "MAIN-RELAX@JVASP-1002"
job = VaspJob(
    poscar=mat,
    incar=inc,
    potcar=pot,
    kpoints=kp,
    vasp_cmd=vasp_cmd,
    copy_files=copy_files,
    jobname=jobname,
)

dumpjson(data=job.to_dict(), filename="job.json")
write_vaspjob(pyname="job.py", job_json="job.json")