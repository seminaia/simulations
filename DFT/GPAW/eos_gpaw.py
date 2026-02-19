from re import A
from ase.build import bulk
from ase.spacegroup import crystal
from ase.optimize import BFGS
from ase.visualize import view
from ase.vibrations import Vibrations
from ase.filters import UnitCellFilter, FrechetCellFilter
from ase.transport.calculators import TransportCalculator
from gpaw import GPAW, PW, Mixer
from ase.eos import calculate_eos
import numpy as np
from atomistics.calculators.ase import evaluate_with_ase
from atomistics.workflows import ElasticMatrixWorkflow
from atomistics.workflows import EnergyVolumeCurveWorkflow
import atomistics

def relax(atoms, calculator_params,
          fmax=0.01, d3=False, fixcell=True,
          logname='opt.log',
          trajname='opt.traj'):

    # set DFT calculator
    calc_dft = GPAW(**calculator_params)

    # magnetize atoms
    atoms.set_initial_magnetic_moments(len(atoms) * [1])
    # non-magnetic calculation:
    # atoms.set_initial_magnetic_moments(len(atoms) * [0])

    # optionally include van der Waals DFT-D3
    if d3:
        from ase.calculators.dftd3 import DFTD3
        calc = DFTD3(dft=calc_dft)
    else:
        calc = calc_dft

    # set calculator
    atoms.calc = calc

    # set configuration to be optimized
    if fixcell:
        # only optimize positions of the atoms
        opt_conf = atoms
    else:
        # setup full relaxation
        # set unit cell filter
        opt_conf = FrechetCellFilter(atoms)

    # setup optimizer
    # specify logfile and trajectory file names
    opt = BFGS(opt_conf, logfile=logname, trajectory=trajname)
    # run the optimization until forces are smaller than fmax
    opt.run(fmax=fmax)

    return atoms

calculator_params = {
    "xc": "PBE",
    "basis": "dzp",
    "mode": {"name": "pw",
             "ecut": 600},
    "kpts": {"size": [5, 5, 5],
             "gamma": True},
    "convergence": {"density": 1e-6,
                    "forces": 1e-4},
    "occupations": {"name": "fermi-dirac",
                    "width": 0.05},
    "mixer": {"method": "fullspin",
              "backend": "pulay"},
    "txt": "rlx.txt",
}

Cr = bulk('Cr','bcc',a=2.97, cubic=True)
view(Cr)
EV_init = EnergyVolumeCurveWorkflow(Cr,
                                      num_points=20,
                                      vol_range=0.05,
                                      axes=['x','y','z'])
elastic_init = ElasticMatrixWorkflow()
#RP = bulk('La2NiO4',a=3.87,c=12.7,orthorhombic=True)
#view(RP)
task_dict = EV_init.generate_structures()
Cr.calc = GPAW(xc='PBE',
                basis='dzp',
                mode=PW(600),
                kpts=(5, 5, 5),
               convergence={'density': 1e-6, 'energy': 1e-6, 'bands': -10},
               txt='Cr.txt')
calc = calculate_eos(Cr, npoints=9,eps=0.5)
V0,E0,B = calc.fit()

calc.plot('Cr_eos.png', show=True)
#result_dict = evaluate_with_ase(structure=Al,
#                               task_dict,
#                               ase_calculator=Al.calc)
#print(result_dict)
fit_dict = EV_init.fit_dict
print(fit_dict)
#dft = Al.calc.dft
uf = UnitCellFilter(Al)
#relax = BFGS(uf)
#relax.run(fmax=0.01)
#a = np.linalg.norm(Al.cell[0])*np.sqrt(2)
#c = np.linalg.norm(Al.cell[2])* np.sqrt(2)
#print('Lattice constant after relaxation: a=%.3f Å, c=%.3f Å' % (a,c))

