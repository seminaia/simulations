from re import A
from ase.build import bulk
from ase.spacegroup import crystal
from ase.optimize import BFGS
from ase.visualize import view
from ase.vibrations import Vibrations
from ase.thermochemistry import CrystalThermo
from ase.phonons import Phonons
from ase.filters import UnitCellFilter, FrechetCellFilter
from ase.transport.calculators import TransportCalculator
from gpaw import GPAW, PW, Mixer
from ase.eos import calculate_eos

import numpy as np
from atomistics.calculators.ase import evaluate_with_ase
from atomistics.workflows import ElasticMatrixWorkflow
from atomistics.workflows import EnergyVolumeCurveWorkflow
import atomistics
import pandas as pd

cryst= crystal('Cr', [(0, 0, 0)], spacegroup=225, cellpar=[2.88, 2.88, 2.88, 90, 90, 90])
calc = GPAW(mode=PW(500),
            xc='PBE',
            kpts=(8, 8, 8),
            mixer=Mixer(0.1, 5, weight=50.0),
            occuypations={'name': 'fermi-dirac', 'width': 0.1},
            txt='Cr.txt')
cryst.calc = calc

opt = BFGS(cryst)
opt.run(fmax=0.01)
pot_energy = cryst.get_potential_energy()

N=2
ph = Phonons(cryst,
             calc,
             supercell=(N, N, N),
             delta=0.01)
ph.run()
ph.read(acoustic=True)
dos = ph.get_dos(kpts=(5, 5, 5)).sample_grid(npts=100)
phonon_energies = dos.get_energies()
phonon_dos =dos.get_weights()

thermo = CrystalThermo(phonon_dos, phonon_energies,potentialenergy=pot_energy)
temperatures = np.linspace(0, 1000, 100)
data = []
for T in temperatures:
    F = thermo.get_helmholtz_energy(temperature=T)
    data.append((T, F))
data = np.array(data)
pd.DataFrame(data, columns=['T', 'F']).to_csv('Cr_thermo.csv', index=False)
