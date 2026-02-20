from scipy import constants as cst
from pint import UnitRegistry
import sys
import multiprocessing
import numpy as np
from MinimizeEnergy import MinimizeEnergy
from MonteCarlo import MonteCarlo

path_to_code = "./"
sys.path.append(path_to_code)
ureg = UnitRegistry()
ureg = UnitRegistry(autoconvert_offset_to_baseunit = True)

kB = cst.Boltzmann*ureg.J/ureg.kelvin # boltzman constant
Na = cst.Avogadro/ureg.mole # avogadro
R = kB*Na # gas constant
tau = np.linspace(0.1, 10, 10) 
epsilon = (119.76*ureg.kelvin*kB*Na).to(ureg.kcal/ureg.mol)
r_star = 3.822*ureg.angstrom
sigma = r_star / 2**(1/6)
m_argon = 39.948*ureg.gram/ureg.mol
T = (55 * ureg.degC).to(ureg.degK)
N_atom = 200
cut_off = sigma*2.5 # angstrom
displace_mc = sigma/5 # angstrom
volume_star = r_star**3 * Na * 2**(-0.5)
volume = N_atom*volume_star*tau/Na
L = volume**(1/3)

def launch_MC_code(tau):

    epsilon = (119.76*ureg.kelvin*kB*Na).to(ureg.kcal/ureg.mol)
    r_star = 3.822*ureg.angstrom
    sigma = r_star / 2**(1/6)
    m_argon = 39.948*ureg.gram/ureg.mol
    T = (55 * ureg.degC).to(ureg.degK)

    N_atom = 200

    cut_off = sigma*2.5 # angstrom
    displace_mc = sigma/5 # angstrom

    volume_star = r_star**3 * Na * 2**(-0.5)
    volume = N_atom*volume_star*tau/Na
    L = volume**(1/3)

    folder = "outputs_tau"+str(tau)+"/"
    em = MinimizeEnergy(
        ureg = ureg,
        maximum_steps=100,
        thermo_period=10,
        dumping_period=10,
        number_atoms=[N_atom],
        epsilon=[epsilon],
        sigma=[sigma],
        atom_mass=[m_argon],
        box_dimensions=[L, L, L],
        cut_off=cut_off,
        data_folder=folder,
        thermo_outputs="Epot-MaxF",
        neighbor=20,
    )
    em.run()
    minimized_positions = em.atoms_positions*em.ref_length
    mc = MonteCarlo(
        ureg = ureg,
        maximum_steps=20000,
        dumping_period=1000,
        thermo_period=1000,
        neighbor=50,
        displace_mc = displace_mc,
        desired_temperature = T,
        number_atoms=[N_atom],
        epsilon=[epsilon],
        sigma=[sigma],
        atom_mass=[m_argon],
        box_dimensions=[L, L, L],
        initial_positions = minimized_positions,
        cut_off=cut_off,
        data_folder=folder,
        thermo_outputs="Epot-press",
    )
    mc.run()
if __name__ == "__main__":
    tau_values = np.round(np.logspace(-0.126, 0.882, 10),2)
    pool = multiprocessing.Pool()
    squared_numbers = pool.map(launch_MC_code, tau_values)
    pool.close()
    pool.join()