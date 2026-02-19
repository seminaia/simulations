import os
from amset.core.run import Runner
from amset.wavefunction.vasp import get_wavefunction_coefficients
from amset.scattering.calculate import calculate_rate
from pymatgen.io.vasp import Vasprun

settings = {
    #"scattering_type":['ADP','IMP','POP'],
    "interpolation_factor":150,
    #"doping":[1e19,-1e19,0],
    #"temperatures":[300,600,900,1200,1500,1800],
#    "deformation_potential":[6.5,6.5],
#    "elastic_constant":[200,200],
}

if __name__ == "__main__":
    #wavefunction= get_wavefunction_coefficients
    #runner = Runner.from_vasprun("La2NiO4_bulk/BS/vasprun.xml", settings)
    #amset_dat = runner.run()
    vr = Vasprun("La2NiO4_bulk/phonon/undisp/vasprun.xml")
    print(vr.get_band_structure())
    
