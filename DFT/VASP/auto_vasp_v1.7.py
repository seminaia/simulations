#!/usr/bin/env python
# -*-coding:utf-8 -*-

from ast import Dict
from pyclbr import Class
import os, sys, getopt, re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pymatgen.io.vasp import inputs, outputs
from pymatgen.core.structure import Structure, IStructure
from pymatgen.core.trajectory import Trajectory

if sys.version_info[1] < 8:
    from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer # for python < 3.8
else:
    from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer

class Utilities():
    def __init__(self) -> None:
        pass

    def get_symbols(self):
        '''
        * Get elemental symbols from POSCAR
        * Update the POTCAR based on this
        '''
        POSCAR = inputs.Poscar.from_file('POSCAR', check_for_POTCAR=False, read_velocities=False)
        symbols = POSCAR.site_symbols
        ary_sv = ['Li', 'K', 'Sr', 'Cs', 'Ba', 'Sc', 'Zr', 'Y', 'Rb', 'V', 'Ca', 'Be']
        ary_pv = ['Na', 'Nb', 'Tc', 'Mo', 'Ta', 'Os', 'Hf', 'W', 'Mn', 'Re', 'Fe', 'Cu', 'Ni', 'Ru', 'Rh', 'Cr', 'Mg', 'Ti', 'Co']
        ary_d = ['Ga', 'Ge', 'In', 'Sn', 'Tl', 'Pb', 'Bi']
        ary_3 = ['Pr', 'Nd', 'Pm', 'Sm', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu']
        for i, symbol in enumerate(symbols):
            if symbol in ary_sv:
                symbols[i] = f'{symbol}_sv'
            elif symbol in ary_pv:
                symbols[i] = f'{symbol}_pv'
            elif symbol in ary_d:
                symbols[i] = f'{symbol}_d'
            elif symbol in ary_3:
                symbols[i] = f'{symbol}_3'
        return symbols
    
    def add_mag(self, file_name='INCAR'):
        '''
        * Add MAGMOM and the other relative tags in INCAR
        * If the materials have magnetc
        '''
        POSCAR = Structure.from_file('POSCAR')
        composition = POSCAR.composition.to_data_dict
        elements = composition['elements']
        mag_str = ''
        for i, ele in enumerate(elements):
            natoms = int(composition['unit_cell_composition'][ele])
            if ele in MAG_DICT.keys():
                mag_str += f'{natoms}*{MAG_DICT[ele]} '
            else:
                mag_str += f'{natoms}*0.0 '
        with open(file_name, 'a') as f:
            mag_str = 'MAGMOM = ' + mag_str
            f.write('\n\n# Magnetic (User)')
            f.write('\nLORBIT = 11')
            f.write('\nISPIN  = 2')
            f.write('\n' + mag_str)
    
    def add_par(self, file_name='INCAR'):
        '''
        * Determine the parallel mode wisely
        * Base on the total nodes & cores
        '''
        total_cores, total_nodes =  int(TOTAL_CORES), int(TOTAL_NODES)
        npar = int(total_cores**0.5)
        while total_cores % npar != 0:
            npar -= 1
        with open(file_name, 'a') as f:
            f.write('\n\n# Parallel (User)')
            f.write(f'\nNPAR = {npar}')
            # f.write(f'\nKPAR = {kpar}')

class ISIFX():
    def __init__(self) -> None:
        self.utilities = Utilities()

    def isifx_bash(self):
        '''
        * Create job scripts for isifx
        '''
        if MAGNETIC:
            with open('update_mag.py', 'w') as f:
                f.write('#!/usr/bin/env python \n')
                f.write('# -*-coding:utf-8 -*- \n')
                f.write('from pymatgen.io.vasp import outputs \n')
                f.write('with open("INCAR", "a") as  f: \n')
                f.write('    OUTCAR = outputs.Outcar("OUTCAR") \n')
                f.write('    mag_str = " ".join([str(m["tot"]) for m in OUTCAR.magnetization]) \n')
                f.write(r'    mag_str = "MAGMOM = " + mag_str ' + '\n')
                f.write(r'    f.write("\n# With magnetic \n")' +  '\n')
                f.write(r'    f.write("LORBIT = 11 \n")' + '\n')
                f.write(r'    f.write("ISPIN  = 2 \n")' + '\n')
                f.write('    f.write(mag_str) \n')
        with open(f'{JOB_NAME}.sh', 'w') as f:
            f.write('#!/usr/bin/sh \n')
            f.write(headers)
            # Do the ISIF7 first
            f.write(f'time {SUBMIT_CMD} \n')
            # Then do the isif5 with the CONTCAR from isif7
            f.write('if [ `grep -c "reached required accuracy" vasp.out` -ne "0" ] \n')
            f.write('then \n')
            f.write('    rm -f INCAR \n')
            f.write('    rm -f POSCAR \n')
            f.write('    cp INCAR_isif5 INCAR \n')
            f.write('    cp OUTCAR OUTCAR_isif7 \n')
            f.write('    cp OSZICAR OSZICAR_isif7 \n')
            f.write('    cp CONTCAR CONTCAR_isif7 \n')
            f.write('    cp vasp.out vasp.out_isif7 \n')
            f.write('    cp CONTCAR POSCAR \n')
            if MAGNETIC:
                f.write('    time python update_mag.py \n')
            f.write(f'    time {SUBMIT_CMD} \n')
            # Then do the isif4 with the CONTCAR from isif5
            f.write('    if [ `grep -c "reached required accuracy" vasp.out` -ne "0" ] \n')
            f.write('    then \n')
            f.write('        rm -f INCAR \n')
            f.write('        rm -f POSCAR \n')
            f.write('        cp INCAR_isif4 INCAR \n')
            f.write('        cp OUTCAR OUTCAR_isif5 \n')
            f.write('        cp OSZICAR OSZICAR_isif5 \n')
            f.write('        cp CONTCAR CONTCAR_isif5 \n')
            f.write('        cp vasp.out vasp.out_isif5 \n')
            f.write('        cp CONTCAR POSCAR \n')
            if MAGNETIC:
                f.write('        time python update_mag.py \n')
            f.write(f'        time {SUBMIT_CMD} \n')
            # Then do the isif2 with the CONTCAR from isif4
            f.write('        if [ `grep -c "reached required accuracy" vasp.out` -ne "0" ] \n')
            f.write('        then \n')
            f.write('            rm -f INCAR \n')
            f.write('            rm -f POSCAR \n')
            f.write('            cp INCAR_isif2 INCAR \n')
            f.write('            cp OUTCAR OUTCAR_isif4 \n')
            f.write('            cp OSZICAR OSZICAR_isif4 \n')
            f.write('            cp CONTCAR CONTCAR_isif4 \n')
            f.write('            cp vasp.out vasp.out_isif4 \n')
            f.write('            cp CONTCAR POSCAR \n')
            if MAGNETIC:
                f.write('            time python update_mag.py \n')
            f.write(f'            time {SUBMIT_CMD} \n')
            # If the isif4 is not finished
            f.write('        else \n')
            f.write('            echo "The ISIF4 calculation is not finished yet!" \n')
            f.write('        fi \n')
            # If the isif5 is not finished
            f.write('    else \n')
            f.write('        echo "The ISIF5 calculation is not finished yet!" \n')
            f.write('    fi \n')
            # If the isif7 is not finished
            f.write('else \n')
            f.write('    echo "The ISIF7 calculation is not finished yet!" \n')
            f.write('fi \n')
            f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh')
    
    def isifx(self):
        '''
        * ISIF7 -> ISIF5 -> ISIF4 -> ISIF2
        * Create sub-folders and submit jobs in batch
        '''
        # Create folder
        if not os.path.exists(PATH + '/isifx'):
            os.makedirs(PATH + '/isifx')
        # Copy INCARs and POSCAR
        os.system(f'cp {PATH}/INCARs/INCAR_isif7 {PATH}/isifx/INCAR_isif7')
        os.system(f'cp {PATH}/INCARs/INCAR_isif5 {PATH}/isifx/INCAR_isif5')
        os.system(f'cp {PATH}/INCARs/INCAR_isif4 {PATH}/isifx/INCAR_isif4')
        os.system(f'cp {PATH}/INCARs/INCAR_isif2 {PATH}/isifx/INCAR_isif2')
        os.system(f'cp {PATH}/POSCAR {PATH}/isifx/POSCAR')
        # Enter the target folder
        os.chdir(PATH + '/isifx')
        # Add PAR
        self.utilities.add_par(file_name='INCAR_isif7')
        self.utilities.add_par(file_name='INCAR_isif5')
        self.utilities.add_par(file_name='INCAR_isif4')
        self.utilities.add_par(file_name='INCAR_isif2')
        # INCAR_relax --> INCAR
        os.system('cp INCAR_isif7 INCAR')
        # POSCAR --> POSCAR_REV --> POSCAR
        with open('POSCAR', 'r') as f:
            content = f.readlines()
        if 'direct' in content[7].lower():
            with open('vk_POSCAR.sh', 'w') as f:
                f.write('412 \n1')
        else:
            print('Please check the POSCAR format! The POSCAR should be in Direct format.')
            sys.exit()
        os.system('vaspkit < vk_POSCAR.sh')
        os.system('rm -f POSCAR')
        os.system('cp POSCAR_REV POSCAR')
        # Create KPOINTS (0.04)
        with open('vk_KPOINTS.sh', 'w') as f:
            f.write('102 \n2 \n0.04')
        os.system('vaspkit < vk_KPOINTS.sh')
        # Replace POTCAR
        symbols = self.utilities.get_symbols()
        with open('vk_POTCAR.sh', 'w') as f:
            f.write('104 \n' + ' \n'.join(symbols))
        os.system('rm POTCAR')
        os.system('vaspkit < vk_POTCAR.sh')
        # Add mag in INCAR
        if MAGNETIC:
            self.utilities.add_mag()
        # Create job scripts & Submit
        self.isifx_bash()
        if AUTO_SUBMIT:
            os.system(f'sbatch {JOB_NAME}.sh')

class EOS():
    def __init__(self) -> None:
        self.utilities = Utilities()

    def eos_relax_bash(self):
        '''
        * Create job scripts for eos-relax
        '''
        if MAGNETIC:
            with open('update_mag.py', 'w') as f:
                f.write('#!/usr/bin/env python \n')
                f.write('# -*-coding:utf-8 -*- \n')
                f.write('from pymatgen.io.vasp import outputs \n')
                f.write('with open("INCAR", "a") as  f: \n')
                f.write('    OUTCAR = outputs.Outcar("OUTCAR") \n')
                f.write('    mag_str = " ".join([str(m["tot"]) for m in OUTCAR.magnetization]) \n')
                f.write(r'    mag_str = "MAGMOM = " + mag_str ' + '\n')
                f.write(r'    f.write("\n# With magnetic \n")' +  '\n')
                f.write(r'    f.write("LORBIT = 11 \n")' + '\n')
                f.write(r'    f.write("ISPIN  = 2 \n")' + '\n')
                f.write('    f.write(mag_str) \n')
        with open(f'{JOB_NAME}_relax.sh', 'w') as f:
            f.write('#!/usr/bin/sh \n')
            f.write(headers)
            # Do the isif2 first
            f.write(f'time {SUBMIT_CMD} \n')
            f.write('if [ `grep -c "reached required accuracy" vasp.out` -ne "0" ] \n')
            f.write('then \n')
            # Then do the isif4 with the CONTCAR from isif2
            f.write('    rm -f INCAR \n')
            f.write('    rm -f POSCAR \n')
            f.write('    cp ../INCAR_relax_isif4 INCAR \n')
            f.write('    cp OUTCAR OUTCAR_relax_isif2 \n')
            f.write('    cp OSZICAR OSZICAR_relax_isif2 \n')
            f.write('    cp CONTCAR CONTCAR_relax_isif2 \n')
            f.write('    cp CONTCAR POSCAR \n')
            if MAGNETIC:
                f.write('    time python update_mag.py \n')
            f.write(f'    time {SUBMIT_CMD} \n')
            f.write('else \n')
            f.write('    echo "The relax-isif2 calculation is not finished yet!" \n')
            f.write('fi \n')
            f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh')
    
    def eos_static_bash(self):
        '''
        * Create job scripts for eos-static
        '''
        with open(f'{JOB_NAME}_static.sh', 'w') as f:
            f.write('#!/usr/bin/sh \n')
            f.write(headers)
            f.write(f'time {SUBMIT_CMD} \n')
            f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh')
    
    def isif3_bash(self):
        '''
        * Create job scripts for isif3
        '''
        with open('get_mid.py', 'w') as f:
            f.write('#!/usr/bin/env python \n')
            f.write('# -*-coding:utf-8 -*- \n')
            f.write('from pymatgen.core.structure import IStructure \n')
            f.write('with open("POSCAR_REV", "r") as f: \n')
            f.write('    content = f.readlines() \n')
            f.write('scaling_factor = float(content[1]) \n')
            f.write('POSCAR = IStructure.from_file("POSCAR_REV") \n')
            f.write('CONTCAR = IStructure.from_file("CONTCAR") \n')
            f.write('ratio = pow(CONTCAR.volume / POSCAR.volume, 1/3) \n')
            f.write('mid = ratio * scaling_factor \n')
            f.write(r'print("ISIF3 calculation results suggest: \n")' + '\n')
            f.write('print("The median value of the scaling factor is about %.6f" % mid) \n')
        with open(f'{JOB_NAME}.sh', 'w') as f:
            f.write('#!/usr/bin/sh \n')
            f.write(headers)
            # f.write('export I_MPI_ADJUST_REDUCE=3 \n')
            f.write(f'time {SUBMIT_CMD} \n')
            f.write('time python get_mid.py \n')
            f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh')

    def isif3(self):
        '''
        * To determine the median value of scaling factor based on ISIF3
        '''
        # Copy INCAR_isif3 --> INCAR; POSCAR --> POSCAR
        if not os.path.exists(PATH + '/isif3'):
            os.makedirs(PATH + '/isif3')
        os.system(f'cp {PATH}/INCARs/INCAR_eos_isif3 {PATH}/isif3/INCAR')
        os.system(f'cp {PATH}/POSCAR {PATH}/isif3/POSCAR')
        # Enter folder
        os.chdir(PATH + '/isif3')
        # Add PAR
        self.utilities.add_par()
        # POSCAR --> POSCAR_REV --> POSCAR
        # check if POSCAR is in direct or cartesian
        with open('POSCAR', 'r') as f:
            content = f.readlines()
        if 'direct' in content[7].lower():
            with open('vk_POSCAR.sh', 'w') as f:
                f.write('412 \n1')
        else:
            print('Please check the POSCAR format! The POSCAR should be in Direct format.')
            sys.exit()
        os.system('vaspkit < vk_POSCAR.sh')
        os.system('rm -f POSCAR')
        os.system('cp POSCAR_REV POSCAR')
        # Create KPOINTS (0.06)
        with open('vk_KPOINTS.sh', 'w') as f:
            f.write('102 \n2 \n0.06')
        os.system('vaspkit < vk_KPOINTS.sh')
        # Replace POTCAR
        symbols = self.utilities.get_symbols()
        with open('vk_POTCAR.sh', 'w') as f:
            f.write('104 \n' + ' \n'.join(symbols))
        os.system('rm POTCAR')
        os.system('vaspkit < vk_POTCAR.sh')
        # Add mag in INCAR
        if MAGNETIC:
            self.utilities.add_mag()
        # Create job scripts & Submit
        self.isif3_bash()
        if AUTO_SUBMIT:
            os.system(f'sbatch {JOB_NAME}.sh')

    def relax_pre(self):
        '''
        * Relax of EOS calculations
        * Create sub-folders and submit jobs in batch
        '''
        # Create folder
        if not os.path.exists(PATH + '/eos_relax'):
            os.makedirs(PATH + '/eos_relax')
        # Copy INCARs and POSCAR
        os.system(f'cp {PATH}/INCARs/INCAR_relax_isif2 {PATH}/eos_relax/INCAR_relax_isif2')
        os.system(f'cp {PATH}/INCARs/INCAR_relax_isif4 {PATH}/eos_relax/INCAR_relax_isif4')
        os.system(f'cp {PATH}/POSCAR {PATH}/eos_relax/POSCAR')
        # Enter the target folder
        os.chdir(PATH + '/eos_relax')
        # Add PAR
        self.utilities.add_par(file_name='INCAR_relax_isif2')
        self.utilities.add_par(file_name='INCAR_relax_isif4')
        # INCAR_relax --> INCAR
        os.system('cp INCAR_relax_isif2 INCAR')
        # POSCAR --> POSCAR_REV --> POSCAR
        with open('POSCAR', 'r') as f:
            content = f.readlines()
        if 'direct' in content[7].lower():
            with open('vk_POSCAR.sh', 'w') as f:
                f.write('412 \n1')
        else:
            print('Please check the POSCAR format! The POSCAR should be in Direct format.')
            sys.exit()
        os.system('vaspkit < vk_POSCAR.sh')
        os.system('rm -f POSCAR')
        os.system('cp POSCAR_REV POSCAR')
        # Create KPOINTS --> KPOINTS_static (0.02); KPOINTS_relax (0.04)
        with open('vk_KPOINTS_relax.sh', 'w') as f:
            f.write('102 \n2 \n0.04')
        os.system('vaspkit < vk_KPOINTS_relax.sh')
        os.system('cp KPOINTS KPOINTS_relax')
        # Replace POTCAR
        symbols = self.utilities.get_symbols()
        with open('vk_POTCAR.sh', 'w') as f:
            f.write('104 \n' + ' \n'.join(symbols))
        os.system('rm POTCAR')
        os.system('vaspkit < vk_POTCAR.sh')
        # Add mag in INCAR if MAGNETIC == True
        if MAGNETIC:
            self.utilities.add_mag()
        # Create VPKIT.in file
        with open('VPKIT.in', 'w') as f:
            f.write(f'1 \n{EOS_TYPE} \n123 321 100 \n{len(SCALING_FACTORS.split())} \n{SCALING_FACTORS}')
        # Create vk.sh to call vaspkit
        with open('vk_eos_prep.sh', 'w') as f:
            f.write('205 \n')
        os.system('vaspkit < vk_eos_prep.sh')
        os.system('rm -f *.sh')
        # Submit jobs
        dirs = os.listdir()
        for d in dirs:
            if 'lattice' in d:
                os.chdir(PATH + f'/eos_relax/{d}')
                self.eos_relax_bash()
                if AUTO_SUBMIT:
                    os.system(f'sbatch {JOB_NAME}_relax.sh')
    
    def static_pre(self):
        '''
        * Pre-process of EOS calculations
        * Create sub-folders and submit jobs in batch
        '''
        # Create folder
        if not os.path.exists(PATH + '/eos_static'):
            os.makedirs(PATH + '/eos_static')
        # Copy INCARs and POSCAR
        os.system(f'cp {PATH}/INCARs/INCAR_static_isif{ISIF} {PATH}/eos_static/INCAR_static')
        os.system(f'cp {PATH}/POSCAR {PATH}/eos_static/POSCAR')
        # Enter the target folder
        os.chdir(PATH + '/eos_static')
        # Add PAR
        self.utilities.add_par(file_name='INCAR_static')
        # INCAR_static --> INCAR
        os.system('cp INCAR_static INCAR')
        # Create KPOINTS --> KPOINTS_static (0.02); KPOINTS_relax (0.04)
        with open('vk_KPOINTS_static.sh', 'w') as f:
            f.write('102 \n2 \n0.02')
        os.system('vaspkit < vk_KPOINTS_static.sh')
        os.system('cp KPOINTS KPOINTS_static')
        # Replace POTCAR
        symbols = self.utilities.get_symbols()
        with open('vk_POTCAR.sh', 'w') as f:
            f.write('104 \n' + ' \n'.join(symbols))
        os.system('rm POTCAR')
        os.system('vaspkit < vk_POTCAR.sh')
        # Create VPKIT.in file
        with open('VPKIT.in', 'w') as f:
            f.write(f'1 \n{EOS_TYPE} \n123 321 100 \n{len(SCALING_FACTORS.split())} \n{SCALING_FACTORS}')
        # Create vk.sh to call vaspkit
        with open('vk_eos_prep.sh', 'w') as f:
            f.write('205 \n')
        os.system('vaspkit < vk_eos_prep.sh')
        os.system('rm -f *.sh')
        # Submit jobs
        dirs = os.listdir()
        for d in dirs:
            if 'lattice' in d:
                # Replace POSCAR with certain CONTCAR in eos_relax
                os.chdir(PATH)
                if ISIF == '2':
                    os.system(f'cp {PATH}/eos_relax/{d}/CONTCAR_relax_isif{ISIF} {PATH}/eos_static/{d}/POSCAR')
                elif ISIF == '4':
                    os.system(f'cp {PATH}/eos_relax/{d}/CONTCAR {PATH}/eos_static/{d}/POSCAR')
                # Update the MAGMOM in INCAR
                if MAGNETIC:
                    OUTCAR = outputs.Outcar(f'{PATH}/eos_relax/{d}/OUTCAR')
                    mag_str = ' '.join([str(m['tot']) for m in OUTCAR.magnetization])
                    mag_str = f'MAGMOM = {mag_str}'
                    with open(f'{PATH}/eos_static/{d}/INCAR', 'r') as f:
                        content = f.readlines()
                    with open(f'{PATH}/eos_static/{d}/INCAR_{d}', 'w') as f:
                        f.writelines(content)
                        f.write('\n\n# With magnetic \n')
                        f.write('LORBIT = 11 \n')
                        f.write('ISPIN = 2 \n')
                        f.write(mag_str)
                    os.system(f'rm {PATH}/eos_static/{d}/INCAR')
                    os.system(f'cp {PATH}/eos_static/{d}/INCAR_{d} {PATH}/eos_static/{d}/INCAR')
                    os.system(f'rm {PATH}/eos_static/{d}/INCAR_{d}')
                # Create bash file and submit jobs
                os.chdir(PATH + f'/eos_static/{d}')
                self.eos_static_bash()
                if AUTO_SUBMIT:
                    os.system(f'sbatch {JOB_NAME}_static.sh')
    
    def relax_post(self):
        # get total free energy vs volume from OUTCAR
        os.chdir(PATH + '/eos_relax')
        with open(f'{PATH}/eos_relax/ENERGY_VOLUME_DFT_ISIF2.dat', 'w') as f:
            f.write('#  Volume (A^3/atom)       Energy (eV/atom)')
        with open(f'{PATH}/eos_relax/ENERGY_VOLUME_DFT_ISIF4.dat', 'w') as f:
            f.write('#  Volume (A^3/atom)       Energy (eV/atom)')
        with open(f'{PATH}/eos_relax/STRUCTURE_ISIF2.dat', 'w') as f:
            f.write('#  Structures information for ISIF = 2')
        with open(f'{PATH}/eos_relax/STRUCTURE_ISIF4.dat', 'w') as f:
            f.write('#  Structures information for ISIF = 4')
        POSCAR = IStructure.from_file(f'{PATH}/eos_relax/POSCAR')
        with open(f'{PATH}/eos_relax/STRUCTURE_ORI.dat', 'w') as f:
            f.write('#  Structures information for original structure')
            f.write('\n' + str(POSCAR) + '\n')
        dirs = sorted(os.listdir())
        for d in dirs:
            if 'lattice' in d:
                if d[-5:-1] in SCALING_FACTORS.split() or d[-5:] in SCALING_FACTORS.split():
                    CONTCAR_ISIF2 = IStructure.from_file(f'{PATH}/eos_relax/{d}/CONTCAR_relax_isif2')
                    CONTCAR_ISIF4 = IStructure.from_file(f'{PATH}/eos_relax/{d}/CONTCAR')
                    natoms = CONTCAR_ISIF2.num_sites
                    OSZICAR_ISIF2 = outputs.Oszicar(f'{PATH}/eos_relax/{d}/OSZICAR_relax_isif2')
                    OSZICAR_ISIF4 = outputs.Oszicar(f'{PATH}/eos_relax/{d}/OSZICAR')
                    volume_isif2, energy_isif2 = CONTCAR_ISIF2.volume / natoms, OSZICAR_ISIF2.final_energy / natoms
                    volume_isif4, energy_isif4 = CONTCAR_ISIF4.volume / natoms, OSZICAR_ISIF4.final_energy / natoms
                    with open(f'{PATH}/eos_relax/ENERGY_VOLUME_DFT_ISIF2.dat', 'a') as f:
                        f.write(f'\n{volume_isif2} {energy_isif2}')
                    with open(f'{PATH}/eos_relax/ENERGY_VOLUME_DFT_ISIF4.dat', 'a') as f:
                        f.write(f'\n{volume_isif4} {energy_isif4}')
                    with open(f'{PATH}/eos_relax/STRUCTURE_ISIF2.dat', 'a') as f:
                        f.write('\n' + str(CONTCAR_ISIF2) + '\n')
                    with open(f'{PATH}/eos_relax/STRUCTURE_ISIF4.dat', 'a') as f:
                        f.write('\n' + str(CONTCAR_ISIF4) + '\n')
        self.relax_post_visualize()

    def static_post(self):
        '''
        * Post-process of static calculations
        * Do the EOS fitting and get POSCAR_EOS
        '''
        os.chdir(PATH + '/eos_static')
        os.system('rm -f *.dat POSCAR_EOS')
        # Get the max volume and min volume
        min_volume, max_volume = 100000, 0
        dirs = os.listdir()
        for d in dirs:
            if 'lattice' in d:
                if d[-5:-1] in SCALING_FACTORS.split() or d[-5:] in SCALING_FACTORS.split():
                    CONTCAR = IStructure.from_file(f'{PATH}/eos_static/{d}/CONTCAR')
                    volume = CONTCAR.volume
                    if volume > max_volume:
                        max_volume = volume
                    if volume < min_volume:
                        min_volume = volume
        min_volume, max_volume = min_volume * 0.97, max_volume * 1.03
        with open('VPKIT.in', 'w') as f:
            f.write(f'2 \n{EOS_TYPE} \n{min_volume} {max_volume} 100 \n{len(SCALING_FACTORS.split())} \n{SCALING_FACTORS}')
        with open('vk_eos_post.sh', 'w') as f:
            f.write('205 \n')
        os.system('vaspkit < vk_eos_post.sh > vaspkit.out')
        for fpath, dirname, fname in os.walk(PATH + '/eos'):
            for f in fname:
                if ('.dat' in f or '_EOS' in f) and (len(dirname) == 0):
                    path = os.path.join(fpath, f)
                    os.system(f'cp -rf {path} {f}')
        os.chdir(PATH + '/eos_static')
        with open('POSCAR_EOS', 'r+') as f:
            content = f.readlines()
            temp_str = (content[1].split('!'))[0].strip()
            content[1] = temp_str + ' \n'
        with open('POSCAR_EOS', 'w') as f:
            f.writelines(content)
        os.system('cp POSCAR_EOS ../POSCAR_EOS')
        os.system('rm -f *.sh')
        self.static_post_visualize()

    def eos_func(self, volume, a, b, c, d):
        energy = a + b * volume**(-2/3) + c * volume**(-4/3) + d * volume**(-2)
        return energy

    def relax_post_visualize(self):
        # 1 eV = 96.485 kJ/mol
        os.chdir(PATH + '/eos_relax')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot()
        data_isif2 = pd.read_table('ENERGY_VOLUME_DFT_ISIF2.dat', sep='\s+')
        data_isif4 = pd.read_table('ENERGY_VOLUME_DFT_ISIF4.dat', sep='\s+')
        ax.scatter(data_isif2.iloc[:, 0], data_isif2.iloc[:, 1], label='DFT (ISIF=2)')
        ax.scatter(data_isif4.iloc[:, 0], data_isif4.iloc[:, 1], label='DFT (ISIF=4)')
        
        
        volumes_isif2, energies_isif2 = data_isif2.iloc[:, 0], data_isif2.iloc[:, 1]
        t_isif2, v_isif2 = curve_fit(self.eos_func, volumes_isif2, energies_isif2)
        volume_fit_isif2 = np.linspace(min(volumes_isif2)*0.97, max(volumes_isif2)*1.03, 100)
        energy_fit_isif2 = self.eos_func(volume_fit_isif2, t_isif2[0], t_isif2[1], t_isif2[2], t_isif2[3]).tolist()
        ax.plot(volume_fit_isif2, energy_fit_isif2, label='FIT (ISIF=2)')
        
        volumes_isif4, energies_isif4 = data_isif4.iloc[:, 0], data_isif4.iloc[:, 1]
        t_isif4, v_isif4 = curve_fit(self.eos_func, volumes_isif4, energies_isif4)
        volume_fit_isif4 = np.linspace(min(volumes_isif4)*0.97, max(volumes_isif4)*1.03, 100)
        energy_fit_isif4 = self.eos_func(volume_fit_isif4, t_isif4[0], t_isif4[1], t_isif4[2], t_isif4[3]).tolist()
        ax.plot(volume_fit_isif4, energy_fit_isif4, label='FIT (ISIF=2)')

        ax.set_xlabel('Volume (A^3/atom)')
        ax.set_ylabel('Energy (eV/atom)')
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('eos_relax.png', dpi=330)

    def static_post_visualize(self):
        '''
        * Visualize the EOS results
        * Volume vs. Energy
        * Volume vs. Pressure
        * Pressure vs. Enthalpy
        '''
        os.chdir(PATH + '/eos_static')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        data_evd = pd.read_table('ENERGY_VOLUME_DFT.dat', sep='\s+')
        data_evf = pd.read_table('ENERGY_VOLUME_FIT.dat', sep='\s+')
        ax[0].scatter(data_evd.iloc[:, 0], data_evd.iloc[:, 1], label='DFT', zorder=2)
        ax[0].plot(data_evf.iloc[:, 0], data_evf.iloc[:, 1], c='r', label='FIT', zorder=1)
        ax[0].set_xlabel('Volume (A^3/atom)')
        ax[0].set_ylabel('Energy (eV/atom)')
        ax[0].legend(frameon=False)
        data_pvd = pd.read_table('PRESSURE_VOLUME_DFT.dat', sep='\s+')
        data_pvf = pd.read_table('PRESSURE_VOLUME_FIT.dat', sep='\s+')
        ax[1].scatter(data_pvd.iloc[:, 0], data_pvd.iloc[:, 1], label='DFT', zorder=2)
        ax[1].plot(data_pvf.iloc[:, 0], data_pvf.iloc[:, 1], c='r', label='FIT', zorder=1)
        ax[1].set_xlabel('Volume (A^3/atom)')
        ax[1].set_ylabel('Pressure (GPa)')
        ax[1].legend(frameon=False)
        data_hvf = pd.read_table('ENTHALPY_VOLUME_FIT.dat', sep='\s+')
        ax[2].plot(data_hvf.iloc[:, 0], data_hvf.iloc[:, 1], c='r', label='FIT')
        ax[2].set_xlabel('Pressure (GPa)')
        ax[2].set_ylabel('Enthalpy (eV/atom)')
        ax[2].legend(frameon=False)
        plt.tight_layout()
        plt.savefig('eos_static.png', dpi=330)

class Relax():
    def __init__(self) -> None:
        self.utilities = Utilities()

    def bash(self):
        '''
        * Create job scripts for fully relaxation
        '''
        with open(f'{JOB_NAME}.sh', 'w') as f:
            f.write('#!/usr/bin/sh \n')
            f.write(headers)
            f.write(f'time {SUBMIT_CMD} \n')
            f.write('cp CONTCAR ../POSCAR_REL \n')
            f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh')
    
    def fully_relaxed(self):
        '''
        * Fully relax the POSCAR based on EOS
        * Get POSCAR_REL
        '''
        # Copy INCAR_frelax --> INCAR; POSCAR_EOS --> POSCAR
        if not os.path.exists(PATH + '/fully_relaxed'):
            os.makedirs(PATH + '/fully_relaxed')
        os.system(f'cp {PATH}/INCARs/INCAR_relax_isif{ISIF} {PATH}/fully_relaxed/INCAR')
        os.system(f'cp {PATH}/POSCAR_EOS {PATH}/fully_relaxed/POSCAR')
        # Enter folder
        os.chdir(PATH + '/fully_relaxed')
        # Add PAR
        self.utilities.add_par()
        # Create KPOINTS (0.02)
        with open('vk_KPOINTS.sh', 'w') as f:
            f.write('102 \n2 \n0.02')
        os.system('vaspkit < vk_KPOINTS.sh')
        # Replace POTCAR
        symbols = self.utilities.get_symbols()
        with open('vk_POTCAR.sh', 'w') as f:
            f.write('104 \n' + ' \n'.join(symbols))
        os.system('rm POTCAR')
        os.system('vaspkit < vk_POTCAR.sh')
        # Add mag in INCAR
        if MAGNETIC:
            self.utilities.add_mag()
        # Submit job
        self.bash()
        if AUTO_SUBMIT:
            os.system(f'sbatch {JOB_NAME}.sh')

class Elastic():
    def __init__(self) -> None:
        self.utilities = Utilities()
    
    def bash(self):
        '''
        * Create job scripts for elastic calculation
        '''
        with open(f'{JOB_NAME}.sh', 'w') as f:
            f.write('#!/usr/bin/sh \n')
            f.write(headers)
            f.write(f'time {SUBMIT_CMD} \n')
            f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh')
    
    def cubic_filter(self, m):
        c11 = (m[0][0] + m[1][1] + m[2][2]) / 3
        c12 = (m[0][1] + m[0][2] + m[1][0] + m[1][2] + m[2][0] + m[2][1]) / 6
        c44 = (m[3][3] + m[4][4] + m[5][5]) / 3
        new = np.zeros((6, 6)).tolist()
        new[0][0] = new[1][1] = new[2][2] = c11
        new[0][1] = new[0][2] = new[1][0] = new[1][2] = new[2][0] = new[2][1] = c12
        new[3][3] = new[4][4] = new[5][5] = c44
        return new
    
    def hexagonal_filter(self, m):
        c11 = (m[0][0] + m[1][1]) / 2
        c12 = (m[0][1] + m[1][0]) / 2
        c13 = (m[0][2] + m[1][2] + m[2][0] + m[2][1]) / 4
        c33 = m[2][2]
        c44 = (m[3][3] + m[4][4]) / 2
        new = np.zeros((6, 6)).tolist()
        new[0][0] = new[1][1] = c11
        new[0][1] = new[1][0] = c12
        new[0][2] = new[1][2] = new[2][0] = new[2][1] = c13
        new[2][2] = c33
        new[3][3] = new[4][4] = c44
        new[5][5] = (((c11 - c12) / 2) + m[5][5]) / 2
        return new

    def trigonal_i_filter(self, m):
        c11 = (m[0][0] + m[1][1]) / 2
        c12 = (m[0][1] + m[1][0]) / 2
        c13 = (m[0][2] + m[1][2] + m[2][0] + m[2][1]) / 4
        c14 = (m[0][3] - m[1][3] + m[3][0] - m[3][1] + m[4][5] + m[5][4]) / 6
        c33 = m[2][2]
        c44 = (m[3][3] + m[4][4]) / 2
        new = np.zeros((6, 6)).tolist()
        new[0][0] = new[1][1] = c11
        new[0][1] = new[1][0] = c12
        new[0][2] = new[1][2] = new[2][0] = new[2][1] = c13
        new[0][3] = new[3][0] = new[4][5] = new[5][4] = c14
        new[1][3] = new[3][1] = -c14
        new[2][2] = c33
        new[3][3] = new[4][4] = c44
        new[5][5] = (((c11 - c12) / 2) + m[5][5]) / 2
        return new
    
    def trigonal_ii_filter(self, m):
        c11 = (m[0][0] + m[1][1]) / 2
        c12 = (m[0][1] + m[1][0]) / 2
        c13 = (m[0][2] + m[1][2] + m[2][0] + m[2][1]) / 4
        c14 = (m[0][3] - m[1][3] + m[3][0] - m[3][1] + m[4][5] + m[5][4]) / 6
        c15 = (m[0][4] - m[1][4] - m[3][5] + m[4][0] - m[4][1] - m[5][3]) / 6
        c33 = m[2][2]
        c44 = (m[3][3] + m[4][4]) / 2
        new = np.zeros((6, 6)).tolist()
        new[0][0] = new[1][1] = c11
        new[0][1] = new[1][0] = c12
        new[0][2] = new[1][2] = new[2][0] = new[2][1] = c13
        new[0][3] = new[3][0] = new[4][5] = new[5][4] = c14
        new[1][3] = new[3][1] = -c14
        new[0][4] = new[4][0] = c15
        new[1][4] = new[3][5] = new[4][1] = new[5][3] = -c15
        new[2][2] = c33
        new[3][3] = new[4][4] = c44
        new[5][5] = (((c11 - c12) / 2) + m[5][5]) / 2
        return new

    def tetragonal_i_filter(self, m):
        c11 = (m[0][0] + m[1][1]) / 2
        c12 = (m[0][1] + m[1][0]) / 2
        c13 = (m[0][2] + m[1][2] + m[2][0] + m[2][1]) / 4
        c33 = m[2][2]
        c44 = (m[3][3] + m[4][4]) / 2
        c66 = m[5][5]
        new = np.zeros((6, 6)).tolist()
        new[0][0] = new[1][1] = c11
        new[0][1] = new[1][0] = c12
        new[0][2] = new[1][2] = new[2][0] = new[2][1] = c13
        new[2][2] = c33
        new[3][3] = new[4][4] = c44
        new[5][5] = c66
        return new
    
    def tetragonal_ii_filter(self, m):
        c11 = (m[0][0] + m[1][1]) / 2
        c12 = (m[0][1] + m[1][0]) / 2
        c13 = (m[0][2] + m[1][2] + m[2][0] + m[2][1]) / 4
        c16 = (m[0][5] - m[1][5] + m[5][0] - m[5][1]) / 4
        c33 = m[2][2]
        c44 = (m[3][3] + m[4][4]) / 2
        c66 = m[5][5]
        new = np.zeros((6, 6)).tolist()
        new[0][0] = new[1][1] = c11
        new[0][1] = new[1][0] = c12
        new[0][2] = new[1][2] = new[2][0] = new[2][1] = c13
        new[0][5] = new[5][0] = c16
        new[1][5] = new[5][1] = -c16
        new[2][2] = c33
        new[3][3] = new[4][4] = c44
        new[5][5] = c66
        return new
    
    def orthorhombic_filter(self, m):
        c11 = m[0][0]
        c12 = (m[0][1] + m[1][0]) / 2
        c13 = (m[0][2] + m[2][0]) / 2
        c22 = m[1][1]
        c23 = (m[1][2] + m[2][1]) / 2
        c33 = m[2][2]
        c44 = m[3][3]
        c55 = m[4][4]
        c66 = m[5][5]
        new = np.zeros((6, 6)).tolist()
        new[0][0] = c11
        new[0][1] = new[1][0] = c12
        new[0][2] = new[2][0] = c13
        new[1][1] = c22
        new[1][2] = new[2][1] = c23
        new[2][2] = c33
        new[3][3] = c44
        new[4][4] = c55
        new[5][5] = c66
        return new

    def monoclinic_filter(self, m):
        c11 = m[0][0]
        c12 = (m[0][1] + m[1][0]) / 2
        c13 = (m[0][2] + m[2][0]) / 2
        c15 = (m[0][4] + m[4][0]) / 2
        c22 = m[1][1]
        c23 = (m[1][2] + m[2][1]) / 2
        c25 = (m[1][4] + m[4][1]) / 2
        c33 = m[2][2]
        c35 = (m[2][4] + m[4][2]) / 2
        c44 = m[3][3]
        c46 = (m[3][5] + m[5][3]) / 2
        c55 = m[4][4]
        c66 = m[5][5]
        new = np.zeros((6, 6)).tolist()
        new[0][0] = c11
        new[0][1] = new[1][0] = c12
        new[0][2] = new[2][0] = c13
        new[0][4] = new[4][0] = c15
        new[1][1] = c22
        new[1][2] = new[2][1] = c23
        new[1][4] = new[4][1] = c25
        new[2][2] = c33
        new[2][4] = new[4][2] = c35
        new[3][3] = c44
        new[3][5] = new[5][3] = c46
        new[4][4] = c55
        new[5][5] = c66
        return new
    
    def triclinic_filter(self, m):
        c11 = m[0][0]
        c12 = (m[0][1] + m[1][0]) / 2
        c13 = (m[0][2] + m[2][0]) / 2
        c14 = (m[0][3] + m[3][0]) / 2
        c15 = (m[0][4] + m[4][0]) / 2
        c16 = (m[0][5] + m[5][0]) / 2
        c22 = m[1][1]
        c23 = (m[1][2] + m[2][1]) / 2
        c24 = (m[1][3] + m[3][1]) / 2
        c25 = (m[1][4] + m[4][1]) / 2
        c26 = (m[1][5] + m[5][1]) / 2
        c33 = m[2][2]
        c34 = (m[2][3] + m[3][2]) / 2
        c35 = (m[2][4] + m[4][2]) / 2
        c36 = (m[2][5] + m[5][2]) / 2
        c44 = m[3][3]
        c45 = (m[3][4] + m[4][3]) / 2
        c46 = (m[3][5] + m[5][3]) / 2
        c55 = m[4][4]
        c56 = (m[4][5] + m[5][4]) / 2
        c66 = m[5][5]
        new = np.zeros((6, 6)).tolist()
        new[0][0] = c11
        new[0][1] = new[1][0] = c12
        new[0][2] = new[2][0] = c13
        new[0][3] = new[3][0] = c14
        new[0][4] = new[4][0] = c15
        new[0][5] = new[5][0] = c16
        new[1][1] = c22
        new[1][2] = new[2][1] = c23
        new[1][3] = new[3][1] = c24
        new[1][4] = new[4][1] = c25
        new[1][5] = new[5][1] = c26
        new[2][2] = c33
        new[2][3] = new[3][2] = c34
        new[2][4] = new[4][2] = c35
        new[2][5] = new[5][2] = c36
        new[3][3] = c44
        new[3][4] = new[4][3] = c45
        new[3][5] = new[5][3] = c46
        new[4][4] = c55
        new[4][5] = new[5][4] = c56
        new[5][5] = c66
        return new

    def pre_process(self):
        '''
        * Pre-process of elastic calculations
        * Create sub-folders and submit jobs in batch
        '''
        # Create folder
        if not os.path.exists(PATH + '/elastic'):
            os.makedirs(PATH + '/elastic')
        # Copy INCARs and POSCAR
        os.system(f'cp {PATH}/INCARs/INCAR_elastic {PATH}/elastic/INCAR')
        os.system(f'cp {PATH}/POSCAR_REL {PATH}/elastic/POSCAR')
        # Enter the target folder
        os.chdir(PATH + '/elastic')
        # Add PAR
        self.utilities.add_par()
        # Create standard conventional cell
        with open('vk_POSCAR.sh', 'w') as f:
            f.write('603 \n')
        os.system('vaspkit < vk_POSCAR.sh')
        os.system('rm -f POSCAR')
        os.system('cp CONVCELL.vasp POSCAR')
        # Create KPOINTS (0.02)
        with open('vk_KPOINT.sh', 'w') as f:
            f.write('102 \n2 \n0.02')
        os.system('vaspkit < vk_KPOINT.sh')
        # Replace POTCAR
        symbols = self.utilities.get_symbols()
        with open('vk_POTCAR.sh', 'w') as f:
            f.write('104 \n' + ' \n'.join(symbols))
        os.system('rm POTCAR')
        os.system('vaspkit < vk_POTCAR.sh')
        # Add mag in INCAR if MAGNETIC == True
        if MAGNETIC:
            self.utilities.add_mag()
        # Elastic methods: vaspkit
        if ELASTIC_METHOD.lower() != 'sqs':
            # Create VPKIT.in file
            with open('VPKIT.in', 'w') as f:
                f.write(f'1 \n3D \n{len(STRAINS.split())} \n{STRAINS}')
            # Create SYMMETRY.in file
            os.system('rm -f SYMMETRY')
            with open('SYMMETRY.in', 'w') as f:
                f.write(f'# Read the symmetry of structure from the SYMMETRY.in file if it exists. \n  {SPACE_GROUP_NUM}          # Space group number of the input structure')
            # Create vk.sh to call vaspkit
            with open('vk_els_prep.sh', 'w') as f:
                f.write(f'{ELASTIC_METHOD_CODE} \n')
            os.system('vaspkit < vk_els_prep.sh')
            os.system('rm -f *.sh')
        # Elastic methods: SQS
        else:
            # To float
            xx = [float(f'{float(i):.3f}') for i in STRAINS_XX.split()]
            yy = [float(f'{float(i):.3f}') for i in STRAINS_YY.split()]
            zz = [float(f'{float(i):.3f}') for i in STRAINS_ZZ.split()]
            xy = [float(f'{float(i):.3f}') for i in STRAINS_XY.split()]
            yz = [float(f'{float(i):.3f}') for i in STRAINS_YZ.split()]
            xz = [float(f'{float(i):.3f}') for i in STRAINS_ZX.split()]
            # Create deformation matrixes
            D_xyz = {f'00_strain_xyz': [[1.000, 0.000, 0.000], [0.000, 1.000, 0.000], [0.000, 0.000, 1.000]]}
            D_xx0 = {f'01_strain_xx_{float(xx[0]):.3f}': [[1.000+xx[0], 0.000, 0.000], [0.000, 1.000, 0.000], [0.000, 0.000, 1.000]]}
            D_yy0 = {f'03_strain_yy_{float(yy[0]):.3f}': [[1.000, 0.000, 0.000], [0.000, 1.000+yy[0], 0.000], [0.000, 0.000, 1.000]]}
            D_zz0 = {f'05_strain_zz_{float(zz[0]):.3f}': [[1.000, 0.000, 0.000], [0.000, 1.000, 0.000], [0.000, 0.000, 1.000+zz[0]]]}
            D_xy0 = {f'07_strain_xy_{float(xy[0]):.3f}': [[1.000, xy[0], 0.000], [xy[0], 1.000, 0.000], [0.000, 0.000, 1.000]]}
            D_yz0 = {f'09_strain_yz_{float(yz[0]):.3f}': [[1.000, 0.000, 0.000], [0.000, 1.000, yz[0]], [0.000, yz[0], 1.000]]}
            D_xz0 = {f'11_strain_xz_{float(xz[0]):.3f}': [[1.000, 0.000, xz[0]], [0.000, 1.000, 0.000], [xz[0], 0.000, 1.000]]}
            D_xx1 = {f'02_strain_xx_{float(xx[1]):.3f}': [[1.000+xx[1], 0.000, 0.000], [0.000, 1.000, 0.000], [0.000, 0.000, 1.000]]}
            D_yy1 = {f'04_strain_yy_{float(yy[1]):.3f}': [[1.000, 0.000, 0.000], [0.000, 1.000+yy[1], 0.000], [0.000, 0.000, 1.000]]}
            D_zz1 = {f'06_strain_zz_{float(zz[1]):.3f}': [[1.000, 0.000, 0.000], [0.000, 1.000, 0.000], [0.000, 0.000, 1.000+zz[1]]]}
            D_xy1 = {f'08_strain_xy_{float(xy[1]):.3f}': [[1.000, xy[1], 0.000], [xy[1], 1.000, 0.000], [0.000, 0.000, 1.000]]}
            D_yz1 = {f'10_strain_yz_{float(yz[1]):.3f}': [[1.000, 0.000, 0.000], [0.000, 1.000, yz[1]], [0.000, yz[1], 1.000]]}
            D_xz1 = {f'12_strain_xz_{float(xz[1]):.3f}': [[1.000, 0.000, xz[1]], [0.000, 1.000, 0.000], [xz[1], 0.000, 1.000]]}
            D_all = [D_xyz, D_xx0, D_yy0, D_zz0, D_xy0, D_yz0, D_xz0, D_xx1, D_yy1, D_zz1, D_xy1, D_yz1, D_xz1]
            # Create sub-folders and POSCARs based on each deformation matrix
            POSCAR_raw = IStructure.from_file('POSCAR')
            lattice_matrix_raw = POSCAR_raw.lattice.matrix
            for D in D_all:
                for id, D_matrix in D.items():
                    D_matrix = np.array(D_matrix)
                    # Create sub-folder and copy INCAR, KPOINTS, and POTCAR into it
                    os.system(f'mkdir {id}')
                    os.system(f'cp INCAR KPOINTS POTCAR {id}')
                    # Create POSCAR
                    lattice_matrix_new = np.dot(lattice_matrix_raw, D_matrix)
                    POSCAR_new = IStructure(lattice_matrix_new, POSCAR_raw.species, POSCAR_raw.frac_coords)
                    POSCAR_new.to('poscar', f'{id}/POSCAR')
        os.system('rm -f *.sh')
        # Submit jobs
        paths = []
        for fpath, dirname, fname in os.walk(PATH + '/elastic'):
            for dir in dirname:
                path = os.path.join(fpath, dir)
                if 'strain' in path:
                    paths.append(path)
        for p in paths:
            os.chdir(p)
            self.bash()
            if AUTO_SUBMIT:
                os.system(f'sbatch {JOB_NAME}.sh')

    def post_process(self):
        '''
        * Post-process of elastic calculations
        * Get the elastic tensor
        '''
        os.chdir(PATH + '/elastic')
        # Elastic methods: vaspkit
        if ELASTIC_METHOD.lower() != 'sqs':
            with open('VPKIT.in', 'w') as f:
                f.write(f'2 \n3D \n{len(STRAINS.split())} \n{STRAINS}')
            with open('vk_els_post.sh', 'w') as f:
                f.write(f'{ELASTIC_METHOD_CODE} \n')
            os.system('vaspkit < vk_els_post.sh > vaspkit.out')
            os.system('cp ELASTIC_TENSOR ../ELASTIC_TENSOR')
            os.system('rm -f *.sh')
        # Elastic methods: SQS
        else:
            dirs = os.listdir()
            sub_paths = []
            for d in dirs:
                if 'strain' in d:
                    sub_paths.append(d)
            sub_paths.sort()
            data_ss, data_ec_raw, data_ec = [], [], []
            for s in sub_paths:
                sub_path = f'{PATH}/elastic/{s}'
                os.chdir(sub_path)
                # print(sub_path)
                with open('OUTCAR', 'r') as f:
                    content = f.readlines()
                for c in content:
                    match = re.search('in kB', c)
                    if match:
                        target = c
                        # print(target)
                target = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", target)
                target = [float(t) for t in target]
                data_ss.append(target)
            for i, d in enumerate(data_ss):
                if i % 2 != 0:
                    temp = (np.array(data_ss[i]) - np.array(data_ss[i+1])) / 0.2
                    data_ec_raw.append(temp.tolist())
            if CRYSTAL_SYSTEM.lower() == 'triclinic':
                data_ec = self.triclinic_filter(data_ec_raw)
            elif CRYSTAL_SYSTEM.lower() == 'monoclinic':
                data_ec = self.monoclinic_filter(data_ec_raw)
            elif CRYSTAL_SYSTEM.lower() == 'orthorhombic':
                data_ec = self.orthorhombic_filter(data_ec_raw)
            elif CRYSTAL_SYSTEM.lower() == 'tetragonal-i':
                data_ec = self.tetragonal_i_filter(data_ec_raw)
            elif CRYSTAL_SYSTEM.lower() == 'tetragonal-ii':
                data_ec = self.tetragonal_ii_filter(data_ec_raw)
            elif CRYSTAL_SYSTEM.lower() == 'trigonal-i':
                data_ec = self.trigonal_i_filter(data_ec_raw)
            elif CRYSTAL_SYSTEM.lower() == 'trigonal-ii':
                data_ec = self.trigonal_ii_filter(data_ec_raw)
            elif CRYSTAL_SYSTEM.lower() == 'hexagonal':
                data_ec = self.hexagonal_filter(data_ec_raw)
            elif CRYSTAL_SYSTEM.lower() == 'cubic':
                data_ec = self.cubic_filter(data_ec_raw)
            else:
                print(f'ERROR: The selected mode ({CRYSTAL_SYSTEM}) is not supported!')
            os.chdir(PATH + '/elastic')
            pd.DataFrame(data_ss).to_csv('outcar_kB.dat', sep=' ', header=False, index=False)
            pd.DataFrame(data_ec_raw).to_csv('elastic_raw.dat', sep=' ', header=False, index=False)
            pd.DataFrame(data_ec).to_csv('ELASTIC_TENSOR.in', sep=' ', header=False, index=False)
            with open('ELASTIC_TENSOR.in', 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write('# comment line (in GPa)\n' + content)
            with open('vk_els_post.sh', 'w') as f:
                f.write('202 \n')
            os.system('vaspkit < vk_els_post.sh')
            os.system('cp ELASTIC_TENSOR ../ELASTIC_TENSOR')
            os.system('rm -f *.sh')

class Ionic_conductivity():
    def __init__(self) -> None:
        self.utilities = Utilities()

    def bash(self):
        '''
        * Create job scripts for AIMD
        '''
        with open(f'{JOB_NAME}.sh', 'w') as f:
            f.write('#!/usr/bin/sh \n')
            f.write(headers)
            f.write(f'time {SUBMIT_CMD} \n')
            f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh')
    
    def aimd_pre(self):
        '''
        * Pre-process of AIMD calculations
        * Create sub-folders for AIMD and submit the jobs in batch
        '''
        # Create folder
        if not os.path.exists(PATH + '/ionic'):
            os.makedirs(PATH + '/ionic')
        # Copy INCARs and POSCAR
        os.system(f'cp {PATH}/INCARs/INCAR_aimd {PATH}/ionic/INCAR')
        os.system(f'cp {PATH}/POSCAR_REL {PATH}/ionic/POSCAR_REL')
        # Enter the target folder
        os.chdir(PATH + '/ionic')
        # Add PAR
        self.utilities.add_par()
        # Cell expansion
        for atom in range(int(MD_ATOMS)):
            temp_structrue = Structure.from_file('POSCAR_REL')
            abc = temp_structrue.lattice.abc
            temp_scaling_matrix = [round(max(abc)/i) * (atom+1) for i in abc]
            temp_structrue.make_supercell(scaling_matrix=temp_scaling_matrix, to_unit_cell=False)
            if temp_structrue.num_sites >= int(MD_ATOMS):
                structure = Structure.from_file('POSCAR_REL')
                scaling_matrix = [round(max(abc)/i) * atom for i in abc]
                structure.make_supercell(scaling_matrix=scaling_matrix, to_unit_cell=False)
                break
        structure.to(fmt='POSCAR', filename='POSCAR')
        # Create KPOINTS (0-single Gamma)
        with open('vk_KPOINT.sh', 'w') as f:
            f.write('102 \n2 \n0')
        os.system('vaspkit < vk_KPOINT.sh')
        # Replace POTCAR
        symbols = self.utilities.get_symbols()
        with open('vk_POTCAR.sh', 'w') as f:
            f.write('104 \n' + ' \n'.join(symbols))
        os.system('rm POTCAR')
        os.system('vaspkit < vk_POTCAR.sh')
        os.system('rm -f *.sh')
        # Add mag in INCAR if MAGNETIC == True
        if MAGNETIC:
            self.utilities.add_mag()
        # Create sub-folders, update INCAR, and submit jobs
        temps = MD_TEMPERATURES.split()
        for t in temps:
            os.makedirs(PATH + f'/ionic/temp_{t}')
            os.chdir(PATH + f'/ionic/temp_{t}')
            os.system('cp ../INCAR INCAR')
            os.system('cp ../POSCAR POSCAR')
            os.system('cp ../POTCAR POTCAR')
            os.system('cp ../KPOINTS KPOINTS')
            with open('INCAR', 'a') as f:
                f.write('\n\n# MD (User)')
                f.write(f'\nTEBEG = {t}')
                f.write(f'\nTEEND = {t}')
                f.write(f'\nPOTIM = {MD_TIMESTEP}')
                f.write(f'\nNSW   = {MD_ALLSTEPS}')
            self.bash()
            if AUTO_SUBMIT:
                os.system(f'sbatch {JOB_NAME}.sh')

    def aimd_post(self):
        '''
        * Post-process of AIMD calculations
        * Analyze and save the results
        '''
        os.chdir(PATH + '/ionic')
        temps = MD_TEMPERATURES.split()
        # From OSZICAR
        for temp in temps:
            with open(f'{PATH}/ionic/temp_{temp}/OSZICAR', 'r') as f:
                lines = f.readlines()
                data = []
                for line in lines:
                    if 'T=' in line:
                        T = float(line.split()[2])
                        E = float(line.split()[4])
                        data.append([T, E])
                pd.DataFrame(data).to_csv(f'{PATH}/ionic/TEMP_ENERGY_{temp}.csv', sep=',', index=False, header=['Temperature (K)', 'Total Energy (eV)'])
        # From XDATCAR
        data = []
        for temp in temps:
            traj = Trajectory.from_file(f'{PATH}/ionic/temp_{temp}/XDATCAR')
            diff = DiffusionAnalyzer.from_structures(traj, MD_SPECIE, float(temp), int(MD_TIMESTEP), 1)
            diff.export_msdt(f'{PATH}/ionic/MSD_{temp}.csv')
            data.append([diff.diffusivity, diff.conductivity])
        pd.DataFrame(data).to_csv(f'{PATH}/ionic/DIFF_COND.csv', sep=',', index=False, header=['Diffusivity (cm^2*s^-1)', 'conductivity (ms*cm*K^-1)'])

    def aimd_visualize(self):
        '''
        * Visualize the AIMD results
        * Time vs. Temperature vs. Energy
        * Time vs. MSD
        * Temperature vs. Diffusivity
        * Temperature vs. Conductivity
        '''
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        os.chdir(PATH + '/ionic')
        temps = MD_TEMPERATURES.split()
        # Time vs. Temperature vs. Energy
        for temp in temps:
            data = pd.read_csv(f'{PATH}/ionic/TEMP_ENERGY_{temp}.csv', sep=',', header=None)
            T = [float(d) for d in data.iloc[1:, 0].to_numpy().tolist()]
            E = [float(d) for d in data.iloc[1:, 1].to_numpy().tolist()]
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            ax[0].plot(range(len(T)), T, label='Temperature (K)', color='blue')
            ax[0].hlines(float(temp), 0, len(T), linestyles='dashed', label='Target Temperature', color='red')
            ax[1].plot(range(len(E)), E, label='Total Energy (eV)', color='green')
            ax[0].set_xlabel('MD Index')
            ax[0].set_ylabel('Temperature (K)')
            ax[1].set_xlabel('MD Index')
            ax[1].set_ylabel('Total Energy (eV)')
            ax[0].set_xlim(0, len(T))
            ax[1].set_xlim(0, len(T))
            ax[0].legend(frameon=False)
            ax[1].legend(frameon=False)
            plt.tight_layout()
            plt.savefig(f'{PATH}/ionic/TEMP_ENERGY_{temp}.png', dpi=330)
            plt.close()
        # Time vs. MSD
        for temp in temps:
            data = pd.read_csv(f'{PATH}/ionic/MSD_{temp}.csv', sep=',', header=None)
            T = [float(d) for d in data.iloc[1:, 0].to_numpy().tolist()]
            MSD = [float(d) for d in data.iloc[1:, 1].to_numpy().tolist()]
            plt.plot(T, MSD, label=f'{temp} K')
        plt.xlabel('Time (fs)')
        plt.ylabel('MSD (${cm^2}$)')
        plt.legend(frameon=False)
        plt.savefig(f'{PATH}/ionic/MSD.png', dpi=330)
        plt.close()
        # Temperature vs. Diffusivity
        data = pd.read_csv(f'{PATH}/ionic/DIFF_COND.csv', sep=',', header=None)
        T = np.array([float(t) for t in temps])
        D = np.array([float(d) for d in data.iloc[1:, 0].to_numpy().tolist()])
        X = 1000 / T
        plf = np.polyfit(X, D, 1)
        func = np.poly1d(plf)
        X_pre = np.linspace(X[0], X[-1], 1000)
        D_pre = func(X_pre)
        plt.scatter(X, D, marker='o', facecolors='none', edgecolors='black', label='Ab Initio')
        plt.plot(X_pre, D_pre, c='r', linestyle='dashed', label='Fitting')
        plt.xlabel('1000 / T (K)')
        plt.ylabel('Diffusion Coefficient (${cm^2}{s^{-1}}$)')
        plt.tight_layout()
        plt.legend(frameon=False)
        plt.savefig(f'{PATH}/ionic/DIFFUSIVITY.png', dpi=330)
        plt.close()
        # Temperature vs. Conductivity
        C = np.array([float(d) for d in data.iloc[1:, 1].to_numpy().tolist()])
        Y = np.log(C * T)
        plf = np.polyfit(X, Y, 1)
        func = np.poly1d(plf)
        X_pre = np.linspace(X[0], X[-1], 1000)
        C_pre = func(X_pre)
        plt.scatter(X, Y, marker='o', facecolors='none', edgecolors='black', label='Ab Initio')
        plt.plot(X_pre, C_pre, c='r', linestyle='dashed', label='Fitting')
        plt.xlabel('1000 / T (K)')
        plt.ylabel('Ln($\delta$T) ($ms*cm*{K^{-1}}$)')
        plt.tight_layout()
        plt.legend(frameon=False)
        plt.savefig(f'{PATH}/ionic/CONDUCTIVITY.png', dpi=330)
        plt.close()

def check(target_path, is_pre):
    '''
    * Check if there already exits the calculation results
    * is_pre = True : pre-process
    * is_pre = False: post-process
    '''
    if is_pre:
        if os.path.exists(target_path):
            print(f'The path <{target_path}> is already existed')
            print('Please remove/rename it and try it again...')
            sys.exit()
    else:
        if not os.path.exists(target_path):
            print('ERROR: No calcualtion results found!')
            sys.exit()

def check_rel():
    '''
    * Check if there exits POSCAR_REL
    '''
    relax = Relax()
    if not os.path.exists('POSCAR_REL'):
        relax.fully_relaxed()
        print('POSCAR_REL is missing, fully relaxation based on POSCAR_EOS will be performed first!')
        print('After you got POSCAR_REL, perform the calculation again...')
        sys.exit()

def restart():
    '''
    * Restart the calculations
    '''
    paths = [f'{PATH}/{folder}' for folder in RESTART_FOLDERS.split()]
    # Create folders for restart calculations
    for path in paths:
        for i in range(100):
            if not os.path.exists(f'{path}_ORI{i}'):
                re_idx = i
                break
        os.system(f'cp -r {path} {path}_ORI{re_idx}')
        # Copy CONTCAR to POSCAR
        os.system(f'rm {path}/POSCAR')
        os.system(f'cp {path}/CONTCAR {path}/POSCAR')

        files_str = ' '.join(os.listdir(path))
        # isif7 is finished, continue with isif5, need to update the bash script
        if 'CONTCAR_isif7' in files_str and 'CONTCAR_isif5' not in files_str and 'CONTCAR_isif4' not in files_str:
            with open(f'{path}/{JOB_NAME}.sh', 'w') as f:
                f.write('#!/usr/bin/sh \n')
                f.write(headers)
                f.write(f'time {SUBMIT_CMD} \n')
                f.write('if [ `grep -c "reached required accuracy" vasp.out` -ne "0" ] \n')
                f.write('then \n')
                f.write('    rm -f INCAR \n')
                f.write('    rm -f POSCAR \n')
                f.write('    cp INCAR_isif4 INCAR \n')
                f.write('    cp OUTCAR OUTCAR_isif5 \n')
                f.write('    cp OSZICAR OSZICAR_isif5 \n')
                f.write('    cp CONTCAR CONTCAR_isif5 \n')
                f.write('    cp vasp.out vasp.out_isif5 \n')
                f.write('    cp CONTCAR POSCAR \n')
                if MAGNETIC:
                    f.write('    time python update_mag.py \n')
                f.write(f'    time {SUBMIT_CMD} \n')
                f.write('    if [ `grep -c "reached required accuracy" vasp.out` -ne "0" ] \n')
                f.write('    then \n')
                f.write('        rm -f INCAR \n')
                f.write('        rm -f POSCAR \n')
                f.write('        cp INCAR_isif2 INCAR \n')
                f.write('        cp OUTCAR OUTCAR_isif4 \n')
                f.write('        cp OSZICAR OSZICAR_isif4 \n')
                f.write('        cp CONTCAR CONTCAR_isif4 \n')
                f.write('        cp vasp.out vasp.out_isif4 \n')
                f.write('        cp CONTCAR POSCAR \n')
                if MAGNETIC:
                    f.write('        time python update_mag.py \n')
                f.write(f'        time {SUBMIT_CMD} \n')
                f.write('    else \n')
                f.write('        echo "The ISIF4 calculation is not finished yet!" \n')
                f.write('    fi \n')
                f.write('else \n')
                f.write('    echo "The ISIF5 calculation is not finished yet!" \n')
                f.write('fi \n')
                f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh')
        # isif5 is finished, continue with isif4, need to update the bash script
        if 'CONTCAR_isif5' in files_str and 'CONTCAR_isif4' not in files_str:
            with open(f'{path}/{JOB_NAME}.sh', 'w') as f:
                f.write('#!/usr/bin/sh \n')
                f.write(headers)
                f.write(f'time {SUBMIT_CMD} \n')
                f.write('if [ `grep -c "reached required accuracy" vasp.out` -ne "0" ] \n')
                f.write('then \n')
                f.write('    rm -f INCAR \n')
                f.write('    rm -f POSCAR \n')
                f.write('    cp INCAR_isif2 INCAR \n')
                f.write('    cp OUTCAR OUTCAR_isif4 \n')
                f.write('    cp OSZICAR OSZICAR_isif4 \n')
                f.write('    cp CONTCAR CONTCAR_isif4 \n')
                f.write('    cp vasp.out vasp.out_isif4 \n')
                f.write('    cp CONTCAR POSCAR \n')
                if MAGNETIC:
                    f.write('    time python update_mag.py \n')
                f.write(f'    time {SUBMIT_CMD} \n')
                f.write('else \n')
                f.write('    echo "The ISIF4 calculation is not finished yet!" \n')
                f.write('fi \n')
                f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh')
        # For eos: relax_isif2 is finished, continue with relax_isif4, need to update the bash script
        if 'CONTCAR_relax_isif2' in files_str:
            with open(f'{path}/{JOB_NAME}_relax.sh', 'r') as f:
                lines = f.readlines()
            with open(f'{path}/{JOB_NAME}_relax.sh', 'w') as f:
                for line in lines:
                    if line.startswith('if'):
                        break
                    f.write(line)
                f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh \n')
        # For isifx: isif3 is finished, continue with isif4, need to update the bash script
        if 'CONTCAR_isif4' in files_str:
            with open(f'{path}/{JOB_NAME}.sh', 'r') as f:
                lines = f.readlines()
            with open(f'{path}/{JOB_NAME}.sh', 'w') as f:
                for line in lines:
                    if line.startswith('if'):
                        break
                    f.write(line)
                f.write('rm -f CHG* PROCAR* vasprun.xml W* vk*.sh \n')

        # Submit job
        if AUTO_SUBMIT:
            os.chdir(f'{path}')
            if 'eos_relax' in path:
                os.system(f'sbatch {JOB_NAME}_relax.sh')
            else:
                os.system(f'sbatch {JOB_NAME}.sh')

def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:", ['ifile='])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> [> <outputfile>]')
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> [> <outputfile>]')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    return inputfile

if __name__ == '__main__':
    PATH = os.getcwd()
    inputfile = main(sys.argv[1:])
    with open(inputfile, 'r') as f:
        content = f.readlines()
    headers = ''
    paras = []
    for line in content:
        if line.startswith('#SBATCH'):
            headers += line
            if '=' not in line:
                if '-N' in line:
                    TOTAL_NODES = int(line.split()[-1])
                if '-n' in line and '-normal' not in line:
                    TOTAL_CORES = int(line.split()[-1])
            else:
                if '--nodes' in line:
                    TOTAL_NODES = int(line.split('=', 1)[-1])
                if '--ntasks-per-node' in line:
                    TOTAL_CORES = int(line.split('=', 1)[-1]) * TOTAL_NODES

        if '=' in line and not line.startswith('#'):
            line = line.rstrip('\n')
            paras.append(line.split('=', 1))
    for d in paras:
        d[0] = d[0].strip()
        d[1] = d[1].strip()
    paras = dict(paras)

    SUBMIT_CMD  = paras['SUBMIT_CMD']
    JOB_NAME    = paras['JOB_NAME']
    AUTO_SUB_STR = paras['AUTO_SUBMIT']

    if AUTO_SUB_STR.lower() == 'false':
        AUTO_SUBMIT = False
    elif AUTO_SUB_STR == 'true':
        AUTO_SUBMIT = True
    else:
        print(f'ERROR: Expect the AUTO_SUBMIT value to be true/false, but got "{AUTO_SUB_STR}"')
    
    MAG_STR = paras['MAGNETIC']
    if MAG_STR.lower() == 'none':
        MAGNETIC = False
    else:
        MAGNETIC = True
        MAG_STR = MAG_STR.split()
        MAG_DICT = {MAG_STR[i]: float(MAG_STR[i+1]) for i in range(0, len(MAG_STR), 2)}

    MODE = paras['MODE']
    if 'isifx' in MODE.lower():
        check(PATH + '/isifx', True)
        isifx = ISIFX()
        isifx.isifx()
    elif 'eos' in MODE.lower():
        eos = EOS()
        SCALING_FACTORS = paras['SCALING_FACTORS']
        EOS_TYPE        = paras['EOS_TYPE']
        if MODE.lower() == 'eos-isif3':
            check(PATH + '/isif3', True)
            eos.isif3()
        elif MODE.lower() == 'eos-relax-pre':
            check(PATH + '/eos_relax', True)
            eos.relax_pre()
        elif MODE.lower() == 'eos-static-pre':
            ISIF = paras['ISIF']
            if ISIF != '4' and ISIF != '2':
                print(f'ERROR: Expect the ISIF value to be 2 or 4, but got "{ISIF}"')
                sys.exit()
            check(PATH + '/eos_static', True)
            eos.static_pre()
        elif MODE.lower() == 'eos-relax-post':
            check(PATH + '/eos_relax', False)
            eos.relax_post()
        elif MODE.lower() == 'eos-static-post':
            check(PATH + '/eos_static', False)
            eos.static_post()
        else:
            print(f'ERROR: The selected mode "{MODE}" is not supported!')
    elif 'elastic' in MODE.lower():
        elastic = Elastic()
        ELASTIC_METHOD = paras['ELASTIC_METHOD']
        ISIF = paras['ISIF']
        if ISIF != '4' and ISIF != '2':
            print(f'ERROR: Expect the ISIF value to be 2 or 4, but got "{ISIF}"')
            sys.exit()
        if ELASTIC_METHOD.lower() == 'stress-strain' or ELASTIC_METHOD.lower() == 'energy-strain':
            STRAINS = paras['STRAINS']
            SPACE_GROUP_NUM = paras['SPACE_GROUP_NUM']
            ELASTIC_METHOD_CODE = 200 if ELASTIC_METHOD.lower() == 'stress-strain' else 201
        elif ELASTIC_METHOD.lower() == 'sqs':
            CRYSTAL_SYSTEM = paras['CRYSTAL_SYSTEM']
            STRAINS_XX = paras['STRAINS_XX']
            STRAINS_YY = paras['STRAINS_YY']
            STRAINS_ZZ = paras['STRAINS_ZZ']
            STRAINS_XY = paras['STRAINS_XY']
            STRAINS_YZ = paras['STRAINS_YZ']
            STRAINS_ZX = paras['STRAINS_ZX']
        else:
            print(f'ERROR: The selected ELASTIC_METHOD "{ELASTIC_METHOD}" is not supported!')
        if MODE.lower() == 'elastic-pre':
            check(PATH + '/elastic', True)
            check_rel()
            elastic.pre_process()
        elif MODE.lower() == 'elastic-post':
            check(PATH + '/elastic', False)
            elastic.post_process()
        else:
            print(f'ERROR: The selected mode "{MODE}" is not supported!')
    elif 'ionic' in MODE.lower():
        ionic = Ionic_conductivity()
        relax = Relax()
        IONIC_METHOD = paras['IONIC_METHOD']
        MD_TEMPERATURES = paras['MD_TEMPERATURES']
        MD_ALLSTEPS = paras['MD_ALLSTEPS']
        MD_TIMESTEP = paras['MD_TIMESTEP']
        MD_ATOMS = paras['MD_ATOMS']
        MD_SPECIE = paras['MD_SPECIE']
        ISIF = paras['ISIF']
        if ISIF != '4' and ISIF != '2':
            print(f'ERROR: Expect the ISIF value to be 2 or 4, but got "{ISIF}"')
            sys.exit()
        if MODE.lower() == 'ionic-pre':
            check(PATH + '/ionic', True)
            check_rel()
            if IONIC_METHOD.lower() == 'aimd':
                ionic.aimd_pre()
        elif MODE.lower() == 'ionic-post':
            check(PATH + '/ionic', False)
            if IONIC_METHOD.lower() == 'aimd':
                ionic.aimd_post()
                ionic.aimd_visualize()
        else:
            print(f'ERROR: The selected mode "{MODE}" is not supported!')
    elif MODE.lower() == 'restart':
        RESTART_FOLDERS = paras['RESTART_FOLDERS']
        restart()
    else:
        print(f'ERROR: The selected mode "{MODE}" is not supported!')
class Magnetism:

    def _parse_magnetic_pairs(mag_str: str) -> Dict[str, float]:
        # "Ni 3.0 Co 2.0" -> {"Ni": 3.0, "Co": 2.0}
        if not mag_str or mag_str.strip().lower() == "none":
            return {}
        toks = mag_str.split()
        if len(toks) % 2 != 0:
            raise ValueError(f"Bad MAGNETIC string: {mag_str}")
        return {toks[i]: float(toks[i+1]) for i in range(0, len(toks), 2)}

    @staticmethod
    def _sublattice_sign_AFM_G(fx, fy, fz, period=(0.5, 0.5, 0.5)) -> int:
        ax = int((fx / period[0]) + 1e-9) % 2
        by = int((fy / period[1]) + 1e-9) % 2
        cz = int((fz / period[2]) + 1e-9) % 2
        return +1 if (ax + by + cz) % 2 == 0 else -1

    @staticmethod
    @staticmethod
    def _sublattice_sign_AFM_A(fz, period_c=0.5) -> int:
        cz = int((fz / period_c) + 1e-9) % 2
        return +1 if cz == 0 else -1

    @staticmethod
    def _sublattice_sign_AFM_C(fx, fy, period=(0.5, 0.5)) -> int:
        ax = int((fx / period[0]) + 1e-9) % 2
        by = int((fy / period[1]) + 1e-9) % 2
        return +1 if (ax + by) % 2 == 0 else -1

    @staticmethod
    def _parse_magnetic_pairs(mag_str: str) -> Dict[str, float]:
        # "Ni 3.0 Co 2.0" -> {"Ni": 3.0, "Co": 2.0}
        if not mag_str or mag_str.strip().lower() == "none":
            return {}
        toks = mag_str.split()
        if len(toks) % 2 != 0:
            raise ValueError(f"Bad MAGNETIC string: {mag_str}")
        return {toks[i]: float(toks[i+1]) for i in range(0, len(toks), 2)}

    @staticmethod
    def apply_magnetism_from_infile(
    struct: Structure,
    cfg: Dict[str, str],
    default_moment: float = 4.0,
    ) -> Dict[str, str]:
    """
    Returns INCAR updates based on:
      - cfg['MAGNETIC_ORDER'] in {NM, FM, AFM, AFM-A, AFM-C, AFM-G}
      - cfg['MAGNETIC'] element->moment pairs (e.g., 'Ni 3.0 Co 2.0')
    """
    order = cfg.get("MAGNETIC_ORDER", "NM").strip().upper()
    elem_mags = Magnetism._parse_magnetic_pairs(cfg.get("MAGNETIC", ""))

    if order == "NM" or not elem_mags:
        return {"ISPIN": "1"}  # ensure collinear off; no MAGMOM written

    magmom = []
    for s in struct.sites:
        elem = s.specie.symbol
        m = elem_mags.get(elem, 0.0)
        if m == 0.0:
            magmom.append(0.0)
            continue
        fx, fy, fz = s.frac_coords
        if order == "FM":
            sign = +1
        elif order in ("AFM", "AFM-G"):
            sign = _sublattice_sign_AFM_G(fx, fy, fz)
        elif order == "AFM-A":
            sign = _sublattice_sign_AFM_A(fz)
        elif order == "AFM-C":
            sign = _sublattice_sign_AFM_C(fx, fy)
        else:
            raise ValueError(f"Unknown MAGNETIC_ORDER: {order}")
        magmom.append(sign * (m or default_moment))

    return {"ISPIN": "2", "MAGMOM": " ".join(f"{x:g}" for x in magmom)}
