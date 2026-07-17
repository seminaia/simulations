"""
List of Free energies of formations for each element

"""
import pandas as pd 
import scipy as sp    

def AgO_formation(T):
    # Free energy of formation for AgO
    Gf = 74.950 + 0.022*10^-3*T*lnT + 1.072xl0- 6 r 2 + 22.650T- 1 - 24.544xl0-'T
    return -11.2  # kJ/mol