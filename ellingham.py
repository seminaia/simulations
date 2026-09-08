#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ellingham.py

Generates separate Ellingham diagrams for Oxides, Carbides, Nitrides,
Fluorides, Chlorides, Hydrides and Sulfides.

Data:
  * O2, N2, F2, Cl2 reference lines and salt data — Reed, T.B., 1971.
    Free Energy of Formation of Binary Compounds. MIT Press, Cambridge, Mass.
  * Carbides — Coltters, R.G., 1985. Thermodynamics of binary metallic
    carbides: a review. Materials Science and Engineering 76, 1-50.
  * Added salts, hydrides and sulfides — two-point linear segments
    (dG = dH - T*dS) compiled from standard thermochemical tables
    (Barin, 1993); slopes follow the reaction entropy. Verify against
    the primary source before publication-quality use.

Storage convention (raw tables below):
  temperatures in K, energies in kcal per mole of gas reference
  (O2, N2, F2, Cl2, H2, S2) or per mole C for carbides.
  convert_units() turns them into degC / kJ before plotting.

Line style code (metal state x compound state):
  metal:   solid '-'   liquid '--'   gas ':'
  compound: solid a=1.0   liquid a=0.6   gas a=0.3

Usage:
  python ellingham.py                        # every family, every element
  python ellingham.py --elements Al,Fe,Mg
  python ellingham.py --families oxides,sulfides
  python ellingham.py --phases ss,ll
"""

import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------
plt.rcParams.update({
    'mathtext.default': 'regular',
    'mathtext.fontset': 'custom',
    'mathtext.it': 'Arial:italic',
    'mathtext.rm': 'Arial',
    'font.family': 'Arial',
})


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def element_from_reaction(rxn_str):
    """First element symbol appearing in a reaction string."""
    clean = re.sub(r'[$_{}\\]', '', str(rxn_str))
    tokens = re.findall(r'[A-Z][a-z]?', clean)
    return tokens[0] if tokens else None


EMPTY = lambda: np.empty((0, 6))


def filter_by_elements(arr, allowed_elements):
    if arr.size == 0:
        return arr
    new_rows = [row for row in arr
                if element_from_reaction(row[4]) in allowed_elements]
    return np.array(new_rows) if new_rows else EMPTY()


def filter_anion_dict(anion_dict, allowed_elements):
    return {phase: filter_by_elements(arr, allowed_elements)
            for phase, arr in anion_dict.items()}


# ----------------------------------------------------------------------
# Element reference: Symbol -> [Name, molar mass]
# ----------------------------------------------------------------------
molarmass_bin = {
    'Ac': ['Actinium', 227], 'Ag': ['Silver', 107.8682], 'Al': ['Aluminum', 26.9815],
    'Am': ['Americium', 243], 'Ar': ['Argon', 39.948], 'As': ['Arsenic', 74.9216],
    'At': ['Astatine', 210], 'Au': ['Gold', 196.9665], 'B': ['Boron', 10.811],
    'Ba': ['Barium', 137.327], 'Be': ['Beryllium', 9.0122], 'Bi': ['Bismuth', 208.9804],
    'Br': ['Bromine', 79.904], 'C': ['Carbon', 12.0107], 'Ca': ['Calcium', 40.078],
    'Cd': ['Cadmium', 112.411], 'Ce': ['Cerium', 140.116], 'Cl': ['Chlorine', 35.453],
    'Co': ['Cobalt', 58.9332], 'Cr': ['Chromium', 51.9961], 'Cs': ['Cesium', 132.9055],
    'Cu': ['Copper', 63.546], 'Dy': ['Dysprosium', 162.5], 'Er': ['Erbium', 167.259],
    'Eu': ['Europium', 151.964], 'F': ['Fluorine', 18.9984], 'Fe': ['Iron', 55.845],
    'Ga': ['Gallium', 69.723], 'Gd': ['Gadolinium', 157.25], 'Ge': ['Germanium', 72.64],
    'H': ['Hydrogen', 1.0079], 'He': ['Helium', 4.0026], 'Hf': ['Hafnium', 178.49],
    'Hg': ['Mercury', 200.59], 'Ho': ['Holmium', 164.9303], 'I': ['Iodine', 126.9045],
    'In': ['Indium', 114.818], 'Ir': ['Iridium', 192.217], 'K': ['Potassium', 39.0983],
    'Kr': ['Krypton', 83.8], 'La': ['Lanthanum', 138.9055], 'Li': ['Lithium', 6.941],
    'Lu': ['Lutetium', 174.967], 'Mg': ['Magnesium', 24.305], 'Mn': ['Manganese', 54.938],
    'Mo': ['Molybdenum', 95.94], 'N': ['Nitrogen', 14.0067], 'Na': ['Sodium', 22.9897],
    'Nb': ['Niobium', 92.9064], 'Nd': ['Neodymium', 144.24], 'Ne': ['Neon', 20.1797],
    'Ni': ['Nickel', 58.6934], 'O': ['Oxygen', 15.9994], 'Os': ['Osmium', 190.23],
    'P': ['Phosphorus', 30.9738], 'Pa': ['Protactinium', 231.0359], 'Pb': ['Lead', 207.2],
    'Pd': ['Palladium', 106.42], 'Pr': ['Praseodymium', 140.9077], 'Pt': ['Platinum', 195.078],
    'Pu': ['Plutonium', 244], 'Ra': ['Radium', 226], 'Rb': ['Rubidium', 85.4678],
    'Re': ['Rhenium', 186.207], 'Rh': ['Rhodium', 102.9055], 'Rn': ['Radon', 222],
    'Ru': ['Ruthenium', 101.07], 'S': ['Sulfur', 32.065], 'Sb': ['Antimony', 121.76],
    'Sc': ['Scandium', 44.9559], 'Se': ['Selenium', 78.96], 'Si': ['Silicon', 28.0855],
    'Sm': ['Samarium', 150.36], 'Sn': ['Tin', 118.71], 'Sr': ['Strontium', 87.62],
    'Ta': ['Tantalum', 180.9479], 'Tb': ['Terbium', 158.9253], 'Tc': ['Technetium', 98],
    'Te': ['Tellurium', 127.6], 'Th': ['Thorium', 232.0381], 'Ti': ['Titanium', 47.867],
    'Tl': ['Thallium', 204.3833], 'Tm': ['Thulium', 168.9342], 'U': ['Uranium', 238.0289],
    'V': ['Vanadium', 50.9415], 'W': ['Tungsten', 183.84], 'Xe': ['Xenon', 131.293],
    'Y': ['Yttrium', 88.9059], 'Yb': ['Ytterbium', 173.04], 'Zn': ['Zinc', 65.39],
    'Zr': ['Zirconium', 91.224],
}

# ======================================================================
# OXIDES  (Reed 1971) — raw: K, kcal/mol O2
# row: [T0, T1, G0, G1, reaction, label-offset]
# ======================================================================

# Metal solid, oxide solid
oxss = np.array([
    [   0,  480,  -14.0,    0.0, '$4Ag + O_2 = 2Ag_2O $', 0],
    [   0,  932, -266.6, -220.0, r'$\frac{4}{3} Al + O_2 = \frac{2}{3} Al_2O_3$', -13],
    [   0,  983, -265.0, -222.0, '$2Ba + O_2 = 2BaO $', 0],
    [   0,  544,  -92.0,  -69.0, r'$\frac{4}{3} Bi + O_2 = \frac{2}{3} Bi_2O_3$', 6],
    [   0,  723, -200.5, -171.5, r'$\frac{4}{3} B + O_2 = \frac{2}{3} B_2O_3$', -3],
    [   0, 1123, -303.0, -249.0, '$2Ca + O_2 = 2CaO $', 0],
    [   0,    0,  -55.6,  -55.6, '$2C + O_2 = 2CO $', 0],
    [   0,    0,  -94.5,  -94.5, '$C + O_2 = CO_2 $', -8],
    [   0,    0,  -38.9,  -38.9, '$C + CO_2 = 2CO$', 0],
    [   0,    0, -133.4, -133.4, '$2CO + O_2 = 2CO_2$', 0],
    [   0,  302, -151.8, -125.0, '$4Cs + O_2 = 2Cs_2O$', -14],
    [   0, 1357,  -80.0,  -33.0, '$4Cu + O_2 = 2Cu_2O $', 0],
    [   0, 1357,  -74.5,  -16.0, '$2Cu + O_2 = 2CuO $', 0],
    [   0,    0, -119.3, -119.3, '$4H + O_2 = 2H_2O$', 12],
    [   0, 1642, -124.1,  -75.0, '$2Fe + O_2 = 2FeO$', -9],
    [   0, 1809, -129.2,  -55.5, r'$\frac{4}{3} Fe + O_2 = \frac{2}{3} Fe_2O_3$', -5],
    [   0,  453, -286.0, -258.0, '$4Li + O_2 = 2Li_2O $', -12],
    [   0, 1068, -120.0,  -77.0, r'$ \frac{2}{3}Mo + O_2 = \frac{2}{3}MoO_3 $', -3],
    [   0,  923, -286.0, -240.0, '$2Mg + O_2 = 2MgO $', 8],
    [   0,    0,  -44.0,  -44.0, '$2Hg + O_2 = 2HgO$', 0],
    [   0, 1764, -181.0, -112.0, r'$\frac{4}{5}Nb + O_2 = \frac{2}{5}Nb_2O_5$', 0],
    [   0,  734,  -32.0,    0.0, r'$\frac{3}{2} Pt + O_2 = \frac{1}{2}Pt_3O_4 $', 0],
    [   0,  336, -172.0, -151.0, '$4K + O_2 = 2K_2O $', -14],
    [   0,  312, -157.8, -138.0, '$4Rb + O_2 = 2Rb_2O$', -8],
    [   0, 1685, -216.5, -145.8, '$Si + O_2 = SiO_2 $', 0],
    [   0,  371, -197.0, -176.0, '$4Na + O_2 = 2Na_2O $', 3],
    [   0, 1725, -114.0,  -44.5, '$2Ni + O_2 = 2NiO$', 0],
    [   0,  904, -111.0,  -74.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 3],
    [   0, 1043, -281.0, -233.0, '$2Sr + O_2 = 2SrO$', 5],
    [   0,  505, -138.8, -114.0, '$Sn + O_2 = SnO_2 $', -11],
    [   0, 1940, -225.5, -142.5, '$Ti + O_2 = TiO_2 $', 0],
    [   0, 1940, -247.5, -161.0, '$2Ti + O_2 = 2TiO$', 0],
    [   0, 1818, -168.0, -100.0, '$V + O_2 = VO_2$', -9],
    [   0,  943, -149.5, -110.0, r'$\frac{4}{5}V + O_2 = \frac{2}{5}V_2O_5$', 2],
    [   0, 1743, -133.0,  -67.0, r'$ \frac{2}{3}W + O_2 = \frac{2}{3}WO_3 $', -10],
    [   0,  693, -166.0, -134.0, '$2Zn + O_2 = 2ZnO $', 4],
    [   0, 2125, -262.0, -166.0, '$Zr + O_2 = ZrO_2 $', 9],
], dtype=object)

# Metal liquid, oxide solid
oxls = np.array([
    [ 932, 2345, -220.0, -147.6, r'$\frac{4}{3} Al + O_2 = \frac{2}{3} Al_2O_3$', 0],
    [ 904,  928,  -74.0,  -73.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 0],
    [ 983, 1895, -222.0, -183.0, '$2Ba + O_2 = 2BaO $', 0],
    [ 544, 1098,  -69.0,  -44.0, r'$\frac{4}{3} Bi + O_2 = \frac{2}{3} Bi_2O_3$', 0],
    [1123, 1756, -249.0, -217.0, '$2Ca + O_2 = 2CaO $', 0],
    [ 302,  763, -125.0,  -84.0, '$4Cs + O_2 = 2Cs_2O$', -16],
    [1357, 1509,  -33.0,  -28.0, '$4Cu + O_2 = 2Cu_2O $', 0],
    [1357, 1609,  -16.0,   -9.5, '$2Cu + O_2 = 2CuO $', 0],
    [ 453, 1597, -258.0, -173.0, '$4Li + O_2 = 2Li_2O $', 0],
    [ 336,  980, -151.0, -107.0, '$4K + O_2 = 2K_2O $', 0],
    [   0,  630,  -44.0,  -10.0, '$2Hg + O_2 = 2HgO$', 0],
    [ 312,  910, -138.0,  -96.0, '$4Rb + O_2 = 2Rb_2O$', -8],
    [1685, 1696, -145.8, -145.4, '$Si + O_2 = SiO_2 $', 0],
    [ 371, 1156, -176.0, -122.0, '$4Na + O_2 = 2Na_2O $', 0],
    [1725, 2257,  -44.5,  -24.5, '$2Ni + O_2 = 2NiO$', 0],
    [1940, 2128, -142.5, -134.5, '$Ti + O_2 = TiO_2 $', 0],
    [1940, 2033, -161.0, -159.0, '$2Ti + O_2 = 2TiO$', 0],
    [ 693, 1180, -134.0, -109.0, '$2Zn + O_2 = 2ZnO $', 0],
    [2125, 2980, -166.0, -130.0, '$Zr + O_2 = ZrO_2 $', 0],
], dtype=object)

# Metal gas, oxide solid
oxgs = np.array([
    [1895, 2191, -183.0, -159.0, '$2Ba + O_2 = 2BaO $', 0],
    [1756, 2887, -217.0, -117.0, '$2Ca + O_2 = 2CaO $', 0],
    [1597, 2000, -173.0, -128.0, '$4Li + O_2 = 2Li_2O $', 0],
    [ 923, 1376, -240.0, -214.0, '$2Mg + O_2 = 2MgO $', 0],
    [ 630,  740,  -10.0,    0.0, '$2Hg + O_2 = 2HgO$', 0],
    [1156, 1193, -122.0, -119.0, '$4Na + O_2 = 2Na_2O $', 0],
    [1180, 2240, -109.0,   -9.0, '$2Zn + O_2 = 2ZnO $', 0],
], dtype=object)

# Metal solid, oxide liquid
oxsl = np.array([
    [ 723, 2313, -171.5, -112.0, r'$\frac{4}{3} B + O_2 = \frac{2}{3} B_2O_3$', 0],
    [1642, 1809,  -75.0,  -71.9, '$2Fe + O_2 = 2FeO$', 0],
    [1068, 1530,  -77.0,  -64.0, r'$\frac{2}{3}Mo + O_2 = \frac{2}{3}MoO_3$', 0],
    [1818, 2190, -100.0,  -96.0, '$V + O_2 = VO_2$', 0],
    [1743, 2100,  -67.0,  -57.0, r'$ \frac{2}{3}W + O_2 = \frac{2}{3}WO_3 $', 0],
], dtype=object)

# Metal liquid, oxide liquid
oxll = np.array([
    [2345, 2736, -147.6, -128.5, r'$\frac{4}{3} Al + O_2 = \frac{2}{3} Al_2O_3$', 0],
    [ 928, 1698,  -73.0,  -45.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 0],
    [1098, 1852,  -44.0,  -12.0, r'$\frac{4}{3} Bi + O_2 = \frac{2}{3} Bi_2O_3$', 0],
    [1809, 2000,  -71.9,  -67.9, '$2Fe + O_2 = 2FeO$', 0],
    [1376, 3125, -214.0,   52.0, '$2Mg + O_2 = 2MgO $', 0],
    [ 763,  915,  -84.0,  -73.0, '$4Cs + O_2 = 2Cs_2O$', -16],
    [1509, 2500,  -28.0,   -9.5, '$4Cu + O_2 = 2Cu_2O $', 0],
    [1609, 1870,   -9.5,    0.0, '$2Cu + O_2 = 2CuO $', 0],
    [2257, 2500,  -24.5,  -15.0, '$2Ni + O_2 = 2NiO$', 0],
    [ 980, 1031, -107.0, -104.0, '$4K + O_2 = 2K_2O $', 0],
    [ 910,  952,  -96.0,  -95.0, '$4Rb + O_2 = 2Rb_2O$', -8],
    [1696, 2500, -145.4, -107.8, '$Si + O_2 = SiO_2 $', 0],
    [2128, 2500, -134.5, -121.5, '$Ti + O_2 = TiO_2 $', 0],
    [2033, 2500, -159.0, -142.5, '$2Ti + O_2 = 2TiO$', 0],
    [2190, 2500,  -96.0,  -81.0, '$V + O_2 = VO_2$', 0],
], dtype=object)

# Metal gas, oxide liquid
oxgl = np.array([
    [2191, 2500, -159.0, -131.0, '$2Ba + O_2 = 2BaO $', 0],
    [1031, 1325, -104.0,  -71.0, '$4K + O_2 = 2K_2O $', 0],
    [1193, 1600, -119.0,  -62.0, '$4Na + O_2 = 2Na_2O $', 0],
    [2240, 2340,   -9.0,    0.0, '$2Zn + O_2 = 2ZnO $', 0],
], dtype=object)

# Metal solid, oxide gas
oxsg = np.array([
    [   0, 3400,  -55.6, -191.9, '$2C + O_2 = 2CO $', 0],
    [   0, 3400,  -94.5,  -94.5, '$C + O_2 = CO_2 $', 0],
    [   0, 3400,  -38.9,  -97.4, '$C + CO_2 = 2CO$', 0],
    [1530, 2500,  -64.0,  -52.0, r'$\frac{2}{3}Mo + O_2 = \frac{2}{3}MoO_3$', 0],
    [2100, 2500,  -57.0,  -52.0, r'$ \frac{2}{3}W + O_2 = \frac{2}{3}WO_3 $', 0],
], dtype=object)

# Metal liquid, oxide gas
oxlg = np.array([
    [ 915,  955,  -73.0,  -72.0, '$4Cs + O_2 = 2Cs_2O$', -16],
    [1698, 1908,  -45.0,  -32.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 0],
], dtype=object)

# Metal gas, oxide gas
oxgg = np.array([
    [   0, 3400, -135.4,   -4.7, '$2CO + O_2 = 2CO_2$', 0],
    [   0, 3400, -119.3,  -26.6, '$4H + O_2 = 2H_2O$', 0],
    [1325, 2160,  -71.0,    0.0, '$4K + O_2 = 2K_2O $', 0],
    [1600, 2250,  -62.0,    0.0, '$4Na + O_2 = 2Na_2O $', 0],
    [1908, 2380,  -32.0,    0.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 0],
], dtype=object)

# ======================================================================
# CARBIDES (Coltters 1985) — raw: K, kcal/mol C
# ======================================================================

# Metal solid, carbide solid
cass = np.array([
    [   0, 1414,   -57.0,   -49.0, '$ Si + C = SiC $', -9],
    [   0, 1750,  -160.0, -150.5, '$ Ti + C = TiC $', -5],
    [   0,  723,    23.0,    -1.0, '$3Fe + C = Fe_3C $', 0],
    [   0, 1290,   -31.0,   -34.0, '$2W + C = W_2C$', -1],
    [   0,  800,   -39.5,   -45.0, '$W + C = WC$', -10],
    [   0, 1000,   -70.0,   -59.0, '$2Mo + C = Mo_2C$', -12],
    [   0,  720,  -183.0,  -175.0, '$Zr + C = ZrC$', 1],
    # --- added (Barin 1993, linear approximations) ---
    [   0, 3083,   -20.5,   -14.3, '$V + C = VC$', 0],
    [   0, 3885,   -32.0,   -24.2, '$Nb + C = NbC$', 26],
    [   0, 4153,   -33.6,   -25.3, '$Ta + C = TaC$', -16],
    [   0, 4200,   -44.0,   -33.0, '$Hf + C = HfC$', -30],
    [   0, 2700,   -16.0,   -10.6, '$4B + C = B_4C$', 0],
], dtype=object)

# Metal liquid, carbide solid
cals = np.array([
    [1414, 2000,   -49.0,   -30.0, '$ Si + C = SiC $', 0],
], dtype=object)

cags = EMPTY()
casl = EMPTY()
call = EMPTY()
cagl = EMPTY()
casg = EMPTY()
calg = EMPTY()
cagg = EMPTY()

# ======================================================================
# NITRIDES (Reed 1971) — raw: K, kcal/mol N2
# ======================================================================

# Metal solid, nitride solid
niss = np.array([
    [   0,  932, -144.3, -101.0, '$2Al + N_2 = 2AlN $', 0],
    [   0, 2300, -121.4,  -20.8, '$2B + N_2 = 2BN$', 0],
    [   0, 1809,   -5.8,   38.5, '$8Fe + N_2 = 2Fe_4N $', 12],
    [   0,  923, -109.6,  -65.8, '$3Mg + N_2 = Mg_3N_2 $', 8],
    [   0, 1150,  -31.9,    0.0, '$4Mo + N_2 = Mo_2N $', 10],
    [   0,    0,  -24.1,  -24.1, '$6H + N_2 = 2NH_3$', 0],
    [   0, 1680,  -90.0,  -22.5, r'$\frac{3}{2}Si + N_2 = \frac{1}{2}Si_3N_4 $', -6],
    [   0, 1940, -160.5,  -73.4, '$2Ti + N_2 = 2TiN $', 1],
    [   0, 2190,  -83.3,    3.6, '$2V + N_2 = 2VN$', -13],
    [   0, 2128, -163.8,  -67.2, '$2Zr + N_2 = 2ZrN $', -2],
    # --- added (Barin 1993, linear approximations) ---
    [   0, 3580, -159.0,  -70.0, '$2Hf + N_2 = 2HfN $', 18],
    [   0, 1800,  -58.0,  -15.0, '$2Cr + N_2 = 2CrN $', 0],
], dtype=object)

# Metal liquid, nitride solid
nils = np.array([
    [2300, 2500,  -20.8,    0.0, '$2B + N_2 = 2BN$', 0],
    [ 923, 1376,  -65.8,  -41.3, '$3Mg + N_2 = Mg_3N_2$', 0],
    [1680, 2130,  -22.5,    0.0, r'$\frac{3}{2}Si + N_2 = \frac{1}{2}Si_3N_4 $', 0],
], dtype=object)

nigs = EMPTY()
nisl = EMPTY()
nill = EMPTY()
nigl = EMPTY()
nisg = EMPTY()
nilg = EMPTY()

# Metal gas, nitride gas
nigg = np.array([
    [0, 2000, -24.1, 85.2, '$6H + N_2 = 2NH_3$', 0],
], dtype=object)

# ======================================================================
# FLUORIDES (Reed 1971) — raw: K, kcal/mol F2
# ======================================================================

# Metal solid, fluoride solid
flss = np.array([
    [   0,  932, -215.3, -181.0, r'$\frac{2}{3}Al + F_2 = \frac{2}{3}AlF_3 $', 0],
    [   0, 1123, -288.0, -245.0, '$ Ca + F_2 = CaF_2 $', 5],
    [   0,    0, -129.8, -129.8, '$2H + F_2 = 2HF $', 0],
    [   0,    0,  -81.2,  -81.2, r'$\frac{1}{2}C + F_2 = \frac{1}{2}CF_4 $', 2],
    [   0,  453, -290.0, -271.0, '$2Li + F_2 = 2LiF $', -8],
    [   0,  336, -270.0, -253.0, '$2K + F_2 = 2KF $', 0.3],
    [   0,  371, -274.0, -255.0, '$2Na + F_2 = 2NaF $', -0.3],
    # --- added (Barin 1993, linear approximations) ---
    [   0,  923, -246.4, -216.0, '$Mg + F_2 = MgF_2 $', 0],
], dtype=object)

# Metal liquid, fluoride solid
flls = np.array([
    [ 932, 1545, -181.0, -156.0, r'$\frac{2}{3}Al + F_2 = \frac{2}{3}AlF_3 $', 0],
    [1123, 1691, -245.0, -224.0, '$ Ca + F_2 = CaF_2 $', 0],
    [ 453, 1120, -271.0, -240.0, '$2Li + F_2 = 2LiF $', 0],
    [ 336, 1031, -253.0, -214.0, '$2K + F_2 = 2KF $', 0],
    [ 371, 1187, -255.0, -214.0, '$2Na + F_2 = 2NaF $', 0],
    # --- added ---
    [ 923, 1363, -216.0, -201.5, '$Mg + F_2 = MgF_2 $', 0],
], dtype=object)

# Metal gas, fluoride solid
flgs = np.array([
    # --- added ---
    [1363, 1536, -201.5, -194.0, '$Mg + F_2 = MgF_2 $', 0],
], dtype=object)

flsl = EMPTY()

# Metal liquid, fluoride liquid
flll = np.array([
    [1545, 2500, -156.0, -157.0, r'$\frac{2}{3}Al + F_2 = \frac{2}{3}AlF_3 $', 0],
    [1691, 1955, -224.0, -222.0, '$ Ca + F_2 = CaF_2 $', 0],
    [1120, 1597, -240.0, -216.0, '$2Li + F_2 = 2LiF $', 0],
    [1031, 1130, -214.0, -209.0, '$2K + F_2 = 2KF $', 0],
    [1187, 1268, -214.0, -209.0, '$2Na + F_2 = 2NaF $', 0],
], dtype=object)

# Metal gas, fluoride liquid
flgl = np.array([
    [1955, 2500, -222.0, -186.0, '$ Ca + F_2 = CaF_2 $', 0],
    [1597, 1954, -216.0, -194.0, '$2Li + F_2 = 2LiF $', 0],
    [1130, 1775, -209.0, -166.0, '$2K + F_2 = 2KF $', 0],
    [1268, 1977, -209.0, -156.0, '$2Na + F_2 = 2NaF $', 0],
    # --- added ---
    [1536, 2300, -194.0, -166.0, '$Mg + F_2 = MgF_2 $', 0],
], dtype=object)

# Metal solid, fluoride gas
flsg = np.array([
    # --- added ---
    [   0, 1687, -186.6, -157.5, r'$\frac{1}{2}Si + F_2 = \frac{1}{2}SiF_4 $', 0],
], dtype=object)

fllg = EMPTY()

# Metal gas, fluoride gas
flgg = np.array([
    [   0, 2500,  -81.2,  -36.0, r'$\frac{1}{2}C + F_2 = \frac{1}{2}CF_4 $', 0],
    [   0,    0, -129.8, -129.8, '$2H + F_2 = 2HF $', 0],
    [1954, 2500, -194.0, -179.0, '$2Li + F_2 = 2LiF $', 0],
    [1775, 2500, -166.0, -150.0, '$2K + F_2 = 2KF $', 0],
    [1977, 2500, -156.0, -150.0, '$2Na + F_2 = 2NaF $', 0],
    [   0, 1287, -129.8, -134.1, '$2H + F_2 = 2HF $', 0],
    # --- added ---
    [1687, 2500, -157.5, -143.7, r'$\frac{1}{2}Si + F_2 = \frac{1}{2}SiF_4 $', 0],
], dtype=object)

# ======================================================================
# CHLORIDES (Reed 1971) — raw: K, kcal/mol Cl2
# ======================================================================

# Metal solid, chloride solid
clss = np.array([
    [   0,  465, -110.9,  -92.9, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', -5],
    [   0, 1055, -188.0, -154.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [   0,    0,  -12.3,  -12.3, r'$ \frac{1}{2}C + Cl_2 = \frac{1}{2}CCl_4 $', 0],
    [   0,    0,  -45.0,  -45.0, '$2H + Cl_2 = 2HCl $', -11],
    [   0,  459, -193.6, -177.6, '$2Li + Cl_2 = 2LiCl $', 0],
    [   0,  336, -209.4, -193.2, '$2K + Cl_2 = 2KCl $', 0],
    [   0,  371, -196.8, -180.0, '$2Na + Cl_2 = 2NaCl $', -5],
    [   0,    0,  -36.1,  -36.1, r'$\frac{1}{3} W + Cl_2 = \frac{1}{3} WCl_6 $', 8],
    # --- added (Barin 1993, linear approximations) ---
    [   0,  923, -141.2, -110.5, '$Mg + Cl_2 = MgCl_2 $', 0],
], dtype=object)

# Metal liquid, chloride solid
clls = np.array([
    [ 459,  887, -177.6, -161.0, '$2Li + Cl_2 = 2LiCl $', 0],
    [ 336, 1031, -193.2, -161.0, '$2K + Cl_2 = 2KCl $', 0],
    [ 371, 1073, -180.0, -149.4, '$2Na + Cl_2 = 2NaCl $', 0],
    # --- added ---
    [ 923,  987, -110.5, -108.4, '$Mg + Cl_2 = MgCl_2 $', 0],
], dtype=object)

clgs = EMPTY()

# Metal solid, chloride liquid
clsl = np.array([
    [ 465,  500,  -92.9,  -91.7, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', 0],
    [1055, 1123, -154.0, -152.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [   0,  548,  -36.1,  -15.0, r'$\frac{1}{3} W + Cl_2 = \frac{1}{3} WCl_6 $', 0],
    # --- added (TiCl4: b.p. 409 K) ---
    [   0,  409,  -95.8,  -88.8, r'$\frac{1}{2}Ti + Cl_2 = \frac{1}{2}TiCl_4 $', 0],
], dtype=object)

# Metal liquid, chloride liquid
clll = np.array([
    [1123, 1755, -152.0, -136.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [ 887, 1597, -161.0, -141.2, '$2Li + Cl_2 = 2LiCl $', 0],
    [1031, 1043, -161.0, -160.0, '$2K + Cl_2 = 2KCl $', 0],
    [1073, 1156, -149.4, -145.6, '$2Na + Cl_2 = 2NaCl $', 0],
    # --- added ---
    [ 987, 1363, -108.4,  -95.9, '$Mg + Cl_2 = MgCl_2 $', 0],
], dtype=object)

# Metal gas, chloride liquid
clgl = np.array([
    [1755, 1900, -136.0, -128.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [1597, 1655, -141.2, -138.4, '$2Li + Cl_2 = 2LiCl $', 0],
    [1043, 1680, -160.0, -122.4, '$2K + Cl_2 = 2KCl $', 0],
    [1156, 1738, -145.6, -110.0, '$2Na + Cl_2 = 2NaCl $', 0],
    # --- added ---
    [1363, 1685,  -95.9,  -83.0, '$Mg + Cl_2 = MgCl_2 $', 0],
], dtype=object)

# Metal solid, chloride gas
clsg = np.array([
    [ 500,  932,  -91.7,  -84.6, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', 0],
    [ 548, 1500,  -15.0,   -0.8, r'$\frac{1}{3} W + Cl_2 = \frac{1}{3} WCl_6 $', 0],
    # --- added ---
    [ 409, 1941,  -88.8,  -62.7, r'$\frac{1}{2}Ti + Cl_2 = \frac{1}{2}TiCl_4 $', 0],
], dtype=object)

# Metal liquid, chloride gas
cllg = np.array([
    [ 932, 2273,  -84.6,  -70.2, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', 0],
    # --- added ---
    [1941, 2500,  -62.7,  -53.0, r'$\frac{1}{2}Ti + Cl_2 = \frac{1}{2}TiCl_4 $', 0],
], dtype=object)

# Metal gas, chloride gas
clgg = np.array([
    [2273, 2500,  -70.2,  -71.6, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', 0],
    [1900, 2500, -128.0, -114.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [   0, 2500,  -12.3,   27.4, r'$ \frac{1}{2}C + Cl_2 = \frac{1}{2}CCl_4 $', 0],
    [   0, 2500,  -45.0,  -53.3, '$2H + Cl_2 = 2HCl $', 0],
    [1655, 2500, -138.4, -118.4, '$2Li + Cl_2 = 2LiCl $', 0],
    [1680, 2500, -122.4, -110.4, '$2K + Cl_2 = 2KCl $', 0],
    [1738, 2500, -110.0,  -96.8, '$2Na + Cl_2 = 2NaCl $', 0],
    # --- added ---
    [1685, 2000,  -83.0,  -71.0, '$Mg + Cl_2 = MgCl_2 $', 0],
], dtype=object)

# ======================================================================
# HYDRIDES (added, Barin 1993) — raw: K, kcal/mol H2
# ======================================================================

# Metal solid, hydride solid
hyss = np.array([
    [   0,  454,  -32.6,  -27.0, '$2Li + H_2 = 2LiH $', 14],
    [   0,  371,  -16.1,  -11.6, '$2Na + H_2 = 2NaH $', 0],
    [   0, 1089,  -33.4,   -6.9, '$Ca + H_2 = CaH_2 $', 0],
    [   0,  561,   -8.4,    0.0, '$Mg + H_2 = MgH_2 $', 0],
    [   0, 1200,  -27.7,   -2.7, '$Ti + H_2 = TiH_2 $', -44],
    [   0, 1500,  -38.0,  -30.2, '$Zr + H_2 = ZrH_2 $', -20],
], dtype=object)

# Metal liquid, hydride solid
hyls = np.array([
    [ 454,  961,  -27.0,   -8.8, '$2Li + H_2 = 2LiH $', 0],
    [ 371,  700,  -11.6,    0.5, '$2Na + H_2 = 2NaH $', 0],
    [1115, 1400,   -5.9,    4.6, '$Ca + H_2 = CaH_2 $', 0],
], dtype=object)

hygs = EMPTY()

# Metal solid, hydride liquid (CaH2 melts at 1089 K, just below Ca m.p.)
hysl = np.array([
    [1089, 1115,   -6.9,   -5.9, '$Ca + H_2 = CaH_2 $', 0],
], dtype=object)

# Metal liquid, hydride liquid
hyll = np.array([
    [ 961, 1150,   -8.8,   -2.1, '$2Li + H_2 = 2LiH $', 0],
], dtype=object)

hygl = EMPTY()
hysg = EMPTY()
hylg = EMPTY()
hygg = EMPTY()

# ======================================================================
# SULFIDES (added, Barin 1993) — raw: K, kcal/mol S2
# ======================================================================

# Metal solid, sulfide solid
suss = np.array([
    [   0, 1422,  -47.2,    0.0, '$2Fe + S_2 = 2FeS $', -38],
    [   0, 1519, -102.8,  -37.4, '$2Mn + S_2 = 2MnS $', 0],
    [   0,  693,  -96.4,  -68.4, '$2Zn + S_2 = 2ZnS $', 0],
    [   0,  923, -165.0, -125.2, '$2Mg + S_2 = 2MgS $', 0],
    [   0, 1115, -225.4, -179.7, '$2Ca + S_2 = 2CaS $', 0],
    [   0, 1358,  -41.2,   13.3, '$4Cu + S_2 = 2Cu_2S $', 12],
    [   0,  601,  -46.8,  -22.7, '$2Pb + S_2 = 2PbS $', 11],
    [   0, 1405,  -45.4,    0.0, '$3Ni + S_2 = Ni_3S_2 $', -20],
    [   0,  933,  -76.9,  -38.7, r'$\frac{4}{3}Al + S_2 = \frac{2}{3}Al_2S_3 $', 0],
], dtype=object)

# Metal liquid, sulfide solid
suls = np.array([
    [1519, 2339,  -37.4,   -2.4, '$2Mn + S_2 = 2MnS $', 0],
    [ 693, 1180,  -68.4,  -46.5, '$2Zn + S_2 = 2ZnS $', 0],
    [ 923, 1363, -125.2, -106.3, '$2Mg + S_2 = 2MgS $', 0],
    [1115, 1757, -179.7, -152.7, '$2Ca + S_2 = 2CaS $', 0],
    [1358, 2000,   13.3,   41.8, '$4Cu + S_2 = 2Cu_2S $', 0],
    [ 601, 1395,  -22.7,    8.7, '$2Pb + S_2 = 2PbS $', 0],
    [ 933, 2000,  -38.7,    5.9, r'$\frac{4}{3}Al + S_2 = \frac{2}{3}Al_2S_3 $', 0],
    [   0,  630,  -23.4,    1.2, '$2Hg + S_2 = 2HgS $', 0],
], dtype=object)

# Metal gas, sulfide solid
sugs = np.array([
    [1180, 1900,  -46.5,  -14.5, '$2Zn + S_2 = 2ZnS $', 0],
    [1363, 2100, -106.3,  -71.5, '$2Mg + S_2 = 2MgS $', 0],
    [1757, 2300, -152.7, -127.7, '$2Ca + S_2 = 2CaS $', 0],
    [ 630,  857,    1.2,   10.4, '$2Hg + S_2 = 2HgS $', 0],
], dtype=object)

susl = EMPTY()

# Metal liquid, sulfide liquid
sull = np.array([
    [1395, 2022,    8.7,   35.7, '$2Pb + S_2 = 2PbS $', 0],
], dtype=object)

sugl = EMPTY()
susg = EMPTY()
sulg = EMPTY()

# Metal gas, sulfide gas
sugg = np.array([
    [   0, 2000,   16.0,   18.0, '$C + S_2 = CS_2 $', 0],
    [   0, 2000,  -16.0,   21.4, '$4H + S_2 = 2H_2S $', 0],
], dtype=object)


# ----------------------------------------------------------------------
# CONVERSION — K to degC, kcal to kJ (label offsets stay UNSCALED)
# ----------------------------------------------------------------------
def convert_units(*arrays):
    for arr in arrays:
        if arr.size == 0:
            continue
        numeric = arr[:, 0:4].astype(float)
        numeric[:, 0:2] -= 273.15        # K -> degC
        numeric[:, 2:4] *= 4.184         # kcal -> kJ
        arr[:, :4] = numeric
        if arr.shape[1] > 5:
            arr[:, 5] = arr[:, 5].astype(float)


# ----------------------------------------------------------------------
# Family registry
# ----------------------------------------------------------------------
def make_phase_dict(ss, ls, gs, sl, ll, gl, sg, lg, gg):
    return {'ss': ss, 'ls': ls, 'gs': gs, 'sl': sl, 'll': ll,
            'gl': gl, 'sg': sg, 'lg': lg, 'gg': gg}


FAMILIES = {
    'oxides': dict(
        color='#d5433c', compound='oxide', gas=r'O$_2$',
        phases=make_phase_dict(oxss, oxls, oxgs, oxsl, oxll, oxgl, oxsg, oxlg, oxgg)),
    'carbides': dict(
        color='#666666', compound='carbide', gas='C',
        phases=make_phase_dict(cass, cals, cags, casl, call, cagl, casg, calg, cagg)),
    'nitrides': dict(
        color='#3b6fd4', compound='nitride', gas=r'N$_2$',
        phases=make_phase_dict(niss, nils, nigs, nisl, nill, nigl, nisg, nilg, nigg)),
    'fluorides': dict(
        color='#2fa84f', compound='fluoride', gas=r'F$_2$',
        phases=make_phase_dict(flss, flls, flgs, flsl, flll, flgl, flsg, fllg, flgg)),
    'chlorides': dict(
        color='#c9a227', compound='chloride', gas=r'Cl$_2$',
        phases=make_phase_dict(clss, clls, clgs, clsl, clll, clgl, clsg, cllg, clgg)),
    'hydrides': dict(
        color='#9a5fc7', compound='hydride', gas=r'H$_2$',
        phases=make_phase_dict(hyss, hyls, hygs, hysl, hyll, hygl, hysg, hylg, hygg)),
    'sulfides': dict(
        color='#b4643c', compound='sulfide', gas=r'S$_2$',
        phases=make_phase_dict(suss, suls, sugs, susl, sull, sugl, susg, sulg, sugg)),
}

# Apply the unit conversion to every table, exactly once.
convert_units(*[arr for fam in FAMILIES.values() for arr in fam['phases'].values()])


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
STYLES = {
    'ss': dict(ls='-',  alpha=1.0),
    'ls': dict(ls='--', alpha=1.0),
    'gs': dict(ls=':',  alpha=1.0),
    'sl': dict(ls='-',  alpha=0.6),
    'll': dict(ls='--', alpha=0.6),
    'gl': dict(ls=':',  alpha=0.6),
    'sg': dict(ls='-',  alpha=0.3),
    'lg': dict(ls='--', alpha=0.3),
    'gg': dict(ls=':',  alpha=0.3),
}


def plot_family(ax, phases, color, title, ylabel, compound,
                xlabel='Temperature (°C)'):
    """Draw one Ellingham diagram. `compound` names the legend box
    ('oxide', 'sulfide', ...)."""
    for phase, arr in phases.items():
        if arr.size == 0:
            continue
        st = STYLES[phase]
        for row in arr:
            ax.plot([float(row[0]), float(row[1])],
                    [float(row[2]), float(row[3])],
                    color=color, ls=st['ls'], alpha=st['alpha'],
                    marker='.', markersize=2.25)

    # ss reaction labels (as in the original figure)
    for row in phases.get('ss', EMPTY()):
        ax.text(float(row[0]) - 25, float(row[2]) + float(row[5]), row[4],
                horizontalalignment='right', verticalalignment='center',
                fontsize=8)

    # ticks, limits, grid
    xticks = list(range(0, 2001, 200))
    yticks = np.arange(-1300, 100, 100)
    ax.set_xlim([-800, 2000]); ax.set_xticks(xticks)
    ax.set_ylim([-1300, 50]);  ax.set_yticks(yticks)
    for line in xticks:
        ax.axvline(line, color='0.5', alpha=0.5, zorder=-9)
    ax.axvline(0, color='k'); ax.axhline(0, color='k')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, x=0.64)
    ax.set_ylabel(ylabel)

    # ---- legend box: metal state x compound state ----
    rectpos = [900, 1970, -1290, -1060]
    rectpos1 = [900, 1970, -1400, -1300]
    ax.add_patch(patches.Rectangle(
        (rectpos[0], rectpos[2]),
        rectpos[1] - rectpos[0], rectpos[3] - rectpos[2],
        facecolor='#ffffff', fill=True, edgecolor='k', linewidth=1))

    ax.text(rectpos[0] + (rectpos[1]-rectpos[0])/2 + 155, rectpos[3]-30,
            'Metal', ha='center', fontsize=9, fontweight='bold')
    ax.text(rectpos[0] + (rectpos[1]-rectpos[0])/4 + 170, rectpos[3]-65,
            'Solid', ha='center', fontsize=9)
    ax.text(rectpos[0] + (rectpos[1]-rectpos[0])/2 + 155, rectpos[3]-65,
            'Liquid', ha='center', fontsize=9)
    ax.text(rectpos[0] + 3*(rectpos[1]-rectpos[0])/4 + 140, rectpos[3]-65,
            'Gas', ha='center', fontsize=9)
    ax.text(rectpos[0]+70, rectpos[3]-200, 'Compound', ha='center',
            fontsize=9, rotation=90, fontweight='bold')
    ax.text(rectpos[0]+290, rectpos[3]-110, 'Solid', ha='right', fontsize=9)
    ax.text(rectpos[0]+290, rectpos[3]-155, 'Liquid', ha='right', fontsize=9)
    ax.text(rectpos[0]+290, rectpos[3]-200, 'Gas', ha='right', fontsize=9)

    # line-style key (uses the family colour and compound name)
    c = color
    key = [
        (1260, '-',  1.0, f'Metal solid, {compound} solid'),
        (1520, '--', 1.0, f'Metal liquid, {compound} solid'),
        (1780, ':',  1.0, f'Metal gas, {compound} solid'),
        (1260, '-',  0.6, f'Metal solid, {compound} liquid'),
        (1520, '--', 0.6, f'Metal liquid, {compound} liquid'),
        (1780, ':',  0.6, f'Metal gas, {compound} liquid'),
        (1260, '-',  0.3, f'Metal solid, {compound} gas'),
        (1520, '--', 0.3, f'Metal liquid, {compound} gas'),
        (1780, ':',  0.3, f'Metal gas, {compound} gas'),
    ]
    for i, (x0, ls, a, label) in enumerate(key):
        y = [-1160, -1208, -1255][i // 3]
        ax.plot([x0, x0 + 140], [y, y], color=c, ls=ls, alpha=a, label=label)

    # ---- sources box ----
    ax.text(rectpos1[0]+300, rectpos1[3]-100, 'Sources',
            fontsize=9, fontweight='bold')
    ax.text(rectpos1[0]+300, rectpos1[3]-110,
            r'$O_2$, $N_2$, $F_2$ and $Cl_2$ data from:', fontsize=9, va='top')
    ax.text(rectpos1[0]+300, rectpos1[3]-120,
            '\nReed, T.B., 1971. Free energy of \nformation of binary compounds. '
            '\nMIT Press, Cambridge, Mass.',
            fontsize=8, va='top', fontstyle='italic')
    ax.text(rectpos1[0]+300, rectpos1[3]-130, '\n\n\n\nC data from:',
            fontsize=9, va='top')
    ax.text(rectpos1[0]+300, rectpos1[3]-140,
            '\n\n\n\n\nColtters, R.G., 1985. Thermodynamics \nof binary metallic '
            'carbides: A review. \nMaterials Science and Engineering \n76, 1–50.',
            fontsize=8, va='top', fontstyle='italic')


def save_family_figure(name):
    fam = FAMILIES[name]
    ylabel = (r'Standard free energy of formation ($\Delta G_f^\circ$) kJ/mol '
              + fam['gas'])
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_family(ax, fam['phases'], fam['color'], name.capitalize(),
                ylabel, fam['compound'])
    plt.tight_layout()
    out = f'ellingham_{name}.pdf'
    plt.savefig(out, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Ellingham diagrams for selected elements.')
    parser.add_argument('--elements', type=str, default='all',
                        help='Comma-separated element symbols, e.g. Al,Fe,Mg '
                             '(case insensitive)')
    parser.add_argument('--phases', type=str, default=None,
                        help='Comma-separated phase codes to plot, e.g. ss,ll '
                             '(all phases if omitted)')
    parser.add_argument('--families', type=str, default='all',
                        help='Comma-separated families: '
                             + ','.join(FAMILIES) + ' (default: all)')
    args = parser.parse_args()

    allowed_elements = None
    if args.elements.lower() != 'all':
        allowed_elements = {e.strip().capitalize()
                            for e in args.elements.split(',')}

    allowed_phases = set(args.phases.split(',')) if args.phases else None

    if args.families.lower() == 'all':
        selected = list(FAMILIES)
    else:
        selected = [f.strip().lower() for f in args.families.split(',')]
        unknown = [f for f in selected if f not in FAMILIES]
        if unknown:
            raise SystemExit(f'Unknown families: {unknown}. '
                             f'Valid: {", ".join(FAMILIES)}')

    for name in selected:
        phases = FAMILIES[name]['phases']
        if allowed_elements is not None:
            phases = filter_anion_dict(phases, allowed_elements)
        if allowed_phases is not None:
            phases = {p: a for p, a in phases.items() if p in allowed_phases}
        if all(arr.size == 0 for arr in phases.values()):
            print(f'Skipping {name} – no matching reactions.')
            continue
        # temporarily swap in the filtered set
        original = FAMILIES[name]['phases']
        FAMILIES[name]['phases'] = phases
        save_family_figure(name)
        FAMILIES[name]['phases'] = original
