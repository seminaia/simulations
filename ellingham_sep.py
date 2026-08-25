#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ellingham_separate.py

Generates separate Ellingham diagrams for Oxides, Carbides, Nitrides,
Fluorides, and Chlorides. Data from Reed (1971) and Coltters (1985).
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # avoids FT_Render_Glyph crash
import matplotlib.pyplot as pl
import matplotlib.patches as patches

# ----------------------------------------------------------------------
# Plot style (same as original)
# ----------------------------------------------------------------------
pl.rcParams.update({
    'mathtext.default': 'regular',
    'mathtext.fontset': 'custom',
    'mathtext.it': 'Arial:italic',
    'mathtext.rm': 'Arial',
    'font.family': 'Arial',
})

# ======================================================================
# DATA – exactly as in your original script
# ======================================================================

# ---------- Oxides ----------
oxss = np.array([
    [   0, 932, -266.6, -220.0, r'$\frac{4}{3} Al + O_2 = \frac{2}{3} Al_2O_3$', -13],
    [   0, 904, -111.0,  -74.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 3],
    [   0, 983, -265.0, -222.0, '$2Ba + O_2 = 2BaO $', 0],
    [   0, 544,  -92.0,  -69.0, r'$\frac{4}{3} Bi + O_2 = \frac{2}{3} Bi_2O_3$', 6],
    [   0, 723, -200.5, -171.5, r'$\frac{4}{3} B + O_2 = \frac{2}{3} B_2O_3$', -3],
    [   0,1123, -303.0, -249.0, '$2Ca + O_2 = 2CaO $', 0],
    [   0,   0,  -55.6,  -55.6, '$2C + O_2 = 2CO $', 0],
    [   0,   0,  -94.5,  -94.5, '$C + O_2 = CO_2 $', -8],
    [   0, 302, -151.8, -125.0, '$4Cs + O_2 = 2Cs_2O$', -14],
    [   0,1357,  -80.0,  -33.0, '$4Cu + O_2 = 2Cu_2O $', 0],
    [   0,1357,  -74.5,  -16.0, '$2Cu + O_2 = 2CuO $', 0],
    [   0,   0, -119.3, -119.3, '$4H + O_2 = 2H_2O$', 12],
    [   0,1642, -124.1,  -75.0, '$2Fe + O_2 = 2FeO$', -9],
    [   0,1809, -129.2,  -55.5, r'$\frac{4}{3} Fe + O_2 = \frac{2}{3} Fe_2O_3$', -5],
    [   0, 453, -286.0, -258.0, '$4Li + O_2 = 2Li_2O $', -12],
    [   0,1068, -120.0,  -77.0, r'$ \frac{2}{3}Mo + O_2 = \frac{2}{3}MoO_3 $', -3],
    [   0, 923, -286.0, -240.0, '$2Mg + O_2 = 2MgO $', 8],
    [   0,   0,  -44.0,  -44.0, '$2Hg + O_2 = 2HgO$', 0],
    [   0,1764, -181.0, -112.0, r'$\frac{4}{5}Nb + O_2 = \frac{2}{5}Nb_2O_5$', 0],
    [   0, 734,  -32.0,    0.0, r'$\frac{3}{2} Pt + O_2 = \frac{1}{2}Pt_3O_4 $', 0],
    [   0, 336, -172.0, -151.0, '$4K + O_2 = 2K_2O $', -14],
    [   0, 312, -157.8, -138.0, '$4Rb + O_2 = 2Rb_2O$', -8],
    [   0,1685, -216.5, -145.8, '$Si + O_2 = SiO_2 $', 0],
    [   0, 480,  -14.0,    0.0, '$4Ag + O_2 = 2Ag_2O $', 0],
    [   0, 371, -197.0, -176.0, '$4Na + O_2 = 2Na_2O $', 3],
    [   0,1043, -281.0, -233.0, '$2Sr + O_2 = 2SrO$', 5],
    [   0, 505, -138.8, -114.0, '$Sn + O_2 = SnO_2 $', -11],
    [   0,1940, -225.5, -142.5, '$Ti + O_2 = TiO_2 $', 0],
    [   0,1940, -247.5, -161.0, '$2Ti + O_2 = 2TiO$', 0],
    [   0,1818, -168.0, -100.0, '$V + O_2 = VO_2$', -9],
    [   0, 943, -149.5, -110.0, r'$\frac{4}{5}V + O_2 = \frac{2}{5}V_2O_5$', 2],
    [   0,1743, -133.0,  -67.0, r'$ \frac{2}{3}W + O_2 = \frac{2}{3}WO_3 $', -10],
    [   0, 693, -166.0, -134.0, '$2Zn + O_2 = 2ZnO $', 4],
    [   0,2125, -262.0, -166.0, '$Zr + O_2 = ZrO_2 $', 9],
])

oxls = np.array([
    [ 932,2345, -220.0, -147.6, r'$\frac{4}{3} Al + O = \frac{2}{3} Al_2O_3$', 0],
    [ 904, 928,  -74.0,  -73.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 0],
    [ 983,1895, -222.0, -183.0, '$2Ba + O_2 = 2BaO $', 0],
    [ 544,1098,  -69.0,  -44.0, r'$\frac{4}{3} Bi + O_2 = \frac{2}{3} Bi_2O_3$', 0],
    [1123,1756, -249.0, -217.0, '$2Ca + O_2 = 2CaO $', 0],
    [ 302, 763, -125.0,  -84.0, '$4Cs + O_2 = 2Cs_2O$', -16],
    [1357,1509,  -33.0,  -28.0, '$4Cu + O_2 = 2Cu_2O $', 0],
    [1357,1609,  -16.0,   -9.5, '$2Cu + O_2 = 2CuO $', 0],
    [ 453,1597, -258.0, -173.0, '$4Li + O_2 = 2Li_2O $', 0],
    [ 336, 980, -151.0, -107.0, '$4K + O_2 = 2K_2O $', 0],
    [   0, 630,  -44.0,  -10.0, '$2Hg + O_2 = 2HgO$', 0],
    [ 312, 910, -138.0,  -96.0, '$4Rb + O_2 = 2Rb_2O$', -8],
    [1685,1696, -145.8, -145.4, '$Si + O_2 = SiO_2 $', 0],
    [ 371,1156, -176.0, -122.0, '$4Na + O_2 = 2Na_2O $', 0],
    [1940,2128, -142.5, -134.5, '$Ti + O_2 = TiO_2 $', 0],
    [1940,2033, -161.0, -159.0, '$2Ti + O_2 = 2TiO$', 0],
    [ 693,1180, -134.0, -109.0, '$2Zn + O_2 = 2ZnO $', 0],
    [2125,2980, -166.0, -130.0, '$Zr + O_2 = ZrO_2 $', 0],
])

oxgs = np.array([
    [1895,2191, -183.0, -159.0, '$2Ba + O_2 = 2BaO $', 0],
    [1756,2887, -217.0, -117.0, '$2Ca + O_2 = 2CaO $', 0],
    [1597,2000, -173.0, -128.0, '$4Li + O_2 = 2Li_2O $', 0],
    [ 923,1376, -240.0, -214.0, '$2Mg + O_2 = 2MgO $', 0],
    [ 630, 740,  -10.0,    0.0, '$2Hg + O_2 = 2HgO$', 0],
    [1156,1193, -122.0, -119.0, '$4Na + O_2 = 2Na_2O $', 0],
    [1180,2240, -109.0,   -9.0, '$2Zn + O_2 = 2ZnO $', 0],
])

oxsl = np.array([
    [ 723,2313, -171.5, -112.0, r'$\frac{4}{3} B + O = \frac{2}{3} B_2O_3$', 0],
    [1642,1809,  -75.0,  -71.9, '$2Fe + O_2 = 2FeO$', 0],
    [1068,1530,  -77.0,  -64.0, '$ something Mo $', 0],
    [1818,2190, -100.0,  -96.0, '$V + O_2 = VO_2$', 0],
    [1743,2100,  -67.0,  -57.0, r'$ \frac{2}{3}W + O_2 = \frac{2}{3}WO_3 $', 0],
])

oxll = np.array([
    [2345,2736, -147.6, -128.5, r'$\frac{4}{3} Al + O = \frac{2}{3} Al_2O_3$', 0],
    [ 928,1698,  -73.0,  -45.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 0],
    [1098,1852, -44.0,   -12.0, r'$\frac{4}{3} Bi + O_2 = \frac{2}{3} Bi_2O_3$', 0],
    [1809,2000, -71.9,   -67.9, '$2Fe + O_2 = 2FeO$', 0],
    [1376,3125, -214.0,   52.0, '$2Mg + O_2 = 2MgO $', 0],
    [ 763, 915,  -84.0,  -73.0, '$4Cs + O_2 = 2Cs_2O$', -16],
    [1509,2500,  -28.0,   -9.5, '$4Cu + O_2 = 2Cu_2O $', 0],
    [1609,1870,   -9.5,      0, '$2Cu + O_2 = 2CuO $', 0],
    [ 980,1031, -107.0, -104.0, '$4K + O_2 = 2K_2O $', 0],
    [ 910, 952,  -96.0,  -95.0, '$4Rb + O_2 = 2Rb_2O$', -8],
    [1696,2500, -145.4, -107.8, '$Si + O_2 = SiO_2 $', 0],
    [2128,2500, -134.5, -121.5, '$Ti + O_2 = TiO_2 $', 0],
    [2033,2500, -159.0, -142.5, '$2Ti + O_2 = 2TiO$', 0],
    [2190,2500,  -96.0,  -81.0, '$V + O_2 = VO_2$', 0],
])

oxgl = np.array([
    [2191,2500, -159.0, -131.0, '$2Ba + O_2 = 2BaO $', 0],
    [1031,1325, -104.0,  -71.0, '$4K + O_2 = 2K_2O $', 0],
    [1193,1600, -119.0,  -62.0, '$4Na + O_2 = 2Na_2O $', 0],
    [2240,2340,   -9.0,    0.0, '$2Zn + O_2 = 2ZnO $', 0],
])

oxsg = np.array([
    [0,3400,  -55.6, -191.9, '$2C + O_2 = 2CO $', 0],
    [0,3400,  -94.5,  -94.5, '$C + O_2 = CO_2 $', 0],
    [1530,2500,  -64.0,  -52.0, '$ something Mo $', 0],
    [2100,2500,  -57.0,  -52.0, r'$ \frac{2}{3}W + O_2 = \frac{2}{3}WO_3 $', 0],
])

oxlg = np.array([
    [ 915, 955,  -73.0,  -72.0, '$4Cs + O_2 = 2Cs_2O$', -16],
    [1698,1908,  -45.0,  -32.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 0],
])

oxgg = np.array([
    [0,3400, -119.3,  -26.6, '$4H + O_2 = 2H_2O$', 0],
    [1325,2160,  -71.0,    0.0, '$4K + O_2 = 2K_2O $', 0],
    [1600,2250,  -62.0,    0.0, '$4Na + O_2 = 2Na_2O $', 0],
    [1908,2380,  -32.0,    0.0, r'$\frac{4}{3} Sb + O_2 = \frac{2}{3} Sb_2O_3$', 0],
])

# ---------- Carbides ----------
cass = np.array([
    [   0, 1414,   -57,    -49, '$ Si + C = SiC $', -9],
    [   0, 1750,  -160, -150.5, '$ Ti + C = TiC $', -5],
    [   0,  723,    23,     -1, '$3Fe + C = Fe_3C $', 0],
    [   0, 1290,   -31,    -34, '$2W + C = W_2C$', -1],
    [   0,  800, -39.5,    -45, '$W + C = WC$', -10],
    [   0, 1000,   -70,    -59, '$2Mo + C = Mo_2C$', -12],
    [   0,  720,  -183,   -175, '$Zr + C = ZrC$', 1],
])
cals = np.array([
    [1414,2000,    -49,    -30, '$ Si + C = SiC $', 0],
])
cags = np.array([[0, 0, 0, 0, ' ', 0]])
casl = np.array([[0, 0, 0, 0, ' ', 0]])
call = np.array([[0, 0, 0, 0, ' ', 0]])
cagl = np.array([[0, 0, 0, 0, ' ', 0]])
casg = np.array([[0, 0, 0, 0, ' ', 0]])
calg = np.array([[0, 0, 0, 0, ' ', 0]])
cagg = np.array([[0, 0, 0, 0, ' ', 0]])

# ---------- Nitrides ----------
niss = np.array([
    [   0, 932, -144.3, -101.0, '$2Al + N_2 = 2AlN $', 0],
    [   0,2300, -121.4,  -20.8, '$2B + N_2 = 2BN$', 0],
    [   0,1809,   -5.8,   38.5, '$8Fe + N_2 = 2Fe_4N $', 12],
    [   0, 923, -109.6,  -65.8, '$3Mg + N_2 = Mg_3N_2 $', 8],
    [   0,1150,  -31.9,    0.0, '$4Mo + N_2 = Mo_2N $', 10],
    [   0,0, -24.1,  -24.1, '$6H + N_2 = 2NH_3$', 0],
    [   0,1680,  -90.0,  -22.5, r'$\frac{3}{2}Si + N_2 = \frac{1}{2}Si_3N_4 $', -6],
    [   0,1940, -160.5,  -73.4, '$2Ti + N_2 = 2TiN $', 1],
    [   0,2190,  -83.3,    3.6, '$2V + N_2 = 2VN$', -13],
    [   0,2128, -163.8,  -67.2, '$2Zr + N_2 = 2ZrN $', -2],
])
nils = np.array([
    [2300,2500, -20.8,       0, '$2B + N_2 = 2BN$', 0],
    [ 923,1376,  -65.8,  -41.3, '$3Mg + N_2 = Mg_3N_2', 0],
    [1680,2130,  -22.5,    0.0, r'$\frac{3}{2}Si + N_2 = \frac{1}{2}Si_3N_4 $', 0],
])
nigs = np.array([[0, 0, 0, 0, ' ', 0]])
nisl = np.array([[0, 0, 0, 0, ' ', 0]])
nill = np.array([[0, 0, 0, 0, ' ', 0]])
nigl = np.array([[0, 0, 0, 0, ' ', 0]])
nisg = np.array([[0, 0, 0, 0, ' ', 0]])
nilg = np.array([[0, 0, 0, 0, ' ', 0]])
nigg = np.array([
    [0,2000,  -24.1,   85.2, '$6H + N_2 = 2NH_3$', 0],
])

# ---------- Fluorides ----------
flss = np.array([
    [   0, 932, -215.3, -181.0, r'$\frac{2}{3}Al + F_2 = \frac{2}{3}AlF_3 $', 0],
    [   0,1123, -288.0, -245.0, '$ Ca + F_2 = CaF_2 $', 5],
    [   0,   0, -129.8, -129.8, '$2H + F_2 = 2HF $', 0],
    [   0,   0,  -81.2,  -81.2, r'$\frac{1}{2}C + F_2 = \frac{1}{2}CF_4 $', 2],
    [   0, 453, -290.0, -271.0, '$2Li + F_2 = 2LiF $', -8],
    [   0, 336, -270.0, -253.0, '$2K + F_2 = 2KF $', +0.3],
    [   0, 371, -274.0, -255.0, '$2Na + F_2 = 2NaF $', -0.3],
])
flls = np.array([
    [ 932,1545, -181.0, -156.0, r'$\frac{2}{3}Al + F_2 = \frac{2}{3}AlF_3 $', 0],
    [1123,1691, -245.0, -224.0, '$ Ca + F_2 = CaF_2 $', 0],
    [ 453,1120, -271.0, -240.0, '$2Li + F_2 = 2LiF $', 0],
    [ 336,1031, -253.0, -214.0, '$2K + F_2 = 2KF $', 0],
    [ 371,1187, -255.0, -214.0, '$2Na + F_2 = 2NaF', 0],
])
flgs = np.array([[0, 0, 0, 0, ' ', 0]])
flsl = np.array([[0, 0, 0, 0, ' ', 0]])
flll = np.array([
    [1545,2500, -156.0, -157.0, r'$\frac{2}{3}Al + F_2 = \frac{2}{3}AlF_3 $', 0],
    [1691,1955, -224.0, -222.0, '$ Ca + F_2 = CaF_2 $', 0],
    [1120,1597, -240.0, -216.0, '$2Li + F_2 = 2LiF $', 0],
    [1031,1130, -214.0, -209.0, '$2K + F_2 = 2KF $', 0],
    [1187,1268, -214.0, -209.0, '$2Na + F_2 = 2NaF', 0],
])
flgl = np.array([
    [1955,2500, -222.0, -186.0, '$ Ca + F_2 = CaF_2 $', 0],
    [1597,1954, -216.0, -194.0, '$2Li + F_2 = 2LiF $', 0],
    [1130,1775, -209.0, -166.0, '$2K + F_2 = 2KF $', 0],
    [1268,1977, -209.0, -156.0, '$2Na + F_2 = 2NaF', 0],
])
flsg = np.array([[0, 0, 0, 0, ' ', 0]])
fllg = np.array([[0, 0, 0, 0, ' ', 0]])
flgg = np.array([
    [   0,2500,  -81.2,  -36.0, r'$\frac{1}{2}C + F_2 = \frac{1}{2}CF_4 $', 0],
    [   0,   0, -129.8, -129.8, '$2H + F_2 = 2HF $', 0],
    [1954,2500, -194.0, -179.0, '$2Li + F_2 = 2LiF $', 0],
    [1775,2500, -166.0, -150.0, '$2K + F_2 = 2KF $', 0],
    [1977,2500, -156.0, -150.0, '$2Na + F_2 = 2NaF', 0],
    [   0,1287, -129.8, -134.1, '$2H + F_2 = 2HF $', 0],
])

# ---------- Chlorides ----------
clss = np.array([
    [   0, 465, -110.9,  -92.9, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', -5],
    [   0,1055, -188.0, -154.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [   0,   0,  -12.3,  -12.3, r'$ \frac{1}{2}C + Cl_2 = \frac{1}{2}CCl_4 $  ', 0],
    [   0,   0,  -45.0,  -45.0, '$2H + Cl_2 = 2HCl $', -11],
    [   0, 459, -193.6, -177.6, '$2Li + Cl_2 = 2LiCl $', 0],
    [   0, 336, -209.4, -193.2, '$2K + Cl_2 = 2KCl $', 0],
    [   0, 371, -196.8, -180.0, '$2Na + Cl_2 = 2NaCl $', -5],
    [   0,   0,  -36.1,  -36.1, r'$\frac{1}{3} W + Cl_2 = \frac{1}{3} WCl_6 $', 8],
])
clls = np.array([
    [ 459, 887, -177.6, -161.0, '$2Li + Cl_2 = 2LiCl $', 0],
    [ 336,1031, -193.2, -161.0, '$2K + Cl_2 = 2KCl $', 0],
    [ 371,1073, -180.0, -149.4, '$2Na + Cl_2 = 2NaCl $', 0],
])
clgs = np.array([[0, 0, 0, 0, ' ', 0]])
clsl = np.array([
    [ 465, 500,  -92.9,  -91.7, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', 0],
    [1055,1123, -154.0, -152.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [   0, 548,  -36.1,  -15.0, r'$\frac{1}{3} W + Cl_2 = \frac{1}{3} WCl_6 $', 0],
])
clll = np.array([
    [1123,1755, -152.0, -136.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [ 887,1597, -161.0, -141.2, '$2Li + Cl_2 = 2LiCl $', 0],
    [1031,1043, -161.0, -160.0, '$2K + Cl_2 = 2KCl $', 0],
    [1073,1156, -149.4, -145.6, '$2Na + Cl_2 = 2NaCl $', 0],
])
clgl = np.array([
    [1755,1900, -136.0, -128.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [1597,1655, -141.2, -138.4, '$2Li + Cl_2 = 2LiCl $', 0],
    [1043,1680, -160.0, -122.4, '$2K + Cl_2 = 2KCl $', 0],
    [1156,1738, -145.6, -110.0, '$2Na + Cl_2 = 2NaCl $', 0],
])
clsg = np.array([
    [ 500, 932,  -91.7,  -84.6, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', 0],
    [ 548,1500,  -15.0,   -0.8, r'$\frac{1}{3} W + Cl_2 = \frac{1}{3} WCl_6 $', 0],
])
cllg = np.array([
    [ 932,2273,  -84.6,  -70.2, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', 0],
])
clgg = np.array([
    [2273,2500,  -70.2,  -71.6, r'$\frac{2}{3}Al + Cl_2 = \frac{2}{3}AlCl_3 $', 0],
    [1900,2500, -128.0, -114.0, '$ Ca + Cl_2 = CaCl_2 $', 0],
    [   0,2500,  -12.3,   27.4, r'$ \frac{1}{2}C + Cl_2 = \frac{1}{2}CCl_4 $', 0],
    [   0,2500,  -45.0,  -53.3, '$2H + Cl_2 = 2HCl $', 0],
    [1655,2500, -138.4, -118.4, '$2Li + Cl_2 = 2LiCl $', 0],
    [1680,2500, -122.4, -110.4, '$2K + Cl_2 = 2KCl $', 0],
    [1738,2500, -110.0,  -96.8, '$2Na + Cl_2 = 2NaCl $', 0],
])
# ------------------------------
# CONVERSION – K to °C, kcal to kJ, but offsets are NOT scaled
# ------------------------------
def convert_units(*arrays):
    for arr in arrays:
        if arr.size == 0:
            continue
        numeric = arr[:, 0:4].astype(float)
        numeric[:, 0:2] -= 273.15      # K → °C
        numeric[:, 2:4] *= 4.184       # kcal → kJ
        arr[:, :4] = numeric
        # OFFSET (column 5) is left UNSCALED – just convert to float if needed
        if arr.shape[1] > 5:
            arr[:, 5] = arr[:, 5].astype(float)   # keep original numbers

# Apply to all arrays
convert_units(oxss, oxls, oxgs, oxsl, oxll, oxgl, oxsg, oxlg, oxgg,
              cass, cals, cags, casl, call, cagl, casg, calg, cagg,
              niss, nils, nigs, nisl, nill, nigl, nisg, nilg, nigg,
              flss, flls, flgs, flsl, flll, flgl, flsg, fllg, flgg,
              clss, clls, clgs, clsl, clll, clgl, clsg, cllg, clgg)

# ------------------------------
# Helper to build anion dictionaries
# ------------------------------
def make_anion_dict(ss, ls, gs, sl, ll, gl, sg, lg, gg):
    return {'ss': ss, 'ls': ls, 'gs': gs, 'sl': sl, 'll': ll,
            'gl': gl, 'sg': sg, 'lg': lg, 'gg': gg}

oxides   = make_anion_dict(oxss, oxls, oxgs, oxsl, oxll, oxgl, oxsg, oxlg, oxgg)
carbides = make_anion_dict(cass, cals, cags, casl, call, cagl, casg, calg, cagg)
nitrides = make_anion_dict(niss, nils, nigs, nisl, nill, nigl, nisg, nilg, nigg)
fluorides= make_anion_dict(flss, flls, flgs, flsl, flll, flgl, flsg, fllg, flgg)
chlorides= make_anion_dict(clss, clls, clgs, clsl, clll, clgl, clsg, cllg, clgg)

# ------------------------------
# Plotting function – labels only for ss phase, offset -25, unscaled offsets
# ------------------------------
def plot_anion(ax, anion_dict, color, title, ylabel, xlabel='Temperature (°C)'):
    styles = {
        'ss': {'ls': '-',  'alpha': 1.0},
        'ls': {'ls': '--', 'alpha': 1.0},
        'gs': {'ls': ':',  'alpha': 1.0},
        'sl': {'ls': '-',  'alpha': 0.6},
        'll': {'ls': '--', 'alpha': 0.6},
        'gl': {'ls': ':',  'alpha': 0.6},
        'sg': {'ls': '-',  'alpha': 0.3},
        'lg': {'ls': '--', 'alpha': 0.3},
        'gg': {'ls': ':',  'alpha': 0.3},
    }
    if isinstance(color, dict):
        phase_colors = color
    else:
        phase_colors = {phase: color for phase in styles.keys()}

    # Plot all lines
    for phase, arr in anion_dict.items():
        if arr.size == 0:
            continue
        style = styles.get(phase, {'ls': '-', 'alpha': 1.0})
        col = phase_colors.get(phase, color)
        for row in arr:
            T0 = float(row[0]); T1 = float(row[1])
            G0 = float(row[2]); G1 = float(row[3])
            ax.plot([T0, T1], [G0, G1],
                    color=col, ls=style['ls'], alpha=style['alpha'],
                    marker='.', markersize=2.25)

    # ----- LABELS ONLY FOR ss PHASE, offset -25, no scaling of offsets -----
    if 'ss' in anion_dict and anion_dict['ss'].size > 0:
        for row in anion_dict['ss']:
            T0 = float(row[0])
            G0 = float(row[2])
            label = row[4]
            off = float(row[5]) if isinstance(row[5], (int, float)) else 0.0
            ax.text(T0 - 25, G0 + off, label,
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontsize=8)

    # Axis settings (same as original)
    xticks = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    yticks = np.arange(-1300, 100, 100)
    ax.set_xlim([-800, 2000])
    ax.set_xticks(xticks)
    ax.set_ylim([-1300, 50])
    ax.set_yticks(yticks)
    for x in xticks:
        ax.axvline(x, color='0.5', alpha=0.5, zorder=-9)
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Legend box (copied from original)
    rectpos = [900, 1970, -1290, -1060]
    ax.add_patch(patches.Rectangle(
        (rectpos[0], rectpos[2]),
        rectpos[1]-rectpos[0],
        rectpos[3]-rectpos[2],
        facecolor='#ffffff', fill=True, edgecolor='k', linewidth=1
    ))
    ax.text((rectpos[1]-rectpos[0])/2 + rectpos[0] + 155,
            rectpos[3]-30, 'Metal', ha='center', fontsize=9, fontweight='bold')
    ax.text((rectpos[1]-rectpos[0])/4 + rectpos[0] + 170,
            rectpos[3]-65, 'Solid', ha='center', fontsize=9)
    ax.text((rectpos[1]-rectpos[0])/2 + rectpos[0] + 155,
            rectpos[3]-65, 'Liquid', ha='center', fontsize=9)
    ax.text(3*(rectpos[1]-rectpos[0])/4 + rectpos[0] + 140,
            rectpos[3]-65, 'Gas', ha='center', fontsize=9)
    ax.text(rectpos[0]+ 70, rectpos[3]-110, 'Compound',
            ha='center', fontsize=9, rotation=90, fontweight='bold')
    ax.text(rectpos[0]+ 290, rectpos[3]-110, 'Solid',
            ha='right', fontsize=9)
    ax.text(rectpos[0]+ 290, rectpos[3]-155, 'Liquid',
            ha='right', fontsize=9)
    ax.text(rectpos[0]+ 290, rectpos[3]-200, 'Gas',
            ha='right', fontsize=9)
    # Line style examples
    ax.plot([1260, 1400], [-1160, -1160], color='k', ls='-',  alpha=1.0)
    ax.plot([1520, 1660], [-1160, -1160], color='k', ls='--', alpha=1.0)
    ax.plot([1780, 1920], [-1160, -1160], color='k', ls=':',  alpha=1.0)
    ax.plot([1260, 1400], [-1208, -1208], color='k', ls='-',  alpha=0.6)
    ax.plot([1520, 1660], [-1208, -1208], color='k', ls='--', alpha=0.6)
    ax.plot([1780, 1920], [-1208, -1208], color='k', ls=':',  alpha=0.6)
    ax.plot([1260, 1400], [-1255, -1255], color='k', ls='-',  alpha=0.3)
    ax.plot([1520, 1660], [-1255, -1255], color='k', ls='--', alpha=0.3)
    ax.plot([1780, 1920], [-1255, -1255], color='k', ls=':',  alpha=0.3)

    ## Sources
    #ax.text(rectpos[0] + 30, rectpos[3]-25, 'Sources',
    #        fontsize=9, fontweight='bold')
    #ax.text(rectpos[0] + 30, rectpos[3]-30,
    #        '$O_2$, $N_2$, $F_2$ and $Cl_2$ data from:',
    #        fontsize=9, va='top')
    #ax.text(rectpos[0] + 30, rectpos[3]-35,
    #        '\nReed, T.B., 1971. Free energy of \nformation of binary compounds. \nMIT Press, Cambridge, Mass.',
    #        fontsize=8, va='top', fontstyle='italic')
    #ax.text(rectpos[0] + 30, rectpos[3]-30,
    #        '\n\n\n\nC data from:',
    #        fontsize=9, va='top')
    #ax.text(rectpos[0] + 30, rectpos[3]-44,
    #        '\n\n\n\n\nColtters, R.G., 1985. Thermodynamics \nof binary metallic carbides: A review. \nMaterials Science and Engineering \n76, 1–50.',
    #        fontsize=8, va='top', fontstyle='italic')
#
# ------------------------------
# Generate figures
# ------------------------------
def save_anion_figure(anion_dict, color, name, ylabel):
    fig, ax = pl.subplots(figsize=(10, 8))
    plot_anion(ax, anion_dict, color,
               title=f'{name.capitalize()}',
               ylabel=ylabel)
    pl.tight_layout()
    pl.savefig(f'ellingham_{name}.pdf', dpi=400, bbox_inches='tight')
    pl.close(fig)
    print(f"Saved ellingham_{name}.pdf")

save_anion_figure(oxides, 'r', 'oxides',
                  r'Standard free energy of formation ($\Delta G_f^\circ$) kJ/mol O$_2$')
save_anion_figure(carbides, '0.4', 'carbides',
                  r'Standard free energy of formation ($\Delta G_f^\circ$) kJ/mol C')
save_anion_figure(nitrides, 'b', 'nitrides',
                  r'Standard free energy of formation ($\Delta G_f^\circ$) kJ/mol N$_2$')
save_anion_figure(fluorides, [0, 1, 0], 'fluorides',
                  r'Standard free energy of formation ($\Delta G_f^\circ$) kJ/mol F$_2$')
save_anion_figure(chlorides, 'g', 'chlorides',
                  r'Standard free energy of formation ($\Delta G_f^\circ$) kJ/mol Cl$_2$')