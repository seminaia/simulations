"""Melting points (K, 1 atm) used to mark phase changes on the Ellingham lines.

Approximate literature values: check them before publication use. Compounds that
decompose or sublime instead of melting are left out on purpose.
Add or edit entries freely; anything missing is simply not marked.
"""

METAL_MELTING_K = {
    'Ag': 1235, 'Al': 933, 'Au': 1337, 'Ba': 1000, 'Be': 1560, 'Bi': 544,
    'Ca': 1115, 'Cd': 594, 'Ce': 1068, 'Co': 1768, 'Cr': 2180, 'Cu': 1358,
    'Fe': 1811, 'Ga': 303, 'Ge': 1211, 'Hf': 2506, 'In': 430, 'K': 337,
    'La': 1193, 'Li': 454, 'Mg': 923, 'Mn': 1519, 'Mo': 2896, 'Na': 371,
    'Nb': 2750, 'Ni': 1728, 'Pb': 601, 'Pd': 1828, 'Pt': 2041, 'Sb': 904,
    'Si': 1687, 'Sn': 505, 'Sr': 1050, 'Ta': 3290, 'Th': 2023, 'Ti': 1941,
    'U': 1405, 'V': 2183, 'W': 3695, 'Y': 1799, 'Zn': 693, 'Zr': 2128,
}

OXIDE_MELTING_K = {
    'Al2O3': 2345, 'BaO': 2196, 'BeO': 2820, 'Bi2O3': 1097, 'CaO': 2886,
    'CeO2': 2673, 'CoO': 2103, 'Cr2O3': 2603, 'Cu2O': 1508, 'CuO': 1599,
    'FeO': 1650, 'Fe3O4': 1870, 'GeO2': 1388, 'HfO2': 3031, 'La2O3': 2588,
    'Li2O': 1711, 'MgO': 3098, 'MnO': 2115, 'Mn3O4': 1835, 'MoO3': 1075,
    'Nb2O5': 1785, 'NiO': 2228, 'PbO': 1161, 'Sb2O3': 929, 'SiO2': 1996,
    'SnO2': 1903, 'SrO': 2804, 'Ta2O5': 2145, 'Ti2O3': 2115, 'TiO2': 2116,
    'V2O3': 2243, 'V2O5': 963, 'VO2': 1818, 'WO3': 1746, 'Y2O3': 2712,
    'ZrO2': 2983, 'Ga2O3': 2013,
}
