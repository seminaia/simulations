#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ellingham.py

Generates separate Ellingham diagrams for Oxides, Carbides, Nitrides,
Fluorides, Chlorides, Hydrides, Sulfides, Tellurides, Selenides,
Iodides and Bromides.

Data:
    Excel data in kcal/mol and K
  * O2, N2, F2 and Cl2 reference lines and salt data — Reed, T.B., 1971.
    Free Energy of Formation of Binary Compounds. MIT Press, Cambridge, Mass.
  * Carbides — Coltters, R.G., 1985. Thermodynamics of binary metallic
    carbides: a review. Materials Science and Engineering 76, 1-50.
  * Added bromides, iodides, hydrides, sulfides, selenides and tellurides
    follow the source workbook piecewise-linear segments. Verify against the
    primary source before publication-quality use.

Storage convention (data/ellingham/ellingham_data.xlsx):
  one sheet per family with columns
  phase_code, T0, T1, G0, G1, reaction, label_offset, element.
  Values stay in K and kcal per mole of the family reference gas
  (O2, N2, F2, Cl2, Br2, I2, H2, S2, Se2, Te2) or per mole C for carbides.
  convert_units() turns them into degC / kJ once, after loading only the
  family sheets requested by --families.

Line style code (metal state x compound state):
  metal:   solid '-'   liquid '--'   gas ':'
  compound: solid a=1.0   liquid a=0.6   gas a=0.3

Usage:
  python ellingham.py
  python ellingham.py --elements Al,Fe,Mg
  python ellingham.py --families oxides,sulfides
  python ellingham.py --families tellurides,selenides,iodides,bromides --elements all
  python ellingham.py --phases ss,ll
  python ellingham.py --no-temp-mark
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# User-editable defaults
# ----------------------------------------------------------------------
# Edit these to change what runs when no CLI flags are given. Any of them
# can still be overridden on the command line, e.g. `--elements all`.
DEFAULT_ELEMENTS = "Al,Fe,Sn,Cu,Y,Zr,Ca,C"   # comma-separated symbols, or "all"
DEFAULT_TEMP_C = 1000.0                      # °C for Richardson lines / P_eq marks
DEFAULT_PRESSURE_PA = 101325.0               # Pa, global partial pressure shift
DEFAULT_FAMILIES = "oxides,carbides,hydrides"

# ----------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------
plt.rcParams.update({
    'mathtext.default': 'regular',
    'mathtext.fontset': 'dejavusans',
    'font.family': 'DejaVu Sans',
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
# Reaction data — loaded lazily from data/ellingham/ellingham_data.xlsx
#
# The workbook stores one sheet per family. Each row is one plotted line
# segment with columns: phase_code, T0, T1, G0, G1, reaction, label_offset,
# and element. The XLS parser regenerates all XLS-backed families while
# preserving the manually-transcribed carbides sheet.
# ======================================================================
DATA_WORKBOOK = Path(__file__).resolve().parent / 'data' / 'ellingham' / 'ellingham_data.xlsx'
WORKBOOK_COLUMNS = ['phase_code', 'T0', 'T1', 'G0', 'G1', 'reaction', 'label_offset', 'element']
PHASE_CODES = ('ss', 'ls', 'gs', 'sl', 'll', 'gl', 'sg', 'lg', 'gg', 'aux')

def boudouard_reaction(T):
    """C + CO2 = 2CO 
    log(K_eq) = 9141/T - 0.000224*T -9.595 
    from Wikipedia """
    R=8.314/1000  # kJ/(mol·K)
    logK = lambda T: 9141/(T) - 0.000224*(T) -9.595
    K_eq = 10**logK(T)
    G_f = -R * (T) * np.log(K_eq) # Gibbs free energy in kJ/mol
    return G_f

def empty_phase_dict():
    return {phase_code: EMPTY() for phase_code in PHASE_CODES}


def load_family_phase_arrays(workbook_path, family_names):
    """Load only the requested family sheets from the XLSX workbook."""
    family_names = list(dict.fromkeys(family_names))
    loaded = {}
    with pd.ExcelFile(workbook_path) as workbook:
        for family in family_names:
            frame = workbook.parse(sheet_name=family)
            missing_columns = [column for column in WORKBOOK_COLUMNS if column not in frame.columns]
            if missing_columns:
                raise ValueError(
                    f"Sheet '{family}' is missing required columns: {', '.join(missing_columns)}"
                )

            phase_rows = {phase_code: [] for phase_code in PHASE_CODES}
            for phase_code, t0, t1, g0, g1, reaction, label_offset, _element in frame[WORKBOOK_COLUMNS].itertuples(index=False, name=None):
                phase_code = str(phase_code).strip().lower()
                if phase_code not in phase_rows:
                    raise ValueError(f"Unknown phase code '{phase_code}' in sheet '{family}'")
                phase_rows[phase_code].append([
                    float(t0),
                    float(t1),
                    float(g0),
                    float(g1),
                    str(reaction),
                    float(label_offset),
                ])

            loaded[family] = {
                phase_code: (np.array(rows, dtype=object) if rows else EMPTY())
                for phase_code, rows in phase_rows.items()
            }
    return loaded


# ----------------------------------------------------------------------
# CONVERSION 
# Excel data is assumed to be in kcal/mol and K; this function converts to kJ/mol and °C.
# ----------------------------------------------------------------------
def convert_units(*arrays, gibbs_already_kj=False):
    """kcal -> kJ only if the sheet's G values aren't
    K to C
    already in kJ. The carbides sheet (transcribed straight from Coltters
    1985 in kJ/mol C)."""
    gibbs_scale = 1.0 if gibbs_already_kj else 4.184
    for arr in arrays:
        if arr.size == 0:
            continue
        numeric = arr[:, 0:4].astype(float)
        numeric[:, 0:2] -= 273.15
        numeric[:, 2:4] *= gibbs_scale
        arr[:, :4] = numeric
        if arr.shape[1] > 5:
            arr[:, 5] = arr[:, 5].astype(float)


# ----------------------------------------------------------------------
# Family registry
# ----------------------------------------------------------------------
# 'eq_kind' controls what add_markings_for_elements()/the equilibrium table
# compute for each family: every family reacts with a diatomic gas anion
# (metal + X_2 = compound) except carbides, which react with solid carbon
# (metal + C = carbide), so carbides get a carbon *activity* instead of a
# gas partial pressure.
FAMILIES = {
    'oxides': dict(color='#d5433c', compound='oxide', gas=r'O$_2$', gas_plain='O2', eq_kind='pressure', phases=None),
    'carbides': dict(color='#666666', compound='carbide', gas='C', gas_plain='C', eq_kind='activity', phases=None),
    'nitrides': dict(color='#3b6fd4', compound='nitride', gas=r'N$_2$', gas_plain='N2', eq_kind='pressure', phases=None),
    'fluorides': dict(color='#2fa84f', compound='fluoride', gas=r'F$_2$', gas_plain='F2', eq_kind='pressure', phases=None),
    'chlorides': dict(color='#c9a227', compound='chloride', gas=r'Cl$_2$', gas_plain='Cl2', eq_kind='pressure', phases=None),
    'hydrides': dict(color='#9a5fc7', compound='hydride', gas=r'H$_2$', gas_plain='H2', eq_kind='pressure', phases=None),
    'sulfides': dict(color='#b4643c', compound='sulfide', gas=r'S$_2$', gas_plain='S2', eq_kind='pressure', phases=None),
    'tellurides': dict(color='#7a5c61', compound='telluride', gas=r'Te$_2$', gas_plain='Te2', eq_kind='pressure', phases=None),
    'selenides': dict(color='#2b8c82', compound='selenide', gas=r'Se$_2$', gas_plain='Se2', eq_kind='pressure', phases=None),
    'iodides': dict(color='#d14e8f', compound='iodide', gas=r'I$_2$', gas_plain='I2', eq_kind='pressure', phases=None),
    'bromides': dict(color='#e07a2d', compound='bromide', gas=r'Br$_2$', gas_plain='Br2', eq_kind='pressure', phases=None),
}


def ensure_family_phase_data(family_names):
    missing = [family for family in dict.fromkeys(family_names) if FAMILIES[family]['phases'] is None]
    if not missing:
        return
    loaded = load_family_phase_arrays(DATA_WORKBOOK, missing)
    for family in missing:
        phases = loaded[family]
        convert_units(*phases.values(), gibbs_already_kj=(family == 'carbides'))
        FAMILIES[family]['phases'] = phases



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


def declutter_label_positions(entries, min_gap):
    """Greedily push overlapping (y, ...) label positions apart.

    `entries` is a list of [x, y, text]; only `y` is adjusted. Labels are
    processed from the top (highest y) down, and each one is pulled below
    the previous one if they'd be closer than `min_gap`. Returns a new list
    of [x, adjusted_y, text] in the original input order.
    """
    order = sorted(range(len(entries)), key=lambda i: entries[i][1], reverse=True)
    adjusted = [list(e) for e in entries]
    previous_y = None
    for i in order:
        y = adjusted[i][1]
        if previous_y is not None and previous_y - y < min_gap:
            y = previous_y - min_gap
        adjusted[i][1] = y
        previous_y = y
    return adjusted


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

    ss_rows = list(phases.get('ss', EMPTY()))
    label_fontsize = 8 if len(ss_rows) <= 20 else (7 if len(ss_rows) <= 40 else 6)
    label_entries = [[float(row[0]) - 25, float(row[2]) + float(row[5]), row[4]]
                      for row in ss_rows]
    placed = declutter_label_positions(label_entries, min_gap=30)
    for row, (label_x, label_y, text) in zip(ss_rows, placed):
        label_x += 273.15
        anchor_y = float(row[2])
        if abs(label_y - anchor_y) > 6:
            ax.plot([label_x, float(row[0])], [label_y, anchor_y],
                    color=color, alpha=0.35, linewidth=0.6, zorder=1)
        ax.text(label_x, label_y, text,
                horizontalalignment='right', verticalalignment='center',
                fontsize=label_fontsize,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                           edgecolor='none', alpha=0.7))

    xticks = list(range(0, 2001, 100))
    yticks = np.arange(-2500, 500, 100)
    ax.set_xlim([-500, 2000]); 
    ax.set_ylim([-2500, 500]);  ax.set_yticks(yticks)
    for line in xticks:
        ax.axvline(line, color='0.5', alpha=0.5, zorder=-9)
    ax.axvline(0, color='k'); ax.axhline(0, color='k')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, x=0.64)
    ax.set_ylabel(ylabel)

    # ---- legend box: metal state x compound state ----
    rectpos = [0, 1000, -2400, -2000]
    rectpos1 = [0, 1000, -2400, -2000]
    ax.add_patch(patches.Rectangle(
        (rectpos[0], rectpos[2]),
        rectpos[1] - rectpos[0], rectpos[3] - rectpos[2]+50,
        facecolor='#ffffff', fill=True, edgecolor='k', linewidth=1))

    ax.text((rectpos[1]-rectpos[0])/2 +50 , rectpos[3],
            'Metal', ha='center', fontsize=9, fontweight='bold')
    ax.text((rectpos[1]-rectpos[0])/4 + 50, rectpos[3]-50,
            'Solid', ha='center', fontsize=9)
    ax.text((rectpos[1]-rectpos[0])/2 +50, rectpos[3]-50,
            'Liquid', ha='center', fontsize=9)
    ax.text(3*(rectpos[1]-rectpos[0])/4 +50, rectpos[3]-50,
            'Gas', ha='center', fontsize=9)
    ax.text(rectpos[0]+50, rectpos[3]-300, 'Compound', ha='center',
            fontsize=9, rotation=90, fontweight='bold')
    ax.text(rectpos[0]+200, rectpos[3]-100, 'Solid', ha='right', fontsize=9)
    ax.text(rectpos[0]+200, rectpos[3]-200, 'Liquid', ha='right', fontsize=9)
    ax.text(rectpos[0]+200, rectpos[3]-300, 'Gas', ha='right', fontsize=9)

    # line-style key (uses the family colour and compound name)
    c = color
    key = [
        ((rectpos[1]-rectpos[0])/2, rectpos[3]-100,'-',  1.0, f'Metal solid, {compound} solid'),
        ((rectpos[1]-rectpos[0])/4, rectpos[3]-100,'--', 1.0, f'Metal liquid, {compound} solid'),
        ((rectpos[1]-rectpos[0])*3/4, rectpos[3]-100, ':',  1.0, f'Metal gas, {compound} solid'),
        ((rectpos[1]-rectpos[0])/2, rectpos[3]-200, '-',  0.6, f'Metal solid, {compound} liquid'),
        ((rectpos[1]-rectpos[0])/4, rectpos[3]-200, '--', 0.6, f'Metal liquid, {compound} liquid'),
        ((rectpos[1]-rectpos[0])*3/4, rectpos[3]-200, ':',  0.6, f'Metal gas, {compound} liquid'),
        ((rectpos[1]-rectpos[0])/2, rectpos[3]-300, '-',  0.3, f'Metal solid, {compound} gas'),
        ((rectpos[1]-rectpos[0])/4, rectpos[3]-300, '--', 0.3, f'Metal liquid, {compound} gas'),
        ((rectpos[1]-rectpos[0])*3/4, rectpos[3]-300, ':',  0.3, f'Metal gas, {compound} gas'),
    ]
    for i, (x0, y0, ls, a, label) in enumerate(key):
        y = [y0, y0]
        ax.plot([x0, x0 + 100], y, color=c, ls=ls, alpha=a, label=label)

    # ---- sources box ----
    ax.text(rectpos1[0]+1000, rectpos1[3], 'Sources:',
            fontsize=8, fontweight='bold')
    ax.text(rectpos1[0]+1000, rectpos1[3]-10,
            r'$O_2$, $N_2$, $F_2$ and $Cl_2$ data from:', fontsize=8, va='top')
    ax.text(rectpos1[0]+1000, rectpos1[3]-20,
            '\nReed, T.B., 1971. Free energy of \nformation of binary compounds. '
            '\nMIT Press, Cambridge, Mass.',
            fontsize=8, va='top', fontstyle='italic')
    ax.text(rectpos1[0]+1000, rectpos1[3]-30, '\n\n\n\nC data from:',
            fontsize=8, va='top')
    ax.text(rectpos1[0]+1000, rectpos1[3]-40,
            '\n\n\n\n\nColtters, R.G., 1985. Thermodynamics \nof binary metallic '
            'carbides: A review. \nMaterials Science and Engineering 76, 1–50.',
            fontsize=8, va='top', fontstyle='italic')

def save_family_figure(name):
    ensure_family_phase_data([name])
    fam = FAMILIES[name]
    ylabel = (r'Standard free energy of formation ($\Delta G_f^\circ$) kJ/mol '
              + fam['gas'])
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_family(ax, fam['phases'], fam['color'], name.capitalize(),
                ylabel, fam['compound'])
    add_pressure_nomograph(ax, T_right=2000, eq_kind=fam['eq_kind'], gas_plain=fam['gas_plain'], gas_label=fam['gas'])
    add_family_gas_references(ax, name)
    plt.tight_layout()
    out = f'ellingham_{name}.pdf'
    fig.savefig(out, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

def apply_pressure_shift(phases, pressure):
    """Shifts Delta G values based on non-standard gas pressure."""
    
    R_kJ = 0.008314
    ln_P = np.log(pressure/101325)
    
    shifted_phases = {}
    for phase, arr in phases.items():
        if arr.size == 0 or phase == 'aux':
            shifted_phases[phase] = arr
            continue
        new_arr = arr.copy()
        T0_K = new_arr[:, 0].astype(float) + 273.15
        T1_K = new_arr[:, 1].astype(float) + 273.15
        new_arr[:, 2] = new_arr[:, 2].astype(float) - R_kJ * T0_K * ln_P
        new_arr[:, 3] = new_arr[:, 3].astype(float) - R_kJ * T1_K * ln_P
        shifted_phases[phase] = new_arr
    return shifted_phases

def add_family_gas_references(ax, name):
    """Adds the classic H/C auxiliary reference lines and ratio nomographs
    appropriate for `name`:
      - oxides:  both H and C lines, plus CO/CO2 and H2/H2O ratio scales
                 (in addition to the existing P(O2) scale) — the full
                 classic combined-Ellingham-diagram nomograph. Also draws
                 the Boudouard reaction (C + CO2 = 2CO) from the carbides
                 sheet's 'aux' row, since it's the same CO/CO2 gas
                 equilibrium the oxide diagram's C reference line and
                 P_CO/P_CO2 scale are calibrated against.
      - carbides: the C line only (2CO+O2=2CO2). The Boudouard reaction
                 is drawn from its own 'aux' row in the carbides sheet
                 by plot_family(), alongside the carbon-activity scale.
      - hydrides: the H line only (2H2+O2=2H2O).
    All other families are left untouched.
    """
    if name == 'oxides':
        add_reference_gas_line(ax, 'H')
        add_reference_gas_line(ax, 'C')
        add_reference_gas_line(ax, 'CO')
        add_ratio_nomograph(ax, 'H', r'$P_{H_2}/P_{H_2O}$', outward_offset=55)
        add_ratio_nomograph(ax, 'CO', r'$P_{CO}/P_{CO_2}$', outward_offset=115)
    elif name == 'carbides':
        add_reference_gas_line(ax, 'C')
    elif name == 'hydrides':
        add_reference_gas_line(ax, 'H')


REFERENCE_GAS_LINES = {
    'H': dict(g0=-119.3*4.184, g1=26.9, T0=0,T1=3400,
              reaction=r'$2H_2 + O_2 = 2H_2O$', color='#3b6fd4'),
    'CO2': dict(g0=-135.0*4.184, g1=4.7, T0=0,T1=3400,
              reaction=r'$2CO + O_2 = 2CO_2$', color='#444444'),
    'CO': dict(g0=-53.4*4.184, g1=-195.9*4.184, T0=0,T1=3400,
              reaction=r'$2C + O_2 = 2CO$', color="#808080"),
    'C': dict(g0=boudouard_reaction(-273), g1=boudouard_reaction(3400), T0=0, T1=3400,reaction=r'$C + CO_2 = 2CO$', color='k',)
}


def add_reference_gas_line(ax, point_label, T_right=2000, T_left=0):
    """Draws one of the classic auxiliary Ellingham reference lines (H or C)
    across the diagram, plus its anchor point/label at T = 0 K. These are
    fixed thermodynamic reference lines (not tied to any particular marked
    element) used to calibrate the H2/H2O and CO/CO2 ratio nomographs.
    """
    line = REFERENCE_GAS_LINES[point_label]
    color = line['color']
    G0 = line['g0'] + (line['g1'] - line['g0']) / (line['T1'] - line['T0']) * (T_left - line['T0'])
    G1 = line['g0'] + (line['g1'] - line['g0']) / (line['T1'] - line['T0']) * (T_right - line['T0'])

    ax.plot([T_left, T_right], [G0, G1], color=color, linestyle='-.',
            linewidth=1.4, alpha=0.85, zorder=6, clip_on=True)
    ax.plot(T_left, G0, 'o', color=color, markersize=7, zorder=10)
    ax.annotate(point_label, (T_left, G0), textcoords='offset points',
                xytext=(9, 0), fontsize=10, fontweight='bold', color=color,
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.85))
    ax.text(T_right - 40, G1 + 20, line['reaction'],
            ha='right', va='bottom', fontsize=8, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=color, alpha=0.85))


def add_ratio_nomograph(ax, point_label, ratio_label, T_right=2000, T_left=0, outward_offset=55):
    """
    Draws a gas-ratio nomograph (H2/H2O or CO/CO2) on an extra right-hand
    axis, offset outward from the main P(O2) scale. Calibrated using the
    matching REFERENCE_GAS_LINES entry evaluated at T_right, per:

        ln(P_X/P_XO) = (ΔG_line(T_right) - G) / (2·R·T_right)

    """
    line = REFERENCE_GAS_LINES[point_label]
    color = line['color']
    R_kJ = 0.008314
    G_line_right = line['g0'] + (line['g1'] - line['g0']) / (line['T1'] - line['T0']) * (T_right - line['T0'])

    major_exps = [-60,-55,-50,-45,-40, -35,-30,-25,-20,-15,-10,-5, 0, 5, 10, 15, 20,25,30,35,40,45,50,55,60]
    ratios = [10.0**e for e in major_exps]
    G_major = [G_line_right - R_kJ * T_right * np.log(r) for r in ratios]

    ymin, ymax = ax.get_ylim()
    ax2 = ax.twinx()
    ax2.set_ylim(ymin, ymax)
    ax2.spines['right'].set_position(('outward', outward_offset))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)

    valid = [(r, g) for r, g in zip(ratios, G_major) if ymin <= g <= ymax]
    if valid:
        r_valid, g_valid = zip(*valid)
        ax2.set_yticks(list(g_valid))
        ax2.set_yticklabels(
            [f'{r:.0e}' if (r < 0.01 or r >= 100) else f'{r:g}' for r in r_valid],
            fontsize=7)

    ax2.set_ylabel(ratio_label, fontsize=9, color=color)
    ax2.tick_params(axis='y', colors=color, labelsize=7)
    ax2.spines['right'].set_color(color)
    return ax2


def add_pressure_nomograph(ax, T_right=2000, T_left=0, minor_step=1, eq_kind='pressure', gas_plain='O2', gas_label=None):
    """
    Draws the pressure/activity nomograph on the right axis with minor ticks
    spaced evenly in the exponent (log10 scale).

    minor_step: interval between minor ticks in log10 units (e.g., 2 means
                ticks at 10^-38, 10^-36, 10^-34, 10^-32 between 10^-40 and 10^-30)
    eq_kind:    'pressure' for gas-forming families (reference state =
                101325 Pa) or 'activity' for carbides, where the reaction
                quotient is a dimensionless carbon activity a(C) referenced
                to pure graphite (reference state = 1, not 101325 Pa).
    gas_label:  the family's own reacting gas/solid species (e.g. 'O$_2$',
                'N$_2$', 'C'), drawn at the pivot point (T=0K, ΔG=0) so the
                marker always names the species that scale is actually for,
                instead of a generic/misleading 'O'.
    """
    R_kJ = 0.008314
    reference = 1.0 if eq_kind == 'activity' else 101325.0

    # Major tick exponents (powers of 10)
    major_exps = [-60,-55,-50,-45,-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    P_major = [10**e for e in major_exps]
    
    # Minor ticks: evenly spaced in the exponent between major ticks
    P_minor = []
    for i in range(len(major_exps) - 1):
        e_start = major_exps[i]
        e_end = major_exps[i + 1]
        # Generate intermediate exponents
        for e in range(e_start + minor_step, e_end, minor_step):
            P_minor.append(10**e)
    
    # Calculate positions using inverse of exp calculation
    G_major = [R_kJ * T_right * np.log(P / reference) for P in P_major]
    G_minor = [R_kJ * T_right * np.log(P / reference) for P in P_minor]
    
    ymin, ymax = ax.get_ylim()
    
    # Create twin axis
    ax2 = ax.twinx()
    ax2.set_ylim(ymin, ymax)
    
    # Filter to visible range
    valid_major = [(P, G) for P, G in zip(P_major, G_major) if ymin <= G <= ymax]
    valid_minor = [G for G in G_minor if ymin <= G <= ymax]
    
    if valid_major:
        P_valid, G_valid = zip(*valid_major)
        ax2.set_yticks(list(G_valid))
        ax2.set_yticklabels([f'{P:.0e}' for P in P_valid], fontsize=8)
    if valid_minor:
        ax2.set_yticks(list(valid_minor), minor=True)



    pivot_T, pivot_G = 0.0, 0.0
    
    if eq_kind == 'activity':
        ax2.set_ylabel(r'Carbon activity $a$(C)', fontsize=10, color='#333333')
    else:
        ax2.set_ylabel(rf'Equilibrium $P$({gas_plain}) (Pa)', fontsize=10, color='#333333')
    ax2.tick_params(axis='y', colors='#333333')
    
    if gas_label:
        ax.plot(pivot_T, pivot_G, 'ko', markersize=7, zorder=10)
        ax.annotate(gas_label, (pivot_T, pivot_G), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=0.8))
        
def add_markings_for_elements(ax, phases, elements_list, temp, family_color, eq_kind='pressure'):
    """
    Draws Richardson lines from O and marks the equilibrium point for each
    element. Returns a list of (element, dG_std, value_str) rows describing
    ΔG° and the corresponding equilibrium quantity — a gas partial pressure
    (Pa) for every family except carbides, which react with solid carbon
    and so get a carbon *activity* instead — for use in a results table
    """
    R_kJ = 0.008314
    T_O = 0.0
    G_O = 0.0
    T_right = 2000.0  # Matches the right edge of the plot

    table_rows = []
    for element in elements_list:
        target_row = None
        for phase, arr in phases.items():
            if arr.size == 0: continue
            for row in arr:
                if element_from_reaction(row[4]).lower() == element.lower():
                    T0, T1 = float(row[0]), float(row[1])
                    if T0 <= temp <= T1:
                        target_row = row
                        break
            if target_row is not None: break
        
        if target_row is None:
            print(f"Could not find valid segment for '{element}' at {temp}°C.")
            continue
        
        T0, T1, G0, G1 = float(target_row[0]), float(target_row[1]), float(target_row[2]), float(target_row[3])
        
        if T1 == T0: dG_std = G0
        else: dG_std = G0 + (G1 - G0) * (temp - T0) / (T1 - T0)

        # ΔG° = RT ln(Q) where Q is the reaction quotient at equilibrium:
        # Q = P_anion (Pa, relative to the 101325 Pa standard state) for
        # every gas-forming family, or Q = a_C (dimensionless carbon
        # activity) for carbides.
        if eq_kind == 'activity':
            value = np.exp(dG_std / (R_kJ * temp))
            value_str = f'{value:.2e}' if value < 1e-2 else f'{value:.3f}'
        else:
            value = 101325 * np.exp(dG_std / (R_kJ * temp))
            value_str = f'{value:.2e}' if value < 1e-3 else f'{value:.1f}'
        table_rows.append((element.upper(), dG_std, value_str))

        # Calculate Richardson line
        slope = (dG_std - G_O) / (temp - T_O)
        G_right = slope * (T_right - T_O)
        
        # Draw the line with clip_on=False so it touches the exact right spine
        ax.plot([T_O, T_right], [G_O, G_right], color=family_color, linestyle='--', 
                linewidth=1.5, alpha=0.7, zorder=5, clip_on=False)
        
        # Mark the exact point on the curve
        ax.plot(temp, dG_std, 'o', color=family_color, markersize=8, zorder=10, 
                markeredgecolor='white', markeredgewidth=1.5)
                
        ax.plot(T_right, G_right, 's', color=family_color, markersize=5, zorder=11, clip_on=False)

    return table_rows


def render_equilibrium_table(table_ax, table_rows, eq_kind, gas_label, temp, family_color):
    """Render the ΔG°/equilibrium-quantity results in a 3-column table
    (Element, ΔG°, equilibrium quantity) instead of cluttering the chart
    with callout boxes."""
    table_ax.axis('off')
    if not table_rows:
        return

    quantity_header = 'Carbon\nactivity a(C)' if eq_kind == 'activity' else f'P({gas_label})\n(Pa)'
    col_labels = ['Element', 'ΔG°\n(kJ/mol)', quantity_header]
    cell_text = [[element, f'{dG_std:.1f}', value_str] for element, dG_std, value_str in table_rows]

    table_ax.set_title(f'Equilibrium values\nat {temp:g} °C', fontsize=10, fontweight='bold', pad=14)
    table = table_ax.table(cellText=cell_text, colLabels=col_labels,
                            colWidths=[0.28, 0.32, 0.40],
                            loc='upper center', cellLoc='center', bbox=[0, 0.05, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_height(cell.get_height() * (1.9 if row == 0 else 1.6))
        if row == 0:
            cell.set_facecolor(family_color)
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)

# ======================================================================
# MAIN EXECUTION
# ======================================================================
def resolve_family_names(family_arg):
    if family_arg.lower() == 'all':
        return list(FAMILIES.keys())

    families = [family.strip().lower() for family in family_arg.split(',') if family.strip()]
    invalid_families = [family for family in families if family not in FAMILIES]
    if invalid_families:
        print(f"Error: Unknown families: {', '.join(invalid_families)}")
        print(f"Available families: {', '.join(FAMILIES.keys())}")
        raise SystemExit(1)
    return list(dict.fromkeys(families))


def main():
    print('=' * 60)
    print('Starting Ellingham Diagram Generator')
    print('=' * 60)

    parser = argparse.ArgumentParser(
        description='Plot Ellingham diagrams with Richardson construction lines.'
    )
    parser.add_argument(
        '--elements', type=str, default=DEFAULT_ELEMENTS,
        help=f"Comma-separated element symbols, e.g. Al,Fe,Ti, or 'all'. Default: {DEFAULT_ELEMENTS}"
    )
    parser.add_argument(
        '--families', type=str, default=DEFAULT_FAMILIES,
        help=f'Comma-separated family names, e.g. oxides,sulfides. Default: {DEFAULT_FAMILIES}'
    )
    parser.add_argument(
        '--phases', type=str, default=None,
        help='Comma-separated phase codes to plot, e.g. ss,ll (all phases if omitted)'
    )
    parser.add_argument(
        '--pressure', type=float, default=DEFAULT_PRESSURE_PA,
        help=f'Global partial pressure shift in Pa. Default: {DEFAULT_PRESSURE_PA:.0f} Pa'
    )
    parser.add_argument(
        '--temp', type=float, default=DEFAULT_TEMP_C,
        help=f'Temperature in °C to draw Richardson lines and mark P_eq (Pa) for all filtered elements. '
             f'Default: {DEFAULT_TEMP_C:.0f} °C. Pass an empty override is not supported; use --no-temp-mark to disable.'
    )
    parser.add_argument(
        '--no-temp-mark', action='store_true',
        help='Disable Richardson-line/P_eq marking even though a default --temp is set.'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Display the plots on screen instead of just saving them.'
    )
    args = parser.parse_args()

    if args.elements.lower() == 'all':
        allowed_elements = None
        mark_elements = []
        print('-> Filtering: All elements')
    else:
        allowed_elements = {element.strip().capitalize() for element in args.elements.split(',')}
        mark_elements = [element.strip() for element in args.elements.split(',')]
        print(f'-> Filtering & Marking elements: {allowed_elements}')

    allowed_families = resolve_family_names(args.families)
    if args.families.lower() == 'all':
        print(f"-> Filtering families: All ({len(allowed_families)} families): {', '.join(allowed_families)}")
    else:
        print(f'-> Filtering families: {allowed_families}')

    ensure_family_phase_data(allowed_families)

    if args.phases is not None:
        allowed_phases = {phase.strip().lower() for phase in args.phases.split(',')}
        print(f'-> Filtering phases: {allowed_phases}')
    else:
        allowed_phases = None
        print('-> Filtering phases: All phases')

    effective_temp = None if args.no_temp_mark else args.temp

    print(f'-> Global pressure shift: {args.pressure} Pa')
    if effective_temp is not None:
        print(f'-> MARKING: Drawing Richardson lines at {effective_temp} °C for specified elements.')
    print('-' * 60)

    generated_count = 0
    for name in allowed_families:
        fam = FAMILIES[name]
        filtered_phases = fam['phases'].copy()

        if allowed_phases is not None:
            filtered_phases = {phase: arr for phase, arr in filtered_phases.items() if phase in allowed_phases}

        if allowed_elements is not None:
            filtered_phases = {
                phase: (arr if phase == 'aux' else filter_by_elements(arr, allowed_elements))
                for phase, arr in filtered_phases.items()
            }

        if all(arr.size == 0 for arr in filtered_phases.values()):
            print(f"Skipping '{name}' – no matching reactions.")
            continue

        final_phases = apply_pressure_shift(filtered_phases, args.pressure)
        print(f"Generating '{name}' diagram...")

        p_str = f'{args.pressure:.0e}' if args.pressure != 101325 else '101325'
        title = rf"{name.capitalize()} Formation ($P_0$ = {p_str} Pa)"
        ylabel = r'Standard free energy of formation ($\Delta G_f^\circ$) kJ/mol ' + fam['gas']

        has_marking = effective_temp is not None and mark_elements
        if has_marking:
            fig = plt.figure(figsize=(13, 9))
            gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.12)
            ax = fig.add_subplot(gs[0])
            table_ax = fig.add_subplot(gs[1])
            fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.08)
            if name == 'oxides':
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0, pos.width * 0.80, pos.height])
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            table_ax = None

        plot_family(ax, final_phases, fam['color'], title, ylabel, fam['compound'])
        add_pressure_nomograph(ax, T_right=2000, eq_kind=fam['eq_kind'], gas_plain=fam['gas_plain'], gas_label=fam['gas'])
        add_family_gas_references(ax, name)

        if has_marking:
            table_rows = add_markings_for_elements(
                ax, final_phases, mark_elements, effective_temp, fam['color'], fam['eq_kind'])
            render_equilibrium_table(
                table_ax, table_rows, fam['eq_kind'], fam['gas_plain'], effective_temp, fam['color'])

        if not has_marking:
            plt.tight_layout()

        out = f'ellingham_{name}.pdf'
        abs_path = os.path.abspath(out)
        fig.savefig(out, dpi=400, bbox_inches='tight')

        if args.show:
            plt.show()
        else:
            plt.close(fig)

        generated_count += 1
        print(f'Saved: {abs_path}')

    print('-' * 60)
    print(f'Generated {generated_count} diagram(s).')


if __name__ == '__main__':
    main()
