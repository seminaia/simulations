#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ellingham_separate_final.py

Generates separate Ellingham diagrams for Oxides, Carbides, Nitrides,
Fluorides, and Chlorides. Data from Reed (1971) and Coltters (1985).
"""

import numpy as np
import matplotlib
matplotlib.use('PDF')          # avoids FT_Render_Glyph crash
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
# (Paste all your arrays here – oxss, oxls, ..., clgg)
# For brevity, I’m showing only the first few lines – you must copy your full data.
# ======================================================================

# ---------- Oxides ----------
oxss = np.array([
    [   0, 932, -266.6, -220.0, r'$\frac{4}{3} Al + O_2 = \frac{2}{3} Al_2O_3$', -13],
    # ... (paste all your oxide rows)
])
oxls = np.array([ ... ])
oxgs = np.array([ ... ])
oxsl = np.array([ ... ])
oxll = np.array([ ... ])
oxgl = np.array([ ... ])
oxsg = np.array([ ... ])
oxlg = np.array([ ... ])
oxgg = np.array([ ... ])

# ---------- Carbides ----------
cass = np.array([
    [   0, 1414,   -57,    -49, '$ Si + C = SiC $', -9],
    # ... etc.
])
cals = np.array([ ... ])
cags = np.array([[0, 0, 0, 0, ' ', 0]])
casl = np.array([[0, 0, 0, 0, ' ', 0]])
call = np.array([[0, 0, 0, 0, ' ', 0]])
cagl = np.array([[0, 0, 0, 0, ' ', 0]])
casg = np.array([[0, 0, 0, 0, ' ', 0]])
calg = np.array([[0, 0, 0, 0, ' ', 0]])
cagg = np.array([[0, 0, 0, 0, ' ', 0]])

# ---------- Nitrides ----------
niss = np.array([ ... ])
nils = np.array([ ... ])
nigs = np.array([[0, 0, 0, 0, ' ', 0]])
nisl = np.array([[0, 0, 0, 0, ' ', 0]])
nill = np.array([[0, 0, 0, 0, ' ', 0]])
nigl = np.array([[0, 0, 0, 0, ' ', 0]])
nisg = np.array([[0, 0, 0, 0, ' ', 0]])
nilg = np.array([[0, 0, 0, 0, ' ', 0]])
nigg = np.array([ ... ])

# ---------- Fluorides ----------
flss = np.array([ ... ])
flls = np.array([ ... ])
flgs = np.array([[0, 0, 0, 0, ' ', 0]])
flsl = np.array([[0, 0, 0, 0, ' ', 0]])
flll = np.array([ ... ])
flgl = np.array([ ... ])
flsg = np.array([[0, 0, 0, 0, ' ', 0]])
fllg = np.array([[0, 0, 0, 0, ' ', 0]])
flgg = np.array([ ... ])

# ---------- Chlorides ----------
clss = np.array([ ... ])
clls = np.array([ ... ])
clgs = np.array([[0, 0, 0, 0, ' ', 0]])
clsl = np.array([ ... ])
clll = np.array([ ... ])
clgl = np.array([ ... ])
clsg = np.array([ ... ])
cllg = np.array([ ... ])
clgg = np.array([ ... ])

# ----------------------------------------------------------------------
# Conversion: K → °C, kcal → kJ, with OPTIONAL scaling of offsets
# ----------------------------------------------------------------------
def convert_units(*arrays, scale_offset=False):
    for arr in arrays:
        if arr.size == 0:
            continue
        numeric = arr[:, :4].astype(float)
        numeric[:, 0:2] -= 273.15      # K → °C
        numeric[:, 2:4] *= 4.184       # kcal → kJ
        arr[:, :4] = numeric
        if arr.shape[1] > 5 and scale_offset:
            arr[:, 5] = arr[:, 5].astype(float) * 4.184   # scale offsets

# Apply: oxides, nitrides, fluorides, chlorides → scale offsets
convert_units(oxss, oxls, oxgs, oxsl, oxll, oxgl, oxsg, oxlg, oxgg, scale_offset=True)
convert_units(niss, nils, nigs, nisl, nill, nigl, nisg, nilg, nigg, scale_offset=True)
convert_units(flss, flls, flgs, flsl, flll, flgl, flsg, fllg, flgg, scale_offset=True)
convert_units(clss, clls, clgs, clsl, clll, clgl, clsg, cllg, clgg, scale_offset=True)

# Carbides: do NOT scale offsets (already in kJ)
convert_units(cass, cals, cags, casl, call, cagl, casg, calg, cagg, scale_offset=False)

# ----------------------------------------------------------------------
# Helper to build anion dictionaries
# ----------------------------------------------------------------------
def make_anion_dict(ss, ls, gs, sl, ll, gl, sg, lg, gg):
    return {'ss': ss, 'ls': ls, 'gs': gs, 'sl': sl, 'll': ll,
            'gl': gl, 'sg': sg, 'lg': lg, 'gg': gg}

oxides   = make_anion_dict(oxss, oxls, oxgs, oxsl, oxll, oxgl, oxsg, oxlg, oxgg)
carbides = make_anion_dict(cass, cals, cags, casl, call, cagl, casg, calg, cagg)
nitrides = make_anion_dict(niss, nils, nigs, nisl, nill, nigl, nisg, nilg, nigg)
fluorides= make_anion_dict(flss, flls, flgs, flsl, flll, flgl, flsg, fllg, flgg)
chlorides= make_anion_dict(clss, clls, clgs, clsl, clll, clgl, clsg, cllg, clgg)

# ----------------------------------------------------------------------
# Plotting function – labels only for ss phase, offset -25, offsets scaled
# Legend box moved to top-right to avoid overlap
# ----------------------------------------------------------------------
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

    # ----- LABELS ONLY FOR ss PHASE, offset -25, offsets already scaled -----
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

    # Axis settings
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

    # Legend box moved to top-right (y ≈ -280 to -60) to avoid data overlap
    rectpos = [1100, 1900, -280, -60]
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
    ax.text(rectpos[0]+ 290, rectpos[3]-110, 'Solid', ha='right', fontsize=9)
    ax.text(rectpos[0]+ 290, rectpos[3]-155, 'Liquid', ha='right', fontsize=9)
    ax.text(rectpos[0]+ 290, rectpos[3]-200, 'Gas', ha='right', fontsize=9)

    # Line style examples (adjusted y to match new rectangle)
    ax.plot([1260, 1400], [-200, -200], color='k', ls='-',  alpha=1.0)
    ax.plot([1520, 1660], [-200, -200], color='k', ls='--', alpha=1.0)
    ax.plot([1780, 1920], [-200, -200], color='k', ls=':',  alpha=1.0)
    ax.plot([1260, 1400], [-230, -230], color='k', ls='-',  alpha=0.6)
    ax.plot([1520, 1660], [-230, -230], color='k', ls='--', alpha=0.6)
    ax.plot([1780, 1920], [-230, -230], color='k', ls=':',  alpha=0.6)
    ax.plot([1260, 1400], [-260, -260], color='k', ls='-',  alpha=0.3)
    ax.plot([1520, 1660], [-260, -260], color='k', ls='--', alpha=0.3)
    ax.plot([1780, 1920], [-260, -260], color='k', ls=':',  alpha=0.3)

    # Sources (shortened, placed inside the legend box)
    ax.text(rectpos[0] + 30, rectpos[3]-25, 'Sources',
            fontsize=9, fontweight='bold')
    ax.text(rectpos[0] + 30, rectpos[3]-30,
            'Reed (1971) and Coltters (1985)',
            fontsize=8, va='top')

# ----------------------------------------------------------------------
# Generate and save each figure as PDF
# ----------------------------------------------------------------------
def save_anion_figure(anion_dict, color, name, ylabel):
    fig, ax = pl.subplots(figsize=(10, 8))
    plot_anion(ax, anion_dict, color,
               title=f'{name.capitalize()}',
               ylabel=ylabel)
    pl.tight_layout()
    pl.savefig(f'ellingham_{name}.pdf', bbox_inches='tight')
    pl.close(fig)
    print(f"Saved ellingham_{name}.pdf")

# Use plain Unicode to avoid font rendering issues
save_anion_figure(oxides, 'r', 'oxides',
                  'Standard free energy of formation (ΔG°) kJ/mol O₂')
save_anion_figure(carbides, '0.4', 'carbides',
                  'Standard free energy of formation (ΔG°) kJ/mol C')
save_anion_figure(nitrides, 'b', 'nitrides',
                  'Standard free energy of formation (ΔG°) kJ/mol N₂')
save_anion_figure(fluorides, [0, 1, 0], 'fluorides',
                  'Standard free energy of formation (ΔG°) kJ/mol F₂')
save_anion_figure(chlorides, 'g', 'chlorides',
                  'Standard free energy of formation (ΔG°) kJ/mol Cl₂')