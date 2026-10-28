"""Ellingham diagram (oxides) from Data_EllinghamDiagram_v6.xlsx.

Every line is labelled with its balanced reaction per mole of O2, e.g.
    2Fe + O2 = 2FeO          (formation from the metal)
    4Fe3O4 + O2 = 6Fe2O3     (oxidation of a lower oxide)

The diagram also carries the classic nomograph scales:
    P(O2)  - equilibrium oxygen pressure
    H2/H2O - hydrogen / steam ratio
    CO/CO2 - carbon monoxide / dioxide ratio
    a(C)   - carbon activity

The raw workbook is converted once into a readable one
(data/ellingham/ellingham_v6_clean.xlsx, see ellingham_v6_data.py) and the
plot is drawn from that.

Usage:
    python ellingham_v6.py                          # defaults set below
    python ellingham_v6.py --elements Fe,Al,Si
    python ellingham_v6.py --elements all --tmax 2500
    python ellingham_v6.py --elements Fe --no-gas-scales   # only the P(O2) scale
    python ellingham_v6.py --rebuild-data                  # regenerate the clean xlsx
    python ellingham_v6.py --elements Fe --show            # open a window too
"""


import argparse
import re
import sys
from pathlib import Path

import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')   # no GUI event loop: avoids editor/debugger Qt hangs
import matplotlib.pyplot as plt
import numpy as np

from ellingham import (add_pressure_nomograph, add_ratio_nomograph,
                       add_reference_gas_line, ensure_family_phase_data,
                       FAMILIES)
from ellingham_v6_data import load_clean

# ----------------------------------------------------------------------
# User-editable defaults (any can be overridden on the command line)
# ----------------------------------------------------------------------
DEFAULT_ELEMENTS = "Sn,Sb,Bi"   # comma-separated symbols, or "all"
DEFAULT_TMAX_K = 2000.0                  # right edge of the diagram (K)
DEFAULT_TEMP_C = 1000.0                  # temperature (°C) for the equilibrium lines/table
R_KJ = 8.314462618e-3                    # kJ/(mol K)
LINE_COLOUR = '#1f4e9c'                  # every reaction line and label
LEFT_MARGIN_K = -700.0                   # room on the left for the reaction labels
PHASE_NAMES = {'s': 'solid', 'l': 'liquid', 'g': 'gas'}
PHASE_STYLES = {
    'ss': ('-', 1.0), 'ls': ('--', 1.0), 'gs': (':', 1.0),
    'sl': ('-', 0.6), 'll': ('--', 0.6), 'gl': (':', 0.6),
    'sg': ('-', 0.3), 'lg': ('--', 0.3), 'gg': (':', 0.3),
}

HERE = Path(__file__).resolve().parent

plt.rcParams.update({
    'mathtext.default': 'regular',
    'mathtext.fontset': 'dejavusans',
    'font.family': 'DejaVu Sans',
})

# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------



def mathify(reaction):
    """'4Fe3O4 + O2 = 6Fe2O3' -> mathtext with subscripts (digits after a letter)."""
    return re.sub(r'(?<=[A-Za-z])(\d+(?:\.\d+)?)', r'$_{\1}$', reaction)


def reaction_key(reaction):
    """Normalize plain-text and mathtext reaction labels for matching."""
    text = str(reaction).replace('$', '').replace('\\', '')
    text = re.sub(r'frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', text)
    text = re.sub(r'_\{([^{}]+)\}', r'\1', text)
    text = re.sub(r'\((\d+/\d+)\)', r'\1', text)
    return re.sub(r'\s+', '', text).lower()


def phase_segments_by_reaction(phase_data):
    segments = {}
    for phase_code, rows in phase_data.items():
        if phase_code not in PHASE_STYLES:
            continue
        for row in rows:
            segments.setdefault(reaction_key(row[4]), []).append(
                (phase_code, float(row[0]) + 273.15, float(row[1]) + 273.15)
            )
    return segments


def phase_segments_by_metal(phase_data):
    segments = {}
    for phase_code, rows in phase_data.items():
        if phase_code not in PHASE_STYLES:
            continue
        for row in rows:
            match = re.search(r'[A-Z][a-z]?', str(row[4]))
            if match is None:
                continue
            metal = match.group()
            segments.setdefault(metal, []).append(
                (phase_code, float(row[0]) + 273.15, float(row[1]) + 273.15)
            )
    return segments


def spread(ys, min_gap):
    """Return y positions with at least min_gap between neighbours, closest to the originals."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    # relax the pile-up back towards the true positions (top-down pass)
    for a, b in zip(order[::-1][1:], order[::-1][:-1]):
        if out[b] - out[a] < min_gap:
            out[a] = out[b] - min_gap
    return out


def mark_equilibria(ax, curves_by_id, temp_k, tmax):
    """Richardson line from the (0 K, 0) pivot through each curve at temp_k, with a marker.
    Returns table rows (reaction, dG kJ, pO2 Pa, colour)."""
    rows = []
    for text, c, T, dG in curves_by_id:
        if not (T[0] <= temp_k <= T[-1]):
            continue
        g = float(np.interp(temp_k, T, dG))
        ax.plot([0, tmax], [0, g / temp_k * tmax], color=c, ls='--', lw=0.8,
                alpha=0.45, zorder=5, clip_on=True)
        ax.plot(temp_k, g, 'o', color=c, ms=7, mec='white', mew=1.2, zorder=10)
        p_pa = 101325.0 * np.exp(g / (R_KJ * temp_k))
        rows.append((text, g, p_pa, c))
    return rows


def render_table(table_ax, rows, temp_c):
    table_ax.axis('off')
    if not rows:
        table_ax.text(0, 0.5, f'No reaction is defined\nat {temp_c:g} °C', fontsize=9)
        return
    rows = sorted(rows, key=lambda r: r[1])
    cells = [[t, f'{g:.1f}', f'{p:.2e}'] for t, g, p, _ in rows]
    table_ax.set_title(f'Equilibrium values\nat {temp_c:g} °C', fontsize=10, fontweight='bold', pad=14)
    tbl = table_ax.table(cellText=cells, colLabels=['Reaction', 'ΔG°\n(kJ/mol O$_2$)', 'P(O$_2$)\n(Pa)'],
                         colWidths=[0.52, 0.22, 0.26], loc='upper right', cellLoc='center',
                         bbox=[0, 0.02, 1, 0.9])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#305496')
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        else:
            cell.set_text_props(color=rows[r - 1][3] if col == 0 else 'black')


def mark_phase_transitions(ax, phase_data, elements, tmax, curves_by_reaction):
    """Mark actual metal/oxide phase-code changes on the diagram."""
    if not phase_data:
        return

    segments_by_reaction = {}
    wanted = {element.lower() for element in elements}
    for phase_code, rows in phase_data.items():
        if phase_code == 'aux' or len(phase_code) != 2:
            continue
        for row in rows:
            reaction = str(row[4])
            metal = re.search(r'[A-Z][a-z]?', reaction)
            if metal is None or metal.group().lower() not in wanted:
                continue
            key = (metal.group(), reaction_key(reaction))
            segments_by_reaction.setdefault(key, []).append(
                (float(row[0]), float(row[1]), phase_code)
            )

    transitions = []
    for (metal, reaction), segments in segments_by_reaction.items():
        segments.sort()
        for previous, current in zip(segments, segments[1:]):
            temperature, previous_end, previous_code = previous
            current_start, _current_end, current_code = current
            if abs(previous_end - current_start) > 1e-3 or previous_code == current_code:
                continue
            temperature = current_start
            if 0 <= temperature + 273.15 <= tmax:
                transitions.append(
                    (temperature, metal, reaction, previous_code, current_code)
                )

    for temperature, metal, reaction, previous_code, current_code in transitions:
        curve = curves_by_reaction.get(reaction_key(reaction))
        if curve is None:
            continue
        curve_temperature, curve_dg, curve_color = curve
        temperature_k = temperature + 273.15
        if not (curve_temperature[0] <= temperature_k <= curve_temperature[-1]):
            continue
        dg = float(np.interp(temperature_k, curve_temperature, curve_dg))
        labels = []
        for name, before, after in (
            ('metal', previous_code[0], current_code[0]),
            ('oxide', previous_code[1], current_code[1]),
        ):
            if before != after:
                labels.append(
                    f'{name} {PHASE_NAMES[before]}→{PHASE_NAMES[after]}'
                )
        if not labels:
            continue
        ax.plot(temperature_k, dg, marker='o', markersize=7,
                markerfacecolor=curve_color, markeredgecolor='white',
                markeredgewidth=1.2, zorder=8)
        ax.annotate(
            f'{metal}: ' + ' / '.join(labels),
            (temperature_k, dg),
            xytext=(5, 7),
            textcoords='offset points',
            fontsize=7,
            color='#444444',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor=curve_color, alpha=0.8),
        )


def plot(reactions, curves, tmax, gas_scales=True, temp_c=None,
         title='Ellingham diagram — oxides', phase_data=None):
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.5], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    table_ax = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.08)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.72, pos.height])

    metals = sorted(reactions['Metal'].unique())

    entries = []   # [x, y, text] for the label declutterer
    entry_colour = []
    by_id = []
    curves_by_reaction = {}
    phase_segments = phase_segments_by_reaction(phase_data or {})
    phase_segments_for_metal = phase_segments_by_metal(phase_data or {})
    for grp in curves.groupby('Reaction_ID', sort=False):    
        grp = grp[1]
        grp = grp.sort_values('T_K')
        grp = grp[grp['T_K'] <= tmax]
        if len(grp) < 2:
            continue
        T = grp['T_K'].to_numpy()
        dG = grp['dG_kJ_per_molO2'].to_numpy()
        # DFT lines follow dG = dH - T*dS, so the 0 K value is the intercept of a fit over the whole line
        slope, intercept = np.polyfit(T, dG, 1)
        if T[0] > 0:
            T = np.concatenate([[0.0], T])
            dG = np.concatenate([[intercept], dG])
        c = LINE_COLOUR
        reaction = reaction_key(grp['Reaction'].iloc[0])
        metal = str(grp['Metal'].iloc[0])
        styled_segments = phase_segments.get(reaction)
        if not styled_segments:
            styled_segments = phase_segments_for_metal.get(metal, [])
        if styled_segments:
            for phase_code, start, end in styled_segments:
                start = max(start, T[0])
                end = min(end, T[-1], tmax)
                if start >= end:
                    continue
                y = np.interp([start, end], T, dG)
                linestyle, alpha = PHASE_STYLES[phase_code]
                ax.plot([start, end], y, color=c, ls=linestyle,
                        alpha=alpha, lw=1.5)
        else:
            ax.plot(T, dG, color=c, lw=1.5)
        curves_by_reaction[reaction] = (T, dG, c)
        by_id.append((mathify(grp['Reaction'].iloc[0]), c, T, dG))
        entries.append([0.0, float(dG[0]), mathify(grp['Reaction'].iloc[0])])
        entry_colour.append(c)

    if not entries:
        raise SystemExit('No curves in the requested temperature range.')

    all_dG = min(float(d.min()) for _, _, _, d in by_id)
    ymin = max(all_dG, -1500.0) - 60   # ignore extreme low-T outliers
    ymax = 120.0
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(LEFT_MARGIN_K, tmax)
    # reaction labels in the empty left-hand margin, joined to each line start
    label_y = spread([e[1] for e in entries], 0.016 * (ymax - ymin))
    for (x0, y0, text), y_l, c in zip(entries, label_y, entry_colour):
        ax.annotate(text, xy=(x0, y0), xytext=(LEFT_MARGIN_K + 10, y_l), fontsize=9,
                    color=c, ha='left', va='center', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=c, alpha=0.35, lw=0.5,
                                    shrinkA=1, shrinkB=0))

    if gas_scales:
        for key in ('H', 'C', 'CO', 'CCO2', 'CO2'):
            add_reference_gas_line(ax, key, T_right=tmax)
    add_pressure_nomograph(ax, T_right=tmax, gas_plain='O2', gas_label=r'O$_2$')
    if gas_scales:
        add_ratio_nomograph(ax, 'H', r'$P_{H_2}/P_{H_2O}$', T_right=tmax, outward_offset=55)
        add_ratio_nomograph(ax, 'CO', r'$P_{CO}/P_{CO_2}$', T_right=tmax, outward_offset=115)
        add_ratio_nomograph(ax, 'C', r'$a_C$', T_right=tmax, outward_offset=175)

    if temp_c is not None:
        rows = mark_equilibria(ax, by_id, temp_c + 273.15, tmax)
        render_table(table_ax, rows, temp_c)
    else:
        table_ax.axis('off')

    for t in range(0, int(tmax) + 1, 200):
        ax.axvline(t, color='0.5', alpha=0.3, zorder=-9)
    ax.axhline(0, color='k', lw=1)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(range(0, int(tmax) + 1, 200))

    secx = ax.secondary_xaxis('top', functions=(lambda k: k - 273.15, lambda c: c + 273.15))
    secx.set_xlabel('Temperature (°C)')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=36)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'$\Delta G^\circ$ (kJ per mol O$_2$)')
    return fig


def main():
    parser = argparse.ArgumentParser(description='Ellingham diagram from Data_EllinghamDiagram_v6.xlsx')
    parser.add_argument('--elements', default=DEFAULT_ELEMENTS,
                        help=f"Comma-separated symbols, e.g. Fe,Al,Si, or 'all'. Default: {DEFAULT_ELEMENTS}")
    parser.add_argument('--tmax', type=float, default=DEFAULT_TMAX_K, help='Maximum temperature (K)')
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMP_C,
                        help=f'Temperature (°C) for the equilibrium lines, markers and table. Default: {DEFAULT_TEMP_C:g}')
    parser.add_argument('--no-temp-mark', action='store_true', help='Disable the equilibrium lines and table')
    parser.add_argument('--no-gas-scales', action='store_true',
                        help='Only the P(O2) scale (omit H2/H2O, CO/CO2, a(C) and their lines)')
    parser.add_argument('--rebuild-data', action='store_true', help='Regenerate the clean workbook first')
    parser.add_argument('--out', default=None, help='Output file (default ellingham_v6_<elements>.pdf)')
    parser.add_argument('--show', action='store_true', help='Also open the plot in a window')
    args = parser.parse_args()

    if not args.show:
        plt.switch_backend('Agg')

    reactions, curves = load_clean(rebuild=args.rebuild_data)
    if args.elements.lower() != 'all':
        wanted = {e.strip().capitalize() for e in args.elements.split(',') if e.strip()}
        reactions = reactions[reactions['Metal'].isin(wanted)]
        tag = '_'.join(sorted(wanted))
        print(f'-> Elements: {sorted(wanted)}')
    else:
        tag = 'all'
        print('-> Elements: all')
    if reactions.empty:
        raise SystemExit('No reactions matched those elements.')
    curves = curves[curves['Reaction_ID'].isin(reactions['Reaction_ID'])]
    ensure_family_phase_data(['oxides'])
    phase_data = FAMILIES['oxides']['phases']

    fig = plot(reactions, curves, args.tmax, gas_scales=not args.no_gas_scales,
               temp_c=None if args.no_temp_mark else args.temp,
               phase_data=phase_data)
    print('-> Saving...', flush=True)
    out = Path(args.out) if args.out else HERE / f'ellingham_v6.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
