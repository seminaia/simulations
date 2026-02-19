import os
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp.outputs import Vasprun, Dos, Eigenval
from pymatgen.electronic_structure.core import OrbitalType, Spin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=EncodingWarning)
warnings.filterwarnings("ignore", message="We strongly encourage explicit")

def read_doscar(doscar_path, csv_file='DOSCAR.csv', atol=1e-6):
    """
    Reads a VASP DOSCAR file and returns:
        - df: DataFrame with columns like:
              ['Energy', 'DOS(up)', 'DOS(down)', 'Int_DOS(up)', 'Int_DOS(down)']
              or ['Energy', 'DOS', 'Int_DOS'] for ISPIN=1
        - header: dict with keys Emax, Emin, NEDOS, Efermi, weight

    Modifications:
      - Ensures a row at Energy == Efermi is present.
      - Sets DOS at Efermi to 0 (DOS(up)=0 and DOS(down)=0 for ISPIN=2, DOS=0 for ISPIN=1).
      - Leaves integrated DOS unchanged for existing rows; for the inserted Efermi
        row, integrated DOS is obtained by linear interpolation between neighbors.
      - Energies are NOT shifted; values are written as read from DOSCAR.
    """
    with open(doscar_path, 'rt', encoding='utf-8',errors='replace') as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError("DOSCAR file too short, expected at least 6 lines.")

    # --- Parse 6th line header ---
    header_tokens = lines[5].split()
    if len(header_tokens) < 4:
        raise ValueError("6th line of DOSCAR is malformed.")

    Emax   = float(header_tokens[0])
    Emin   = float(header_tokens[1])
    NEDOS  = int(float(header_tokens[2]))  # sometimes written as 1000.000
    Efermi = float(header_tokens[3])
    weight = float(header_tokens[4]) if len(header_tokens) > 4 else 1.0

    header = {'Emax': Emax, 'Emin': Emin, 'NEDOS': NEDOS, 'Efermi': Efermi, 'weight': weight}

    # --- Read DOS data section ---
    dos_lines = lines[6:6 + NEDOS]  # DOS data follows immediately
    data = [list(map(float, l.split())) for l in dos_lines]

    # Determine number of columns
    num_cols = len(data[0])
    if num_cols == 5:
        # Spin-polarized DOS: energy, DOS(up), DOS(down), int DOS(up), int DOS(down)
        columns = ['Energy', 'DOS(up)', 'DOS(down)', 'Int_DOS(up)', 'Int_DOS(down)']
    elif num_cols == 3:
        # Non-spin-polarized DOS: energy, DOS, int DOS
        columns = ['Energy', 'DOS', 'Int_DOS']
    else:
        raise ValueError(f"Unexpected number of columns in DOSCAR: {num_cols}")

    df = pd.DataFrame(data, columns=columns)

    # Ensure monotonic energies (VASP usually is). If not, sort to be safe.
    if not np.all(np.diff(df['Energy'].values) >= 0):
        df = df.sort_values('Energy', kind='mergesort').reset_index(drop=True)

    energies = df['Energy'].values

    # Helper: set DOS to zero at a given index without touching integrated DOS
    def _zero_dos_at(idx):
        if 'DOS(up)' in df.columns:
            df.at[idx, 'DOS(up)'] = 0.0
            df.at[idx, 'DOS(down)'] = 0.0
        else:
            df.at[idx, 'DOS'] = 0.0

    # Check if Efermi already exists within tolerance
    close_mask = np.isclose(energies, Efermi, atol=atol)
    if np.any(close_mask):
        # Zero DOS at the (first) matching row, integrated DOS unchanged
        idx = int(np.where(close_mask)[0][0])
        _zero_dos_at(idx)
    else:
        # Insert a new row at Efermi with DOS=0 and Int_DOS interpolated
        i = int(np.searchsorted(energies, Efermi, side='left'))

        # Pick neighbors for interpolation (clamp to edges if outside range)
        if i <= 0:
            i0, i1 = 0, min(1, len(df) - 1)
        elif i >= len(df):
            i0, i1 = max(len(df) - 2, 0), len(df) - 1
        else:
            i0, i1 = i - 1, i

        x0, x1 = df.at[i0, 'Energy'], df.at[i1, 'Energy']
        # Avoid division by zero if two energies are identical (shouldn't happen)
        if np.isclose(x0, x1, atol=1e-15):
            t = 0.0
        else:
            t = (Efermi - x0) / (x1 - x0)

        new_row = {'Energy': Efermi}
        if 'DOS(up)' in df.columns:
            new_row['DOS(up)'] = 0.0
            new_row['DOS(down)'] = 0.0
            # Linear interpolation for integrated DOS to maintain continuity
            for col in ['Int_DOS(up)', 'Int_DOS(down)']:
                y0, y1 = df.at[i0, col], df.at[i1, col]
                new_row[col] = (1 - t) * y0 + t * y1
        else:
            new_row['DOS'] = 0.0
            y0, y1 = df.at[i0, 'Int_DOS'], df.at[i1, 'Int_DOS']
            new_row['Int_DOS'] = (1 - t) * y0 + t * y1

        # Insert while preserving order
        df = pd.concat([df.iloc[:i], pd.DataFrame([new_row]), df.iloc[i:]], ignore_index=True)

    # Write CSV (no index column)
    df.to_csv(csv_file, index=False)

    return df, header

CONFIG = {
    "material": "La2NiO4",
    "elements": ["La", "Ni", "O"],
    "e_above_hull": 0.08,
    "processes": 4,
    "plot_style": {
        "interstitials": {"color": "blue", "marker": "o"},
        "vacancies": {"color": "red", "marker": "s"}
    }
}

def load_scf_reference(scf_vasprun_path):
    """
    Load EF, VBM, CBM from a *uniform-mesh SCF* vasprun.
    Returns (ef_scf, vbm_scf, cbm_scf, gap_scf).
    """
    vs = Vasprun(scf_vasprun_path, parse_potcar_file=False,parse_projected_eigen=False, separate_spins=False)
    # uniform-mesh SCF
    complete_dos = vs.eigenvalue_band_properties

    ef_scf = vs.efermi
    vbm_scf = complete_dos[2]
    cbm_scf = complete_dos[1]
    gap_scf = complete_dos[0]
    return ef_scf, vbm_scf, cbm_scf, gap_scf

def plot_dos(dos_dirs, expt_gaps=None, expt_labels=None, labels=None, colors=None, output_file='TDOS.pdf',
             xlim=None, ylim=None, el_ref=None, metal_tol=1e-3):
    """
    Plot TDOS from many folders by reading vasprun.xml in each.
    If el_ref is provided (from SCF), align all curves to that EF.
    Legend labels include DOS-based band gap for each curve.
    Experimental band gaps are marked with vertical dashed lines.
    
    Args:
        expt_gaps: List of experimental band gaps (eV)
        expt_labels: List of labels for experimental gaps (e.g., ['exp1', 'exp2'])
    """
    #plotter = DosPlotter(zero_at_efermi=True)
    band_gaps = []
    
    # Handle experimental gaps defaults
    if expt_gaps is None:
        expt_gaps = []
    if expt_labels is None:
        expt_labels = [f'exp_gap_{i+1}' for i in range(len(expt_gaps))]
    
    # sane defaults for calculated DOS
    
    if labels is None or len(labels) < len(dos_dirs):
        labels = (labels or []) + [f"calc{i+1}" for i in range(len(dos_dirs)-len(labels or []))]
    if colors is None or len(colors) < len(dos_dirs):
        colors = (colors or []) + [None]*(len(dos_dirs)-len(colors or []))
    dos_up_list = []
    dos_down_list = []
    full_labels = []
    for i, d in enumerate(dos_dirs):
        try:
            eig = Eigenval(os.path.join(d, "EIGENVAL"), separate_spins=False)
            #vr = Vasprun(os.path.join(d, "vasprun.xml"), parse_projected_eigen=False, separate_spins=False)
        except Exception as e:
            print(f"Skipping {d}: {e}")
            continue
        df, header = read_doscar(os.path.join(d, "DOSCAR"), csv_file=os.path.join(d, "DOSCAR.csv"))
        
        efermi = header['Efermi']
        #efermi = vr.efermi
        energy = df['Energy'].values
        if 'DOS(up)' in df.columns and 'DOS(down)' in df.columns:
            dos_up = df['DOS(up)'].values
            dos_down = df['DOS(down)'].values
        elif 'DOS' in df.columns:
            dos_up = df['DOS'].values / 2
            dos_down = df['DOS'].values / 2
        else:
            print(f"Skipping {d}: No DOS columns found")
            continue

        tdos_up = Dos(efermi=efermi, energies=energy, densities={Spin.up: dos_up})
        tdos_down = Dos(efermi=efermi, energies=energy, densities={Spin.down: dos_down})
        
        cbm= eig.eigenvalue_band_properties[1]
        vbm = eig.eigenvalue_band_properties[2]
        eg = cbm - vbm
        band_gaps.append((labels[i], eg))
        #spd_dos = vr.complete_dos.get_spd_dos()
        #tdos = vr.tdos
        #tdos_up = vr.complete_dos.get_densities(spin=Spin.up)
        #tdos_down = vr.complete_dos.get_densities(spin=Spin.down)
        #dos_up = Dos(efermi=vr.efermi, energies=vr.complete_dos.energies, densities={Spin.up: tdos_up})
        #dos_down = Dos(efermi=vr.efermi, energies=vr.complete_dos.energies, densities={Spin.down: tdos_down})

        if eg < metal_tol:
            eg_str = f"$E_g≈0 (metal)$"
        else:
            eg_str = f"$E_g={eg:.2f} eV$"
        label_with_gap = f"{labels[i]} ({eg_str})"

        #plotter.add_dos(label_with_gap, tdos_up)
        #plotter.add_dos(f'{labels[i]} (down)', dos=tdos_down)
        dos_up_list.append(tdos_up)
        dos_down_list.append(tdos_down)
        full_labels.append(label_with_gap)
        full_labels.append(f'{labels[i]} (down)')

        print(f"Loaded DOS from {d}: EF={efermi:.3f} eV, CBM={cbm:.3f} eV, VBM={vbm:.3f} eV, Eg={eg:.3f} eV")
    #print(f"Initial Structure: {vr.initial_structure.get_space_group_info()[0]}, final Structure: {vr.final_structure.get_space_group_info()[0]}")
    # plot
    plt.figure(figsize=(15, 12))
    for i, (dos_up, dos_down) in enumerate(zip(dos_up_list, dos_down_list)):
    #plotter.get_plot(xlim=xlim, ylim=ylim, beta_dashed=True)
        energies = dos_up.energies - dos_up.efermi
        plt.plot(energies, dos_up.densities[Spin.up], label=full_labels[2*i], color=colors[i] if colors[i] else None, lw= 2)
        plt.plot(energies, -dos_down.densities[Spin.down], label=full_labels[2*i+1], color=colors[i] if colors[i] else None, linestyle='--',lw=2)
    plt.ylim(ylim if ylim else None)
    plt.xlim(xlim if xlim else None)
    ax = plt.gca()
    handles, labels_legend = ax.get_legend_handles_labels()
    
    ax.set_xlabel('Energy - E$_F$ (eV)', fontsize=15)
    ax.set_ylabel('DOS (states/eV)', fontsize=15)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    # Add experimental gap lines
    for gap, lbl, col in zip(expt_gaps, expt_labels, color):
        ax.axvline(gap, color=col, linestyle='--', alpha=0.7, label=f'{lbl} $(E_g = {gap:.2f} eV)$')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    ax.grid(False)
    plt.title('Total DOS', fontsize=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # save a quick summary
    with open(os.path.splitext(output_file)[0] + '_bandgaps.txt', 'w') as f:
        f.write("Band Gap Summary (DOS-based)\n")
        for lbl, eg in band_gaps:
            f.write(f"{lbl}: {eg:.3f} eV\n")

    print(f"Plotted {len(band_gaps)} DOS curves → {output_file}")

if __name__ == '__main__':
    # Path to your *uniform-mesh SCF* vasprun (not the line-mode band run)
    #scf_path = "PBE_DOS/DOS_6.2/vasprun.xml"    # E<-- update to your actual SCF file
    #ef_scf, vbm_scf, cbm_scf, gap_scf = load_scf_reference(scf_path)
    #print(f"[SCF reference] EF = {ef_scf:.3f} eV, VBM = {vbm_scf:.3f} eV, CBM = {cbm_scf:.3f} eV, Eg = {gap_scf:.3f} eV")
    # Example for a current-dir DOS plot (NSCF DOS run)

    bulk_dirs = [
        "PBE_DOS/DOS_6",
        "PBE_DOS/DOS_6.2",
        "PBE_DOS/DOS_7",
        "PBE_DOS/DOS_8",
        "PBE_DOS/DOS_9",
        "PBE_DOS/DOS_10",
    ]
    
    conv_dirs = [
        "conv_dos/U_4",
        "conv_dos/U_5",
        "conv_dos/U_6",
        "conv_dos/U_7",
        "conv_dos/U_8",
    ]
    prim_dirs = [
        "prim_dos/U_4",
        "prim_dos/U_5",
        "prim_dos/U_6",
        "prim_dos/U_7",
        "prim_dos/U_8",
    ]
    
    
    O_i_C2v_0_dirs = [
        "O_i_C2v_0/U_4",
        "O_i_C2v_0/U_5",
        "O_i_C2v_0/U_6",
        "O_i_C2v_0/U_7",
        "O_i_C2v_0/U_8",
    ]
    O_i_C2v_1_dirs = [
        "O_i_C2v_-1/U_4",
        "O_i_C2v_-1/U_5",
        "O_i_C2v_-1/U_6",
        "O_i_C2v_-1/U_7",
        "O_i_C2v_-1/U_8",
    ]
    O_i_C2v_2_dirs = [
        "O_i_C2v_-2/U_4",
        "O_i_C2v_-2/U_5",
        "O_i_C2v_-2/U_6",
        "O_i_C2v_-2/U_7",
        "O_i_C2v_-2/U_8",
    ]
    
    all_dirs = bulk_dirs + O_i_C2v_0_dirs + O_i_C2v_1_dirs + O_i_C2v_2_dirs

    labels = (
        [f"Bulk U={x}" for x in ["4","5","6","7","8"]] +
        [f"O$_i$ C2v 0 U={x}" for x in ["4","5","6","7","8"]] +
        [f"O$_i$ C2v -1 U={x}" for x in ["4","5","6","7","8"]] +
        [f"O$_i$ C2v -2 U={x}" for x in ["4","5","6","7","8"]]
    )
    dos_labels = labels[:5]
    pbe_colors = ['blue', 'cyan', 'green', 'orange', 'red', 'violet']
    O_i_C2v_0_colors = ['navy', 'teal', 'darkgreen', 'darkorange', 'darkred', 'pink']
    O_i_C2v_1_colors = ['navy', 'teal', 'darkgreen', 'darkorange', 'darkred', 'pink']
    O_i_C2v_2_colors = ['navy', 'teal', 'darkgreen', 'darkorange', 'darkred', 'pink']
    all_colors = pbe_colors + O_i_C2v_0_colors + O_i_C2v_1_colors + O_i_C2v_2_colors
    expt_gaps = [0.8, 1.87, 1.51]
    expt_labels = ['Su, et al. 2023', 'Lahmar, et al. 2020', 'Laouici, et al. 2021']

    plot_dos(
        dos_dirs=prim_dirs,
        expt_gaps=expt_gaps,
        expt_labels=expt_labels,
        labels=labels[:5],
        colors=pbe_colors,
        output_file='prim-TDOS.jpg',
        xlim=(-5, 5),
        ylim=(-5, 5),
    )
    
    plot_dos(
        dos_dirs=conv_dirs,
        expt_gaps=expt_gaps,
        expt_labels=expt_labels,
        labels=labels[:5],
        colors=pbe_colors,
        output_file='conv-TDOS.jpg',
        xlim=(-5, 5),
        ylim=(-5, 5),
    )
    #plot_dos(
    #    dos_dirs=O_i_C2v_0_dirs,
    #    expt_gaps=expt_gaps,
    #    expt_labels=expt_labels,
    #    labels=labels[5:10],
    #    colors=O_i_C2v_0_colors,
    #    output_file='O_i_C2v_0-TDOS.jpg',
    #    xlim=(-8, 5),
    #    ylim=(-15, 15),
    #)
    #
    #plot_dos(
    #    dos_dirs=O_i_C2v_1_dirs,
    #    expt_gaps=expt_gaps,
    #    expt_labels=expt_labels,
    #    labels=labels[10:15],
    #    colors=O_i_C2v_1_colors,
    #    output_file='O_i_C2v_-1-TDOS.jpg',
    #    xlim=(-8, 5),
    #    ylim=(-15, 15),
    #)
    #plot_dos(
    #    dos_dirs=O_i_C2v_2_dirs,
    #    expt_gaps=expt_gaps,
    #    expt_labels=expt_labels,
    #    labels=labels[15:20],
    #    colors=O_i_C2v_2_colors,
    #    output_file='O_i_C2v_-2-TDOS.jpg',
    #    xlim=(-8, 5),
    #    ylim=(-15, 15),
    #)
