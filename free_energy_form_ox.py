import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# ============================================================
# 1. Choose elements – edit this list, or set to None for all
# ============================================================
elements_to_plot = ['Al','Fe','Mg','Ca','Ti','Zr','Si','Mn','Ni','C',
                    'Co','Cr','V','W','Mo','Sn','Cu']
# elements_to_plot = None   # uncomment to plot all

# ============================================================
# 2. Load and filter data
# ============================================================
df = pd.read_csv("oxide_data.csv")
if elements_to_plot is not None:
    df = df[df['Element'].isin(elements_to_plot)]
print(f"Plotting {df['Element'].nunique()} elements.")

# ============================================================
# 3. Fit segments (split at phase markers)
# ============================================================
segments = []      # (T0, T1, a, b, elem, react, phase)
phase_events = []  # (T, DG, elem, phase, react)

for (elem, react), grp in df.groupby(['Element', 'Reaction']):
    grp = grp.sort_values('T (K)').reset_index(drop=True)
    seg = []
    for _, row in grp.iterrows():
        phase = row['Phase'] if pd.notna(row['Phase']) else ''
        if phase != '' and len(seg) > 0:
            seg.append(row)
            if len(seg) >= 2:
                sdf = pd.DataFrame(seg)
                x = sdf['T (K)'].values
                y = sdf['DeltaG (kJ/mol)'].values
                coeffs = np.polyfit(x, y, 1)
                a, b = coeffs[1], coeffs[0]
                segments.append((x.min(), x.max(), a, b, elem, react, phase))
                phase_events.append((row['T (K)'], row['DeltaG (kJ/mol)'], elem, phase, react))
            seg = [row]
        else:
            seg.append(row)
    if len(seg) >= 2:
        sdf = pd.DataFrame(seg)
        x = sdf['T (K)'].values
        y = sdf['DeltaG (kJ/mol)'].values
        coeffs = np.polyfit(x, y, 1)
        a, b = coeffs[1], coeffs[0]
        segments.append((x.min(), x.max(), a, b, elem, react, ''))

# ============================================================
# 4. Save parameters
# ============================================================
params = []
for T0, T1, a, b, elem, react, phase in segments:
    params.append({
        'Element': elem,
        'Reaction': react,
        'T_start': T0,
        'T_end': T1,
        'DeltaH (kJ/mol)': a,
        'DeltaS (kJ/mol·K)': -b,
        'Phase_marker': phase if phase else ''
    })
pd.DataFrame(params).to_csv("params.csv", index=False)

# ============================================================
# 5. Plot
# ============================================================
plt.figure(figsize=(22, 16))

line_colour = 'black'

for T0, T1, a, b, elem, react, phase in segments:
    T = np.linspace(T0, T1, 100)
    G = a + b * T
    plt.plot(T, G, color=line_colour, linewidth=0.8, alpha=0.6)

    # Reaction label above midpoint
    T_mid = (T0 + T1) / 2
    G_mid = a + b * T_mid
    norm = np.array([-b, 1])
    norm = norm / np.linalg.norm(norm)
    offset_dist = 12.0
    T_label = T_mid + offset_dist * norm[0]
    G_label = G_mid + offset_dist * norm[1]
    label = react if len(react) <= 30 else react[:27] + '…'
    plt.text(T_label, G_label, label, fontsize=5.5, color='black',
             ha='center', va='center', rotation=0,
             bbox=dict(boxstyle="round,pad=0.15", facecolor='white', alpha=0.7, edgecolor='none'))

# Phase transitions: symbol placed exactly at the point (no offset)
for T, DG, elem, phase, react in phase_events:
    plt.text(T, DG, phase, fontsize=8, color='red',
             ha='center', va='center', weight='bold',
             bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor='none'))

# ---- Legend: no markers, just text ----
phase_meanings = {
    'm': 'melting of pure element',
    'M': 'melting of oxide',
    'b': 'boiling of pure element',
    'B': 'boiling of oxide'
}
unique_phases = sorted({phase for _, _, _, phase, _ in phase_events})
if unique_phases:
    handles = [plt.Line2D([0], [0], color='none', marker='none') for _ in unique_phases]
    labels = [f"{p} = {phase_meanings.get(p, p)}" for p in unique_phases]
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1),
               fontsize=9, title="Phase transition symbols", framealpha=0.8)

# Axis limits
allT = df['T (K)']; allG = df['DeltaG (kJ/mol)']
plt.xlim(allT.min() - 50, allT.max() + 50)
plt.ylim(allG.min() - 30, allG.max() + 30)

plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel(r'$\Delta G^\circ$ (kJ/mol O$_2$)', fontsize=12)
plt.title('Ellingham Diagram – Reaction Labels Above Segments', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.savefig("ellingham.png", format='png', dpi=300)
print("✅ Plot saved as ellingham.png – open in browser.")