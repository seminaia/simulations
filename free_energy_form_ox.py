import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load the cleaned data
# ------------------------------------------------------------------
df = pd.read_excel("clean_oxide_data.xlsx")
print(f"Loaded {len(df)} data points.")

# ------------------------------------------------------------------
# 2. Choose elements (edit this list, or set to all)
# ------------------------------------------------------------------
elements_to_plot = ['Al', 'Fe', 'Mg', 'Ca', 'Ti', 'Zr', 'Si', 'Mn',
                    'Ni', 'Co', 'Cr', 'V', 'W', 'Mo', 'Nb', 'Ta',
                    'Hf', 'U', 'Th', 'Cu', 'Ag', 'Pb', 'Zn']

# If you want ALL elements, use this instead:
# elements_to_plot = df['Element'].unique().tolist()

df = df[df['Element'].isin(elements_to_plot)]
if df.empty:
    print("No data for selected elements. Check your element list.")
    exit()

# ------------------------------------------------------------------
# 3. Plot: one line per (Element, Reaction)
# ------------------------------------------------------------------
plt.figure(figsize=(14, 10))

# Assign a unique color per element (not per reaction, to keep legend clean)
elements = sorted(df['Element'].unique())
color_map = {e: plt.cm.tab20(i % 20) for i, e in enumerate(elements)}

# Group by Element and Reaction (in case an element has multiple reactions)
for (elem, react), group in df.groupby(['Element', 'Reaction']):
    group = group.sort_values('T (K)')
    plt.plot(group['T (K)'], group['DeltaG (kJ/mol)'],
             color=color_map[elem], marker='o', markersize=3,
             linewidth=1.5, label=elem)

# Legend – one entry per element (remove duplicates)
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), loc='best', fontsize=9, ncol=2)

plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel(r'$\Delta G^\circ$ (kJ/mol O$_2$)', fontsize=12)
plt.title('Ellingham Diagram (Raw Data)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# ------------------------------------------------------------------
# 4. Save
# ------------------------------------------------------------------
try:
    plt.savefig("ellingham_raw.png", dpi=200)
    print("✅ Saved as ellingham_raw.png")
except Exception as e:
    print(f"Error saving PNG: {e}")
    print("Saving as SVG instead...")
    import matplotlib
    matplotlib.use('svg')
    plt.savefig("ellingham_raw.svg")
    print("✅ Saved as ellingham_raw.svg")