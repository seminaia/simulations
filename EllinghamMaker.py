import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Use SVG backend to avoid any rasterisation issues
matplotlib.use('svg')

# ----------------------------------------------------------------------
# 1. Parse the Excel file (unchanged)
# ----------------------------------------------------------------------
file_path = "Oxide_formation.xls"
df = pd.read_excel(file_path, sheet_name=0, header=None, dtype=str)

def is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

reactions = []
current_element = None
current_reaction = None
current_data = []

for idx, row in df.iterrows():
    a = row[0] if pd.notna(row[0]) else ""
    b = row[1] if pd.notna(row[1]) else ""
    c = row[2] if pd.notna(row[2]) else ""
    d = row[3] if pd.notna(row[3]) else ""

    if not any([a, b, c, d]):
        if current_element and current_reaction and current_data:
            reactions.append({
                'element': current_element,
                'reaction': current_reaction,
                'data': current_data
            })
            current_element = None
            current_reaction = None
            current_data = []
        continue

    if (b and "=" not in b and 
        c.strip() == "T, K" and d.strip() == "DGo , Kcal/mol"):
        if current_element and current_reaction and current_data:
            reactions.append({
                'element': current_element,
                'reaction': current_reaction,
                'data': current_data
            })
        current_element = b.strip()
        current_reaction = None
        current_data = []
        continue

    if current_element is not None and current_reaction is None:
        if "=" in b and "O2" in b:
            current_reaction = b.strip()
        continue

    if current_element is not None and current_reaction is not None:
        if is_float(c) and is_float(d):
            T = float(c)
            DG = float(d)
            current_data.append((T, DG))

if current_element and current_reaction and current_data:
    reactions.append({
        'element': current_element,
        'reaction': current_reaction,
        'data': current_data
    })

reactions = [r for r in reactions if len(r['data']) >= 2]
print(f"Parsed reactions: {reactions}")
print(f"Plotting {len(reactions)} reactions")

# ----------------------------------------------------------------------
# 2. Plot everything (no legend, just inline labels)
# ----------------------------------------------------------------------
plt.figure(figsize=(16, 12))

num_lines = len(reactions)
colors = plt.cm.tab20(np.linspace(0, 1, num_lines))

for i, r in enumerate(reactions):
    data = r['data']
    sorted_data = sorted(data, key=lambda x: x[0])
    T_vals = [p[0] for p in sorted_data]
    DG_vals = [p[1] for p in sorted_data]
    plt.plot(T_vals, DG_vals, color=colors[i % len(colors)],
             linewidth=1.2, marker='o', markersize=2)

    # Add element symbol at the rightmost point
    last_T = T_vals[-1]
    last_DG = DG_vals[-1]
    plt.text(last_T + 15, last_DG, r['element'],
             fontsize=6, alpha=0.8, verticalalignment='center')

# Axis limits with margins
all_T = [p[0] for r in reactions for p in r['data']]
all_DG = [p[1] for r in reactions for p in r['data']]
if all_T and all_DG:
    x_min, x_max = min(all_T), max(all_T)
    y_min, y_max = min(all_DG), max(all_DG)
    x_margin = 0.05 * (x_max - x_min) if x_max != x_min else 50
    y_margin = 0.05 * (y_max - y_min) if y_max != y_min else 10
    plt.xlim(x_min - x_margin, x_max + 2*x_margin)  # extra room for labels
    plt.ylim(y_min - y_margin, y_max + y_margin)

# Reduce tick labels to a reasonable number
plt.xticks(np.arange(0, 4000, 500))
plt.yticks(np.arange(-350, 100, 20))
plt.tick_params(axis='both', labelsize=8)

plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel(r'$\Delta G^\circ$ (kcal/mol O$_2$)', fontsize=12)
plt.title('Ellingham Diagram – All Oxides', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# ----------------------------------------------------------------------
# 3. Save as SVG (no rasterisation, guaranteed no overflow)
# ----------------------------------------------------------------------
svg_filename = 'Ellingham_diagram.svg'
plt.savefig(svg_filename, bbox_inches='tight')
print(f"SVG saved as '{svg_filename}'")

# ----------------------------------------------------------------------
# 4. (Optional) Convert SVG to PNG using cairosvg (if installed)
# ----------------------------------------------------------------------
try:
    import cairosvg
    png_filename = 'Ellingham_diagram.png'
    cairosvg.svg2png(url=svg_filename, write_to=png_filename, dpi=150)
    print(f"PNG also saved as '{png_filename}' (via cairosvg)")
except ImportError:
    print("cairosvg not installed – PNG not created. You can open the SVG in a browser.")
    