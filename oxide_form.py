import pandas as pd

# ----------------------------------------------------------------------
# 1. Read the original Excel file
# ----------------------------------------------------------------------
file_path = "Oxide_formation.xls"
df = pd.read_excel(file_path, sheet_name=0, header=None, dtype=str)

# Helper to check if a string is a number
def is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

# ----------------------------------------------------------------------
# 2. Parse and build a clean table with ΔG in kJ/mol
# ----------------------------------------------------------------------
clean_rows = []

current_element = None
current_reaction = None

for idx, row in df.iterrows():
    a = row[0] if pd.notna(row[0]) else ""
    b = row[1] if pd.notna(row[1]) else ""
    c = row[2] if pd.notna(row[2]) else ""
    d = row[3] if pd.notna(row[3]) else ""

    if not any([a, b, c, d]):
        continue

    # Detect a new header
    if (b and "=" not in b and 
        c.strip() == "T, K" and d.strip() == "DGo , Kcal/mol"):
        current_element = b.strip()
        current_reaction = None
        continue

    # After header, the next row with '=' and 'O2' is the reaction
    if current_element is not None and current_reaction is None:
        if "=" in b and "O2" in b:
            current_reaction = b.strip()
        continue

    # Now we are inside a block and have a reaction – look for numeric data
    if current_element is not None and current_reaction is not None:
        if is_float(c) and is_float(d):
            T = float(c)
            DG_kcal = float(d)
            DG_kJ = DG_kcal * 4.184   # convert to kJ/mol
            clean_rows.append({
                'Element': current_element,
                'Reaction': current_reaction,
                'T (K)': T,
                'DeltaG (kJ/mol)': DG_kJ
            })

# ----------------------------------------------------------------------
# 3. Write to a new Excel file
# ----------------------------------------------------------------------
clean_df = pd.DataFrame(clean_rows)
clean_df.sort_values(['Element', 'T (K)'], inplace=True)

output_file = "clean_oxide_data.xlsx"
clean_df.to_excel(output_file, index=False)
print(f"Converted {len(clean_df)} data points to '{output_file}' (ΔG in kJ/mol)")