import pandas as pd

# ----------------------------------------------------------------------
# 1. Read the original Excel file
# ----------------------------------------------------------------------
file_path = "EllinghamMaker_v12-5.xls"
df = pd.read_csv(file_path)

def is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

# ----------------------------------------------------------------------
# 2. Parse and build a clean table with ΔG in kJ and phase column
# ----------------------------------------------------------------------
clean_rows = []

current_element = None
current_reaction = None

for idx, row in df.iterrows():
    print(row)
    a = row[0] if pd.notna(row[0]) else ""
    b = row[1] if pd.notna(row[1]) else ""
    c = row[2] if pd.notna(row[2]) else ""
    d = row[3] if pd.notna(row[3]) else ""

    if not any([a, b, c, d]):
        continue

    # Detect a new header (element symbol, and columns C & D are labels)
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

    # Now we are inside a block – look for numeric data in columns C and D
    if current_element is not None and current_reaction is not None:
        if is_float(c) and is_float(d):
            T = float(c)
            DG_kcal = float(d)
            DG_kJ = DG_kcal * 4.184   # convert to kJ/mol

            # Capture phase label from column B if it's one of the known symbols
            phase = ''
            if b.strip() in ['m', 'M', 'b', 'B']:
                phase = b.strip()
            # Also, some rows have empty or other strings, but we only care about these.

            clean_rows.append({
                'Element': current_element,
                'Reaction': current_reaction,
                'T (K)': T,
                'DeltaG (kJ/mol)': DG_kJ,
                'Phase': phase
            })

# ----------------------------------------------------------------------
# 3. Write to a new Excel file
# ----------------------------------------------------------------------
clean_df = pd.DataFrame(clean_rows)
clean_df.sort_values(['Element', 'T (K)'], inplace=True)

output_file = "clean_oxide_data.csv"
clean_df.to_csv(output_file, index=False)
print(f"Cleaned data has been written to {output_file}")