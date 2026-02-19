import pandas as pd

# === USER SETTINGS ===
acf_file = "ACF.dat"    # Path to your Bader output file
output_csv = "bader_net_charges.csv"

# POSCAR composition order
elements = ["La", "Ni", "O"]  
counts = [4, 2, 8]  # From POSCAR: La 4, Ni 2, O 8

# Valence electrons from POTCAR (Ni_pv assumed)
valence_dict = {
    "La": 9,
    "Ni": 16,
    "O": 6
}

# === READ ACF.DAT ===
# Skip the header (2 lines) and footer (4 lines)
df = pd.read_csv(acf_file, delim_whitespace=True, skiprows=2, skipfooter=4, engine='python')
# Rename columns for clarity
df.columns = ["Index", "X", "Y", "Z", "Bader_e", "Min_dist", "Atomic_vol"]

# Map atom indices to elements
atom_labels = []
idx = 0
for elem, count in zip(elements, counts):
    atom_labels.extend([elem] * count)
df["Element"] = atom_labels

# Compute net charge = valence - Bader_e
df["Valence_e"] = df["Element"].map(valence_dict)
df["Net_charge"] = df["Valence_e"] - df["Bader_e"]

# Save to CSV
df.to_csv(output_csv, index=False)

# Print summary
print("Net charges per element type:")
print(df.groupby("Element")["Net_charge"].mean())
print(f"\nDetailed table saved to: {output_csv}")
print("Bader charge analysis complete.")