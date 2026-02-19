import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import brentq

def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)

def load_defect_data(defect_csv):
    """Load defect data from the CSV file."""
    if not os.path.exists(defect_csv):
        raise FileNotFoundError(f"The defect CSV file '{defect_csv}' was not found. Please check the file path.")
    return pd.read_csv(defect_csv)

def filter_defects(defects_df, pure_components, compound_components):
    """Filter out rows corresponding to pure components and pure compounds."""
    return defects_df[~defects_df["Folder"].isin(pure_components + compound_components)]

def parse_formula(formula):
    """Parse the stoichiometry from a chemical formula."""
    pattern = re.compile(r'([A-Z][a-z]*)(\d*)')
    parsed = pattern.findall(formula)
    stoich = {}
    for element, count in parsed:
        stoich[element] = stoich.get(element, 0) + int(count) if count else 1
    return stoich

def extract_stoichiometry(folder_name):
    """Infer stoichiometric changes from the defect folder name."""
    stoichiometry = {"Sr": 0, "Ti": 0, "O": 0, "La": 0}

    if "La_Sr" in folder_name:
        stoichiometry["La"] -= 1
        stoichiometry["Sr"] += 1
    elif "O_Sr" in folder_name:
        stoichiometry["O"] -= 1
        stoichiometry["Sr"] += 1
    elif "O_Ti" in folder_name:
        stoichiometry["O"] -= 1
        stoichiometry["Ti"] += 1
    elif "V_O" in folder_name:
        stoichiometry["O"] += 1
    elif "V_Sr" in folder_name:
        stoichiometry["Sr"] += 1
    elif "V_Ti" in folder_name:
        stoichiometry["Ti"] += 1
    return stoichiometry


def calculate_mu_sum(row):
    """Calculate mu_sum based on inferred stoichiometry and chemical potentials."""
    folder_name = row["Folder"]
    stoichiometry = extract_stoichiometry(folder_name)
    mu_sum = sum(
        coeff * row.get(f"mu_{element}(eV/atom)", 0)
        for element, coeff in stoichiometry.items()
    )
    return mu_sum

def calculate_formation_energy(defects_df, bulk_energy):
    """Calculate formation energies dynamically based on defect-specific VBM and CBM."""
    formation_energies = {}
    defect_fermi_levels = {}

    for _, row in defects_df.iterrows():
        if row["Folder"] == "Bulk/charge_0":
            continue

        defect_name = row["Folder"].split("/")[0]
        charge = row["Charge(q)"]
        defect_energy = row["Energy(eV)"]
        mu_sum = calculate_mu_sum(row)

        # Extract defect-specific VBM and BandGap
        vbm = row["E_vbm(eV)"]
        band_gap = row["BandGap(eV)"]
        cbm = vbm + band_gap

        # Generate Fermi levels for this defect
        fermi_levels = np.linspace(0, cbm-vbm, 10)
        defect_fermi_levels[defect_name] = fermi_levels

        # Compute formation energy for the defect
        energies = defect_energy - bulk_energy + mu_sum + charge * (fermi_levels)

        if defect_name not in formation_energies:
            formation_energies[defect_name] = {}
        formation_energies[defect_name][charge] = energies

    return formation_energies, defect_fermi_levels

def find_transition_levels(formation_energies, defect_fermi_levels):
    """Identify transition levels dynamically for each defect."""
    transition_data = []

    for defect_name, charges in formation_energies.items():
        charge_states = sorted(charges.keys())
        fermi_levels = defect_fermi_levels[defect_name]

        for i in range(len(charge_states) - 1):
            q1, q2 = charge_states[i], charge_states[i + 1]
            energies_q1 = charges[q1]
            energies_q2 = charges[q2]

            for j in range(len(fermi_levels) - 1):
                if (energies_q1[j] - energies_q2[j]) * (energies_q1[j + 1] - energies_q2[j + 1]) < 0:
                    try:
                        transition_level = brentq(
                            lambda x: np.interp(x, fermi_levels, energies_q1) - np.interp(x, fermi_levels, energies_q2),
                            fermi_levels[j], fermi_levels[j + 1]
                        )
                        transition_data.append({
                            "Defect Name": defect_name,
                            "Charge Transition": f"{q1}→{q2}",
                            "Fermi Level (eV)": transition_level
                        })
                    except ValueError as e:
                        print(f"Brentq error for {defect_name} between q={q1} and q={q2}: {e}")

    return pd.DataFrame(transition_data)
def plot_formation_energy(formation_energies, defect_fermi_levels, output_plot):
    """Plot defect formation energies with combined labels for all charge states."""
    plt.figure(figsize=(14, 8))

    for defect_name, charges in formation_energies.items():
        fermi_levels = defect_fermi_levels[defect_name]

        # Plot all charge states for this defect
        for charge, energies in charges.items():
            plt.plot(
                fermi_levels, energies,
                linestyle="--" if charge < 0 else "-", linewidth=1, alpha=0.6
            )

        # Add a combined label for the defect
        plt.plot(
            fermi_levels, [np.min([charges[c][i] for c in charges]) for i in range(len(fermi_levels))],
            label=f"{defect_name} (all charges)", linewidth=2
        )

    plt.xlabel("Fermi Level (eV)", fontsize=14)
    plt.ylabel("Formation Energy (eV)", fontsize=14)
    plt.title("Defect Formation Energy Diagram (Grouped Charges)", fontsize=16)

    # Adjust legend
    plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to leave space for the legend
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"Plot saved to: {output_plot}")

def save_to_csv(formation_energies, fermi_levels, transitions, output_dir):
    """Save formation energies and transitions to CSV."""
    formation_data = []
    for defect_name, charges in formation_energies.items():
        for charge, energies in charges.items():
            for fermi_level, energy in zip(fermi_levels, energies):
                formation_data.append({
                    "Defect Name": defect_name,
                    "Charge": charge,
                    "Fermi Level (eV)": fermi_level,
                    "Formation Energy (eV)": energy
                })

    pd.DataFrame(formation_data).to_csv(os.path.join(output_dir, "formation_energies.csv"), index=False)
    print("Formation energies saved to 'formation_energies.csv'.")

    transitions.to_csv(os.path.join(output_dir, "transition_levels.csv"), index=False)
    print("Transition levels saved to 'transition_levels.csv'.")
def main():
    # Configuration
    defect_csv = "defect_output_chemi.csv"
    output_dir = "./results"
    output_plot = os.path.join(output_dir, "formation_energy_diagram.png")

    create_output_dir(output_dir)

    # Load defect data
    defects_df = load_defect_data(defect_csv)

    # Filter out pure components and compounds
    pure_components = ["La", "Sr", "Ti", "O"]
    compound_components = ["La2O3", "SrTiO3", "TiO2", "LaTiO3", "Sr2TiO4", "Sr1O1"]
    defects_df = filter_defects(defects_df, pure_components, compound_components)

    # Extract bulk energy
    bulk_row = defects_df.loc[defects_df["Folder"] == "Bulk/charge_0"]
    bulk_energy = bulk_row["Energy(eV)"].values[0]

    # Ensure required columns exist
    if "E_vbm(eV)" not in defects_df.columns or "BandGap(eV)" not in defects_df.columns:
        raise KeyError("The CSV must contain 'E_vbm(eV)' and 'BandGap(eV)' columns.")

    # Calculate formation energies and defect-specific Fermi levels
    formation_energies, defect_fermi_levels = calculate_formation_energy(defects_df, bulk_energy)

    # Find transition levels
    transitions = find_transition_levels(formation_energies, defect_fermi_levels)

    # Plot formation energy diagram
    plot_formation_energy(formation_energies, defect_fermi_levels, output_plot)

    # Save results to CSV
    save_to_csv(formation_energies, defect_fermi_levels, transitions, output_dir)

if __name__ == "__main__":
    main()