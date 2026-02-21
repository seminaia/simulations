#!/usr/bin/env python3
"""
Extract the final Epot value from simulation.log files in folders outputs_tau*
and plot Epot vs tau.

Assumes simulation.log contains columns: step, Epot, press (space-separated).
"""

import glob
import os
import re
import matplotlib.pyplot as plt

# List everything in the current directory
all_items = os.listdir('.')
print("All files and folders:")
for item in sorted(all_items):
    print(f"  - {item}")

# Specifically check for folders matching the pattern
matching = glob.glob('outputs_tau*')
print(f"\nFolders matching 'outputs_tau*': {matching}")

def get_final_epot(log_path):
    """
    Read simulation.log and return the Epot value from the last data line.
    The file is assumed to have a header and then lines with at least two numeric columns.
    Returns None if no valid Epot is found.
    """
    final_epot = None
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Expect at least 2 columns; try to convert the second to float
                if len(parts) >= 2:
                    try:
                        epot = float(parts[1])
                        final_epot = epot   # keep updating to last value
                    except ValueError:
                        # Skip lines that don't have a number in column 2 (e.g., header)
                        pass
    except Exception as e:
        print(f"Warning: Could not read {log_path}: {e}")
    return final_epot

def get_final_press(log_path):
    """
    Read simulation.log and return the press value from the last data line.
    The file is assumed to have a header and then lines with at least three numeric columns.
    Returns None if no valid press is found.
    """
    final_press = None
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Expect at least 3 columns; try to convert the third to float
                if len(parts) >= 3:
                    try:
                        press = float(parts[2])
                        final_press = press   # keep updating to last value
                    except ValueError:
                        # Skip lines that don't have a number in column 3 (e.g., header)
                        pass
    except Exception as e:
        print(f"Warning: Could not read {log_path}: {e}")
    return final_press

def main():
    # Find all folders matching "outputs_tau*"
    folders = glob.glob("outputs_tau*")
    if not folders:
        print("No folders found matching 'outputs_tau*'")
        return

    tau_values = []
    epot_values = []
    press_values = []

    for folder in sorted(folders):
        # Extract tau from folder name (e.g., "outputs_tau0.5" -> 0.5)
        match = re.search(r"outputs_tau([\d\.]+(?:[eE][-+]?\d+)?)", folder)
        if not match:
            print(f"Skipping {folder}: could not parse tau value")
            continue
        tau = float(match.group(1))

        log_path = os.path.join(folder, "simulation.log")
        if not os.path.isfile(log_path):
            print(f"Skipping {folder}: simulation.log not found")
            continue

        epot = get_final_epot(log_path)
        if epot is None:
            print(f"Skipping {folder}: no Epot value found in log")
            continue

        tau_values.append(tau)
        epot_values.append(epot)
        press = get_final_press(log_path)
        if press is None:
            print(f"Skipping {folder}: no press value found in log")
            continue
        press_values.append(press)
        print(f"tau = {tau:.4f}, Epot = {epot:.6f}, Press = {press:.6f}")

    if not tau_values:
        print("No data collected.")
        return

    # Sort by tau for a clean plot
    tau_values, epot_values, press_values = zip(*sorted(zip(tau_values, epot_values, press_values)))
    # Create a figure with two subplots side by side (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left subplot: Epot vs tau
    ax1.plot(tau_values, epot_values, marker='o', linestyle='-', color='b')
    ax1.set_xlabel('tau')
    ax1.set_xscale('log')  # Set x-axis to logarithmic scale
    ax1.set_ylabel('Final Epot (kcal/mol)')   # Replace with your units
    ax1.set_title('Potential Energy vs tau')
    ax1.grid(True)

    # Right subplot: Pressure vs tau
    ax2.plot(tau_values, press_values, marker='s', linestyle='--', color='r')
    ax2.set_xlabel('tau')
    ax2.set_xscale('log')  # Set x-axis to logarithmic scale
    ax2.set_ylabel('Final Pressure (kcal/A^3)') # Replace with your units
    ax2.set_yscale('log')  # Set y-axis to logarithmic scale if needed
    ax2.set_title('Pressure vs tau')
    ax2.grid(True)

    plt.suptitle('Argon EOS: Final Properties vs tau')  # Optional overall title
    plt.tight_layout()  # Adjust spacing to prevent overlap
    plt.savefig('epot_press_subplots.png', dpi=300)
    plt.show()
if __name__ == "__main__":
    main()