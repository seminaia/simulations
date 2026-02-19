import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

# Define file paths for both ISIF2 and ISIF4 data
isif2_file_path = os.path.expanduser('./mag_vol_isif2.csv')
isif4_file_path = os.path.expanduser('./mag_vol_isif4.csv')

# Load data with correct delimiter
data_isif2 = pd.read_csv(isif2_file_path, delimiter=',')
data_isif4 = pd.read_csv(isif4_file_path, delimiter=',')

# Assign column names for both datasets
data_isif2 = data_isif2[['Volume (A^3/atom)', 'Magnetization (microB)', 'Ion']]
data_isif4 = data_isif4[['Volume (A^3/atom)', 'Magnetization (microB)', 'Ion']]

# Extract columns and group by Ion for ISIF2
ion_groups_isif2 = data_isif2.groupby('Ion')
ion_groups_isif4 = data_isif4.groupby('Ion')

def BM(V, a, b, c, d, e):
    """Birch-Murnaghan equation for fitting."""
    return a + b * V**(-2/3) + c * V**(-4/3) + d * V**(-6/3) + e * V**(-8/3)

# Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# Loop through ISIF2 ion groups to plot Volume vs Magnetization
for ion_name, ion_data in ion_groups_isif2:
    # Scatter plot for each ion in ISIF2 with increased transparency
    axs[0].scatter(ion_data['Volume (A^3/atom)'], ion_data['Magnetization (microB)'], label=ion_name, alpha=0.5)

    # Perform curve fitting
    popt, pcov = curve_fit(BM, ion_data['Volume (A^3/atom)'], ion_data['Magnetization (microB)'])
    
    # Create fit line
    Vol1 = np.linspace(ion_data['Volume (A^3/atom)'].min(), ion_data['Volume (A^3/atom)'].max(), 1000)
    fit = BM(Vol1, *popt)
    
    # Plot fit line
    axs[0].plot(Vol1, fit, label=f'Fit for {ion_name}')

# Finalize ISIF2 plot
axs[0].set_title("ISIF2: Magnetization (microB) vs Volume (A^3/atom)")
axs[0].set_xlabel("Volume (A^3/atom)")
axs[0].set_ylabel("Magnetization (microB)")
axs[0].legend(title="Ions and Fits")
axs[0].grid(True)

# Loop through ISIF4 ion groups to plot Volume vs Magnetization
for ion_name, ion_data in ion_groups_isif4:
    # Scatter plot for each ion in ISIF4 with increased transparency
    axs[1].scatter(ion_data['Volume (A^3/atom)'], ion_data['Magnetization (microB)'], label=ion_name, alpha=0.8)

    # Perform curve fitting
    popt, pcov = curve_fit(BM, ion_data['Volume (A^3/atom)'], ion_data['Magnetization (microB)'])
    
    # Create fit line
    Vol1 = np.linspace(ion_data['Volume (A^3/atom)'].min(), ion_data['Volume (A^3/atom)'].max(), 1000)
    fit = BM(Vol1, *popt)
    
    # Plot fit line
    axs[1].plot(Vol1, fit, label=f'Fit for {ion_name}')

# Finalize ISIF4 plot
axs[1].set_title("ISIF4: Magnetization (microB) vs Volume (A^3/atom)")
axs[1].set_xlabel("Volume (A^3/atom)")
axs[1].set_ylabel("Magnetization (microB)")
axs[1].legend(title="Ions and Fits")
axs[1].grid(True)

# Adjust layout and show plot
plt.savefig('LNO_mag.pdf', dpi=500)
plt.tight_layout()
plt.show()
