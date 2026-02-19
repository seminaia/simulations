import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

def read_dope_trace(path):
    # Read the header line manually
    with open(path, "r") as f:
        header = f.readline()

    # Remove leading '#' and split on whitespace
    columns = header.lstrip("#").split()

    # Now read the numerical data
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        names=columns,
        engine="python"
    )

    return df
    
# ================================
# Experimental data files
# ================================
exp_files = {
    "Huang et al.": "Huang_LNO3.dat",
    "Shen et al.": "shen_LNO1.dat",
    "Sadykov et al.": "sadykov_LNO4.dat",
    "Tsivinkinberg et al.": "Tsivinkinberg_LNO2.dat",
}

# ================================
# Model file (your dope.trace CSV)
# ================================
model_file = "interpolation.dope.trace"

# ================================
# Load experimental data
# ================================
exp_data = {}
for label, path in exp_files.items():
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    T = df.iloc[:, 0].values            # K
    sigma = df.iloc[:, 1].values        # S/cm
    exp_data[label] = (T, sigma)

# ================================
# Load model data (CORRECTLY)
# ================================
df_model = read_dope_trace(model_file)
print(df_model.head())
print(df_model["sigma/tau0[1/(ohm*m*s)]"].head())
print(df_model["L0_h"].head())
print(df_model["L0_e"].head())
print(df_model["tau(s)"].head())
print(df_model["T[K]"].head())


T_model = df_model["T[K]"].values

sigma_over_tau = df_model["sigma/tau0[1/(ohm*m*s)]"].values
tau = df_model["tau(s)"].values

# Physical conductivity in S/m
sigma_model = sigma_over_tau * tau/100

# ================================
# Plot 1: Conductivity vs Temperature
# ================================
plt.figure(figsize=(7, 5))

for label, (T, sigma) in exp_data.items():
    plt.plot(T, sigma, "o-", label=label)

plt.plot(
    T_model,
    sigma_model,
    "--",
    linewidth=2.5,
    label="This work (model)"
)

plt.xlabel("Temperature (K)")
plt.ylabel("Conductivity (S/cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================
# Plot 2: Arrhenius plot
# ================================
plt.figure(figsize=(7, 5))

for label, (T, sigma) in exp_data.items():
    plt.plot(
        1000 / T,
        np.log10(sigma),
        "o",
        label=label
    )

plt.plot(
    1000 / T_model,
    np.log10(sigma_model),
    "--",
    linewidth=2.5,
    label="This work (model)"
)

plt.xlabel(r"$1000 / T$ (K$^{-1}$)")
plt.ylabel(r"$\log_{10}(\sigma\ /\ \mathrm{S\,cm^{-1}})$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
