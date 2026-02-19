# -*- coding: utf-8 -*-
import numpy as np
import re
import pandas as pd
from scipy.optimize import lsq_linear

# Function to parse the stoichiometry from a chemical formula
def parse_formula(formula):
    pattern = re.compile(r'([A-Z][a-z]*)(\d*)')
    parsed = pattern.findall(formula)
    stoich = {}
    for element, count in parsed:
        stoich[element] = stoich.get(element, 0) + int(count) if count else 1
    return stoich

# Load the defect data to check for existing chemical potentials
defect_file_path = 'defect_output.csv'  # Replace with defect file path
defect_data = pd.read_csv(defect_file_path)

# Load the formation energies data
file_path = 'all_results_with_formation_energies.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Check if chemical potentials exist in the defect data
if 'ChemicalPotential(eV/atom)' in defect_data.columns:
    print("Chemical potential data already exists in defect_output.csv. Overwriting it...")
    defect_data = defect_data.drop(columns=['ChemicalPotential(eV/atom)'])
else:
    print("No chemical potential data found in defect_output.csv. Adding it...")

# Define the main phase and competing phases
main_phase = "La2NiO4"
competing_phases = ["La2O3", "O2"]  # Includes a pure element Sr

# Filter the relevant rows for main phase and competing phases
filtered_data = data[data['Folder'].isin([main_phase] + competing_phases)].reset_index(drop=True)

# Extract phase names and formation energies
phases = filtered_data['Folder'].tolist()
formation_energies = filtered_data['FormationEnergy(eV)'].values

# Identify pure elements and set their chemical potentials to 0
pure_elements = []
pure_element_indices = []

for i, phase in enumerate(phases):
    if len(parse_formula(phase)) == 1:  # Pure elements have only one unique element
        pure_elements.append(list(parse_formula(phase).keys())[0])
        pure_element_indices.append(i)

# Parse stoichiometries for all phases
parsed_stoichiometries = [parse_formula(phase) for phase in phases]

# Determine all unique elements
unique_elements = sorted(set(el for stoich in parsed_stoichiometries for el in stoich))

# Build the coefficient matrix and adjust for pure elements
coeff_matrix = np.zeros((len(phases), len(unique_elements)))

for i, stoich in enumerate(parsed_stoichiometries):
    for j, element in enumerate(unique_elements):
        coeff_matrix[i, j] = stoich.get(element, 0)

# Adjust formation energies and coefficients for pure elements
for idx, pure_el in zip(pure_element_indices, pure_elements):
    formation_energies[idx] = 0  # Set pure element formation energy to 0
# Parse stoichiometries for all phases
    coeff_matrix[idx, unique_elements.index(pure_el)] = 1  # Pure element contribution is 1

# Add constraint: Each element's chemical potential <= 0
lower_bounds = [-np.inf] * len(unique_elements)  # No lower bound on chemical potentials
upper_bounds = [0] * len(unique_elements)       # Chemical potentials must be <= 0

# Solve for chemical potentials using least squares with bounds
result = lsq_linear(coeff_matrix, formation_energies, bounds=(lower_bounds, upper_bounds), lsmr_tol='auto')
chemical_potentials = result.x

# Format results
chemical_potentials_dict = {unique_elements[i]: chemical_potentials[i] for i in range(len(unique_elements))}

# Print results
print("Chemical Potentials (in eV/atom):")
for element, mu in chemical_potentials_dict.items():
    print(f"{element}: {mu:.6f}")

# Verify the solution by calculating residuals
residuals = coeff_matrix @ chemical_potentials - formation_energies
print("\nVerification: Residuals of each equation:")
for i, res in enumerate(residuals):
    print(f"Equation {i + 1} ({phases[i]}): Residual = {res:.6e}")

if np.any(np.abs(residuals) > 1e-6):
    print("\nWarning: Significant residuals detected! Check input data or numerical precision.")

# Add or update the chemical potentials in the defect DataFrame
for element in unique_elements:
    defect_data[f"mu_{element}(eV/atom)"] = chemical_potentials_dict[element]

# Save updated defect data to a new file
output_file_path = defect_file_path.replace('.csv', '_chemi.csv')
defect_data.to_csv(output_file_path, index=False)
print(f"Updated defect file saved as: {output_file_path}")