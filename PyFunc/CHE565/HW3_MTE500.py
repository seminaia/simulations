from networkx import omega
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.constants as const
import sympy as sp

def plot_gibbs_energy(T, T1, output_file):
    plt.figure(figsize=(8, 6))
    # EXACT requirement: x_A from 0 to 1 with step 0.05
    x_A = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    new_R= const.R/(const.N_A * const.e)  # J/mol to eV/atom

    # Build the mixing term safely (handle endpoints via limit -> 0)
    term = np.zeros_like(x_A, dtype=float)
    mask = (x_A > 0) & (x_A < 1)
    term[mask] = x_A[mask] * np.log(x_A[mask]) + (1 - x_A[mask]) * np.log(1 - x_A[mask])

    # Calculate Gibbs free energy of mixing in J/mol
    G_mix  = new_R * T  * term
    G_mix1 = new_R * T1 * term

    plt.plot(x_A, G_mix,  'o-', label=f'T={T}K')
    plt.plot(x_A, G_mix1, 's-', label=f'T={T1}K')
    plt.title('Gibbs Free Energy of Mixing vs Mole Fraction of A')
    plt.xlabel('Mole Fraction of A')
    plt.ylabel('Gibbs Free Energy of Mixing (eV/atom)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file, dpi=200)
    plt.close()

def differentiate_gibbs_energy(T, x_A_values):
    # Define symbolic variables
    x = sp.symbols('x')
    T_sym = sp.symbols('T')
    new_R= const.R/(const.N_A * const.e)  # J/mol to eV/atom
    print(f"R (eV/atom·K): {new_R}")
    # Define the Gibbs free energy of mixing function (in J/mol)
    G_mix_sym = new_R* T_sym * (x * sp.log(x) + (1 - x) * sp.log(1 - x))

    # Compute the first derivative analytically
    dG_dx_sym = sp.diff(G_mix_sym, x)


    # Chemical potentials for an ideal solution (excess parts in eV/atom)
    mu_A_sym = new_R * T_sym * sp.log(x)
    mu_B_sym = new_R * T_sym * sp.log(1 - x)

    # Convert symbolic expressions to numerical functions
    G_mix_func  = sp.lambdify((x, T_sym), G_mix_sym, 'numpy')
    dG_dx_func  = sp.lambdify((x, T_sym), dG_dx_sym, 'numpy')
    mu_A_func   = sp.lambdify((x, T_sym), mu_A_sym, 'numpy')
    mu_B_func   = sp.lambdify((x, T_sym), mu_B_sym, 'numpy')

    # Smooth curve for visualization (not required by the prompt, but helpful)
    x_plot = np.linspace(0.01, 0.99, 100)
    G_mix_plot = G_mix_func(x_plot, T)
    dG_dx_plot = dG_dx_func(x_plot, T)

    # Calculate and store results for each x_A value
    results = {}
    for x_A_val in x_A_values:
        if x_A_val == 0:
            G_mix_val = 0.0
            dG_dx_val = np.inf
            mu_A_val = -np.inf
            mu_B_val = 0.0
        elif x_A_val == 1:
            G_mix_val = 0.0
            dG_dx_val = -np.inf
            mu_A_val = 0.0
            mu_B_val = -np.inf
        else:
            G_mix_val = G_mix_func(x_A_val, T)
            dG_dx_val = dG_dx_func(x_A_val, T)
            mu_A_val  = mu_A_func(x_A_val, T)
            # FIX: evaluate μ_B at x_A directly (μ_B = RT ln(1 - x_A))
            mu_B_val  = mu_B_func(x_A_val, T)

        results[x_A_val] = {
            'G_mix': G_mix_val,
            'dG_dx': dG_dx_val,
            'mu_A': mu_A_val,
            'mu_B': mu_B_val
        }

    # Create plots (optional visualization, matches your original style)
    plt.figure(figsize=(12, 8))

    # Plot Gibbs free energy and its derivative
    plt.subplot(2, 1, 1)
    plt.plot(x_plot, G_mix_plot, label='Gibbs Free Energy of Mixing (eV/atom)')
    plt.plot(x_plot, dG_dx_plot, label='dG/dx (eV/atom)')
    for x_A_val in x_A_values:
        if 0 < x_A_val < 1:
            plt.plot(x_A_val, results[x_A_val]['G_mix'], 'ro')
            plt.text(x_A_val, results[x_A_val]['G_mix'], f'x={x_A_val}', va='bottom')
    plt.title(f'Gibbs Free Energy of Mixing and Its Derivative at T={T}K')
    plt.xlabel('Mole Fraction of A')
    plt.ylabel('Energy (eV/atom)')
    plt.legend()
    plt.grid(True)

    # Plot chemical potentials
    mu_A_plot = mu_A_func(x_plot, T)
    mu_B_plot = mu_B_func(x_plot, T)  # FIX here as well
    plt.subplot(2, 1, 2)
    plt.plot(x_plot, mu_A_plot, label='Chemical Potential of A (μ_A)')
    plt.plot(x_plot, mu_B_plot, label='Chemical Potential of B (μ_B)')
    for x_A_val in x_A_values:
        if 0 < x_A_val < 1:
            plt.plot(x_A_val, results[x_A_val]['mu_A'], 'ro')
            plt.text(x_A_val, results[x_A_val]['mu_A'], f'μ_A(x={x_A_val})', va='bottom')
            plt.plot(x_A_val, results[x_A_val]['mu_B'], 'go')
            plt.text(x_A_val, results[x_A_val]['mu_B'], f'μ_B(x={x_A_val})', va='top')
    plt.title(f'Chemical Potentials at T={T}K')
    plt.xlabel('Mole Fraction of A')
    plt.ylabel('Chemical Potential (eV/atom)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('Gibbs_Free_energy_derivative.png', dpi=200)
    plt.close()

    # Print the analytical expressions
    print("Analytical Expressions:")
    print(f"G_mix(x, T) = {G_mix_sym}")
    print(f"dG/dx = {dG_dx_sym}")
    print(f"μ_A = {mu_A_sym}")
    print(f"μ_B = {mu_B_sym}")

    return results

def gibbs_mixing_MgO(x_A, T):
    new_R= const.R/(const.N_A * const.e)  # J/mol to eV/atom
    if x_A == 0 or x_A == 1:
        return 0.0
    else:
        g_A0 = 0.0  # Pure A reference
        g_B0 = 0.0  # Pure B reference
        
        g_mix_A = new_R * T * (x_A * np.log(x_A))
        g_mix_B = new_R * T * ((1 - x_A) * np.log(1 - x_A))
        
        g_excess = omega * x_A * (1 - x_A)
        g_tot = x_A * g_A0 + (1 - x_A) * g_B0 + g_mix_A + g_mix_B + g_excess
        return g_mix_A, g_mix_B, g_tot
if __name__ == "__main__":
    
    # Part 1: Plot Gibbs free energy for T=300 K and T=1000 K with step 0.05 in x_A
    plot_gibbs_energy(300, 1000, 'Gibbs_Free_energy.png')

    # Part 2: Derivatives and chemical potentials at x_A = 0.5 and 0.3 for T=300 K
    x_A_values = [0.5, 0.3]
    T_values =[2700,2500,2400,1900]
    results = differentiate_gibbs_energy(300, x_A_values)
    x_A = np.linspace(0, 1, 100)
    G_mix_A, G_mix_B, G_mix_tot = gibbs_mixing(x_A, T_values)
    plt.plot(x_A, G_mix_A, label='Gibbs Free Energy of Mixing with Excess Term')
    plt.plot(x_A, G_mix_B, label='Gibbs Free Energy of Mixing without Excess Term')
    plt.plot(x_A, G_mix_tot, label='Total Gibbs Free Energy with Excess Term')
    plt.xlabel('Mole Fraction of A')
    plt.ylabel('Gibbs Free Energy of Mixing (eV/atom)')
    plt.title('Gibbs Free Energy of Mixing with Excess Term (ω=1 eV/atom)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Gibbs_Free_energy_with_excess.png', dpi=200)

    # Report results (answers exactly what’s asked)
    print(f"\nResults for T = 300K:")
    for x_A in x_A_values:
        print(f"\nFor x_A = {x_A}:")
        print(f"Gibbs Free Energy of Mixing: {results[x_A]['G_mix']:.6f} eV/atom")
        print(f"Slope (dG/dx): {results[x_A]['dG_dx']:.6f} eV/atom")
        print(f"Chemical Potential of A (μ_A): {results[x_A]['mu_A']:.6f} eV/atom")
        print(f"Chemical Potential of B (μ_B): {results[x_A]['mu_B']:.6f} eV/atom")
