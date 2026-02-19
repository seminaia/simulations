import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ===================================================================
# Enhanced Data Loading with Diagnostics
# ===================================================================
def load_data(file_path):
    """Load and validate data with detailed error reporting"""
    try:
        # Try different CSV parsing strategies
        for header_option in [None, 0]:
            try:
                data = pd.read_csv(file_path, 
                                 header=header_option, 
                                 comment='#',
                                 engine='python')
                print("\nRaw data preview:")
                print(data.head())
                
                # Validate expected columns
                if data.shape[1] < 3:
                    raise ValueError("CSV must contain at least 3 columns")
                    
                # Use first 3 columns if unnamed
                data = data.iloc[:, :3]
                data.columns = ['Directory', 'Volume', 'Energy']
                
                # Convert numeric types
                data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
                data['Energy'] = pd.to_numeric(data['Energy'], errors='coerce')
                data.dropna(inplace=True)
                
                if data.empty:
                    raise ValueError("No valid numeric data found after cleaning")
                    
                print("\nCleaned data:")
                print(data)
                return data
                
            except pd.errors.ParserError:
                continue

        raise ValueError("Could not parse CSV file")

    except Exception as e:
        print(f"\nCRITICAL ERROR IN DATA LOADING: {str(e)}")
        print("Please verify:")
        print("1. File exists at specified path")
        print("2. CSV format is consistent")
        print("3. Numeric columns contain only numbers")
        print("4. There are at least 3 columns")
        raise

# ===================================================================
# Enhanced Birch-Murnaghan Fit with Fallbacks
# ===================================================================
def birch_murnaghan(V, E0, V0, B0, B0_prime):
    """Equation of state with parameter bounds"""
    eta = (V0 / V) ** (2/3)
    return E0 + (9/16) * B0 * V0 * (
        (eta - 1)**3 * B0_prime + 
        (eta - 1)**2 * (6 - 4 * eta)
    )

def safe_curve_fit(xdata, ydata):
    """Robust fitting with multiple strategies"""
    # Initial guesses from data features
    V0_guess = xdata[np.argmin(ydata)]
    E0_guess = np.min(ydata)
    
    strategies = [
        {'p0': [E0_guess, V0_guess, 10, 4], 'maxfev': 10000},
        {'p0': [E0_guess, V0_guess, 50, 2], 'bounds': (
            [E0_guess-1, V0_guess*0.9, 1, 1], 
            [E0_guess+1, V0_guess*1.1, 100, 5]
        )}
    ]
    
    for strategy in strategies:
        try:
            popt, pcov = curve_fit(birch_murnaghan, xdata, ydata, **strategy)
            print(f"\nSuccessful fit with strategy: {strategy}")
            return popt, pcov
        except Exception as e:
            print(f"Fit attempt failed: {str(e)}")
            continue
            
    raise RuntimeError("All fitting strategies failed")

# ===================================================================
# Main Analysis Workflow
# ===================================================================
if __name__ == "__main__":
    # Load data with diagnostics
    print("\n=== DATA LOADING PHASE ===")
    data = load_data('energy_vol_isif2.csv')
    material = 'La2NiO4'
    # Extract validated data
    Vol = data['Volume'].values
    Energy = data['Energy'].values
    
    # Auto-generate scaling factors if needed
    if len(Vol) == 7:  # Common case for volume scaling
        scaling_factors = np.array([0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03])
    else:
        scaling_factors = np.linspace(0.97, 1.03, len(Vol))
    
    print(f"\nNumber of data points: {len(Vol)}")
    print(f"Scaling factors used: {scaling_factors}")

    # Perform robust fitting
    print("\n=== ENERGY FITTING PHASE ===")
    try:
        popt_energy, _ = safe_curve_fit(Vol, Energy)
        print("\nFit parameters [E0, V0, B0(GPa), B0_prime]:")
        print(popt_energy)
    except Exception as e:
        print(f"\nFATAL: Energy fitting failed: {e}")
        print("Possible solutions:")
        print("1. Check energy/volume data follows EOS curve")
        print("2. Verify units (eV/atom and Å³/atom)")
        print("3. Try manual initial guesses")
        raise

    # Polynomial fit for scaling factors
    print("\n=== SCALING FACTOR FITTING ===")
    sf_coeffs = np.polyfit(Vol, scaling_factors, 1)  # Linear fit
    print(f"Scaling factor coefficients: {sf_coeffs}")

    # Find minimum energy state
    Vol_fine = np.linspace(min(Vol), max(Vol), 1000)
    E_fine = birch_murnaghan(Vol_fine, *popt_energy)
    min_idx = np.argmin(E_fine)
    
    results = {
        'min_volume': Vol_fine[min_idx],
        'min_energy': E_fine[min_idx],
        'scaling_factor': np.polyval(sf_coeffs, Vol_fine[min_idx])
    }

    # ===================================================================
    # Plotting
    # ===================================================================
   
    plt.figure(figsize=(10, 6))

    # Energy plot with equilibrium volume annotation - Only plot once!
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(Vol, Energy, 'bo', label='Raw Data')
    plt.plot(Vol_fine, E_fine, 'r-', label='Birch-Murnaghan Fit')
    plt.plot(results['min_volume'], results['min_energy'], 'g*', 
            markersize=15, label='Minimum Energy')

    # Equilibrium volume annotation
    plt.axvline(results['min_volume'], color='gray', linestyle='--', alpha=0.7)
    plt.text(results['min_volume'], np.median(Energy),  # Better vertical position
            f'Equilibrium Volume: {results["min_volume"]:.4f} Å³/atom',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10,
            color='darkred',
            bbox=dict(facecolor='white', alpha=0.8))

    plt.ylabel('Energy (eV/atom)')
    plt.title('Energy-Volume Relationship Diagnostics')
    plt.legend()

    # Scaling factor plot with equilibrium volume annotation - Only plot once!
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(Vol, scaling_factors, 'ms', label='Scaling Factors')
    plt.plot(Vol_fine, np.polyval(sf_coeffs, Vol_fine), 'k--',
            label='Linear Fit')
    plt.plot(results['min_volume'], results['scaling_factor'], 'g*',
            markersize=15, label='Optimal Scaling Factor')

    plt.tight_layout()
    plt.savefig(f'{material}_EOS.png', dpi=300)
    print("\nDiagnostic plot saved to diagnostic_plot.png")

    # Final output
    print("\n=== FINAL RESULTS ===")
    print(f"Equilibrium Volume: {results['min_volume']:.4f} Å³/atom")
    print(f"Minimum Energy: {results['min_energy']:.4f} eV/atom")
    print(f"Optimal Scaling Factor: {results['scaling_factor']:.4f}")