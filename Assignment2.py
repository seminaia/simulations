import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
# Constants
F = 96485.0          # C/mol
n = 4                # electrons for OR1
H = 90700.0          # J/mol  (ΔH° for CH3OH -> CO + 2H2)
S = 201.2            # J/mol/K (ΔS° for CH3OH -> CO + 2H2)

# Temperature range: 0 to 1000 °C, converted to Kelvin
T_C = np.linspace(0, 1000, 500)
T_K = T_C + 273.15

# Gibbs energy and equilibrium potential for OR1
G = H - S * T_K                      # J/mol
E_eq = G / (n * F)                   # V

# Feasibility boundary
Phi_boundary = E_eq

# Plot the T–Φ feasibility map
plt.figure(figsize=(8, 5))
plt.plot(T_C, Phi_boundary, 'b-', linewidth=2,
         label=r'$\Phi = E_{\mathrm{eq,OR1}}(T)$')

# Shade the feasible region: Φ > E_eq
plt.fill_between(T_C, Phi_boundary, 1.5, color='green', alpha=0.25,
                 label='Feasible region (OR1 spontaneous)')
plt.fill_between(T_C, -0.5, Phi_boundary, color='red', alpha=0.15,
                 label='Not feasible')

# Mark where E_eq crosses zero
T_zero = H / S - 273.15              # °C
plt.axvline(T_zero, color='k', linestyle='--', linewidth=1)
plt.text(T_zero + 15, 0.9, f'$T \\approx {T_zero:.0f}$ °C',
         fontsize=10)

plt.xlabel('Temperature (°C)')
plt.ylabel(r'Electrode potential $\Phi$ (V)')
plt.title('Feasibility map for OR1: CH$_3$OH $\\rightleftharpoons$ CO + 4H$^+$ + 4e$^-$')
plt.xlim(0, 1000)
plt.ylim(-0.3, 1.5)
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('HW2.pdf', dpi=120)
print("saved HW2.pdf")
# Print key values
print(f"ΔS° = {S:.1f} J/mol/K")
print(f"E_eq at 25 °C  = {H/(n*F) - S*298.15/(n*F):.4f} V")
print(f"E_eq crosses 0 at T = {H/S:.1f} K = {T_zero:.1f} °C")