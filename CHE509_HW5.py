import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import solve_ivp

R = constants.R
T0 = 473.0  # K

# Feed mole fractions
xCO0 = 0.01
xO20 = 0.025
xH20 = 0.75

# Flow / reactor parameters
mdot = 80.0       # g/s
Mw = 29.0         # g/mol, as in your code/Table 10.1
ndot = mdot / Mw  # mol/s

L = 10.0          # cm
Ab = 60.0         # cm^2
am = 268.95       # cm^2 Pt / cm^3 reactor
eps_r = 1.0
am_eff = am * eps_r

# Kinetics
Ar = 6.699e9
# Er/R = 12556
k1 = Ar * np.exp(-12556.0 / T0)
k2 = k1 / 50.0

# CO adsorption equilibrium constant
dS_co = 4.182 # ds/R = 4.182
dH_co = -961.0 # dh/R = -961.0 K
Kco = np.exp(dS_co - dH_co /  T0)

# Dimensionless prefactor
P = Ab * L * am_eff / ndot

def rhs(zeta, y):
    xCO, xO2 = y
    denom = (1.0 + Kco * xCO)**2

    # Dimensionless rates: d x_i / d zeta contributions
    r1 = P * k1 * xO2 * xCO / denom
    r2 = P * k2 * xO2 * xH20 / denom

    dxCO = -r1
    dxO2 = -0.5 * (r1 + r2)

    return [dxCO, dxO2]

zeta = np.linspace(0.0, 1.0, 400)

y0 = [xCO0, xO20]

sol = solve_ivp(
    rhs,
    [0.0, 1.0],
    y0,
    t_eval=zeta,
    rtol=1e-10,
    atol=1e-14
)

xCO = sol.y[0]
xO2 = sol.y[1]

# Dimensionless concentrations
CO_ratio = xCO / xCO0
O2_ratio = xO2 / xO20

# Selectivity
S_CO = (1 - xCO/xCO0)*xCO0/(2*(1 - xO2/xO20)*xO20)

# Plot
plt.figure(figsize=(7, 5))

plt.plot(zeta, CO_ratio, label=r"$x_{\mathrm{CO}}/x_{\mathrm{CO},0}$")
plt.plot(zeta, O2_ratio, label=r"$x_{\mathrm{O_2}}/x_{\mathrm{O_2},0}$")
plt.plot(zeta, S_CO, label=r"$S_{\mathrm{CO}}$")

plt.xlabel(r"$\zeta=z/L$")
plt.ylabel(r"$x_i/x_{i,0},\ S_{\mathrm{CO}}$")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("CHE509_HW5.png")
plt.show()