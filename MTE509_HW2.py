import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, e, m_e

# ============ Part (a): Symbolic derivation ============
beta, Cs, lam = sp.symbols('beta Cs lambda', positive=True)

# Radius function
r = sp.sqrt((0.61 * lam / beta)**2 + (Cs * beta**3)**2)

# Differentiate r with respect to beta
dr = sp.diff(r, beta)
dr = sp.simplify(dr)

# Solve dr = 0 for beta (find optimal beta)
beta_opt_sym = sp.solve(dr, beta)
# Filter positive real solutions
beta_opt_sym = [sol for sol in beta_opt_sym if sol.is_real and sol > 0][0]
beta_opt_sym = sp.simplify(beta_opt_sym)

# Calculate r_min
r_min_sym = sp.simplify(r.subs(beta, beta_opt_sym))

print("="*50)
print("Part (a): Symbolic Results")
print("="*50)
print("β_opt =")
sp.pprint(beta_opt_sym)
print("\nr_min =")
sp.pprint(r_min_sym)

# ============ Part (b): Numerical evaluation ============
print("\n" + "="*50)
print("Part (b): Numerical Values")
print("="*50)

# Define voltage symbol (in volts)
V = sp.symbols('V', positive=True)

# Relativistic wavelength (in nm)
# lambda = h/sqrt(2*m_e*e*V*(1+e*V/(2*m_e*c**2)))*1e9  # Convert to nm
lam_expr = h * (2*m_e*e*V*(1+e*V/(2*m_e*c**2)))**(-sp.Rational(1,2))*1e9

# Cs values in nm (1 mm = 1e6 nm)
Cs_values_nm = [1e6, 3e6]

# Voltage values to evaluate
V_vals = [100e3, 200e3, 400e3]  # in volts

# Create functions for each Cs
results = {}
for i, Cs_nm in enumerate(Cs_values_nm):
    # Substitute Cs with numeric value
    beta_expr = beta_opt_sym.subs(Cs, Cs_nm)*1000  # Convert to mrad
    r_expr = r_min_sym.subs(Cs, Cs_nm)
    
    # Substitute lambda expression
    beta_expr = beta_expr.subs(lam, lam_expr)
    r_expr = r_expr.subs(lam, lam_expr)
    
    # Store expressions for plotting
    results[f'Cs_{i}'] = {
        'beta': beta_expr,
        'r': r_expr,
        'label': f'Cs = {Cs_nm/1e6:.0f} mm'
    }
    
    # Evaluate at specific voltages
    print(f"\nFor Cs = {Cs_nm/1e6:.0f} mm:")
    for V_val in V_vals:
        beta_val = beta_expr.subs(V, V_val).evalf()
        r_val = r_expr.subs(V, V_val).evalf()
        print(f"  V = {V_val/1e3:.0f} kV: β_opt = {beta_val:.3f} mrad, r_min = {r_val:.3f} nm")

# ============ Plotting with Matplotlib ============
print("\n" + "="*50)
print("Generating Plots")
print("="*50)

# Create voltage range for plotting (50 kV to 500 kV)
V_range = np.linspace(50e3, 500e3, 200)

# Convert SymPy expressions to NumPy functions
beta_funcs = []
r_funcs = []
labels = []

for i, Cs_nm in enumerate(Cs_values_nm):
    # Substitute Cs and lambda
    beta_expr = beta_opt_sym.subs(Cs, Cs_nm).subs(lam, lam_expr)*1000  # Convert to mrad
    r_expr = r_min_sym.subs(Cs, Cs_nm).subs(lam, lam_expr)
    sp.pprint(beta_expr)
    sp.pprint(r_expr)
    # Lambdify for NumPy
    beta_func = sp.lambdify(V, beta_expr, 'numpy')
    r_func = sp.lambdify(V, r_expr, 'numpy')
    
    beta_funcs.append(beta_func)
    r_funcs.append(r_func)
    labels.append(f'Cs = {Cs_nm/1e6:.0f} mm')

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot beta_opt
for beta_func, label in zip(beta_funcs, labels):
    beta_values = beta_func(V_range)  # Already in mrad
    ax1.plot(V_range/1e3, beta_values, linewidth=2, label=label)
ax1.set_xlabel('Accelerating Voltage (kV)', fontsize=12)
ax1.set_ylabel(r'$\beta_{\mathrm{opt}}$ (mrad)', fontsize=12)
ax1.set_title('Optimal Aperture Angle vs Voltage', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot r_min
for r_func, label in zip(r_funcs, labels):
    r_values = r_func(V_range)  # Already in nm
    ax2.plot(V_range/1e3, r_values, linewidth=2, label=label)
ax2.set_xlabel('Accelerating Voltage (kV)', fontsize=12)
ax2.set_ylabel(r'$r_{\min}$ (nm)', fontsize=12)
ax2.set_title('Minimum Resolution vs Voltage', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============ Part (c): Conclusions ============
print("\n" + "="*50)
print("Part (c): Conclusions")
print("="*50)
print("""From the plots and numerical results:

1. Increasing accelerating voltage reduces both β_opt and r_min
   - Higher voltage → shorter wavelength → better resolution
   - β_opt ∝ V^(-1/4) approximately 

2. Larger Cs values:
   - Require smaller β_opt to balance spherical aberration
   - Result in larger r_min (worse resolution)

3. The improvement in r_min with voltage is sub-linear
   - r_min ∝ V^(-3/8) 
   - Relativistic effects make the decrease slightly slower

4. At typical TEM voltages (100-400 kV):
   - β_opt ranges from ~3-7 mrad depending on Cs
   - r_min ranges from ~0.2-0.6 nm for Cs = 1-3 mm
   - Higher voltage is beneficial but with diminishing returns
""")

V1=100e3
V2 = 300e3
B=1
E1 = e*V1 + m_e*c**2
E2 = e*V2 + m_e*c**2

p1 = ((E1/c)**2 - (m_e*c)**2)**0.5
p2 = ((E2/c)**2 - (m_e*c)**2)**0.5

r1=(p1/(e*B)*1e3)
r2=(p2/(e*B)*1e3)
print(f"r1: {r1}, r2: {r2}")