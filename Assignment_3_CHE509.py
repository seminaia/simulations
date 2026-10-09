import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- symbols ---
Xf = sp.Function('X')
W  = sp.symbols('W', positive=True)
Xv = sp.symbols('Xv', real=True)
T  = sp.symbols('T', real=True)

# --- your data (x_D0 fixed to 0.41) ---
x_A0, x_B0, x_C0, x_D0 = 0.0177, 0.33, 0.11, 0.41
n_T0, rho_b = 20460, 90
n_A0  = n_T0 * x_A0
thB, thC, thD = x_B0/x_A0, x_C0/x_A0, x_D0/x_A0

T0, Cpm, H_rxn = 874.67, 7.98, -17165
Cp0 = Cpm / x_A0
T_of_X = T0 + (-H_rxn / Cp0) * Xv

xA = x_A0*(1 - Xv);  xB = x_A0*(thB - Xv)
xC = x_A0*(thC + Xv); xD = x_A0*(thD + Xv)

k = sp.exp(12.88 - sp.Rational(3340)/T)
K = sp.exp(-4.72 + sp.Rational(8640)/T)

rate = sp.Rational(4, 379*rho_b) * k.subs(T, T_of_X) * xA * xB * (1 - xC*xD/(xA*xB*K.subs(T, T_of_X)))
rhs  = rate / n_A0
x_target = 0.83

# --- integrate with scipy (dense smooth grid) ---
f = sp.lambdify(Xv, rhs, 'numpy')

W_grid = np.linspace(0, 4e5, 5000)
sol = solve_ivp(lambda w, y: [f(y[0])], [0, 4e5], [0], t_eval=W_grid)

W_arr = sol.t
X_arr = sol.y[0]
idx = np.where(X_arr >= x_target)[0][0]
W_target = np.interp(x_target, X_arr[idx-1:idx+1], W_arr[idx-1:idx+1])
T_target = float(T_of_X.subs(Xv, x_target))
print(f"W = {W_target:,.0f} lb,  T = {T_target:.2f} R = {T_target-459.67:.2f} F")
T_fun = sp.lambdify(Xv, T_of_X, 'numpy')
# --- plot ---
fig, (ax1,ax2) = plt.subplots(figsize=(8, 5), nrows=2, sharex=True)
fig.suptitle('X and T vs W')
ax1.plot(W_arr, X_arr, 'b-', lw=2)
ax1.set_ylabel('Conversion  X', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(alpha=0.3)
ax1.annotate(f'X = {x_target}, W = {W_target:,.0f} lb', xy=(W_target, x_target), xytext=(W_target, x_target-0.1))

ax2.plot(W_arr, T_fun(X_arr), 'r--', lw=2)
ax2.set_ylabel('Temperature  (°R)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_xlabel('W  (lb catalyst)')
ax2.annotate(f'T = {T_target:.2f} R, W = {W_target:,.0f} lb', xy=(W_target, T_target), xytext=(W_target, T_target-5))

ax1.axvline(W_target, color='b', ls=':', lw=1.5)
ax1.axhline(x_target, color='b', ls=':', lw=1.5)
ax2.axhline(T_target, color='r', ls=':', lw=1.5)
ax2.axvline(W_target, color='r', ls=':', lw=1.5)
ax2.grid(alpha=0.3)
ax1.plot(W_target, x_target, 'bo', ms=6)

plt.tight_layout()
plt.savefig('Assignment_3_CHE509.png')
plt.show()