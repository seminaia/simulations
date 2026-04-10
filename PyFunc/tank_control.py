import control as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate._ivp.radau import P
import sympy as sp

# xdot = Ax + Bu in deviation, where x = h - hbar, u = qin - qbar
# y = Cx + Du  # output is just the height deviation, so C = 1, D = 0

t = sp.symbols('t', real=True)
h, qin = sp.symbols('h qin', positive=True, real=True)
A, cv = sp.symbols('A cv', positive=True, real=True)
f_expr = (qin - cv*sp.sqrt(h)) / A
print("Nonlinear RHS:")
sp.pprint(f_expr)

# Steady-state condition and linearization
hbar, qbar = sp.symbols('hbar qbar', positive=True, real=True)
ss_eq = sp.simplify(sp.Eq(0, f_expr.subs({h: hbar, qin: qbar})))

print("\nSteady-state condition:")
sp.pprint(ss_eq)
qbar_sol = sp.solve(ss_eq, qbar)[0]
print("\nSteady-state inflow qbar:")
sp.pprint(qbar_sol)
A_sym = sp.diff(f_expr, h).subs({h: hbar, qin: qbar})
B_sym = sp.diff(f_expr, qin).subs({h: hbar, qin: qbar})

print("\nLinearized A and B:")
print("A:")
sp.pprint(A_sym)
print("\nB:")
sp.pprint(B_sym)

# -------------------------
# Numerical parameters
# -------------------------
A0 = 2.0
cv0 = 0.05
hbar0 = 1.0

# To linearize about hbar0, choose the consistent steady inflow
qbar0 = sp.lambdify((hbar, cv), qbar_sol)(hbar0, cv0)
print(f"\nChosen steady state: hbar = {hbar0}, qbar = {qbar0}")

# Evaluate A, B
a = float(A_sym.subs({A: A0, cv: cv0, hbar: hbar0}))
b = float(B_sym.subs({A: A0, cv: cv0, hbar: hbar0}))
c = 1.0
d = 0.0
print(f"\nNumerical A: {a}")
print(f"Numerical B: {b}")

# Linear state-space system in deviation variables:
# x = h - hbar
# u = qin - qbar
# xdot = A x + B u
# y = C x + D u
sys = ct.ss(a, b, c, d)

# -------------------------
# Nonlinear simulation
# -------------------------
def f_nonlinear(h, qin, A, cv):
    return (qin - cv*np.sqrt(max(h, 0.0))) / A

tspan = np.linspace(0, 100, 1000)
dt = tspan[1] - tspan[0]

# Step in inflow around steady state
du = 0.01                 # small step
qin_step = qbar0 + du

h_nl = hbar0
hspan = []

for _ in tspan:
    hdot = f_nonlinear(h_nl, qin_step, A0, cv0)
    h_nl += hdot * dt
    hspan.append(h_nl)

hspan = np.array(hspan)
x_nl = hspan - hbar0      # deviation from steady state

# -------------------------
# Linear step response
# -------------------------
# Since system uses deviation input u, apply a step of size du
t_lin, y_lin = ct.step_response(du * sys, T=tspan)

# -------------------------
# Plot comparison
# -------------------------
plt.figure()
plt.plot(tspan, x_nl, label='Nonlinear deviation: h - hbar')
plt.plot(t_lin, y_lin, '--', label='Linearized deviation')
plt.xlabel('Time [s]')
plt.ylabel('Height deviation [m]')
plt.title('Tank level: nonlinear vs linearized response')
plt.grid(True)
plt.legend()
plt.show()