import control as ct
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# -------------------------
# Symbolic setup
# -------------------------
t = sp.symbols('t', real=True)
h, qin = sp.symbols('h qin', positive=True, real=True)
A, cv = sp.symbols('A cv', positive=True, real=True)

f_expr = (qin - cv*sp.sqrt(h)) / A
print("Nonlinear RHS:")
sp.pprint(f_expr)

# Steady-state symbols
hbar, qbar = sp.symbols('hbar qbar', positive=True, real=True)

# Steady-state condition
ss_eq = sp.Eq(0, f_expr.subs({h: hbar, qin: qbar}))
print("\nSteady-state condition:")
sp.pprint(ss_eq)

# Jacobians for linearization
a_sym = sp.diff(f_expr, h).subs({h: hbar, qin: qbar})
b_sym = sp.diff(f_expr, qin).subs({h: hbar, qin: qbar})

print("\na = df/dh at steady state:")
sp.pprint(sp.simplify(a_sym))

print("\nb = df/dqin at steady state:")
sp.pprint(sp.simplify(b_sym))

# -------------------------
# Numerical parameters
# -------------------------
A0 = 2.0
cv0 = 0.05
hbar0 = 1.0

# To linearize about hbar0, choose the consistent steady inflow
qbar0 = cv0 * np.sqrt(hbar0)

print(f"\nChosen steady state: hbar = {hbar0}, qbar = {qbar0}")

# Evaluate A, B
a = float(a_sym.subs({A: A0, cv: cv0, hbar: hbar0}))
b = float(b_sym.subs({A: A0, cv: cv0, hbar: hbar0}))
c = 1.0
d = 0.0

print(f"a = {a}")
print(f"b = {b}")

# Linear state-space system in deviation variables:
# x = h - hbar
# u = qin - qbar
# xdot = a x + b u
sys = ct.ss(a, b, c, d)

# -------------------------
# Nonlinear simulation
# -------------------------
def f_nonlinear(h, qin, A, cv):
    return (qin - cv*np.sqrt(max(h, 0.0))) / A

tspan = np.linspace(0, 100, 1000)
dt = tspan[1] - tspan[0]

# Step in inflow around steady state
du = 0.02                 # small step
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

# Also plot absolute height if wanted
plt.figure()
plt.plot(tspan, hspan, label='Nonlinear h(t)')
plt.plot(t_lin, hbar0 + y_lin, '--', label='Linearized h(t)')
plt.xlabel('Time [s]')
plt.ylabel('Height [m]')
plt.title('Absolute tank level response')
plt.grid(True)
plt.legend()
plt.show()