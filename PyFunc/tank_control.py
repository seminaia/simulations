import control as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate._ivp.radau import P
import sympy as sp
from sympy.abc import x,t,s,u
from IPython.display import display, Math
sp.init_printing()
# xdot = Ax + Bu in deviation, where x = h - hbar, u = qin - qbar
# y = Cx + Du  # output is just the height deviation, so C = 1, D = 0

h, qin= sp.symbols(names='h q_in', positive=True, real=True)
alpha = sp.symbols('alpha', positive=True, real=True)
qout = alpha*sp.sqrt(h)
f_expr = (qin - qout)
f_lamda = sp.lambdify((h, qin, alpha), f_expr)

hdot = sp.symbols(names='hdot', real=True)

print("Nonlinear:")
sp.pprint(sp.Eq(hdot, f_expr))

# Steady-state condition and linearization
hbar, qbar = sp.symbols('hbar qbar', positive=True, real=True)
ss_eq = sp.Eq(0, f_expr.subs({h: hbar, qin: qbar}))

print("\nSteady-state:")
sp.pprint(ss_eq)
qbar_sol = sp.solve(ss_eq, qbar)[0]

print("qbar as a function of hbar:")
sp.pprint(sp.Eq(qbar, qbar_sol))

A_sym = sp.diff(f_expr, h).subs({h: hbar, qin: qbar}) # Linearized by taking the Jacobian of the RHS with respect to h and qin, then evaluating at the steady state
B_sym = sp.diff(f_expr, qin).subs({h: hbar, qin: qbar})
xp = h - hbar
up = qin - qbar
up_eq = sp.Eq(u, qin - qbar)

rhs = A_sym*xp + B_sym*up

h_taylor = qout.series(h, hbar, n=2).removeO()  # Taylor expansion of the outflow around the steady state, to visualize the nonlinearity
lin_sys =  qbar_sol + up - h_taylor 

hdot_jac = sp.Eq(sp.symbols(names='xdot'), rhs)
hdot_taylor = sp.Eq(sp.symbols(names='xdot'),lin_sys)
y = sp.Eq(sp.symbols(names='y'), sp.symbols(names='x'))  # since C = 1, D = 0, output is just the deviation in height

print("\nLinearized system (xdot = A x + B u):")
sp.pprint(sp.Eq(sp.diff(xp, t,evaluate=False),lin_sys))
sp.pprint(sp.Eq(sp.diff(xp, t,evaluate=False), rhs))
print("x= {} u= {}".format(xp, up))
sp.pprint(sp.Eq(sp.diff(xp, t,evaluate=False), hdot_jac))
sp.pprint(sp.Eq(sp.diff(xp, t,evaluate=False), hdot_taylor))
print("\nOutput equation (y = C x + D u):")
sp.pprint(y)

print("\nLinearized A and B:")
print("A:")
sp.pprint(A_sym)
print("\nB:")
sp.pprint(B_sym)

# -------------------------
# Numerical parameters
# -------------------------
alpha0 = 0.05
hbar0 = 1.0

# To linearize about hbar0, choose the consistent steady inflow
qbar0 = sp.lambdify((hbar, alpha), qbar_sol)(hbar0, alpha0)
print(f"\nChosen steady state: hbar = {hbar0}, qbar = {qbar0}")

# Evaluate A, B
a = float(A_sym.subs({alpha: alpha0, hbar: hbar0}))
b = float(B_sym.subs({alpha: alpha0, hbar: hbar0}))
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
tspan = np.linspace(0, 100, 1000)
dt = tspan[1] - tspan[0]
A0=2
# Step in inflow around steady state
du = 0.01                 # small step
qin_step = qbar0 + du  # step up from steady state inflow

h_nl = hbar0
hspan = []

for _ in tspan:
    hdot_lam = f_lamda(h_nl, qin_step, alpha0)
    h_nl += hdot_lam *dt
    hspan.append(h_nl)

hspan = np.array(hspan)
x_nl = hspan - hbar0      # deviation from steady state

# -------------------------
# Linear step response
# -------------------------
# Since system uses deviation input u, apply a step of size du
t_lin, y_lin = ct.step_response(sys*du, tspan)

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