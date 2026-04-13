from re import A
from IPython.core import display_functions
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x,t,s,u, o
from scipy.constants import g
sp.init_printing()
# xdot = Ax + Bu in deviation, where x = h - hbar, u = qin - qbar
# y = Cx + Du  # output is just the height deviation, so C = 1, D = 0
A0=2

h, qin= sp.symbols(names='h q_in', positive=True, real=True)
cv, Kv, alpha = sp.symbols('C_v K_v alpha', positive=True, real=True)
dPv = sp.N(g, 4)*h

hbar, qbar = sp.symbols(names='hbar qbar', positive=True, real=True)
hdot = sp.symbols(names='hdot', real=True)

f_v = cv*Kv*alpha**(o-1)
qout = f_v*sp.sqrt(dPv)
f_expr = (qin - qout)
f_lamda = sp.lambdify((h, qin, alpha, cv, Kv, o), f_expr)


print("Nonlinear:")
hdot_nl_eq = sp.Eq(hdot, f_expr)
sp.pprint(hdot_nl_eq)

print("\nSteady-state:")
ss_eq = sp.Eq(0, f_expr.subs({h: hbar, qin: qbar}))
sp.pprint(ss_eq)
qbar_sol = sp.solve(ss_eq, qbar)[0]
print("qbar as a function of hbar:")
sp.pprint(sp.Eq(qbar, qbar_sol))
sp.pprint(sp.Eq(qbar, qout.subs(h, hbar)))
A_sym = sp.diff(f_expr, h).subs({h: hbar, qin: qbar})
B_sym = sp.diff(f_expr, qin).subs({h: hbar, qin: qbar})
xp = h - hbar
up = qin - qbar
rhs = A_sym*xp + B_sym*up
h_taylor = qout.series(h, hbar,n=2).removeO()
lin_sys = (up - (h_taylor - qout.subs(h, hbar)))
hdot_jac = sp.Eq(sp.symbols(names='xdot'), rhs).subs({xp: x, up: u})  
hdot_taylor = sp.Eq(sp.symbols(names='xdot'),lin_sys).subs({xp: x, up: u}) 
y = sp.Eq(sp.symbols(names='y'), sp.symbols(names='x'))  # since C = 1, D = 0

print("\nLinearized system (xdot = A x + B u):")
print("x= {}, u= {}".format(xp, up))
sp.pprint(sp.Eq(sp.symbols(names='A'), A_sym))
sp.pprint(sp.Eq(sp.symbols(names='B'), B_sym))
sp.pprint(hdot_jac)
sp.pprint(hdot_taylor)
print("\nOutput equation (y = C x + D u):")
sp.pprint(y)

# -------------------------
# Numerical parameters
# -------------------------
Kv0=2.0
cv0=1.0
o0=0.7
alpha0 = 20
hbar0 = 1.0

# To linearize about hbar0, choose the consistent steady inflow
qbar0 = sp.lambdify((hbar, alpha), qbar_sol)(hbar0, alpha0)
print(f"\nChosen steady state: hbar = {hbar0}, qbar = {qbar0}")

# Evaluate A, B
a = sp.N(A_sym.subs({alpha: alpha0, hbar: hbar0, Kv: Kv0, cv: cv0, o: o0}),4)/A0
b = sp.N(B_sym.subs({alpha: alpha0, hbar: hbar0}),4)/A0
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
tspan = np.linspace(1, 100, 1000)
dt = tspan[1]-tspan[0]
omega = 0.05
A_amp = 0.2
# Step in inflow around steady state
du = 1
usin = A_amp * np.sin(omega*tspan)  

# Unit step response
qin_step = qbar0 + du  # step up from steady state inflow
qin_sin = qbar0 + usin     # sinusoidal variation around steady state inflow
h_nl = hbar0
hspan = []

for i, _ in enumerate(tspan):
    hdot_nl = f_lamda(h_nl,qin_step, alpha0, cv0, Kv0, o0)/A0
    h_nl += hdot_nl * dt
    hspan.append(h_nl)
hspan = np.array(hspan)
x_nl = hspan - hbar0      # deviation from steady state

# -------------------------
# Linear step response
# -------------------------
# Since system uses deviation input u, apply a step of size du
t_lin, y_lin = ct.forced_response(sys, tspan,U=du)

# -------------------------
# Plot comparison
# -------------------------
plt.figure()
plt.plot(tspan, x_nl, label='Nonlinear deviation: x = h - hbar')
plt.plot(t_lin, y_lin, '--', label='Linearized deviation')
plt.xlabel(xlabel='Time [s]')
plt.ylabel('Height deviation [m]')
plt.title(label='Tank level: nonlinear vs linearized response')
plt.grid(True)
plt.legend()
plt.show()