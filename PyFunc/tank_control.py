from re import M
from numpy.random import laplace
import scipy
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import sympy as sp
from sympy import Function, dsolve, diff, checkodesol
from sympy.physics.mechanics import linearize
from sympy.abc import F, t, s
from IPython.display import display, Math
sp.init_printing()

h = Function('h')(t)
H = Function('H')(s)
q_in = Function('q_in')(t)
Q_in = Function('Q_in')(s)
A, cv = sp.symbols(names='A cv', positive=True)
hbar = sp.symbols('hbar', positive=True)  # Steady-state
q_out = cv * sp.sqrt(h)
ode = sp.Eq(diff(h, t), (q_in - q_out) / A)
sp.pprint(ode)
ode_rhs = sp.simplify(ode.rhs)
sp.pprint(ode_rhs)
f = sp.lambdify((h, q_in, A, cv),ode_rhs, modules='numpy')
h_lin = Function('h_lin')(t)
h_linbar = sp.symbols('h_linbar', positive=True)
A0= 2
cv = 0.05
h0 = 1
V0 = A0 * h0
qin = 1 
tspan = np.linspace(0, stop=100, 1000)
dt = tspan[1] - tspan[0]

hspan =[]
for t in tspan:
    hdot = f(h0, qin, A0, cv)
    h0 += hdot * dt
    hspan.append(h0)
plt.plot(tspan, hspan)
plt.xlabel('Time [s]')
plt.ylabel('Height h [m]')
plt.title('Tank Level Response to Initial Condition')
#sys = ct.ss(a, b, c, d, inputs='q_in', outputs='h')
plt.show()
# Step response (input magnitude = 1 by default)
response = ct.step_response(sysdata=sys, X0=[1])
t = response.time
x = response.states   # height deviation

# Compute theoretical DC gain

# Plot
plt.plot(t, x[0], 'b', label='Height deviation')
plt.xlabel('Time [s]')
plt.ylabel('Height deviation x = h - h0 [m]')
plt.title('Tank Level Step Response (Linearized Model)')
plt.legend()
plt.grid(True)
plt.show()