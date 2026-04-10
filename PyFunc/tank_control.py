from re import M
from numpy.random import laplace
import scipy
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import sympy as sp
from sympy import Function, dsolve, diff, checkodesol
from sympy.integrals.transforms import laplace_transform, inverse_laplace_transform, 
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
ode_lin = linearize(ode, [h, q_in], [hbar, 0])
ode_lhs = ode.lhs
ode_rhs = ode.rhs
sp.pprint(ode)
f = sp.lambdify((h, q_in, A, cv),ode_rhs, modules='numpy')
cv = 0.05 
A = 0.5
h0 = 1
a =- cv/(2*A*np.sqrt(h0))
b= 1/A
c=1
d=0
lt_rhs = sp.laplace_transform(ode_rhs,t, s, noconds=True, simplify=True)
lt_lhs = sp.laplace_transform(ode_lhs, t, s, noconds=True, simplify=True)
lt = sp.Eq(lt_lhs, lt_rhs)
laplace_correspondence(lt, {h: H, q_in: Q_in})
sp.pprint(lt)
inv_lt = sp.inverse_laplace_transform(lt, s, t, )
sp.pprint(inv_lt)
sys = ct.ss(a, b, c, d, inputs='q_in', outputs='h')
plt.plot(solve, solve.y[0], 'b', label='Height h(t)')
plt.xlabel('Time [s]')
plt.ylabel('Height h [m]')
plt.title(label='Tank Level Response to Initial Condition')
plt.grid(True)
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