from math import e
from os import name
from re import A
import re
from urllib import response
from IPython.core import display_functions
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.integrals import inverse_laplace_transform
from sympy.abc import B, G, x,t,s
from scipy.constants import g
from tbcontrol import blocksim
from scipy import signal
sp.init_printing()
# xdot = Ax + Bu in deviation, where x = h - hbar, u = qin - qbar
# y = Cx + Du  # output is just the height deviation, so C = 1, D = 0
tau_i, tau_D, tau_p = sp.symbols('tau_i tau_D tau_p')
K_c,K_p = sp.symbols('K_c K_p')
s = ct.tf('s')
taui = 1
taup = 5
tauD = 1
Kc = 1
Kp = 1
I = 1/taui
Gp_block = blocksim.LTI('Gp(s)','P','Y',[Kp],[taup, 1],delay=tauD)
Gc_block = blocksim.PI('Gc(s)','E','Yc',Kc,taui)
blocks = [Gp_block, Gc_block]
sums = {'E':['+Ysp','-Y'],
        'P':['+Yc', '+Yd']}
inputs = {'Ysp':blocksim.step(),
          'Yd':blocksim.step()}
t_vals = np.arange(0, 50, 0.1)
diagram = blocksim.Diagram(blocks=blocks, sums=sums, inputs=inputs)
simulation = diagram.simulate(t_vals,True)

Ut = sp.Heaviside(t)
Ut_func = sp.lambdify(t, Ut, 'numpy')
Gsp = 1/s
Gd = 1/s
Gt = sp.Heaviside(t)
Gt_func = sp.lambdify(t, Gt, 'numpy')
GD_numer, GD_denom = ct.delay.pade(tauD,3)
Gp = Kp/(taup*s + 1)
Gc = Kc*(1+I/(s))
Ut_vals = Ut_func(t_vals)
Gt_vals = Gt_func(t_vals)
G_c_ct = ct.tf(Gc, name='Gc(s)', inputs='E', outputs='Yc')
G_p_ct = ct.tf(Gp, name='Gp(s)', inputs='P', outputs='Yp')
G_d_ct = ct.tf(Gd, name='Gd(s)', inputs='D', outputs='Yd')
G_D_ct = ct.tf(GD_numer, GD_denom, name='GD(s)', inputs='Yp', outputs='Y')
sum1 = ct.summing_junction(['Ysp','-Y'], ['E'], name='Sum1')
sum2 = ct.summing_junction(['Yc', 'Yd'], ['P'], name='Sum2')
system = ct.interconnect([G_c_ct, G_p_ct, G_d_ct, G_D_ct, sum1, sum2], inplist=['Ysp', 'Yd'], outlist=['Y'])
print("Closed-loop transfer function G(s):")
print(system)
print("Controller transfer function Gc(s):")
print(G_c_ct)
print("Plant transfer function Gp(s):")
print(G_p_ct)
print("Disturbance transfer function Gd(s):")
print(G_d_ct)
print("Delay Pade approximation")
print(GD_numer, GD_denom)
print("Delay G_D(s):")
print(G_D_ct)

results = simulation['Y']
print(f"Inputs : {Ut}, {Gt}")
response = ct.forced_response(sysdata=system, T=t_vals, U=[Ut_vals, Gt_vals])

plt.plot(t_vals, results, label=f'Block Diagram Simulation: Y', linestyle='-')
plt.plot(response.time, response.outputs[0], label='Step Response from Control Library', linestyle='--')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('g(t)')
plt.title('Block Diagram Response')
plt.grid()
plt.show()