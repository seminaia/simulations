"""
HW5_CHE565.py
=============
CHE 565 – Homework 5
Cascade Control and Disturbance Rejection
Results are written to HW5_CHE565.txt, HW5_CHE565.tex, and HW5_CHE565.pdf
"""

from math import tau
from re import S


import numpy as np
import matplotlib
import scipy as sc
from scipy.optimize import minimize
from scipy.signal import step
import sympy as sp
matplotlib.use(backend="Agg")
import matplotlib.pyplot as plt
from doc_builder import DocumentBuilder
import control as ct

OUTPUT_FILE = "HW5_CHE565"
PLOT_FILE = "HW5_CHE565_plot.png"
report_lines = []
xdot = sp.MatrixSymbol('xdot', 6, 1)
x = sp.MatrixSymbol('x', 6, 1)
y = sp.MatrixSymbol('y', 1, 1)
u = sp.MatrixSymbol('u', 2, 1)
A = sp.MatrixSymbol('A', 6, 6)
B = sp.MatrixSymbol('B', 6, 2)
C = sp.MatrixSymbol('C', 1, 6)
D = sp.MatrixSymbol('D', 1, 2)

doc = DocumentBuilder(
    OUTPUT_FILE,
    title="CHE 565 -- Homework 5",
    author="Soki Sem",
)
# convenience aliases
p = doc.p
line = doc.line
eq = doc.eq
a = doc.align
table = doc.table
figlog = doc.figure
subfiglog = doc.subfigures
px = doc.px
im = doc.im
lst = doc.listings
doc.maketitle(True)
doc.toc(False)

# =============================================================================
# Problem data
# =============================================================================
Kp1 = 5.0
taup1 = 5.0
Kp2 = 2.0
taup2 = 10.0
Kc1 = 1.0
Kc2 = 1.0
theta = 1.0
pade_order = 3
tauI1 = 1.0
tauI2 = 1.0
# Lambda rules
lam1 = max(taup1 / 3.0, theta)
lam2 = max(taup2 / 3.0, theta)
Kc1_nom = taup1 / (Kp1 * (lam1 + theta))
Kc2_nom = taup2 / (Kp2 * (lam2 + theta))
tauI1_nom = taup1
tauI2_nom = taup2
I1_nom = 1.0 / tauI1_nom
I2_nom = 1.0 / tauI2_nom
xdot_eq = sp.Eq(xdot,A*x + B*u)
y_eq = sp.Eq(y, C*x + D*u)
sp.pprint(xdot_eq)
sp.pprint(y_eq)
# Time vector
tvals = np.linspace(0, 50, 1000)
step_on = np.ones_like(tvals)
step_off = np.zeros_like(tvals)


def build_closed_loop(Kc1, Kc2, tauI1, tauI2, cascade=False):
    
    s = ct.tf('s')
    t = sp.symbols('t', real=True)
    I1 = 1.0 / tauI1
    I2 = 0
    numD,denD = ct.delay.pade(theta, pade_order)

    Gc1 = Kc1 * (1 + I1 / s)
    Gc2 = Kc2 * (1 + I2 / s)
    Gp1 = Kp1 / (taup1 * s + 1)
    Gp2 = Kp2 / (taup2 * s + 1)
    Gd = ct.tf([1], [1])    # direct disturbance addition
    GD= ct.tf(numD, denD, name='GD', inputs='Yp2', outputs='Y')
    
    # State Space Representation of the blocks for interconnection
    # Note: the control library's interconnect function works better with state-space models, so we convert the transfer functions to state-space form.
    # xdot = Ax + Bu
    # y = Cx + Du
    
    Gc1_blk = ct.ss(Gc1, name='Gc1', inputs='E1', outputs='Yc1')
    Gc2_blk = ct.ss(Gc2, name='Gc2', inputs='E2', outputs='Yc2')
    Gp1_blk = ct.ss(Gp1, name='Gp1', inputs='Yc2', outputs='Yp1')
    Gp2_blk = ct.ss(Gp2, name='Gp2', inputs='P', outputs='Yp2')
    Gd_blk = ct.ss(Gd, name='Gd', inputs='D', outputs='Yd')     # direct disturbance addition
    GD_blk = ct.ss(GD, name='GD', inputs='Yp2', outputs='Y')
    sum1 = ct.summing_junction(inputs=['Ysp', '-Yp2'], output='E1', name='Sum1')
    if cascade:
         sum2 = ct.summing_junction(inputs=['Yc1', '-P'], output='E2', name='Sum2')
    else:
        sum2 = ct.summing_junction(inputs=['Yc1'], output='E2', name='Sum2')
    sum3 = ct.summing_junction(inputs=['Yp1', 'Yd'], output='P', name='Sum3')
    
    sys = ct.interconnect(
        [Gc1_blk, Gc2_blk, Gp1_blk, Gp2_blk, Gd_blk, GD_blk, sum1, sum2, sum3],
        input=['Ysp', 'D'],
        output=['Y'],
        input_prefix = ["Ysp", "D"],
        output_prefix = ["Y"],
    )
    sys.inputs = ['Ysp', 'D']
    sys.outputs = ['Y']
    
    return sys

def calculate_IAE(t, y, ysp):
    error = ysp - y
    iae = np.trapezoid(np.abs(error), t)
    return iae

def tuning_lqr(sys, Q, R):
    # Convert to state-space if not already
    if not isinstance(sys, ct.StateSpace):
        sys = ct.ss(sys)
    
    # Get A, B, C, D matrices
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    
    # Solve the continuous-time algebraic Riccati equation
    K, S, E = ct.lqr(A, B, Q, R,)
    
    return K, S, E
def simulate_case(sys, t, ysp_input, d_input):
    U = np.vstack([ysp_input, d_input])
    resp = ct.forced_response(sys, T=t, U=U, squeeze=True)
    return resp.time, resp.outputs


# ---------------------------
# Simulation helper
# ---------------------------



def save_plot(filename, t, y, title, ysp=None, d=None):
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, y, label='Output y_p2(t)', linewidth=2)
    if ysp is not None:
        plt.plot(t, ysp, '--', label='Setpoint', linewidth=1.5)

    if d is not None:
        plt.plot(t, d, ':', label='Disturbance', linewidth=1.5)

    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# =============================================================================
# Simulate Response to Step Disturbance with No Cascade Control and No Setpoint Change
# =============================================================================

sys1 = build_closed_loop(Kc1, Kc2, tauI1, tauI2)
A1, B1, C1, D1 = sys1.A, sys1.B, sys1.C, sys1.D

t1, y1 = simulate_case(sys1, tvals, step_off, step_on)
iae1 = calculate_IAE(t1, y1, step_off)
def objective_PI(params):
    P, I = params
    if P <= 0 or I <= 0:
        return np.inf  # Penalize non-positive parameters
    tauI = 1 / I
    sys = build_closed_loop(P, Kc2, tauI, tauI2)
    if not np.all(np.real(ct.poles(sys)) < 0):
        return np.inf
    
    t, y = simulate_case(sys, tvals, step_off, step_on)
    y = np.asarray(y).squeeze()
    t = np.asarray(t).squeeze()
    ysp = np.asarray(step_off).squeeze()
    
    e = ysp - y
    eint = np.trapezoid(np.abs(e), t)
    u_control = P*(e+I*eint)
    Q = 1
    R = 1
    cost = Q * e**2 + R * u_control**2
    J = np.trapezoid(cost, t)
    iae = np.trapezoid(np.abs(e), t)
    print(f"Evaluating P={P:.4f}, I={I:.4f}, IAE={iae:.4f}, Cost J={J:.4f}")
    return J
initial_guess = [Kc1_nom, I1_nom]
result = minimize(objective_PI, initial_guess)
P_opt, I_opt = result.x
sys_opt = build_closed_loop(P_opt, Kc2, 1 / I_opt, tauI2)
t_opt, y_opt = simulate_case(sys_opt, tvals, step_off, step_on)
iae_opt = calculate_IAE(t_opt, y_opt, step_off)
save_plot(
    "closed_loop_no_cascade_optimized.png",
    t_opt,
    y_opt,
    "Closed-loop response to unit step disturbance with optimized PI parameters",
    ysp=step_off,
    d=step_on,)

print(f"Optimal P: {P_opt:.4f}, Optimal I: {I_opt:.4f}")
A1_sp = sp.Matrix(A1)
B1_sp = sp.Matrix(B1)
C1_sp = sp.Matrix(C1)
D1_sp = sp.Matrix(D1)
n = A1.shape[0]
m = B1.shape[1]
Q = np.eye(n)
R = np.eye(m)
[K1, S1, E1] = tuning_lqr(sys1, Q, R)
K1_sp = sp.Matrix(K1)
S1_sp = sp.Matrix(S1)
E1_sp = sp.Matrix(E1)
print("LQR Gain K1:")
sp.pprint(K1_sp)
print("Solution to Riccati Equation S1:")
sp.pprint(S1_sp)
print("Closed-loop eigenvalues E1:")
sp.pprint(E1_sp)
save_plot("closed_loop_no_cascade.png",
          t1,
          y1,
          "Closed-loop response to unit step in disturbance", ysp=step_off, d=step_on)
# =============================================================================
# Simulate Response to Step Disturbance with No Cascade Control and No Setpoint Change using Simulink autotuned values
# =============================================================================
P1 = 0.183711730708738
I1 = 0.0816496580927727
sys2 = build_closed_loop(P1, Kc2, 1 / I1, tauI2)
t2, y2 = simulate_case(sys2, tvals, step_off, step_on)
iae2 = calculate_IAE(t2, y2, step_off)
save_plot(
    "closed_loop_no_cascade_autotuning_python.png",
    t2,
    y2,
    "Closed-loop response to unit step disturbance with PI autotuning",
    ysp=step_off,
    d=step_on,
)

#=============================================================================
# Simulate Response to step setpoint change with no cascade control and no disturbance change
# =============================================================================
t3, y3 = simulate_case(sys1, tvals, step_on, step_off)
iae3 = calculate_IAE(t3, y3, step_on)
save_plot(
    "closed_loop_no_cascade_setpoint.png",
    t3,
    y3,
    "Closed-loop response to unit step setpoint change",
    ysp=step_on,
    d=step_off,
)
P2 =P1
I2 = I1
sys3 = build_closed_loop(P2, Kc2, 1 / I2, tauI2)
t4, y4 = simulate_case(sys3, tvals, step_on, step_off)
iae4 = calculate_IAE(t4, y4, step_on)

save_plot(
    "closed_loop_no_cascade_setpoint_autotuning.png",
    t4,
    y4,
    "Closed-loop response to unit step setpoint change with PI autotuning",
    ysp=step_on,
    d=step_off,
)

images = [
    [("closed_loop_no_cascade_simulink.png","Simulink"), ("closed_loop_no_cascade.png", "Python")],
    [("closed_loop_no_cascade_simulink_autotuning.png", "Simulink"), ("closed_loop_no_cascade_autotuning.png", "Python")],
    [("closed_loop_no_cascade_setpoint_simulink.png", "Simulink"), ("closed_loop_no_cascade_setpoint.png", "Python")],
    [("closed_loop_no_cascade_setpoint_simulink_autotuning.png", "Simulink"), ("closed_loop_no_cascade_setpoint_autotuning.png", "Python")],
]
num_approx, den_approx = ct.delay.pade(theta, pade_order)
# =============================================================================
# Document writeup
# =============================================================================
doc.section("Introduction")
px(
f" The following homework was done with Simulink and the Control Systems Library in Python. The block diagram of the system is shown in ", doc.figref("fig:block_diagram"),
". ",
"The block diagram was created in simulink and then I built the same diagram as a python function using the control library. ")
p(
"Note: sum2 in the block diagram is the summing junction that takes the first controller output and subtract the P stream in order to form the inner loop. But, first in order to simulate without the inner loop, so I set cascade=False. This just converts the controller output (Yc1) to the error signal (E2) for the second controller. "
"The disturbance D is added directly to the output of Gp1 (Yp1) " 
)
px(
"The transport delay transfer function is approximated using a Pade approximation of order ", 
pade_order,im(r'\ '),
" and a delay of ",
im(r"\theta_d = "), 
theta)
line(
" which gives the following transfer function approximation: ")
eq(
rf"e^{{-\theta_d \cdot s}} \approx \frac{{{num_approx[0]}s^3 + {num_approx[1]}s^2 + {num_approx[2]}s + {num_approx[3]}}}{{{den_approx[0]}s^3 + {den_approx[1]}s^2 + {den_approx[2]}s + {den_approx[3]}}}")

lst(["""
import control as ct

def build_closed_loop(Kc1, Kc2, tauI1, tauI2, cascade=False):
    
    s = ct.tf('s')
    t = sp.symbols('t', real=True)
    I1 = 1.0 / tauI1
    I2 = 0
    numD,denD = ct.delay.pade(theta, pade_order)

    Gc1 = Kc1 * (1 + I1 / s)
    Gc2 = Kc2 * (1 + I2 / s)
    Gp1 = Kp1 / (taup1 * s + 1)
    Gp2 = Kp2 / (taup2 * s + 1)
    Gd = ct.tf([1], [1])    # direct disturbance addition
    GD= ct.tf(numD, denD, name='GD', inputs='Yp2', outputs='Y')
    # State Space Representation of the blocks for interconnection
    # Note: the control library's interconnect function works better with state-space models, so we convert the transfer functions to state-space form.
    # xdot = Ax + Bu
    # y = Cx + Du
    #
    Gc1_blk = ct.ss(Gc1, name='Gc1', inputs='E1', outputs='Yc1')
    Gc2_blk = ct.ss(Gc2, name='Gc2', inputs='E2', outputs='Yc2')
    Gp1_blk = ct.ss(Gp1, name='Gp1', inputs='Yc2', outputs='Yp1')
    Gp2_blk = ct.ss(Gp2, name='Gp2', inputs='P', outputs='Yp2')
    Gd_blk = ct.ss(Gd, name='Gd', inputs='D', outputs='Yd')     # direct disturbance addition
    GD_blk = ct.ss(GD, name='GD', inputs='Yp2', outputs='Y')
    sum1 = ct.summing_junction(inputs=['Ysp', '-Yp2'], output='E1', name='Sum1')
    if cascade:
         sum2 = ct.summing_junction(inputs=['Yc1', '-P'], output='E2', name='Sum2')
    else:
        sum2 = ct.summing_junction(inputs=['Yc1'], output='E2', name='Sum2')
    sum3 = ct.summing_junction(inputs=['Yp1', 'Yd'], output='P', name='Sum3')
    sys = ct.interconnect(
        [Gc1_blk, Gc2_blk, Gp1_blk, Gp2_blk, Gd_blk, sum1, sum2, sum3],
        input=['Ysp', 'D'],
        output=['Yp2'],
        input_prefix = ["Ysp", "D"],
        output_prefix = ["Yp2"],
    )
    return sys

def calculate_IAE(t, y, ysp):
    error = ysp - y
    iae = np.trapezoid(np.abs(error), t)
    return iae

# ---------------------------
# Simulation helper
# ---------------------------
def simulate_case(sys, t, ysp_input, d_input):
    U = np.vstack([ysp_input, d_input])
    resp = ct.forced_response(sys, T=t, U=U, squeeze=True)
    return resp.time, resp.outputs

"""])
doc.section("Problem 1")
doc.subsection("Closed-loop Response to Step Disturbance")

figlog(
    "HW5_CHE565_block_diagram.png",
    caption="Block diagram of the closed-loop system without cascade control and with a unit step disturbance from Simulink.",
    label="fig:block_diagram",
    width=r"0.8\textwidth",
    position="H",
)
figlog(
    "closed_loop_no_cascade_optimized.png",
    "Closed-loop response to unit step disturbance with optimized PI parameters.",
    label="fig:optimized_response",
    width=r"0.8\textwidth",
)
px("The Simulink block diagram is shown in ", doc.figref("fig:block_diagram"), ".")

p(
    f"Case 1: Step disturbance with no setpoint change gave and step in disturbance with no setpoint change gave "
    f"IAE = {iae1:.4f} in Python and IAE = 725.8109 in Simulink."
)

subfiglog(
    images[0],
    caption="Closed-loop response to no step change in setpoint and a unit step change in disturbance with no cascade control and no autotuning.",
    label="fig:no_cascade_step_disturbance",
    width=r"0.45\textwidth",
)

px(
    "The closed-loop disturbance response and no cascade loop is shown in ",
    doc.figref("fig:no_cascade_step_disturbance"),
    "."
)


p(
    f"The PI controller was then tuned using the autotuning feature in Simulink, "
    f"which gave P = {P1:.4f} and I = {I1:.4f}."
)
p(
    f"Case 2: Step disturbance with no setpoint change and PI autotuning gave "
    f"IAE = {iae2:.4f} in Python and IAE = 13.0463 in Simulink."
)

subfiglog(
    images[1],
    caption="Closed-loop response to no step change in setpoint and a unit step change in disturbance with autotuning.",
    label="fig:no_cascade_autotuning_step_disturbance",
    width=r"0.45\textwidth",
    )


px(
    "The response to a unit step change in disturbance with no cascade control and PI autotuning is shown in ",
    doc.figref("fig:no_cascade_autotuning_step_disturbance"),
    "."
)



doc.subsection("Closed-loop Response to Step Setpoint Change")

subfiglog(
    images[2],
    caption="Closed-loop response to step change in setpoint and no step disturbance.",
    width=r"0.45\textwidth",
    label="fig:no_cascade_step_setpoint",
)

px(
    "The response to the step change in setpoint with no step disturbance is shown in ",
    doc.figref("fig:no_cascade_step_setpoint"),
    "."
)
p(
    f"Case 3: Step setpoint change with no disturbance change gave "
    f"IAE = {iae3:.4f} in Python and IAE = 2450.8205 in Simulink."
)

subfiglog(
    images[3],
    caption="Closed-loop response to step change in setpoint and no step disturbance with PI autotuning.",
    label="fig:no_cascade_autotuning_step_setpoint",
    width=r"0.45\textwidth",
)


px(
    "The response to a unit step change in setpoint with no disturbance change using PI autotuning is shown in ",
    doc.figref("fig:no_cascade_autotuning_step_setpoint"),
    "."
)
p(
    f"Case 4: Step setpoint change with no disturbance change and PI autotuning gave "
    f"IAE = {iae4:.4f} in Python and IAE = 8.1939 in Simulink."
)

txt_file, tex_file, pdf_file = doc.save_all(runs=2)
print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX file: {tex_file}")
print(f"Wrote PDF report: {pdf_file}")