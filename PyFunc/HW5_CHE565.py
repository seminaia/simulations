"""
HW5_CHE565.py
=============
CHE 565 – Homework 5
Cascade Control and Disturbance Rejection
Results are written to HW5_CHE565.txt, HW5_CHE565.tex, and HW5_CHE565.pdf
"""

from math import tau

import numpy as np
import matplotlib
import scipy as sc
from scipy.optimize import minimize
from scipy.signal import step
import sympy as sp
import matplotlib.pyplot as plt
from doc_builder import DocumentBuilder
import control as ct
import control.optimal as ct_opt
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
pade_order = 1
tauI1 = 1.0
tauI2 = 1.0
xdot_eq = sp.Eq(xdot,A*x + B*u)
y_eq = sp.Eq(y, C*x + D*u)
sp.pprint(xdot_eq)
sp.pprint(y_eq)
# Time vector
tvals = np.linspace(0, 100, 100)
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
    GD1= ct.tf(numD, denD, name='GD1', inputs='Yp1', outputs='YD')
    GD2= ct.tf(numD, denD, name='GD2', inputs='Yp2', outputs='Y')
    # State Space Representation of the blocks for interconnection
    # Note: the control library's interconnect function works better with state-space models, so we convert the transfer functions to state-space form.
    # xdot = Ax + Bu
    # y = Cx + Du
    
    Gc1_blk = ct.ss(Gc1, name='Gc1', inputs='E1', outputs='Yc1')
    Gc2_blk = ct.ss(Gc2, name='Gc2', inputs='E2', outputs='Yc2')
    Gp1_blk = ct.ss(Gp1, name='Gp1', inputs='Yc2', outputs='Yp1')
    Gp2_blk = ct.ss(Gp2, name='Gp2', inputs='P', outputs='Y')
    Gd_blk = ct.ss(Gd, name='Gd', inputs='D', outputs='Yd')     # direct disturbance addition
    #GD1_blk = ct.ss(GD1, name='GD1', inputs='Yp1', outputs='YD')
    #GD2_blk = ct.ss(GD2, name='GD2', inputs='Yp2', outputs='Y')
    sum1 = ct.summing_junction(inputs=['Ysp', '-Y'], output='E1', name='Sum1')
    if cascade:
         sum2 = ct.summing_junction(inputs=['Yc1', '-P'], output='E2', name='Sum2')
    else:
        sum2 = ct.summing_junction(inputs=['Yc1'], output='E2', name='Sum2')
    sum3 = ct.summing_junction(inputs=['Yp1', 'Yd'], output='P', name='Sum3')
    control_blks = [Gc1_blk, Gc2_blk]
    plant_blks = [Gp1_blk, Gp2_blk]
    disturbance_blks = [Gd_blk]
    #delay_blks = [GD1_blk, GD2_blk]
    sum_blks = [sum1, sum2, sum3]
    blocks = control_blks + plant_blks + disturbance_blks + sum_blks
    sys = ct.interconnect(
        blocks,
        input=['Ysp', 'D'],
        output=['Y'],
        input_prefix = ["Ysp", "D"],
        output_prefix = ["Y"],
    )
    return sys

    
def calculate_IAE(t, y, ysp):
    error = ysp - y
    iae = np.trapezoid(np.abs(error), t)
    return iae
def simulate_case(sys, t, ysp_input, d_input):
    U = np.vstack([ysp_input, d_input])
    resp = ct.forced_response(sys, T=t, U=U, squeeze=True,return_states=True)
    return resp.time, resp.outputs, resp.inputs, resp.states
def lambda_tuning(taup, thetad, Kp, PI=True, PID=False):
    # Lambda tuning rules for FOPDT processes
    # Lambda choice rules: 
    # Self-Regulating Process: lam > 0.8*thetad & lambda > 0.1*taup: lambda =thetad
    # Integrating Process: thetad < lambda < taup: lambda = max(taup/3,thetad)
    if not PI and not PID:
        raise ValueError("Must specify either PI or PID control.")
    if PI and PID:
        raise ValueError("Cannot specify both PI and PID control.")
    
    if taup < 4*thetad:
        # Self-Regulating process FOPDT
        lam = thetad
        tauI = max(thetad/4, taup)
        P = taup / (Kp * (lam + thetad))
        if PI:
            tauD = 0
        elif PID:
            tauD = min(taup/4, thetad/2)
        else:
            raise ValueError("Must specify either PI or PID control.")
    else:
        # Integrating process FOPDT
        lam = max(taup/3, thetad)
        tauI = max(4*thetad,2*lam+thetad)
        Kprime = Kp/taup
        P = taup / (Kprime * (lam + thetad)**2)
        if PI:
            tauD = 0
        elif PID:
            tauD = min(taup/4, thetad/2)
        else:
            raise ValueError("Must specify either PI or PID control.")
    return P, tauI, tauD, lam
def tuning_lqr(sys, Q, R):
    # Convert to state-space if not already
    if not isinstance(sys, ct.StateSpace):
        sys = ct.ss(sys)
    
    # Get A, B, C, D matrices
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    
    # Solve the continuous-time algebraic Riccati equation
    K, S, E = ct.lqr(A, B, Q, R,)
    
    return K, S, E

def optimize_PI(sys, t, ysp_input, d_input, initial_guess):
    def objective(params):
        Kc, tauI = params
        tuned_sys = build_closed_loop(Kc, Kc2, tauI, tauI2)
        t_sim, y_sim, _, _ = simulate_case(tuned_sys, t, ysp_input, d_input)
        iae = calculate_IAE(t_sim, y_sim, ysp_input)
        return iae
    result = minimize(objective,initial_guess, )
    return result.x  # returns the optimal Kc and tauI
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
print(sys1)
A1, B1, C1, D1 = sys1.A, sys1.B, sys1.C, sys1.D
n = A1.shape[0]

t1, y1, u1, x1 = simulate_case(sys1, tvals, step_off, step_on)
iae1 = calculate_IAE(t1, y1, step_off)
save_plot(
    "closed_loop_no_cascade.png",
    t1,
    y1,
    "Closed-loop response to unit step disturbance with no cascade control and no autotuning",
    ysp=step_off,
    d=step_on,
)

Gc_x1 = x1[0, :]
Gp1_x1 = x1[1, :]
Gp2_x1 = x1[2, :]
A1_sp = sp.Matrix(A1)
B1_sp = sp.Matrix(B1)
C1_sp = sp.Matrix(C1)
D1_sp = sp.Matrix(D1)

# =============================================================================
# Simulate Response to Step Disturbance with No Cascade Control and No Setpoint Change using Simulink autotuned values
# =============================================================================
P1 = 0.183711730708738
I1 = 0.0816496580927727
sys2 = build_closed_loop(P1, Kc2, 1 / I1, tauI2)
t2, y2, u2, x2 = simulate_case(sys2, tvals, step_off, step_on)
iae2 = calculate_IAE(t2, y2, step_off)
save_plot(
    "closed_loop_no_cascade_autotuning.png",
    t2,
    y2,
    "Closed-loop response to unit step disturbance with PI autotuning",
    ysp=step_off,
    d=step_on,
)

P2, tauI_2, D2, lam2 = lambda_tuning(taup1, theta, Kp1)
I2 = 1.0 / tauI_2
sys3 = build_closed_loop(P2, Kc2, tauI_2, tauI2)
t3, y3, u3, x3 = simulate_case(sys3, tvals, step_off, step_on)
iae3 = calculate_IAE(t3, y3, step_off)
save_plot(
    "closed_loop_no_cascade_lambda_tuning.png",
    t3,
    y3,
    "Closed-loop response to unit step disturbance with lambda tuning",
    ysp=step_off,
    d=step_on,
)

[P_opt1, tauI_opt1] = optimize_PI(sys1, tvals, step_off, step_on, initial_guess=[Kc1, tauI1])
sys_opt = build_closed_loop(P_opt1, Kc2, tauI_opt1, tauI2)
t_opt1, y_opt1, u_opt1, x_opt1 = simulate_case(sys_opt, tvals, step_off, step_on)
I_opt1 = 1.0 / tauI_opt1
iae_opt1 = calculate_IAE(t_opt1, y_opt1, step_off)
save_plot(
    "closed_loop_no_cascade_optimized.png",
    t_opt1,
    y_opt1,
    "Closed-loop response to unit step disturbance with optimized PI parameters",
    ysp=step_off,
    d=step_on,
)
print(f"Optimized Kc: {P_opt1:.4f}, Optimized I: {I_opt1:.4f}, IAE: {iae_opt1:.4f}")
sys4 = build_closed_loop(Kc1, Kc2, tauI1, tauI2)
t4, y4, u4, x4 = simulate_case(sys4, tvals, step_on, step_off)
iae4 = calculate_IAE(t4, y4, step_on)
save_plot(
    "closed_loop_no_cascade_setpoint.png",
    t4,
    y4,
    "Closed-loop response to no step disturbance with no cascade control and setpoint change",
    ysp=step_on,
    d=step_off,
)
P3 = P1
I3 = I1
sys5 = build_closed_loop(P3, Kc2, 1 / I3, tauI2)
t5, y5, u5, x5 = simulate_case(sys5, tvals, step_on, step_off)
iae5 = calculate_IAE(t5, y5, step_on)
save_plot(
    "closed_loop_no_cascade_setpoint_autotuning.png",
    t5,
    y5,
    "Closed-loop response to no step disturbance with no cascade control and setpoint change with autotuning",
    ysp=step_on,
    d=step_off,
)
P4, tauI_4, D4, lam4 = lambda_tuning(taup1, theta, Kp1)
I4 = 1.0 / tauI_4
sys6 = build_closed_loop(P4, Kc2, tauI_4, tauI2)
t6, y6, u6, x6 = simulate_case(sys6, tvals, step_on, step_off)
iae6 = calculate_IAE(t6, y6, step_on)
save_plot(
    "closed_loop_no_cascade_setpoint_lambda_tuning.png",
    t6,
    y6,
    "Closed-loop response to no step disturbance with no cascade control and setpoint change with lambda tuning",
    ysp=step_on,
    d=step_off,
)
print(f"Setpoint change with lambda tuning: P = {P4:.4f}, I = {I4:.4f}, D = {D4:.4f}, lambda = {lam4:.4f}, IAE = {iae6:.4f}")
[P_opt2, tauI_opt2] = optimize_PI(sys1, tvals, step_on, step_off, initial_guess=[Kc1, tauI1])
sys_opt = build_closed_loop(P_opt2, Kc2, tauI_opt2, tauI2)
t_opt2, y_opt2, u_opt2, x_opt2 = simulate_case(sys_opt, tvals, step_on, step_off)
I_opt2 = 1.0 / tauI_opt2
iae_opt2 = calculate_IAE(t_opt2, y_opt2, step_on)
save_plot(
    "closed_loop_no_cascade_optimized_setpoint.png",
    t_opt2,
    y_opt2,
    "Closed-loop response to setpoint change with optimized PI parameters",
    ysp=step_on,
    d=step_off,
)
print(f"Optimized Kc: {P_opt2:.4f}, Optimized I: {I_opt2:.4f}, IAE: {iae_opt2:.4f}")
images = [
    [("closed_loop_no_cascade_simulink.png","Simulink"), ("closed_loop_no_cascade.png", "Python")],
    [("closed_loop_no_cascade_autotuning_simulink.png", "Simulink"), ("closed_loop_no_cascade_autotuning.png", "Python")],
    [("closed_loop_no_cascade_lambda_tuning_simulink.png", "Simulink"), ("closed_loop_no_cascade_lambda_tuning.png", "Python")],
    [("closed_loop_no_cascade_optimized_simulink.png", "Simulink"), ("closed_loop_no_cascade_optimized.png", "Python")],
    [("closed_loop_no_cascade_setpoint_simulink.png", "Simulink"), ("closed_loop_no_cascade_setpoint.png", "Python")],
    [("closed_loop_no_cascade_setpoint_autotuning_simulink.png", "Simulink"), ("closed_loop_no_cascade_setpoint_autotuning.png", "Python")],
    [("closed_loop_no_cascade_setpoint_lambda_tuning_simulink.png", "Simulink"), ("closed_loop_no_cascade_setpoint_lambda_tuning.png", "Python")],
]
# =============================================================================
# Document writeup
# =============================================================================
doc.section("Introduction")
px(
f" The following homework was done with Simulink and the Control Systems Library in Python. The block diagram of the system is shown in ", doc.figref("fig:block_diagram"),
". ",
"The block diagram was created in simulink and then I built the same diagram as a python function using the control library. Simulink was mainly used as a sanity check for the response of the system in the python code. ")
p(
"Note: sum2 in the block diagram is the summing junction that takes the first controller output and subtract the P stream in order to form the inner loop. But, first in order to simulate without the inner loop, so I set cascade=False. This just converts the controller output (Yc1) to the error signal (E2) for the second controller. The disturbance D is added directly to the output of Gp1 (Yp1) " 
)
px(
"The transport delay transfer function is approximated using a Pade approximation of order ", 
pade_order,"." ," Although it should be relatively straightforward to simulate with delay in simulink and in python, I commented through the delay blocks in simulink and in python. The assignment didn't specifiy the delay time. Also, the control library has the delay function, but it only uses the Pade approximation.",)
line(
" Which gives the following transfer function approximation: ")
num_approx_sym = 1 - (sp.symbols('theta_d'))/2 * sp.symbols('s')
den_approx_sym = 1 + (sp.symbols('theta_d'))/2 * sp.symbols('s')
eq(
sp.latex(
            sp.Eq(
            sp.exp(-sp.Symbol("theta_d") *sp.symbols('s')),
            num_approx_sym / den_approx_sym
            )
         )
)

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
    GD1= ct.tf(numD, denD, name='GD1', inputs='Yp1', outputs='YD')
    GD2= ct.tf(numD, denD, name='GD2', inputs='Yp2', outputs='Y')
    # State Space Representation of the blocks for interconnection
    # Note: the control library's interconnect function works better with state-space models, so we convert the transfer functions to state-space form.
    # xdot = Ax + Bu
    # y = Cx + Du
    
    Gc1_blk = ct.ss(Gc1, name='Gc1', inputs='E1', outputs='Yc1')
    Gc2_blk = ct.ss(Gc2, name='Gc2', inputs='E2', outputs='Yc2')
    Gp1_blk = ct.ss(Gp1, name='Gp1', inputs='Yc2', outputs='Yp1')
    Gp2_blk = ct.ss(Gp2, name='Gp2', inputs='P', outputs='Y')
    Gd_blk = ct.ss(Gd, name='Gd', inputs='D', outputs='Yd')     # direct disturbance addition
    #GD1_blk = ct.ss(GD1, name='GD1', inputs='Yp1', outputs='YD')
    #GD2_blk = ct.ss(GD2, name='GD2', inputs='Yp2', outputs='Y')
    sum1 = ct.summing_junction(inputs=['Ysp', '-Y'], output='E1', name='Sum1')
    if cascade:
         sum2 = ct.summing_junction(inputs=['Yc1', '-P'], output='E2', name='Sum2')
    else:
        sum2 = ct.summing_junction(inputs=['Yc1'], output='E2', name='Sum2')
    sum3 = ct.summing_junction(inputs=['Yp1', 'Yd'], output='P', name='Sum3')
    control_blks = [Gc1_blk, Gc2_blk]
    plant_blks = [Gp1_blk, Gp2_blk]
    disturbance_blks = [Gd_blk]
    #delay_blks = [GD1_blk, GD2_blk]
    sum_blks = [sum1, sum2, sum3]
    blocks = control_blks + plant_blks + disturbance_blks + sum_blks
    sys = ct.interconnect(
        blocks,
        input=['Ysp', 'D'],
        output=['Y'],
        input_prefix = ["Ysp", "D"],
        output_prefix = ["Y"],
    )
    return sys

    
def calculate_IAE(t, y, ysp):
    error = ysp - y
    iae = np.trapezoid(np.abs(error), t)
    return iae
def simulate_case(sys, t, ysp_input, d_input):
    U = np.vstack([ysp_input, d_input])
    resp = ct.forced_response(sys, T=t, U=U, squeeze=True,return_states=True)
    return resp.time, resp.outputs, resp.inputs, resp.states
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

px("The Simulink block diagram is shown in ", doc.figref("fig:block_diagram"), ".")

subfiglog(
    images[0],
    caption="Closed-loop response to no step change in setpoint and a unit step change in disturbance with no cascade control and no autotuning.",
    label="fig:no_cascade_step_disturbance",
    width=r"0.45\textwidth",
)

px(
    f"Case 1: Step disturbance with no setpoint change gave and step in disturbance with no setpoint change gave ",
    f"IAE = {iae1:.4f} in Python and IAE = 453752.63065 in Simulink.",
     "The closed-loop disturbance response and no cascade loop is shown in ",
    doc.figref("fig:no_cascade_step_disturbance"),
    "."
)

line(
    f"The PI controller was then tuned using the autotuning feature in Simulink, "
)

subfiglog(
    images[1],
    caption="Closed-loop response to no step change in setpoint and a unit step change in disturbance with no cascade control and PI autotuning.",
    label="fig:no_cascade_autotuning_step_disturbance",
    width=r"0.45\textwidth",
)
px(
    f"Case 2: Step disturbance with no setpoint change and PI autotuning gave IAE = {iae2:.4f} in Python and IAE = 13.0463 in Simulink. The response is shown in ",
    doc.figref("fig:no_cascade_autotuning_step_disturbance"),
    ".",im(r"\ "), f"Proportional gain P = {P1:.4f} and integral gain I = {I1:.4f}.",
)

subfiglog(
    images[2],
    caption="Closed-loop response to no step change in setpoint and a unit step change in disturbance with no cascade control and lambda tuning.",
    label="fig:no_cascade_lambda_tuning_step_disturbance",
    width=r"0.45\textwidth",
)

px(
    f"Case 3: No step change in setpoint and step disturbance change with",im(r"\ \lambda \ ") ,f" tuning gave IAE = {iae3:.4f} in Python and IAE = 8.5578 in Simulink. The disturbance response is shown in ",
    doc.figref(label="fig:no_cascade_lambda_tuning_step_disturbance"),
    ".",im(r"\ "), f" Proportional gain P = {P2:.4f}, integral gain I = {I2:.4f}, derivative gain D = {D2:.4f}, and lambda = {lam2:.4f}."
)

subfiglog(
    images[3],
    caption="Closed-loop response to unit step disturbance with optimized PI parameters.",
    label="fig:no_cascade_optimized_step_disturbance",
    width=r"0.45\textwidth",
)

px(
    f"Case 4: Step disturbance with no setpoint change and optimized PI parameters gave IAE = {iae_opt1:.4f} in Python and IAE = 0.2192. The response is shown in ",
    doc.figref("fig:no_cascade_optimized_step_disturbance"), 
    ".",im(r"\ "),
    f"Optimized proportional gain P = {P_opt:.4f} and integral gain I = {I_opt:.4f}.",
    im(r"\ "), f"Note: The optimized parameters were found by minimizing the IAE to the disturbance"
)

subfiglog(
    images[4],
    caption="Closed-loop response to setpoint change with no step disturbance and no cascade control.",
    label="fig:no_cascade_setpoint_change",
    width=r"0.45\textwidth",
)
px(
    f"Case 5: No step change in disturbance and step change in setpoint with no cascade control gave IAE = {iae4:.4f} in Python and IAE = 1241005.8219 in Simulink. The response is shown in ",
    doc.figref("fig:no_cascade_setpoint_change"),
    ".",im(r"\ "),
)
subfiglog(
    images[5],
    caption="Closed-loop response to setpoint change with no step disturbance and no cascade control with PI autotuning.",
    label="fig:no_cascade_setpoint_change_autotuning",
    width=r"0.45\textwidth",
)
px(
    f"Case 6: No step change in disturbance and step change in setpoint with no cascade control and PI autotuning gave IAE = {iae5:.4f} in Python and IAE = 8.2136 in Simulink. The response is shown in ",
    doc.figref("fig:no_cascade_setpoint_change_autotuning"),
    ".",im(r"\ "),
    f"Proportional gain P = {P3:.4f} and integral gain I = {I3:.4f}.",
)
subfiglog(
    images[6],
    caption="Closed-loop response to setpoint change with no step disturbance and no cascade control with lambda tuning.",
    label="fig:no_cascade_setpoint_change_lambda_tuning",
    width=r"0.45\textwidth",
)
px(
    f"Case 7: No step change in disturbance and step change in setpoint with no cascade control and lambda tuning gave IAE = {iae6:.4f} in Python and IAE = 16.6484 in Simulink. The response is shown in ",
    doc.figref("fig:no_cascade_setpoint_change_lambda_tuning"),im(r"\ "),
    ".",im(r"\ "),
    f"Proportional gain P = {P4:.4f}, integral gain I = {I4:.4f}, derivative gain D = {D4:.4f}, and lambda = {lam4:.4f}.",
)
txt_file, tex_file, pdf_file = doc.save_all(runs=2)
print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX file: {tex_file}")
print(f"Wrote PDF report: {pdf_file}")