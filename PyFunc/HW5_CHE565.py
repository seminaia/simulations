"""
HW5_CHE565.py
=============
CHE 565 – Homework 5
Cascade Control and Disturbance Rejection
Results are written to HW5_CHE565.txt, HW5_CHE565.tex, and HW5_CHE565.pdf
"""

from math import tau

import os
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
tvals = np.linspace(0, 100, 1000)
step_on = np.ones_like(tvals)
step_off = np.zeros_like(tvals)


def build_closed_loop(Kc1, Kc2, tauI1, tauI2, cascade=False):
    
    s = ct.tf('s')
    t = sp.symbols('t', real=True)
    I1 = 1.0 / tauI1
    numD,denD = ct.pade(theta, pade_order)

    Gc1 = Kc1 * (1 + I1 / s)
    Gc2 = ct.tf([Kc2], [1])    # P-only controller as a transfer function
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
        inputs=['Ysp', 'D'],
        outputs=['Y'],
    )
    return sys

    
def calculate_IAE(t, y, ysp):
    y = np.ravel(np.asarray(y))
    ysp = np.asarray(ysp)
    if ysp.ndim == 0:
        ysp = np.ones_like(t) * float(ysp)
    error = ysp - y
    iae = np.trapezoid(np.abs(error), t)
    return float(iae)
def simulate_case(sys, t, ysp_input, d_input):
    U = np.vstack([ysp_input, d_input])
    resp = ct.forced_response(sys, T=t, U=U, squeeze=True,return_states=True)
    return resp.time, resp.outputs, resp.inputs, resp.states

def tuning_lqr(sys, Q, R):
    # Convert to state-space if not already
    if not isinstance(sys, ct.StateSpace):
        sys = ct.ss(sys)
    
    # Get A, B, C, D matrices
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    
    # Solve the continuous-time algebraic Riccati equation
    K, S, E = ct.lqr(A, B, Q, R,)
    
    return K, S, E



def lambda_tuning(taup, thetad, Kp, PI=True, PID=False):
    # Lambda tuning for a self-regulating FOPDT approximation.
    if not PI and not PID:
        raise ValueError("Must specify either PI or PID control.")
    if PI and PID:
        raise ValueError("Cannot specify both PI and PID control.")

    lam = max(taup / 3.0, thetad)
    Kc = taup / (Kp * (lam + thetad))
    tauI = taup

    if PI:
        tauD = 0.0
    else:
        tauD = min(taup / 4.0, thetad / 2.0)

    return Kc, tauI, tauD, lam


def equivalent_outer_process(cascade=False, Kc2=1.0):
    # Approximate equivalent process seen by the outer controller.
    if not cascade:
        K_eq = Kp1 * Kp2
        tau_eq = taup1 + taup2
    else:
        K_inner = (Kc2 * Kp1) / (1.0 + Kc2 * Kp1)
        tau_inner = taup1 / (1.0 + Kc2 * Kp1)
        K_eq = K_inner * Kp2
        tau_eq = tau_inner + taup2
    return K_eq, tau_eq


def optimize_PI(cascade, Kc2_value, t, ysp_input, d_input, initial_guess):
    # Optimize Kc and tauI by minimizing IAE. Optimization is last in each comparison.
    ysp_input = np.asarray(ysp_input)
    d_input = np.asarray(d_input)

    def objective(log_params):
        Kc_trial = float(np.exp(log_params[0]))
        tauI_trial = float(np.exp(log_params[1]))

        try:
            tuned_sys = build_closed_loop(
                Kc_trial,
                Kc2_value,
                tauI_trial,
                tauI2,
                cascade=cascade,
            )
            t_sim, y_sim, _, _ = simulate_case(tuned_sys, t, ysp_input, d_input)

            if np.any(~np.isfinite(y_sim)):
                return 1e12
            if np.max(np.abs(y_sim)) > 1e6:
                return 1e12

            return calculate_IAE(t_sim, y_sim, ysp_input)

        except Exception:
            return 1e12

    result = minimize(
        objective,
        np.log(np.asarray(initial_guess, dtype=float)),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )

    Kc_opt = float(np.exp(result.x[0]))
    tauI_opt = float(np.exp(result.x[1]))
    I_opt = 1.0 / tauI_opt
    iae_opt = float(result.fun)

    return Kc_opt, tauI_opt, I_opt, iae_opt


def rga(K):
    K = np.asarray(K, dtype=float)
    return K * np.linalg.inv(K).T


def matrix_rows_with_labels(M):
    rows = []
    M = np.asarray(M)
    for i in range(M.shape[0]):
        rows.append([f"y{i+1}"] + [float(M[i, j]) for j in range(M.shape[1])])
    return rows


def latex_matrix(M, fmt=".4f"):
    M = np.asarray(M)
    body = r" \\ ".join(
        " & ".join(format(float(v), fmt) for v in row)
        for row in M
    )
    return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"


def fopdt(K, taup, theta_delay, pade_order_local=1):
    s = ct.tf("s")
    numD, denD = ct.pade(theta_delay, pade_order_local)
    return K * ct.tf(numD, denD) / (taup * s + 1)


def build_mimo_process(include_cross_terms=True):
    G11 = fopdt(5, 4, 5, pade_order)
    G22 = fopdt(6, 10, 3, pade_order)

    if include_cross_terms:
        G12 = fopdt(2, 8, 4, pade_order)
        G21 = fopdt(3, 12, 3, pade_order)
    else:
        G12 = ct.tf([0], [1])
        G21 = ct.tf([0], [1])

    G11_blk = ct.ss(G11, name='G11', inputs='u1', outputs='y11')
    G12_blk = ct.ss(G12, name='G12', inputs='u2', outputs='y12')
    G21_blk = ct.ss(G21, name='G21', inputs='u1', outputs='y21')
    G22_blk = ct.ss(G22, name='G22', inputs='u2', outputs='y22')

    sum_y1 = ct.summing_junction(inputs=['y11', 'y12'], output='y1', name='SumY1')
    sum_y2 = ct.summing_junction(inputs=['y21', 'y22'], output='y2', name='SumY2')

    sys = ct.interconnect(
        [G11_blk, G12_blk, G21_blk, G22_blk, sum_y1, sum_y2],
        inputs=['u1', 'u2'],
        outputs=['y1', 'y2'],
    )

    return sys


def build_mimo_closed_loop(Kc_1, tauI_1, Kc_2, tauI_2, include_cross_terms=True, decoupler='none'):
    s = ct.tf("s")

    C1 = Kc_1 * (1 + 1 / (tauI_1 * s))
    C2 = Kc_2 * (1 + 1 / (tauI_2 * s))

    C1_blk = ct.ss(C1, name='C1', inputs='e1', outputs='v1')
    C2_blk = ct.ss(C2, name='C2', inputs='e2', outputs='v2')

    sum_e1 = ct.summing_junction(inputs=['r1', '-y1'], output='e1', name='SumE1')
    sum_e2 = ct.summing_junction(inputs=['r2', '-y2'], output='e2', name='SumE2')

    blocks = [C1_blk, C2_blk, sum_e1, sum_e2]

    if decoupler == 'none':
        blocks += [
            ct.ss(ct.tf([1], [1]), name='D11', inputs='v1', outputs='u1'),
            ct.ss(ct.tf([1], [1]), name='D22', inputs='v2', outputs='u2'),
        ]

    elif decoupler == 'static':
        blocks += [
            ct.ss(ct.tf([1], [1]), name='D11', inputs='v1', outputs='u11'),
            ct.ss(ct.tf([-2 / 5], [1]), name='D12', inputs='v2', outputs='u12'),
            ct.ss(ct.tf([-3 / 6], [1]), name='D21', inputs='v1', outputs='u21'),
            ct.ss(ct.tf([1], [1]), name='D22', inputs='v2', outputs='u22'),
            ct.summing_junction(inputs=['u11', 'u12'], output='u1', name='SumU1'),
            ct.summing_junction(inputs=['u21', 'u22'], output='u2', name='SumU2'),
        ]

    elif decoupler == 'dynamic':
        D12 = -(2 / 5) * (4 * s + 1) / (8 * s + 1)
        D21 = -(3 / 6) * (10 * s + 1) / (12 * s + 1)

        blocks += [
            ct.ss(ct.tf([1], [1]), name='D11', inputs='v1', outputs='u11'),
            ct.ss(D12, name='D12', inputs='v2', outputs='u12'),
            ct.ss(D21, name='D21', inputs='v1', outputs='u21'),
            ct.ss(ct.tf([1], [1]), name='D22', inputs='v2', outputs='u22'),
            ct.summing_junction(inputs=['u11', 'u12'], output='u1', name='SumU1'),
            ct.summing_junction(inputs=['u21', 'u22'], output='u2', name='SumU2'),
        ]

    else:
        raise ValueError("decoupler must be 'none', 'static', or 'dynamic'.")

    Pmimo = build_mimo_process(include_cross_terms=include_cross_terms)
    blocks.append(Pmimo)

    sys = ct.interconnect(
        blocks,
        inputs=['r1', 'r2'],
        outputs=['y1', 'y2'],
    )

    return sys


def simulate_mimo_case(sys, t, r1, r2):
    U = np.vstack([r1, r2])
    resp = ct.forced_response(sys, T=t, U=U, squeeze=True)
    return resp.time, np.asarray(resp.outputs)


def save_mimo_plot(filename, t, y, title):
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, y[0, :], label='y1', linewidth=2)
    plt.plot(t, y[1, :], label='y2', linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Outputs")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_plot(filename, t, y, title, ysp=None, d=None):
    y = np.ravel(np.asarray(y))
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
constraints =[ct_opt.input_range_constraint(sys1, [0,0],[0,1])]
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

#=============================================================================
# Simulate Response to step setpoint change with no cascade control and no disturbance change
# =============================================================================
t3, y3, u3, x3 = simulate_case(sys1, tvals, step_on, step_off)
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
t4, y4, u4, x4 = simulate_case(sys3, tvals, step_on, step_off)
iae4 = calculate_IAE(t4, y4, step_on)

save_plot(
    "closed_loop_no_cascade_setpoint_autotuning.png",
    t4,
    y4,
    "Closed-loop response to unit step setpoint change with PI autotuning",
    ysp=step_on,
    d=step_off,
)


# =============================================================================
# Additional Problem 1 Cases: Lambda Tuning and IAE Optimization
# =============================================================================

K_eq_no, tau_eq_no = equivalent_outer_process(cascade=False, Kc2=Kc2)
P_lambda_no, tauI_lambda_no, D_lambda_no, lam_lambda_no = lambda_tuning(tau_eq_no, theta, K_eq_no)
I_lambda_no = 1.0 / tauI_lambda_no

sys4 = build_closed_loop(P_lambda_no, Kc2, tauI_lambda_no, tauI2)
t5, y5, u5, x5 = simulate_case(sys4, tvals, step_off, step_on)
iae5 = calculate_IAE(t5, y5, step_off)
save_plot(
    "closed_loop_no_cascade_lambda_tuning.png",
    t5,
    y5,
    "Closed-loop response to unit step disturbance with lambda tuning",
    ysp=step_off,
    d=step_on,
)

P_opt1, tauI_opt1, I_opt1, iae_opt1 = optimize_PI(
    cascade=False,
    Kc2_value=Kc2,
    t=tvals,
    ysp_input=step_off,
    d_input=step_on,
    initial_guess=[P1, 1 / I1],
)
sys5 = build_closed_loop(P_opt1, Kc2, tauI_opt1, tauI2)
t6, y6, u6, x6 = simulate_case(sys5, tvals, step_off, step_on)
iae_opt1 = calculate_IAE(t6, y6, step_off)
save_plot(
    "closed_loop_no_cascade_optimized.png",
    t6,
    y6,
    "Closed-loop response to unit step disturbance with IAE-optimized PI",
    ysp=step_off,
    d=step_on,
)

sys6 = build_closed_loop(P_lambda_no, Kc2, tauI_lambda_no, tauI2)
t7, y7, u7, x7 = simulate_case(sys6, tvals, step_on, step_off)
iae7 = calculate_IAE(t7, y7, step_on)
save_plot(
    "closed_loop_no_cascade_setpoint_lambda_tuning.png",
    t7,
    y7,
    "Closed-loop response to unit step setpoint change with lambda tuning",
    ysp=step_on,
    d=step_off,
)

P_opt2, tauI_opt2, I_opt2, iae_opt2 = optimize_PI(
    cascade=False,
    Kc2_value=Kc2,
    t=tvals,
    ysp_input=step_on,
    d_input=step_off,
    initial_guess=[P1, 1 / I1],
)
sys7 = build_closed_loop(P_opt2, Kc2, tauI_opt2, tauI2)
t8, y8, u8, x8 = simulate_case(sys7, tvals, step_on, step_off)
iae_opt2 = calculate_IAE(t8, y8, step_on)
save_plot(
    "closed_loop_no_cascade_setpoint_optimized.png",
    t8,
    y8,
    "Closed-loop response to unit step setpoint change with IAE-optimized PI",
    ysp=step_on,
    d=step_off,
)


# =============================================================================
# Problem 2: Cascade Control
# =============================================================================

Kc2_cascade = 0.4

sys8 = build_closed_loop(Kc1, Kc2_cascade, tauI1, tauI2, cascade=True)
t9, y9, u9, x9 = simulate_case(sys8, tvals, step_off, step_on)
iae9 = calculate_IAE(t9, y9, step_off)
save_plot(
    "closed_loop_cascade.png",
    t9,
    y9,
    "Cascade response to unit step disturbance with no autotuning",
    ysp=step_off,
    d=step_on,
)

K_eq_cas, tau_eq_cas = equivalent_outer_process(cascade=True, Kc2=Kc2_cascade)
P_lambda_cas, tauI_lambda_cas, D_lambda_cas, lam_lambda_cas = lambda_tuning(tau_eq_cas, theta, K_eq_cas)
I_lambda_cas = 1.0 / tauI_lambda_cas

P_c_auto = P_lambda_cas
I_c_auto = I_lambda_cas

sys9 = build_closed_loop(P_c_auto, Kc2_cascade, 1 / I_c_auto, tauI2, cascade=True)
t10, y10, u10, x10 = simulate_case(sys9, tvals, step_off, step_on)
iae10 = calculate_IAE(t10, y10, step_off)
save_plot(
    "closed_loop_cascade_autotuning.png",
    t10,
    y10,
    "Cascade response to unit step disturbance with PI autotuning",
    ysp=step_off,
    d=step_on,
)

sys10 = build_closed_loop(P_lambda_cas, Kc2_cascade, tauI_lambda_cas, tauI2, cascade=True)
t11, y11, u11, x11 = simulate_case(sys10, tvals, step_off, step_on)
iae11 = calculate_IAE(t11, y11, step_off)
save_plot(
    "closed_loop_cascade_lambda_tuning.png",
    t11,
    y11,
    "Cascade response to unit step disturbance with lambda tuning",
    ysp=step_off,
    d=step_on,
)

P_opt3, tauI_opt3, I_opt3, iae_opt3 = optimize_PI(
    cascade=True,
    Kc2_value=Kc2_cascade,
    t=tvals,
    ysp_input=step_off,
    d_input=step_on,
    initial_guess=[P_c_auto, 1 / I_c_auto],
)
sys11 = build_closed_loop(P_opt3, Kc2_cascade, tauI_opt3, tauI2, cascade=True)
t12, y12, u12, x12 = simulate_case(sys11, tvals, step_off, step_on)
iae_opt3 = calculate_IAE(t12, y12, step_off)
save_plot(
    "closed_loop_cascade_optimized.png",
    t12,
    y12,
    "Cascade response to unit step disturbance with IAE-optimized PI",
    ysp=step_off,
    d=step_on,
)

t13, y13, u13, x13 = simulate_case(sys8, tvals, step_on, step_off)
iae13 = calculate_IAE(t13, y13, step_on)
save_plot(
    "closed_loop_cascade_setpoint.png",
    t13,
    y13,
    "Cascade response to unit step setpoint change with no autotuning",
    ysp=step_on,
    d=step_off,
)

sys12 = build_closed_loop(P_c_auto, Kc2_cascade, 1 / I_c_auto, tauI2, cascade=True)
t14, y14, u14, x14 = simulate_case(sys12, tvals, step_on, step_off)
iae14 = calculate_IAE(t14, y14, step_on)
save_plot(
    "closed_loop_cascade_setpoint_autotuning.png",
    t14,
    y14,
    "Cascade response to unit step setpoint change with PI autotuning",
    ysp=step_on,
    d=step_off,
)

sys13 = build_closed_loop(P_lambda_cas, Kc2_cascade, tauI_lambda_cas, tauI2, cascade=True)
t15, y15, u15, x15 = simulate_case(sys13, tvals, step_on, step_off)
iae15 = calculate_IAE(t15, y15, step_on)
save_plot(
    "closed_loop_cascade_setpoint_lambda_tuning.png",
    t15,
    y15,
    "Cascade response to unit step setpoint change with lambda tuning",
    ysp=step_on,
    d=step_off,
)

P_opt4, tauI_opt4, I_opt4, iae_opt4 = optimize_PI(
    cascade=True,
    Kc2_value=Kc2_cascade,
    t=tvals,
    ysp_input=step_on,
    d_input=step_off,
    initial_guess=[P_c_auto, 1 / I_c_auto],
)
sys14 = build_closed_loop(P_opt4, Kc2_cascade, tauI_opt4, tauI2, cascade=True)
t16, y16, u16, x16 = simulate_case(sys14, tvals, step_on, step_off)
iae_opt4 = calculate_IAE(t16, y16, step_on)
save_plot(
    "closed_loop_cascade_setpoint_optimized.png",
    t16,
    y16,
    "Cascade response to unit step setpoint change with IAE-optimized PI",
    ysp=step_on,
    d=step_off,
)


# =============================================================================
# Problems 4 and 5: RGA
# =============================================================================

K_problem4 = np.array(
    [
        [0.43, 0.43, 0.23, 0.22],
        [-0.33, 0.32, -0.20, 0.20],
        [0.22, 0.23, 0.42, 0.41],
        [-0.22, 0.22, -0.32, 0.32],
    ]
)
RGA4 = rga(K_problem4)

problem4_pairings = [
    ["y1", "u2", RGA4[0, 1]],
    ["y2", "u1", RGA4[1, 0]],
    ["y3", "u3", RGA4[2, 2]],
    ["y4", "u4", RGA4[3, 3]],
]

K_problem5 = np.array(
    [
        [5.0, 2.0],
        [3.0, 6.0],
    ]
)
RGA5 = rga(K_problem5)
problem5_pairings = [
    ["y1", "u1", RGA5[0, 0]],
    ["y2", "u2", RGA5[1, 1]],
]


# =============================================================================
# Problems 6--10: MIMO control and decoupling
# =============================================================================

Kc11, tauI11, D11_tmp, lam11 = lambda_tuning(4.0, 5.0, 5.0)
Kc22, tauI22, D22_tmp, lam22 = lambda_tuning(10.0, 3.0, 6.0)

sys_mimo_no_cross = build_mimo_closed_loop(Kc11, tauI11, Kc22, tauI22, include_cross_terms=False, decoupler='none')
tm1, ym1 = simulate_mimo_case(sys_mimo_no_cross, tvals, step_on, step_off)
save_mimo_plot("mimo_no_cross_r1_step.png", tm1, ym1, "Problem 6: no cross terms, step in r1")
tm2, ym2 = simulate_mimo_case(sys_mimo_no_cross, tvals, step_off, step_on)
save_mimo_plot("mimo_no_cross_r2_step.png", tm2, ym2, "Problem 6: no cross terms, step in r2")

sys_mimo_cross = build_mimo_closed_loop(Kc11, tauI11, Kc22, tauI22, include_cross_terms=True, decoupler='none')
tm3, ym3 = simulate_mimo_case(sys_mimo_cross, tvals, step_on, step_off)
save_mimo_plot("mimo_cross_r1_step.png", tm3, ym3, "Problem 7: cross terms, step in r1")
tm4, ym4 = simulate_mimo_case(sys_mimo_cross, tvals, step_off, step_on)
save_mimo_plot("mimo_cross_r2_step.png", tm4, ym4, "Problem 7: cross terms, step in r2")

sys_mimo_static = build_mimo_closed_loop(Kc11, tauI11, Kc22, tauI22, include_cross_terms=True, decoupler='static')
tm5, ym5 = simulate_mimo_case(sys_mimo_static, tvals, step_on, step_off)
save_mimo_plot("mimo_static_r1_step.png", tm5, ym5, "Problem 8: static decoupler, step in r1")
tm6, ym6 = simulate_mimo_case(sys_mimo_static, tvals, step_off, step_on)
save_mimo_plot("mimo_static_r2_step.png", tm6, ym6, "Problem 8: static decoupler, step in r2")

sys_mimo_dynamic = build_mimo_closed_loop(Kc11, tauI11, Kc22, tauI22, include_cross_terms=True, decoupler='dynamic')
tm7, ym7 = simulate_mimo_case(sys_mimo_dynamic, tvals, step_on, step_off)
save_mimo_plot("mimo_dynamic_r1_step.png", tm7, ym7, "Problem 9: dynamic decoupler, step in r1")
tm8, ym8 = simulate_mimo_case(sys_mimo_dynamic, tvals, step_off, step_on)
save_mimo_plot("mimo_dynamic_r2_step.png", tm8, ym8, "Problem 9: dynamic decoupler, step in r2")



images = [
    [("closed_loop_no_cascade_simulink.png","Simulink"), ("closed_loop_no_cascade.png", "Python")],
    [("closed_loop_no_cascade_simulink_autotuning.png", "Simulink"), ("closed_loop_no_cascade_autotuning.png", "Python")],
    [("closed_loop_no_cascade_setpoint_simulink.png", "Simulink"), ("closed_loop_no_cascade_setpoint.png", "Python")],
    [("closed_loop_no_cascade_setpoint_simulink_autotuning.png", "Simulink"), ("closed_loop_no_cascade_setpoint_autotuning.png", "Python")],
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
    numD,denD = ct.pade(theta, pade_order)

    Gc1 = Kc1 * (1 + I1 / s)
    Gc2 = ct.tf([Kc2], [1])    # P-only controller as a transfer function
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
        inputs=['Ysp', 'D'],
        outputs=['Y'],
    )
    return sys

    
def calculate_IAE(t, y, ysp):
    y = np.ravel(np.asarray(y))
    ysp = np.asarray(ysp)
    if ysp.ndim == 0:
        ysp = np.ones_like(t) * float(ysp)
    error = ysp - y
    iae = np.trapezoid(np.abs(error), t)
    return float(iae)
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
    f"IAE = {iae1:.4f} in Python and IAE = 725.8109 in Simulink.",
     "The closed-loop disturbance response and no cascade loop is shown in ",
    doc.figref("fig:no_cascade_step_disturbance"),
    "."
)

#figlog(
#    "closed_loop_no_cascade_optimized.png",
#    caption="Closed-loop response to unit step disturbance with optimized PI parameters.",
#    label="fig:no_cascade_step_disturbance_optimized",
#    width=r"0.8\textwidth",
#)
# px(
    # "The closed-loop disturbance response with optimized PI parameters is shown in ",
    # doc.figref("fig:no_cascade_step_disturbance_optimized"),
    # ".",
    # f" The optimization reduced the IAE from {iae1:.4f} to  in Python. ",
# )



p(
    f"The PI controller was then tuned using the autotuning feature in Simulink, "
    f"which gave P = {P1:.4f} and I = {I1:.4f}."
)


subfiglog(
    images[1],
    caption="Closed-loop response to no step change in setpoint and a unit step change in disturbance with autotuning.",
    label="fig:no_cascade_autotuning_step_disturbance",
    width=r"0.45\textwidth",
    )


<<<<<<< Updated upstream
px(
    f"Case 2: Step disturbance with no setpoint change and PI autotuning gave ",
    f"IAE = {iae2:.4f} in Python and IAE = 13.0463 in Simulink.",
    "The response to a unit step change in disturbance with no cascade control and PI autotuning is shown in ",
    doc.figref("fig:no_cascade_autotuning_step_disturbance"),
    "."
)
=======
def run_problems_1_to_3():
    tvals = np.linspace(0, 100, 1001)
    step_on = np.ones_like(tvals)
    step_off = np.zeros_like(tvals)

    doc.section("Problems 1--3: Cascade Control")

    # Problem 1: no cascade, inner gain 1
    doc.subsection("Problem 1: System Without Cascade Control")
    Kc2_no = 1.0
    P_no = 1.0
    tauI_no = 1.0
    I_no = 1 / tauI_no

    sys_no = build_closed_loop(P_no, Kc2_no, tauI_no, cascade=False)

    t1, y1, _, _ = simulate_case(sys_no, tvals, step_off, step_on)
    iae1 = calculate_IAE(t1, y1, step_off)
    f1 = PLOT_DIR / "P1_no_cascade_disturbance.png"
    save_plot(f1, t1, y1, "No cascade: unit step disturbance", ysp=step_off, d=step_on)

    t2, y2, _, _ = simulate_case(sys_no, tvals, step_on, step_off)
    iae2 = calculate_IAE(t2, y2, step_on)
    f2 = PLOT_DIR / "P1_no_cascade_setpoint.png"
    save_plot(f2, t2, y2, "No cascade: unit step setpoint change", ysp=step_on, d=step_off)

    px(
        "For the no-cascade case, the inner feedback path was disconnected and the P-only inner controller gain was set to ",
        im(r"K_{c,2}=1"), ". The outer PI controller was then tuned by minimizing the IAE."
    )
    eq(rf"K_{{c,1}} = {P_no:.4f}, \qquad \tau_{{I,1}} = {tauI_no:.4f}, \qquad I_1 = {I_no:.4f}")
    table(
        headers=["Case", NoEscape(r"$K_{c,1}$"), NoEscape(r"$\tau_{I,1}$"), "IAE"],
        rows=[
            ["Disturbance", P_no, tauI_no, iae1],
            ["Setpoint", P_no, tauI_no, iae2],
        ],
        caption="Problem 1 results without cascade control.",
        label="tab:problem1_no_cascade",
    )
    subfiglog(
        [(str(f1), "Disturbance response"), (str(f2), "Setpoint response")],
        caption="Problem 1 responses without cascade control.",
        label="fig:problem1_no_cascade",
        width=r"0.48\textwidth",
    )

    # Problem 2: cascade, inner gain 0.4
    doc.subsection("Problem 2: System With Cascade Control")
    Kc2_cas = 0.4
    P_cas = 1.0
    tauI_cas = 1.0
    I_cas = 1 / tauI_cas

    sys_cas = build_closed_loop(P_cas, Kc2_cas, tauI_cas, cascade=True)

    t3, y3, _, _ = simulate_case(sys_cas, tvals, step_off, step_on)
    iae3 = calculate_IAE(t3, y3, step_off)
    f3 = PLOT_DIR / "P2_cascade_disturbance.png"
    save_plot(f3, t3, y3, "Cascade: unit step disturbance", ysp=step_off, d=step_on)

    t4, y4, _, _ = simulate_case(sys_cas, tvals, step_on, step_off)
    iae4 = calculate_IAE(t4, y4, step_on)
    f4 = PLOT_DIR / "P2_cascade_setpoint.png"
    save_plot(f4, t4, y4, "Cascade: unit step setpoint change", ysp=step_on, d=step_off)

    px(
        "For the cascade case, the inner feedback path was connected and the inner P-only controller gain was set to ",
        im(r"K_{c,2}=0.4"), ". The outer controller was tuned again because the outer loop sees a different equivalent process."
    )
    eq(rf"K_{{c,1}} = {P_cas:.4f}, \qquad \tau_{{I,1}} = {tauI_cas:.4f}, \qquad I_1 = {I_cas:.4f}")
    table(
        headers=["Case", NoEscape(r"$K_{c,2}$"), NoEscape(r"$K_{c,1}$"), NoEscape(r"$\tau_{I,1}$"), "IAE"],
        rows=[
            ["Disturbance", Kc2_cas, P_cas, tauI_cas, iae3],
            ["Setpoint", Kc2_cas, P_cas, tauI_cas, iae4],
        ],
        caption="Problem 2 results with cascade control.",
        label="tab:problem2_cascade",
    )
    subfiglog(
        [(str(f3), "Disturbance response"), (str(f4), "Setpoint response")],
        caption="Problem 2 responses with cascade control.",
        label="fig:problem2_cascade",
        width=r"0.48\textwidth",
    )

    # Problem 3 comments
    doc.subsection("Problem 3: Comparison of Cascade and No-Cascade Results")
    table(
        headers=["Test", "No Cascade IAE", "Cascade IAE", "Change"],
        rows=[
            ["Disturbance", iae1, iae3, iae3 - iae1],
            ["Setpoint", iae2, iae4, iae4 - iae2],
        ],
        caption="Comparison of no-cascade and cascade-control performance.",
        label="tab:problem3_comparison",
    )
    p(
        "Cascade control is expected to make the largest difference for disturbance rejection. "
        "The disturbance enters at the intermediate process variable, so the inner loop can react before the effect fully propagates to the final output. "
        "For setpoint tracking, cascade control may still change the response, but the benefit is usually less direct because the setpoint must pass through both the outer and inner loops."
    )
>>>>>>> Stashed changes



doc.subsection("Closed-loop Response to Step Setpoint Change")

subfiglog(
    images[2],
    caption="Closed-loop response to step change in setpoint and no step disturbance.",
    width=r"0.45\textwidth",
    label="fig:no_cascade_step_setpoint",
)

px( 
    f"Case 3: Step setpoint change with no disturbance change gave ",
    f"IAE = {iae3:.4f} in Python and IAE = 2450.8205 in Simulink.",
    "The response is shown in ",
    doc.figref("fig:no_cascade_step_setpoint"),
    "."
)


subfiglog(
    images[3],
    caption="Closed-loop response to step change in setpoint and no step disturbance with PI autotuning.",
    label="fig:no_cascade_autotuning_step_setpoint",
    width=r"0.45\textwidth",
)


px(
    f"Case 4: Step setpoint change with no disturbance change and PI autotuning gave IAE = {iae4:.4f} in Python and IAE = 8.1939 in Simulink. The response is shown in ",
    doc.figref("fig:no_cascade_autotuning_step_setpoint"),
    ". ",
)
   


# =============================================================================
# Additional Document Writeup for Problems 2--10
# =============================================================================

doc.subsection("Additional No-Cascade Tuning Comparisons")

subfiglog(
    [("closed_loop_no_cascade_lambda_tuning.png", "Lambda tuning"),
     ("closed_loop_no_cascade_optimized.png", "IAE optimized")],
    caption="No-cascade disturbance response with lambda tuning and IAE optimization.",
    label="fig:no_cascade_lambda_optimized_disturbance",
    width=r"0.45\textwidth",
)

px(
    "Case 5: Step disturbance with lambda tuning gave ",
    f"IAE = {iae5:.4f}. ",
    "Case 6: Step disturbance with IAE optimization gave ",
    f"IAE = {iae_opt1:.4f}. ",
    f"The optimized proportional gain is P = {P_opt1:.4f} and the optimized integral gain is I = {I_opt1:.4f}."
)

subfiglog(
    [("closed_loop_no_cascade_setpoint_lambda_tuning.png", "Lambda tuning"),
     ("closed_loop_no_cascade_setpoint_optimized.png", "IAE optimized")],
    caption="No-cascade setpoint response with lambda tuning and IAE optimization.",
    label="fig:no_cascade_lambda_optimized_setpoint",
    width=r"0.45\textwidth",
)

px(
    "Case 7: Step setpoint change with lambda tuning gave ",
    f"IAE = {iae7:.4f}. ",
    "Case 8: Step setpoint change with IAE optimization gave ",
    f"IAE = {iae_opt2:.4f}. ",
    f"The optimized proportional gain is P = {P_opt2:.4f} and the optimized integral gain is I = {I_opt2:.4f}."
)

table(
    headers=["Case", r"$P$", r"$I$", r"$\tau_I$", "IAE"],
    rows=[
        ["Default disturbance", Kc1, 1.0 / tauI1, tauI1, iae1],
        ["Autotune disturbance", P1, I1, 1.0 / I1, iae2],
        ["Lambda disturbance", P_lambda_no, I_lambda_no, tauI_lambda_no, iae5],
        ["IAE optimized disturbance", P_opt1, I_opt1, tauI_opt1, iae_opt1],
        ["Default setpoint", Kc1, 1.0 / tauI1, tauI1, iae3],
        ["Autotune setpoint", P2, I2, 1.0 / I2, iae4],
        ["Lambda setpoint", P_lambda_no, I_lambda_no, tauI_lambda_no, iae7],
        ["IAE optimized setpoint", P_opt2, I_opt2, tauI_opt2, iae_opt2],
    ],
    caption="Problem 1 tuning summary without cascade control.",
    label="tab:no_cascade_summary",
    alignment="lcccc",
)


doc.section("Problem 2")
doc.subsection("Closed-loop Response with Cascade Control")

p(
    "For the cascade-control case, the inner feedback loop was reconnected and the inner P-only controller gain was set to 0.4. Since this changes the effective process seen by the outer loop, the outer PI controller was retuned."
)

p(
    "The cascade autotune row currently uses the lambda-tuned value as a placeholder unless the Simulink cascade autotuned values are entered manually."
)

subfiglog(
    [("closed_loop_cascade.png", "Default"),
     ("closed_loop_cascade_autotuning.png", "Autotune")],
    caption="Cascade disturbance response with default and autotuned PI controllers.",
    label="fig:cascade_default_autotune_disturbance",
    width=r"0.45\textwidth",
)

subfiglog(
    [("closed_loop_cascade_lambda_tuning.png", "Lambda tuning"),
     ("closed_loop_cascade_optimized.png", "IAE optimized")],
    caption="Cascade disturbance response with lambda tuning and IAE optimization.",
    label="fig:cascade_lambda_optimized_disturbance",
    width=r"0.45\textwidth",
)

table(
    headers=["Case", r"$K_{c2}$", r"$P$", r"$I$", r"$\tau_I$", "IAE"],
    rows=[
        ["Default", Kc2_cascade, Kc1, 1.0 / tauI1, tauI1, iae9],
        ["Autotune", Kc2_cascade, P_c_auto, I_c_auto, 1.0 / I_c_auto, iae10],
        ["Lambda", Kc2_cascade, P_lambda_cas, I_lambda_cas, tauI_lambda_cas, iae11],
        ["IAE optimized", Kc2_cascade, P_opt3, I_opt3, tauI_opt3, iae_opt3],
    ],
    caption="Cascade disturbance-response controller comparison.",
    label="tab:cascade_disturbance_summary",
    alignment="lccccc",
)

doc.subsection("Closed-loop Response to Step Setpoint Change with Cascade Control")

subfiglog(
    [("closed_loop_cascade_setpoint.png", "Default"),
     ("closed_loop_cascade_setpoint_autotuning.png", "Autotune")],
    caption="Cascade setpoint response with default and autotuned PI controllers.",
    label="fig:cascade_default_autotune_setpoint",
    width=r"0.45\textwidth",
)

subfiglog(
    [("closed_loop_cascade_setpoint_lambda_tuning.png", "Lambda tuning"),
     ("closed_loop_cascade_setpoint_optimized.png", "IAE optimized")],
    caption="Cascade setpoint response with lambda tuning and IAE optimization.",
    label="fig:cascade_lambda_optimized_setpoint",
    width=r"0.45\textwidth",
)

table(
    headers=["Case", r"$K_{c2}$", r"$P$", r"$I$", r"$\tau_I$", "IAE"],
    rows=[
        ["Default", Kc2_cascade, Kc1, 1.0 / tauI1, tauI1, iae13],
        ["Autotune", Kc2_cascade, P_c_auto, I_c_auto, 1.0 / I_c_auto, iae14],
        ["Lambda", Kc2_cascade, P_lambda_cas, I_lambda_cas, tauI_lambda_cas, iae15],
        ["IAE optimized", Kc2_cascade, P_opt4, I_opt4, tauI_opt4, iae_opt4],
    ],
    caption="Cascade setpoint-response controller comparison.",
    label="tab:cascade_setpoint_summary",
    alignment="lccccc",
)


doc.section("Problem 3")
doc.subsection("Comment on Cascade Results")

table(
    headers=["Test", "No cascade autotune IAE", "Cascade autotune IAE", "No cascade optimized IAE", "Cascade optimized IAE"],
    rows=[
        ["Disturbance", iae2, iae10, iae_opt1, iae_opt3],
        ["Setpoint", iae4, iae14, iae_opt2, iae_opt4],
    ],
    caption="Comparison of no-cascade and cascade-control results.",
    label="tab:cascade_comparison",
    alignment="lcccc",
)

p(
    "Cascade control is expected to make the largest difference for the disturbance response because the disturbance enters between the two process blocks. The inner loop can react to the intermediate variable before the disturbance fully propagates to the final output. For setpoint changes, the improvement is usually smaller because the setpoint enters through the outer loop rather than directly inside the process."
)


doc.section("Problem 4")
doc.subsection("Relative Gain Array for the 4 by 4 Process")

eq(r"\Lambda = K \circ \left(K^{-1}\right)^T")
eq(r"K = " + latex_matrix(K_problem4))
eq(r"\Lambda = " + latex_matrix(RGA4))

table(
    headers=[r"$y_i$", r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"],
    rows=matrix_rows_with_labels(RGA4),
    caption="RGA for the 4 by 4 gain matrix.",
    label="tab:rga4",
    alignment="lcccc",
)

table(
    headers=["Output", "Manipulated variable", r"$\lambda_{ij}$"],
    rows=problem4_pairings,
    caption="Suggested pairings for the 4 by 4 process.",
    label="tab:rga4_pairings",
    alignment="ccc",
)

p(
    "The pairings were selected using positive RGA elements that are close to one while avoiding repeated inputs and outputs. Negative RGA elements were avoided because they indicate unfavorable control-loop interaction."
)


doc.section("Problem 5")
doc.subsection("Relative Gain Array for the 2 by 2 Process")

eq(r"K = \begin{bmatrix} 5 & 2 \\ 3 & 6 \end{bmatrix}")
eq(r"\Lambda = " + latex_matrix(RGA5))

table(
    headers=[r"$y_i$", r"$u_1$", r"$u_2$"],
    rows=matrix_rows_with_labels(RGA5),
    caption="RGA for the 2 by 2 gain matrix.",
    label="tab:rga5",
    alignment="lcc",
)

table(
    headers=["Output", "Manipulated variable", r"$\lambda_{ij}$"],
    rows=problem5_pairings,
    caption="Suggested pairings for the 2 by 2 process.",
    label="tab:rga5_pairings",
    alignment="ccc",
)

p(
    "The diagonal pairing is preferred because the diagonal RGA values are positive while the off-diagonal values are negative. Therefore, y1 should be paired with u1 and y2 should be paired with u2."
)


doc.section("Problem 6")
doc.subsection("Two Single-Loop Controllers with Cross Terms Neglected")

a(
    r"G_{11}(s) = \frac{5e^{-5s}}{4s+1}",
    r"G_{22}(s) = \frac{6e^{-3s}}{10s+1}",
)

table(
    headers=["Loop", r"$K_c$", r"$\tau_I$", r"$\lambda$"],
    rows=[
        ["G11", Kc11, tauI11, lam11],
        ["G22", Kc22, tauI22, lam22],
    ],
    caption="PI tuning values for the two diagonal loops.",
    label="tab:mimo_pi_tuning",
    alignment="lccc",
)

subfiglog(
    [("mimo_no_cross_r1_step.png", "Step in r1"),
     ("mimo_no_cross_r2_step.png", "Step in r2")],
    caption="Problem 6 responses when cross terms are neglected.",
    label="fig:mimo_no_cross",
    width=r"0.45\textwidth",
)

p(
    "When the cross terms are neglected, the two loops behave independently. A change in r1 affects y1, and a change in r2 affects y2."
)


doc.section("Problem 7")
doc.subsection("Adding Cross Terms")

a(
    r"G_{12}(s) = \frac{2e^{-4s}}{8s+1}",
    r"G_{21}(s) = \frac{3e^{-3s}}{12s+1}",
)

subfiglog(
    [("mimo_cross_r1_step.png", "Step in r1"),
     ("mimo_cross_r2_step.png", "Step in r2")],
    caption="Problem 7 responses after adding cross terms.",
    label="fig:mimo_cross",
    width=r"0.45\textwidth",
)

p(
    "Adding the cross terms causes interaction between the two loops. A step change in one input affects both output variables."
)


doc.section("Problem 8")
doc.subsection("Static Decouplers")

a(
    r"u_1 = v_1 - \frac{K_{12}}{K_{11}}v_2 = v_1 - 0.4v_2",
    r"u_2 = v_2 - \frac{K_{21}}{K_{22}}v_1 = v_2 - 0.5v_1",
)

subfiglog(
    [("mimo_static_r1_step.png", "Step in r1"),
     ("mimo_static_r2_step.png", "Step in r2")],
    caption="Problem 8 responses using static decouplers.",
    label="fig:mimo_static",
    width=r"0.45\textwidth",
)

p(
    "The static decoupler reduces steady-state interaction because it is based on the steady-state gain matrix."
)


doc.section("Problem 9")
doc.subsection("Dynamic Decouplers")

a(
    r"D_{12}(s) = -\frac{G_{12}}{G_{11}} = -\frac{2}{5}\frac{4s+1}{8s+1}e^{s}",
    r"D_{21}(s) = -\frac{G_{21}}{G_{22}} = -\frac{3}{6}\frac{10s+1}{12s+1}",
)

p(
    "The term e^{s} in D12 is noncausal because it corresponds to a negative delay. Therefore, the realizable dynamic decoupler uses the rational part only."
)

eq(r"D_{12,realizable}(s) = -\frac{2}{5}\frac{4s+1}{8s+1}")

subfiglog(
    [("mimo_dynamic_r1_step.png", "Step in r1"),
     ("mimo_dynamic_r2_step.png", "Step in r2")],
    caption="Problem 9 responses using dynamic decouplers.",
    label="fig:mimo_dynamic",
    width=r"0.45\textwidth",
)


doc.section("Problem 10")
doc.subsection("Comments on MIMO Results")

p(
    "When the cross terms are neglected, the loops appear independent. Once the cross terms are added, a step in one loop causes motion in the other output, which shows loop interaction. Static decouplers reduce steady-state interaction, but they do not fully remove transient interaction because the transfer functions have different time constants and delays. Dynamic decouplers include more of the process dynamics, so they should reduce interaction more effectively. However, any noncausal positive exponential term must be removed or approximated before implementation."
)



txt_file, tex_file, pdf_file = doc.save_all(runs=2)
print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX file: {tex_file}")
print(f"Wrote PDF report: {pdf_file}")