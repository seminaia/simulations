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
import scipy
import scipy.integrate
from sympy import true
from sympy.matrices.expressions.matadd import rules
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from doc_builder import DocumentBuilder
from pylatex_doc_builder import PyLatexDocumentBuilder
from pylatex import NoEscape
from pylatex import Math
import control as ct

OUTPUT_FILE = "HW5_CHE565"
PLOT_FILE = "HW5_CHE565_plot.png"
report_lines = []

doc = DocumentBuilder(
    OUTPUT_FILE,
    title="CHE 565 -- Homework 5",
    author="Soki Sem",
)
# convenience aliases
p = doc.p
line = doc.line
m = doc.eq
a = doc.align
t = doc.table
figlog = doc.figure
px = doc.px
im = doc.im
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

# Time vector
tvals = np.linspace(0, 50, 1000)
step_on = np.ones_like(tvals)
step_off = np.zeros_like(tvals)


def build_closed_loop(Kc1, Kc2, tauI1, tauI2):
    
    s = ct.tf('s')
    I1 = 1.0 / tauI1
    I2 = 0

    Gc1 = Kc1 * (1 + I1 / s)
    Gc2 = Kc2 * (1 + I2 / s)
    Gp1 = Kp1 / (taup1 * s + 1)
    Gp2 = Kp2 / (taup2 * s + 1)
    Gd = 1 * (1 + 0 * s)    # direct disturbance addition
    numD,denD = ct.delay.pade(theta, pade_order)    
    # Named blocks
    Gc1_blk = ct.tf(Gc1, name='Gc1', inputs='E1', outputs='Yc1')
    Gc2_blk = ct.tf(Gc2, name='Gc2', inputs='E2', outputs='Yc2')
    Gp1_blk = ct.tf(Gp1, name='Gp1', inputs='Yc2', outputs='Yp1')
    Gp2_blk = ct.tf(Gp2, name='Gp2', inputs='P', outputs='Yp2')
    Gd_blk = ct.tf(Gd, name='Gd', inputs='D', outputs='Yd')     # direct disturbance addition
    GD_blk = ct.tf(numD, denD, name='GD', inputs='Yp2', outputs='Y')
    sum1 = ct.summing_junction(inputs=['Ysp', '-Yp2'], output='E1', name='Sum1')
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

sys1 = build_closed_loop(Kc1, Kc2, tauI1, tauI2)
t1, y1 = simulate_case(sys1, tvals, step_off, step_on)
iae1 = calculate_IAE(t1, y1, step_off)
save_plot("closed_loop_no_cascade.png", t1, y1, "Closed-loop response to unit step in disturbance", ysp=step_off, d=step_on)


P = 0.183711730708738
I = 0.0816496580927727
# Run Python simulation using Simulink autotuned values
sys2 = build_closed_loop(P, Kc2, 1 / I, tauI2)
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
# =============================================================================
# Document writeup
# =============================================================================

doc.section("Problem 1")
doc.subsection("Closed-loop Response to Step Disturbance")

figlog(
    "HW5_CHE565_block_diagram.png",
    caption="Block diagram of the closed-loop system without cascade control and with a unit step disturbance from Simulink.",
    label="fig:block_diagram",
    width=NoEscape(r"0.8\linewidth"),
    height=NoEscape(r"0.5\textheight"),
    position="h"
)

px("The Simulink block diagram is shown in ", doc.figref("fig:block_diagram"), ".")

figlog(
    "closed_loop_no_cascade_simulink.png",
    caption="Closed-loop response to no step change in setpoint and a unit step change in disturbance from Simulink.",
    label="fig:no_cascade_simulink",
    width=NoEscape(r"0.8\textwidth"),
    height=NoEscape(r"0.8\textheight"),
)

px(
    "The Simulink closed-loop disturbance response is shown in ",
    doc.figref("fig:no_cascade_simulink"),
    "."
)

figlog(
    "closed_loop_no_cascade.png",
    caption="Closed-loop response to no step change in setpoint and a unit step change in disturbance from Python.",
    label="fig:no_cascade_python",
    width=NoEscape(r"0.8\textwidth"),
    height=NoEscape(r"0.8\textheight"),
)

px(
    "The corresponding Python response is shown in ",
    doc.figref("fig:no_cascade_python"),
    "."
)

p(
    f"Case 1: Step disturbance with no setpoint change gave "
    f"IAE = {iae1:.4f} in Python and IAE = 725.8109 in Simulink."
)


p(
    f"The PI controller was then tuned using the autotuning feature in Simulink, "
    f"which gave P = {P:.4f} and I = {I:.4f}."
)



figlog(
    "closed_loop_no_cascade_autotuning_python.png",
    caption="Closed-loop response to no step change in setpoint and a unit step disturbance using the Simulink autotuned PI values in Python.",
    width=NoEscape(r"0.8\textwidth"),
    height=NoEscape(r"0.8\textheight"),
    label="fig:autotuned_python",
)

px(
    "The Python response using the autotuned PI values is shown in ",
    doc.figref("fig:autotuned_python"),
    "."
)

figlog(
    "closed_loop_no_cascade_simulink_autotuning.png",
    caption="Closed-loop response to no step change in setpoint and a unit step disturbance using PI autotuning from Simulink.",
    label="fig:autotuned_simulink",
    width=NoEscape(r"0.8\textwidth"),
    height=NoEscape(r"0.8\textheight"),
)

px(
    "The corresponding Simulink response using PI autotuning is shown in ",
    doc.figref("fig:autotuned_simulink"),
    "."
)

p(
    f"Case 2: Step disturbance with no setpoint change and PI autotuning gave "
    f"IAE = {iae2:.4f} in Python and IAE = 13.0463 in Simulink."
)

txt_file, tex_file, pdf_file = doc.save_all(runs=2)
print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX file: {tex_file}")
print(f"Wrote PDF report: {pdf_file}")