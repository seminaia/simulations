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
from sympy import true
from sympy.matrices.expressions.matadd import rules
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from doc_builder import DocumentBuilder
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


# =============================================================================
# Build closed-loop system
# Disturbance is added after the controller:
# E = Ysp - Y
# Yc = Gc E
# P = Yc + d
# Y = delay * [Kp/(taup s + 1)] * P
# =============================================================================
def build_closed_loop(Kc1, Kc2, tauI1, tauI2):
    
    s = ct.tf('s')
    I1 = 1.0 / tauI1
    I2 = 0

    Gc1 = Kc1 * (1 + I1 / s)
    Gc2 = Kc2 * (1 + I2 / s)
    Gp1 = Kp1 / (taup1 * s + 1)
    Gp2 = Kp2 / (taup2 * s + 1)
    Gd = 1/s
    numD,denD = ct.delay.pade(theta, pade_order)    
    # Named blocks
    Gc1_blk = ct.tf(Gc1, name='Gc1', inputs='E1', outputs='Yc1')
    Gc2_blk = ct.tf(Gc2, name='Gc2', inputs='E2', outputs='Yc2')
    Gp1_blk = ct.tf(Gp1, name='Gp1', inputs='Yc2', outputs='Yp1')
    Gp2_blk = ct.tf(Gp2, name='Gp2', inputs='P', outputs='Yp2')
    Gd_blk = ct.tf(Gd, name='Gd', inputs='D', outputs='Yd')     # direct disturbance addition
    GD_blk = ct.tf(numD, denD, name='GD', inputs='Yp2', outputs='Y')
    print(Gd_blk)
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
    print(sys)
    return sys

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
save_plot("closed_loop_no_cascade.png", t1, y1, "Closed-loop response to step setpoint change", ysp=step_off, d=step_off)
doc.subsection("Closed-loop response to step disturbance")
figlog("closed_loop_no_cascade.png",
       caption="Closed-loop response to no step change in setpoint and a unit step change in disturbance",
       label="fig:fig1")

txt_file, tex_file, pdf_file = doc.save_all()
print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX file: {tex_file}")
print(f"Wrote PDF report: {pdf_file}")
