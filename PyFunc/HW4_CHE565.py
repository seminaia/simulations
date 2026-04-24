"""
HW4_CHE565.py
=============
CHE 565 – Homework 4
Closed-loop PI control for Gp(s) = exp(-s)/(5s+1), with disturbance added after the controller.
Results are written to HW4_CHE565.txt, HW4_CHE565.tex, and HW4_CHE565.pdf
"""

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

OUTPUT_FILE = "HW4_CHE565"
PLOT_FILE = "HW4_CHE565_plot.png"
report_lines = []

doc = DocumentBuilder(
    OUTPUT_FILE,
    title="CHE 565 -- Homework 4",
    author="Soki Sem",
)
# convenience aliases
w = doc.p
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
Kp = 1.0
taup = 5.0
theta = 1.0
pade_order = 3

# Lambda rules
lam = max(taup / 4.0, theta)
Kc_nom = taup / (Kp * (lam + theta))
tauI_nom = taup
I_nom = 1.0 / tauI_nom

# Variations
Kc_double = 2.0 * Kc_nom
Kc_half = 0.5 * Kc_nom
tauI_large = 2.0 * tauI_nom
tauI_small = 0.5 * tauI_nom

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
def build_closed_loop(Kc, tauI):
    s = ct.tf('s')
    I = 1.0 / tauI

    Gc = Kc * (1 + I / s)
    Gp_nodelay = Kp / (taup * s + 1)

    numD, denD = ct.pade(theta, pade_order)
    Gdelay = ct.tf(numD, denD)

    # Named blocks
    Gc_blk = ct.tf(Gc, name='Gc', inputs='E', outputs='Yc')
    Gp_blk = ct.tf(Gp_nodelay, name='Gp', inputs='P', outputs='Yp')
    Gd_blk = ct.ss([], [], [], [[1]], name='Gd', inputs='d', outputs='Yd')     # direct disturbance addition
    GD_blk = ct.tf(Gdelay, name='GD', inputs='Yp', outputs='Y')

    sum1 = ct.summing_junction(inputs=['Ysp', '-Y'], output='E', name='Sum1')
    sum2 = ct.summing_junction(inputs=['Yc', 'Yd'], output='P', name='Sum2')

    sys = ct.interconnect(
        [Gc_blk, Gp_blk, Gd_blk, GD_blk, sum1, sum2],
        inplist=['Ysp', 'd'],
        outlist=['Y']
    )
    return sys


# ---------------------------
# Simulation helper
# ---------------------------
def simulate_case(sys, t, ysp_input, d_input):
    U = np.vstack([ysp_input, d_input])
    resp = ct.forced_response(sys, T=t, U=U)
    y = np.squeeze(resp.outputs)
    return resp.time, y


def save_plot(filename, t, y, title, ysp=None, d=None):
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, y, label='Output y(t)', linewidth=2)

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
# Simulations
# =============================================================================
# Nominal
sys_nom = build_closed_loop(Kc_nom, tauI_nom)
t1, y1 = simulate_case(sys_nom, tvals, step_on, step_off)   # setpoint step
t2, y2 = simulate_case(sys_nom, tvals, step_off, step_on)   # disturbance step

# Double gain
sys_kc_double = build_closed_loop(Kc_double, tauI_nom)
t3, y3 = simulate_case(sys_kc_double, tvals, step_on, step_off)
t4, y4 = simulate_case(sys_kc_double, tvals, step_off, step_on)

# Half gain
sys_kc_half = build_closed_loop(Kc_half, tauI_nom)
t5, y5 = simulate_case(sys_kc_half, tvals, step_on, step_off)
t6, y6 = simulate_case(sys_kc_half, tvals, step_off, step_on)

# Larger tauI
sys_tauI_large = build_closed_loop(Kc_nom, tauI_large)
t7, y7 = simulate_case(sys_tauI_large, tvals, step_on, step_off)
t8, y8 = simulate_case(sys_tauI_large, tvals, step_off, d_input=step_on)
tauI_big = 2 * tauI_nom
tauI_small = 0.5 * tauI_nom

sys_tauI_big = build_closed_loop(Kc_nom, tauI_big)
sys_tauI_small = build_closed_loop(Kc_nom, tauI_small)
t9, y9 = simulate_case(sys_tauI_big, tvals, step_on, step_off)
t10, y10 = simulate_case(sys_tauI_small, tvals, step_off, step_on)

doc.subsection(NoEscape(r"Controller Design using  $\lambda$  -Rules"))

m(r"\lambda = \max\left(\frac{\tau_p}{3}, \theta \right)")

a(
    r"\lambda &= \max\left(\frac{5}{3}, 1\right)",
    r"&= \frac{5}{3} \approx 1.667"
)

m(r"K_c = \frac{\tau_p}{K_p(\lambda + \theta)}")

a(
    r"K_c &= \frac{5}{1(1.667 + 1)}",
    r"&= \frac{5}{2.667}",
    r"&\approx 1.875"
)

m(r"\tau_I = \tau_p")

a(
    r"\tau_I &= 5"
)

m(r"I = \frac{1}{\tau_I}")

a(
    r"I &= \frac{1}{5}",
    r"&= 0.2"
)
doc.subsection("Controller Transfer Function")

m(r"G_c(s) = K_c \left(1 + \frac{1}{\tau_I s} \right)")

a(
    r"G_c(s) &= 1.875 \left(1 + \frac{1}{5s} \right)",
    r"&= 1.875 + \frac{0.375}{s}"
)

doc.subsection("Closed-Loop System")

w("The control system is arranged in a standard feedback configuration.")

m(r"E = Y_{sp} - Y")

m(r"Y_c = G_c(s)E")

w("The disturbance is added after the controller output.")

m(r"P = Y_c + d")

m(r"Y = G_p(s) P")
doc.subsection("Simulation Cases")

w("The following simulations were performed:")

w("1. Step change in setpoint (magnitude 1), no disturbance")
w("2. Step change in disturbance (magnitude 1), no setpoint change")
w("3. Double controller gain")
w("4. Half controller gain")
w("5. Increased and decreased integral time constant")

doc.subsection("Disturbance Response")

w("A unit step disturbance was applied with no setpoint change.")

m(r"Y_{sp}(t) = 0, \quad d(t) = 1")

doc.subsection("Effect of Controller Gain")

w("The controller gain was varied to observe its effect on system response.")

m(r"K_c = 2K_c^{\text{nominal}}")

m(r"K_c = 0.5K_c^{\text{nominal}}")

# =============================================================================
# Save figures
# =============================================================================
save_plot(
    "part2_setpoint_nominal.png",
    t1, y1,
    "Part 2: Setpoint Step, No Disturbance (Nominal PI)",
    ysp=step_on, d=step_off
)

save_plot(
    "part3_disturbance_nominal.png",
    t2, y2,
    "Part 3: Disturbance Step, No Setpoint Change (Nominal PI)",
    ysp=step_off, d=step_on
)

doc.subsection("Effect of Integral Time Constant")
save_plot(
    "part4_setpoint_kc_double.png",
    t3, y3,
    "Part 4: Setpoint Step with Double Controller Gain",
    ysp=step_on, d=step_off
)

save_plot(
    "part4_disturbance_kc_double.png",
    t4, y4,
    "Part 4: Disturbance Step with Doubled Controller Gain",
    ysp=step_off, d=step_on
)

save_plot(
    "part4_setpoint_kc_half.png",
    t5, y5,
    "Part 4: Setpoint Step with Half Controller Gain",
    ysp=step_on, d=step_off
)

save_plot(
    "part4_disturbance_kc_half.png",
    t6, y6,
    "Part 4: Disturbance Step with Half Controller Gain",
    ysp=step_off, d=step_on
)

save_plot(
    "part5_setpoint_tauI_large.png",
    t7, y7,
    "Part 5: Setpoint Step with Increased Integral Time Constant",
    ysp=step_on, d=step_off
)

save_plot(
    "part5_disturbance_tauI_large.png",
    t8, y8,
    "Part 5: Disturbance Step with Increased Integral Time Constant",
    ysp=step_off, d=step_on
)

save_plot(
    "part5_setpoint_tauI_small.png",
    t9, y9,
    "Part 5: Setpoint Step with Decreased Integral Time Constant",
    ysp=step_on, d=step_off
)

save_plot(
    "part5_disturbance_tauI_small.png",
    t10, y10,
    "Part 5: Disturbance Step with Decreased Integral Time Constant",
    ysp=step_off, d=step_on
)

# Comparison plots
plt.figure(figsize=(8, 4.8))
plt.plot(t1, y1, label="Nominal")
plt.plot(t3, y3, label="Double $K_c$")
plt.plot(t5, y5, label="Half $K_c$")
plt.plot(tvals, step_on, '--', label="Setpoint")
plt.xlabel("Time")
plt.ylabel("Response")
plt.title("Comparison of Setpoint Responses for Different Controller Gains")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compare_kc_setpoint.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4.8))
plt.plot(t2, y2, label="Nominal")
plt.plot(t4, y4, label="Double $K_c$")
plt.plot(t6, y6, label="Half $K_c$")
plt.xlabel("Time")
plt.ylabel("Response")
plt.title("Comparison of Disturbance Responses for Different Controller Gains")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compare_kc_disturbance.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4.8))
plt.plot(t1, y1, label="Nominal")
plt.plot(t7, y7, label="Increased $\\tau_I$")
plt.plot(t9, y9, label="Decreased $\\tau_I$")
plt.plot(tvals, step_on, '--', label="Setpoint")
plt.xlabel("Time")
plt.ylabel("Response")
plt.title("Comparison of Setpoint Responses for Different Integral Time Constants")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compare_tauI_setpoint.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4.8))
plt.plot(t2, y2, label="Nominal")
plt.plot(t8, y8, label="Increased $\\tau_I$")
plt.plot(t10, y10, label="Decreased $\\tau_I$")
plt.xlabel("Time")
plt.ylabel("Response")
plt.title("Comparison of Disturbance Responses for Different Integral Time Constants")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compare_tauI_disturbance.png", dpi=150)
plt.close()


# =============================================================================
# Document writeup
# =============================================================================
doc.section("Problem 1")

w("A closed-loop control system was constructed for the process")
m(r"G_p(s)=\frac{e^{-s}}{5s+1}")
w("with a disturbance added after the controller. The controller was set to ideal PI form, as required in the assignment.")

doc.subsection(NoEscape(r"1. Controller Settings from the $\lambda$-Rules"))

w("For the given process model, the process gain, time constant, and dead time are identified as")
m(r"K_p=1,\qquad \tau_p=5,\qquad \theta=1")

w("The assignment specifies that")
m(r"\lambda=\max\left(\frac{\tau_p}{3},\theta\right)")

a(
    r"\lambda &= \max\left(\frac{5}{3},1\right)",
    r"&= \frac{5}{3}\approx 1.667"
)

w("For PI control using the lambda tuning rules for a first-order plus dead-time model,")
m(r"K_c=\frac{\tau_p}{K_p(\lambda+\theta)}, \qquad \tau_I=\tau_p")

a(
    r"K_c &= \frac{5}{1(1.667+1)}",
    r"&= \frac{5}{2.667}",
    rf"&= {Kc_nom:.4f}"
)

a(
    rf"\tau_I &= {tauI_nom:.4f}"
)

w("Because Simulink uses the reset rate instead of the integral time constant,")
m(r"I=\frac{1}{\tau_I}")

a(
    rf"I &= \frac{{1}}{{{tauI_nom:.4f}}}",
    rf"&= {I_nom:.4f}"
)

w("Therefore, the controller settings entered into Simulink were")
a(
    rf"P &= K_c = {Kc_nom:.4f}",
    rf"I &= {I_nom:.4f}"
)

doc.subsection("2. Step Change in Setpoint, No Disturbance")

w("A unit step was applied to the setpoint while the disturbance was held at zero.")
m(r"Y_{{sp}}(t)=1,\qquad d(t)=0")

figlog(
    "part2_setpoint_nominal.png",
    caption="Closed-loop response for a unit step change in setpoint with no disturbance, using the nominal PI settings.",
    label="fig:fig1",
)
m(r"\text{The response of the closed-loop system is shown in Figure} \ref{fig:fig1} \text{for the nominal PI settings obtained from the $\lambda$ rules.}")

doc.subsection("3. Step Change in Disturbance, No Setpoint Change")

w("A unit step was applied to the disturbance while the setpoint was held constant at zero.")
m(r"Y_{{sp}}(t)=0,\qquad d(t)=1")

figlog(
    "part3_disturbance_nominal.png",
    caption="Closed-loop response for a unit step change in disturbance with no setpoint change, using the nominal PI settings.",
    label="fig:fig2",
)
m(r"\text{The response of the closed-loop system is shown in Figure} \ref{fig:fig2}.")

doc.subsection("4. Effect of Doubling and Halving the Controller Gain")

w("The controller gain was first doubled from its nominal value and the simulations of Parts 2 and 3 were repeated.")
a(
    rf"K_c^{{\text{{nominal}}}} &= {Kc_nom:.4f}",
    rf"K_c^{{\text{{double}}}} &= 2({Kc_nom:.4f}) = {Kc_double:.4f}"
)

figlog(
    "part4_setpoint_kc_double.png",
    caption="Closed-loop response to a unit step in setpoint with doubled controller gain.",
    label="fig:fig3",
)
figlog(
    "part4_disturbance_kc_double.png",
    caption="Closed-loop response to a unit step in disturbance with doubled controller gain.",
    label="fig:fig4",
)
m(r"\text{For the doubled controller gain, the setpoint response is shown in Figure} \ref{fig:fig3}  \text{, and the disturbance response is shown in Figure} \ref{fig:fig4}.")

w("The controller gain was then reduced to half of its nominal value and the simulations of Parts 2 and 3 were repeated.")
a(
    rf"K_c^{{\text{{half}}}} &= 0.5({Kc_nom:.4f}) = {Kc_half:.4f}"
)


figlog(
    "part4_setpoint_kc_half.png",
    caption="Closed-loop response to a unit step in setpoint with half the nominal controller gain.",
    label="fig:fig5",
)
figlog(
    "part4_disturbance_kc_half.png",
    caption="Closed-loop response to a unit step in disturbance with half the nominal controller gain.",
    label="fig:fig6",
)
m(r"\text{For the halved controller gain, the setpoint response is shown in Figure} \ref{fig:fig5} \text{, and the disturbance response is shown in Figure} \ref{fig:fig6}.")
figlog(
    "compare_kc_setpoint.png",
    caption="Comparison of setpoint responses for nominal, doubled, and halved controller gain.",
    label="fig:fig7",
)
figlog(
    "compare_kc_disturbance.png",
    caption="Comparison of disturbance responses for nominal, doubled, and halved controller gain.",
    label="fig:fig8",
)
m(r"\text{A direct comparison of the gain variations is shown in Figures \ref{fig:fig7} and \ref{fig:fig8}.}")

doc.subsection("5. Effect of Changing the Integral Time Constant")

w("The controller gain was returned to its nominal value, and the integral time constant was varied above and below its nominal value.")
a(
    rf"K_c &= {Kc_nom:.4f}",
    rf"\tau_I^{{\text{{nominal}}}} &= {tauI_nom:.4f}"
)

w("First, the integral time constant was increased.")
a(
    rf"\tau_I^{{\text{{large}}}} &= 2({tauI_nom:.4f}) = {tauI_large:.4f}",
    rf"I^{{\text{{large}}}} &= \frac{{1}}{{{tauI_large:.4f}}} = {1.0/tauI_large:.4f}"
)


figlog(
    "part5_setpoint_tauI_large.png",
    caption="Closed-loop response to a unit step in setpoint with increased integral time constant.",
    label="fig:fig9",
)
figlog(
    "part5_disturbance_tauI_large.png",
    caption="Closed-loop response to a unit step in disturbance with increased integral time constant.",
    label="fig:fig10",
)
m(r"\text{The corresponding setpoint and disturbance responses are shown in Figures \ref{fig:fig9} and \ref{fig:fig10}.")
w("Next, the integral time constant was decreased.")
a(
    rf"\tau_I^{{\text{{small}}}} &= 0.5({tauI_nom:.4f}) = {tauI_small:.4f}",
    rf"I^{{\text{{small}}}} &= \frac{{1}}{{{tauI_small:.4f}}} = {1.0/tauI_small:.4f}"
)

figlog(
    "part5_setpoint_tauI_small.png",
    caption="Closed-loop response to a unit step in setpoint with decreased integral time constant.",
    label="fig:fig11",
)
figlog(
    "part5_disturbance_tauI_small.png",
    caption="Closed-loop response to a unit step in disturbance with decreased integral time constant.",
    label="fig:fig12",
)
m(r"\text{The corresponding setpoint and disturbance responses are shown in Figures \ref{fig:fig11} and \ref{fig:fig12}.}")

figlog(
    "compare_tauI_setpoint.png",
    caption="Comparison of setpoint responses for nominal, increased, and decreased integral time constant.",
    label="fig:fig13",
)
figlog(
    "compare_tauI_disturbance.png",
    caption="Comparison of disturbance responses for nominal, increased, and decreased integral time constant.",
    label="fig:fig14",
)
m(r"\text{A direct comparison of the integral time constant variations is shown in Figures \ref{fig:fig13} and \ref{fig:fig14}.}")

doc.subsection("6. Comments on the Effects of Changing Controller Parameters")

w("Increasing the controller gain produced a faster response for both setpoint tracking and disturbance rejection, but the responses became more oscillatory and exhibited greater overshoot.")

w("Decreasing the controller gain produced a slower and more sluggish response, but the behavior was generally smoother and less oscillatory.")

w("Increasing the integral time constant weakened the integral action because the reset rate decreased. As a result, offset was removed more slowly and the overall response became less aggressive.")

w("Decreasing the integral time constant strengthened the integral action because the reset rate increased. This caused the controller to remove offset more quickly, but it also increased the tendency toward oscillation.")

w("Overall, the nominal tuning obtained from the lambda rules provided a reasonable balance between response speed and stability.")

doc.subsection("Final Controller Parameters")

a(
    rf"K_c &= {Kc_nom:.4f}",
    rf"\tau_I &= {tauI_nom:.4f}",
    rf"I &= {I_nom:.4f}"
)
txt_file, tex_file, pdf_file = doc.save_all()
print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX file: {tex_file}")
print(f"Wrote PDF report: {pdf_file}")
