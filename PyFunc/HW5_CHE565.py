"""
HW5_CHE565_latex.py
===================
CHE 565 – Homework 5
Cascade Control, RGA, MIMO Control, and Decoupling

This version is written in the same style as your original script:
    - uses DocumentBuilder
    - writes equations in LaTeX
    - saves figures
    - generates a PDF/LaTeX/text report through doc.save_all()

Required local files:
    - doc_builder.py
    - Simulink screenshots, if you want to include them manually

Generated files:
    - HW5_CHE565.tex
    - HW5_CHE565.txt
    - HW5_CHE565.pdf
    - PNG figures
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import control as ct
from scipy.optimize import minimize
from doc_builder import DocumentBuilder


# =============================================================================
# Output setup
# =============================================================================

OUTPUT_FILE = "HW5_CHE565"
doc = DocumentBuilder(
    OUTPUT_FILE,
    title="CHE 565 -- Homework 5",
    author="Soki Sem",
)

p = doc.p
px = doc.px
line = doc.line
eq = doc.eq
align = doc.align
table = doc.table
fig = doc.figure
subfigs = doc.subfigures
im = doc.im
lst = doc.listings

doc.maketitle(True)
doc.toc(False)


# =============================================================================
# Helper functions
# =============================================================================

def iae(t, y, ysp):
    """Integral absolute error."""
    t = np.asarray(t)
    y = np.asarray(y).squeeze()
    if np.isscalar(ysp):
        ysp = np.ones_like(t) * ysp
    else:
        ysp = np.asarray(ysp).squeeze()
    return float(np.trapezoid(np.abs(ysp - y), t))


def save_plot(filename, t, y, title, ysp=None, d=None, ylabels=None):
    plt.figure(figsize=(8, 4.8))

    y = np.asarray(y)
    if y.ndim == 1:
        plt.plot(t, y, linewidth=2, label="Output")
    else:
        for i in range(y.shape[0]):
            label = ylabels[i] if ylabels is not None else f"$y_{i+1}$"
            plt.plot(t, y[i, :], linewidth=2, label=label)

    if ysp is not None:
        if np.isscalar(ysp):
            ysp = np.ones_like(t) * ysp
            plt.plot(t, ysp, "--", linewidth=1.5, label="Setpoint")
        else:
            ysp = np.asarray(ysp)
            if ysp.ndim == 1:
                plt.plot(t, ysp, "--", linewidth=1.5, label="Setpoint")

    if d is not None:
        if np.isscalar(d):
            d = np.ones_like(t) * d
        plt.plot(t, d, ":", linewidth=1.5, label="Disturbance")

    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


def add_figure_if_exists(filename, caption, label, width=r"0.75\textwidth"):
    if os.path.exists(filename):
        fig(filename, caption=caption, label=label, width=width, position="H")
    else:
        p(f"Figure file {filename} was not found.")


def response(sys, t, U):
    resp = ct.forced_response(sys, T=t, U=U, squeeze=True, return_states=True)
    return resp.time, np.asarray(resp.outputs), np.asarray(resp.states)


def ideal_pi(Kc, tauI):
    s = ct.tf("s")
    return Kc * (1 + 1 / (tauI * s))


def p_only(Kc):
    # Important: keep the P-only controller as a transfer function
    # so ct.ss() can convert it cleanly.
    return ct.tf([Kc], [1])


def optimize_pi(objective_builder, initial=(1.0, 1.0)):
    """
    Positive-constrained PI optimization using log variables.

    objective_builder(Kc, tauI) must return a scalar objective.
    """
    def obj(log_params):
        Kc = float(np.exp(log_params[0]))
        tauI = float(np.exp(log_params[1]))
        try:
            val = objective_builder(Kc, tauI)
            if not np.isfinite(val):
                return 1e20
            return val
        except Exception:
            return 1e20

    result = minimize(
        obj,
        np.log(np.asarray(initial, dtype=float)),
        method="Nelder-Mead",
        options={"maxiter": 1200, "xatol": 1e-9, "fatol": 1e-9},
    )
    return float(np.exp(result.x[0])), float(np.exp(result.x[1])), float(result.fun)


# =============================================================================
# Problems 1--3: Cascade control system
# =============================================================================

Kp1 = 5.0
taup1 = 5.0
Kp2 = 2.0
taup2 = 10.0


def build_cascade_system(Kc_outer, tauI_outer, Kc_inner=1.0, cascade=False):
    """
    Build the cascade-control system from the assignment.

    Inputs:
        Ysp : setpoint
        D   : disturbance added after Gp1 and before Gp2

    Output:
        Y

    Without cascade:
        the inner feedback path is removed.
        The inner P-only block remains. If Kc_inner = 1, it is a pass-through.

    With cascade:
        E2 = Yc1 - P
    """
    s = ct.tf("s")

    Gc1 = ideal_pi(Kc_outer, tauI_outer)
    Gc2 = p_only(Kc_inner)
    Gp1 = Kp1 / (taup1 * s + 1)
    Gp2 = Kp2 / (taup2 * s + 1)
    Gd = ct.tf([1], [1])

    Gc1_blk = ct.ss(Gc1, name="Gc1", inputs="E1", outputs="Yc1")
    Gc2_blk = ct.ss(Gc2, name="Gc2", inputs="E2", outputs="Yc2")
    Gp1_blk = ct.ss(Gp1, name="Gp1", inputs="Yc2", outputs="Yp1")
    Gp2_blk = ct.ss(Gp2, name="Gp2", inputs="P", outputs="Y")
    Gd_blk = ct.ss(Gd, name="Gd", inputs="D", outputs="Yd")

    sum1 = ct.summing_junction(inputs=["Ysp", "-Y"], output="E1", name="Sum1")
    if cascade:
        sum2 = ct.summing_junction(inputs=["Yc1", "-P"], output="E2", name="Sum2")
    else:
        sum2 = ct.summing_junction(inputs=["Yc1"], output="E2", name="Sum2")
    sum3 = ct.summing_junction(inputs=["Yp1", "Yd"], output="P", name="Sum3")

    return ct.interconnect(
        [Gc1_blk, Gc2_blk, Gp1_blk, Gp2_blk, Gd_blk, sum1, sum2, sum3],
        inputs=["Ysp", "D"],
        outputs=["Y"],
        name="cascade_system",
    )


def simulate_cascade_case(sys, t, ysp, d):
    U = np.vstack([ysp, d])
    tout, y, x = response(sys, t, U)
    return tout, np.ravel(y), x


def tune_outer_for_case(cascade, Kc_inner, t, ysp, d, initial=(0.2, 10.0)):
    def objective(Kc, tauI):
        sys = build_cascade_system(Kc, tauI, Kc_inner=Kc_inner, cascade=cascade)
        tout, y, _ = simulate_cascade_case(sys, t, ysp, d)
        if np.any(~np.isfinite(y)) or np.max(np.abs(y)) > 1e8:
            return 1e20
        return iae(tout, y, ysp)

    return optimize_pi(objective, initial=initial)


doc.section("Problems 1--3: Cascade Control")
p(
    "The first part of the assignment compares the closed-loop response with and without cascade control. "
    "The outer controller is an ideal PI controller and the inner controller is P-only. "
    "The P-only controller is still written as a transfer function in Python so that it can be converted using "
    "the control library state-space conversion."
)

s = sp.symbols("s")
K_c, tau_I, K_2 = sp.symbols("K_c tau_I K_2")
eq(sp.latex(sp.Eq(sp.Symbol("G_{c,outer}"), K_c * (1 + 1 / (tau_I * s)))))
eq(sp.latex(sp.Eq(sp.Symbol("G_{c,inner}"), K_2)))

eq(sp.latex(sp.Eq(sp.Symbol("G_{p1}"), sp.Rational(5, 1) / (5 * s + 1)))))
eq(sp.latex(sp.Eq(sp.Symbol("G_{p2}"), sp.Rational(2, 1) / (10 * s + 1)))))

p(
    "The performance index used for comparison is the integral absolute error:"
)
t_sym = sp.symbols("t")
e_t = sp.Function("e")(t_sym)
eq(sp.latex(sp.Eq(sp.Symbol("IAE"), sp.Integral(sp.Abs(e_t), (t_sym, 0, sp.oo)))))


t = np.linspace(0, 100, 1001)
step_on = np.ones_like(t)
step_off = np.zeros_like(t)

# Problem 1: no cascade
Kc_inner_no = 1.0
Kc_no, tauI_no, _ = tune_outer_for_case(
    cascade=False,
    Kc_inner=Kc_inner_no,
    t=t,
    ysp=step_off,
    d=step_on,
    initial=(0.2, 10.0),
)
sys_no = build_cascade_system(Kc_no, tauI_no, Kc_inner=Kc_inner_no, cascade=False)

t_no_d, y_no_d, _ = simulate_cascade_case(sys_no, t, step_off, step_on)
IAE_no_d = iae(t_no_d, y_no_d, step_off)
save_plot(
    "P1_no_cascade_disturbance.png",
    t_no_d,
    y_no_d,
    "No cascade: unit step disturbance",
    ysp=step_off,
    d=step_on,
)

t_no_sp, y_no_sp, _ = simulate_cascade_case(sys_no, t, step_on, step_off)
IAE_no_sp = iae(t_no_sp, y_no_sp, step_on)
save_plot(
    "P1_no_cascade_setpoint.png",
    t_no_sp,
    y_no_sp,
    "No cascade: unit step setpoint change",
    ysp=step_on,
)

doc.subsection("Problem 1: System Without Cascade Control")
p(
    "For the no-cascade case, the inner feedback path is removed and the inner P-only controller gain is set to one. "
    "The outer PI controller is then tuned by minimizing the IAE."
)
eq(sp.latex(sp.Eq(sp.Symbol("K_{c,inner}"), Kc_inner_no)))
eq(sp.latex(sp.Eq(sp.Symbol("K_{c,outer}"), round(Kc_no, 6))))
eq(sp.latex(sp.Eq(sp.Symbol(r"\tau_{I,outer}"), round(tauI_no, 6))))
eq(sp.latex(sp.Eq(sp.Symbol("IAE_{disturbance}"), round(IAE_no_d, 6))))
eq(sp.latex(sp.Eq(sp.Symbol("IAE_{setpoint}"), round(IAE_no_sp, 6))))

add_figure_if_exists(
    "P1_no_cascade_disturbance.png",
    "Closed-loop response without cascade control for a unit step disturbance.",
    "fig:p1_no_cascade_disturbance",
)
add_figure_if_exists(
    "P1_no_cascade_setpoint.png",
    "Closed-loop response without cascade control for a unit step setpoint change.",
    "fig:p1_no_cascade_setpoint",
)

# Problem 2: with cascade
Kc_inner_cas = 0.4
Kc_cas, tauI_cas, _ = tune_outer_for_case(
    cascade=True,
    Kc_inner=Kc_inner_cas,
    t=t,
    ysp=step_off,
    d=step_on,
    initial=(0.2, 10.0),
)
sys_cas = build_cascade_system(Kc_cas, tauI_cas, Kc_inner=Kc_inner_cas, cascade=True)

t_cas_d, y_cas_d, _ = simulate_cascade_case(sys_cas, t, step_off, step_on)
IAE_cas_d = iae(t_cas_d, y_cas_d, step_off)
save_plot(
    "P2_cascade_disturbance.png",
    t_cas_d,
    y_cas_d,
    "Cascade control: unit step disturbance",
    ysp=step_off,
    d=step_on,
)

t_cas_sp, y_cas_sp, _ = simulate_cascade_case(sys_cas, t, step_on, step_off)
IAE_cas_sp = iae(t_cas_sp, y_cas_sp, step_on)
save_plot(
    "P2_cascade_setpoint.png",
    t_cas_sp,
    y_cas_sp,
    "Cascade control: unit step setpoint change",
    ysp=step_on,
)

doc.subsection("Problem 2: System With Cascade Control")
p(
    "For the cascade case, the inner loop is reconnected and the inner P-only gain is set to 0.4. "
    "The outer PI controller is tuned again because the primary controller sees a different equivalent process."
)
eq(sp.latex(sp.Eq(sp.Symbol("K_{c,inner}"), Kc_inner_cas)))
eq(sp.latex(sp.Eq(sp.Symbol("K_{c,outer}"), round(Kc_cas, 6))))
eq(sp.latex(sp.Eq(sp.Symbol(r"\tau_{I,outer}"), round(tauI_cas, 6))))
eq(sp.latex(sp.Eq(sp.Symbol("IAE_{disturbance}"), round(IAE_cas_d, 6))))
eq(sp.latex(sp.Eq(sp.Symbol("IAE_{setpoint}"), round(IAE_cas_sp, 6))))

add_figure_if_exists(
    "P2_cascade_disturbance.png",
    "Closed-loop response with cascade control for a unit step disturbance.",
    "fig:p2_cascade_disturbance",
)
add_figure_if_exists(
    "P2_cascade_setpoint.png",
    "Closed-loop response with cascade control for a unit step setpoint change.",
    "fig:p2_cascade_setpoint",
)

doc.subsection("Problem 3: Discussion")
p(
    "Cascade control is expected to make the largest difference for disturbance rejection. "
    "The disturbance enters between the two process blocks, so the inner loop can respond to the intermediate process variable before the disturbance fully propagates to the final output. "
    "For setpoint changes, both the cascade and no-cascade structures must still move the final output to a new target, so the improvement is usually smaller than for disturbances."
)


# =============================================================================
# Problems 4--5: RGA
# =============================================================================

def rga(K):
    K = np.asarray(K, dtype=float)
    return K * np.linalg.inv(K).T


doc.section("Problems 4--5: Relative Gain Array")
p("The relative gain array is calculated from the steady-state gain matrix using")
Kmat_sym = sp.MatrixSymbol("K", 2, 2)
eq(r"\Lambda = K \circ \left(K^{-1}\right)^T")
p("where the symbol " + r"$\circ$" + " denotes element-by-element multiplication.")

# Problem 4
K4 = np.array([
    [0.43, 0.43, 0.23, 0.22],
    [-0.33, 0.32, -0.20, 0.20],
    [0.22, 0.23, 0.42, 0.41],
    [-0.22, 0.22, -0.32, 0.32],
], dtype=float)
Lambda4 = rga(K4)

doc.subsection("Problem 4")
p("The gain matrix is")
eq(sp.latex(sp.Matrix(K4).evalf(4)))
p("The relative gain array is")
eq(sp.latex(sp.Matrix(Lambda4).evalf(4)))

# Pairing by diagonal dominance / values near +1
pairs4 = [(1, 1, Lambda4[0, 0]), (2, 2, Lambda4[1, 1]), (3, 3, Lambda4[2, 2]), (4, 4, Lambda4[3, 3])]
p("The diagonal elements are positive and closest to the desired pairing values, so a reasonable pairing is")
table(
    [["Output", "Manipulated variable", r"$\lambda_{ij}$"]] +
    [[f"$y_{i}$", f"$u_{j}$", f"{val:.4f}"] for i, j, val in pairs4],
    caption="Suggested pairings for Problem 4.",
    label="tab:p4_pairing",
)

# Problem 5
K5 = np.array([
    [5.0, 2.0],
    [3.0, 6.0],
], dtype=float)
Lambda5 = rga(K5)

doc.subsection("Problem 5")
p("For the two-input/two-output system, the steady-state gain matrix is obtained by evaluating each transfer function at " + r"$s=0$" + ".")
eq(sp.latex(sp.Eq(sp.Symbol("K"), sp.Matrix(K5))))
p("The RGA is")
eq(sp.latex(sp.Matrix(Lambda5).evalf(4)))
p(
    "Because the diagonal RGA values are positive and closer to one than the off-diagonal values, "
    "the preferred pairing is " + r"$y_1$ with $u_1$ and $y_2$ with $u_2$."
)


# =============================================================================
# Problems 6--10: MIMO system, cross terms, and decoupling
# =============================================================================

def fopdt(K, tau, theta, pade_order=1):
    s = ct.tf("s")
    numD, denD = ct.pade(theta, pade_order)
    delay = ct.tf(numD, denD)
    return K * delay / (tau * s + 1)


def build_mimo_process(include_cross_terms=True, pade_order=1):
    G11 = fopdt(5, 4, 5, pade_order)
    G12 = fopdt(2, 8, 4, pade_order) if include_cross_terms else ct.tf([0], [1])
    G21 = fopdt(3, 12, 3, pade_order) if include_cross_terms else ct.tf([0], [1])
    G22 = fopdt(6, 10, 3, pade_order)

    blocks = [
        ct.ss(G11, name="G11", inputs="u1", outputs="y11"),
        ct.ss(G12, name="G12", inputs="u2", outputs="y12"),
        ct.ss(G21, name="G21", inputs="u1", outputs="y21"),
        ct.ss(G22, name="G22", inputs="u2", outputs="y22"),
        ct.summing_junction(inputs=["y11", "y12"], output="y1", name="sum_y1"),
        ct.summing_junction(inputs=["y21", "y22"], output="y2", name="sum_y2"),
    ]

    return ct.interconnect(blocks, inputs=["u1", "u2"], outputs=["y1", "y2"], name="P")


def tune_single_loop(G, t, initial=(0.2, 10.0)):
    r = np.ones_like(t)

    def objective(Kc, tauI):
        C = ideal_pi(Kc, tauI)
        Tcl = ct.feedback(C * G, 1)
        tout, y, _ = response(Tcl, t, r)
        y = np.ravel(y)
        if np.any(~np.isfinite(y)) or np.max(np.abs(y)) > 1e8:
            return 1e20
        return iae(tout, y, r)

    return optimize_pi(objective, initial=initial)


def build_mimo_closed_loop(Kc1, tauI1, Kc2, tauI2, include_cross_terms=True, decoupler="none"):
    s = ct.tf("s")

    C1 = ideal_pi(Kc1, tauI1)
    C2 = ideal_pi(Kc2, tauI2)

    blocks = [
        ct.ss(C1, name="C1", inputs="e1", outputs="v1"),
        ct.ss(C2, name="C2", inputs="e2", outputs="v2"),
        ct.summing_junction(inputs=["r1", "-y1"], output="e1", name="sum_e1"),
        ct.summing_junction(inputs=["r2", "-y2"], output="e2", name="sum_e2"),
    ]

    if decoupler == "none":
        blocks += [
            ct.ss(ct.tf([1], [1]), name="D11", inputs="v1", outputs="u1"),
            ct.ss(ct.tf([1], [1]), name="D22", inputs="v2", outputs="u2"),
        ]

    elif decoupler == "static":
        # Static decoupler for diagonal pairing:
        # u1 = v1 - (K12/K11)v2
        # u2 = v2 - (K21/K22)v1
        blocks += [
            ct.ss(ct.tf([1], [1]), name="D11", inputs="v1", outputs="u11"),
            ct.ss(ct.tf([-2 / 5], [1]), name="D12", inputs="v2", outputs="u12"),
            ct.ss(ct.tf([-3 / 6], [1]), name="D21", inputs="v1", outputs="u21"),
            ct.ss(ct.tf([1], [1]), name="D22", inputs="v2", outputs="u22"),
            ct.summing_junction(inputs=["u11", "u12"], output="u1", name="sum_u1"),
            ct.summing_junction(inputs=["u21", "u22"], output="u2", name="sum_u2"),
        ]

    elif decoupler == "dynamic":
        # Dynamic decoupler for diagonal pairing:
        # D12 = -G12/G11 = -(2/5)*(4s+1)/(8s+1)*exp(+s)
        # The exp(+s) term is noncausal, so the realizable part is used.
        # D21 = -G21/G22 = -(3/6)*(10s+1)/(12s+1)
        D12 = -(2 / 5) * (4 * s + 1) / (8 * s + 1)
        D21 = -(3 / 6) * (10 * s + 1) / (12 * s + 1)

        blocks += [
            ct.ss(ct.tf([1], [1]), name="D11", inputs="v1", outputs="u11"),
            ct.ss(D12, name="D12", inputs="v2", outputs="u12"),
            ct.ss(D21, name="D21", inputs="v1", outputs="u21"),
            ct.ss(ct.tf([1], [1]), name="D22", inputs="v2", outputs="u22"),
            ct.summing_junction(inputs=["u11", "u12"], output="u1", name="sum_u1"),
            ct.summing_junction(inputs=["u21", "u22"], output="u2", name="sum_u2"),
        ]

    else:
        raise ValueError("decoupler must be 'none', 'static', or 'dynamic'")

    Psys = build_mimo_process(include_cross_terms=include_cross_terms)
    blocks.append(Psys)

    return ct.interconnect(blocks, inputs=["r1", "r2"], outputs=["y1", "y2"], name="mimo_closed_loop")


def simulate_mimo_case(sys, t, r1, r2):
    U = np.vstack([r1, r2])
    tout, y, _ = response(sys, t, U)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(2, -1)
    return tout, y


doc.section("Problems 6--10: MIMO Control and Decoupling")
p("The transfer-function matrix is")
eq(r"""
G(s)=
\begin{bmatrix}
\dfrac{5e^{-5s}}{4s+1} & \dfrac{2e^{-4s}}{8s+1}\\[6pt]
\dfrac{3e^{-3s}}{12s+1} & \dfrac{6e^{-3s}}{10s+1}
\end{bmatrix}
""")

t_mimo = np.linspace(0, 150, 1501)
G11 = fopdt(5, 4, 5)
G22 = fopdt(6, 10, 3)

Kc_1, tauI_1, _ = tune_single_loop(G11, t_mimo, initial=(0.2, 8.0))
Kc_2, tauI_2, _ = tune_single_loop(G22, t_mimo, initial=(0.2, 10.0))

doc.subsection("Problem 6: Independent Loop Tuning")
p("Neglecting the cross terms, the diagonal loops are tuned independently with PI controllers.")
eq(sp.latex(sp.Eq(sp.Symbol("G_{11}"), sp.Symbol(r"\frac{5e^{-5s}}{4s+1}"))))
eq(sp.latex(sp.Eq(sp.Symbol("G_{22}"), sp.Symbol(r"\frac{6e^{-3s}}{10s+1}"))))
eq(sp.latex(sp.Eq(sp.Symbol("K_{c1}"), round(Kc_1, 6))))
eq(sp.latex(sp.Eq(sp.Symbol(r"\tau_{I1}"), round(tauI_1, 6))))
eq(sp.latex(sp.Eq(sp.Symbol("K_{c2}"), round(Kc_2, 6))))
eq(sp.latex(sp.Eq(sp.Symbol(r"\tau_{I2}"), round(tauI_2, 6))))

r1_step = np.ones_like(t_mimo)
r2_zero = np.zeros_like(t_mimo)
r1_zero = np.zeros_like(t_mimo)
r2_step = np.ones_like(t_mimo)

sys_no_cross = build_mimo_closed_loop(Kc_1, tauI_1, Kc_2, tauI_2, include_cross_terms=False, decoupler="none")

tout, y = simulate_mimo_case(sys_no_cross, t_mimo, r1_step, r2_zero)
save_plot("P6_no_cross_r1.png", tout, y, "Problem 6: no cross terms, step in r1", ylabels=[r"$y_1$", r"$y_2$"])
add_figure_if_exists("P6_no_cross_r1.png", "No cross terms with a unit step in " + r"$r_1$" + ".", "fig:p6_r1")

tout, y = simulate_mimo_case(sys_no_cross, t_mimo, r1_zero, r2_step)
save_plot("P6_no_cross_r2.png", tout, y, "Problem 6: no cross terms, step in r2", ylabels=[r"$y_1$", r"$y_2$"])
add_figure_if_exists("P6_no_cross_r2.png", "No cross terms with a unit step in " + r"$r_2$" + ".", "fig:p6_r2")


doc.subsection("Problem 7: Cross Terms Added")
sys_cross = build_mimo_closed_loop(Kc_1, tauI_1, Kc_2, tauI_2, include_cross_terms=True, decoupler="none")

tout, y = simulate_mimo_case(sys_cross, t_mimo, r1_step, r2_zero)
IAE_cross_y1 = iae(tout, y[0], 1)
INT_cross_y2 = float(np.trapezoid(np.abs(y[1]), tout))
save_plot("P7_cross_r1.png", tout, y, "Problem 7: cross terms, step in r1", ylabels=[r"$y_1$", r"$y_2$"])
eq(sp.latex(sp.Eq(sp.Symbol("IAE_{y1,r1}"), round(IAE_cross_y1, 6))))
eq(sp.latex(sp.Eq(sp.Symbol("Interaction_{y2,r1}"), round(INT_cross_y2, 6))))
add_figure_if_exists("P7_cross_r1.png", "Cross terms included with a unit step in " + r"$r_1$" + ".", "fig:p7_r1")

tout, y = simulate_mimo_case(sys_cross, t_mimo, r1_zero, r2_step)
INT_cross_y1 = float(np.trapezoid(np.abs(y[0]), tout))
IAE_cross_y2 = iae(tout, y[1], 1)
save_plot("P7_cross_r2.png", tout, y, "Problem 7: cross terms, step in r2", ylabels=[r"$y_1$", r"$y_2$"])
eq(sp.latex(sp.Eq(sp.Symbol("Interaction_{y1,r2}"), round(INT_cross_y1, 6))))
eq(sp.latex(sp.Eq(sp.Symbol("IAE_{y2,r2}"), round(IAE_cross_y2, 6))))
add_figure_if_exists("P7_cross_r2.png", "Cross terms included with a unit step in " + r"$r_2$" + ".", "fig:p7_r2")


doc.subsection("Problem 8: Static Decoupling")
p("For diagonal pairing, the static decoupler is")
eq(r"""
D_s =
\begin{bmatrix}
1 & -K_{12}/K_{11}\\
-K_{21}/K_{22} & 1
\end{bmatrix}
=
\begin{bmatrix}
1 & -2/5\\
-3/6 & 1
\end{bmatrix}
""")

sys_static = build_mimo_closed_loop(Kc_1, tauI_1, Kc_2, tauI_2, include_cross_terms=True, decoupler="static")

tout, y = simulate_mimo_case(sys_static, t_mimo, r1_step, r2_zero)
save_plot("P8_static_r1.png", tout, y, "Problem 8: static decoupler, step in r1", ylabels=[r"$y_1$", r"$y_2$"])
add_figure_if_exists("P8_static_r1.png", "Static decoupling with a unit step in " + r"$r_1$" + ".", "fig:p8_r1")

tout, y = simulate_mimo_case(sys_static, t_mimo, r1_zero, r2_step)
save_plot("P8_static_r2.png", tout, y, "Problem 8: static decoupler, step in r2", ylabels=[r"$y_1$", r"$y_2$"])
add_figure_if_exists("P8_static_r2.png", "Static decoupling with a unit step in " + r"$r_2$" + ".", "fig:p8_r2")


doc.subsection("Problem 9: Dynamic Decoupling")
p("For diagonal pairing, the ideal dynamic decouplers are")
eq(r"""
D_{12}(s)=-\frac{G_{12}}{G_{11}}
=
-\frac{2}{5}\frac{4s+1}{8s+1}e^{s}
""")
eq(r"""
D_{21}(s)=-\frac{G_{21}}{G_{22}}
=
-\frac{3}{6}\frac{10s+1}{12s+1}
""")
p(
    "The term " + r"$e^{s}$" + " is noncausal because it corresponds to a negative delay. "
    "Therefore, the realizable dynamic decoupler uses the proper transfer-function part without the noncausal delay."
)
eq(r"""
D_{12,realizable}(s)=
-\frac{2}{5}\frac{4s+1}{8s+1}
""")

sys_dynamic = build_mimo_closed_loop(Kc_1, tauI_1, Kc_2, tauI_2, include_cross_terms=True, decoupler="dynamic")

tout, y = simulate_mimo_case(sys_dynamic, t_mimo, r1_step, r2_zero)
save_plot("P9_dynamic_r1.png", tout, y, "Problem 9: dynamic decoupler, step in r1", ylabels=[r"$y_1$", r"$y_2$"])
add_figure_if_exists("P9_dynamic_r1.png", "Dynamic decoupling with a unit step in " + r"$r_1$" + ".", "fig:p9_r1")

tout, y = simulate_mimo_case(sys_dynamic, t_mimo, r1_zero, r2_step)
save_plot("P9_dynamic_r2.png", tout, y, "Problem 9: dynamic decoupler, step in r2", ylabels=[r"$y_1$", r"$y_2$"])
add_figure_if_exists("P9_dynamic_r2.png", "Dynamic decoupling with a unit step in " + r"$r_2$" + ".", "fig:p9_r2")


doc.subsection("Problem 10: Discussion")
p(
    "When the cross terms are included, the two loops interact: a setpoint change in one loop causes motion in the other output. "
    "Static decoupling reduces the steady-state interaction because it cancels the steady-state gain effects of the off-diagonal elements. "
    "However, it does not fully cancel the dynamic interaction because the off-diagonal transfer functions have different delays and time constants. "
    "Dynamic decoupling accounts for those dynamics and should reduce interaction more effectively, but any noncausal positive-exponential delay terms must be removed or approximated with a realizable transfer function."
)


# =============================================================================
# Save
# =============================================================================

txt_file, tex_file, pdf_file = doc.save_all(runs=2)

print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX file: {tex_file}")
print(f"Wrote PDF report: {pdf_file}")
