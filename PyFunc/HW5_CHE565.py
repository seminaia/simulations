"""
HW5_CHE565_soki_fixed.py
========================
CHE 565 – Homework 5
Cascade Control, RGA, MIMO control, and decoupling.

This version keeps the original report workflow:
    - DocumentBuilder
    - LaTeX equations using doc.eq() / doc.align()
    - doc.table() with explicit row lists
    - generated figures inserted with doc.figure() / doc.subfigures()

Important fixes:
    - Gc2 is P-only but still a transfer function: ct.tf([Kc2], [1])
    - No manual $$ delimiters are used inside doc.eq()
    - Table rows are real lists, not zip/generator objects
    - Undefined optimized variables are fixed
    - Problems 1–10 are included
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate._ivp.radau import P
import sympy as sp
from scipy.optimize import minimize
import control as ct
from pylatex import NoEscape
from doc_builder import DocumentBuilder


# =============================================================================
# Output names
# =============================================================================
OUTPUT_FILE = "HW5_CHE565"
PLOT_DIR = Path("HW5_plots")
PLOT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Document setup
# =============================================================================
doc = DocumentBuilder(
    OUTPUT_FILE,
    title="CHE 565 -- Homework 5",
    author="Soki Sem",
)

# Convenience aliases, matching your original style
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

# Your DocumentBuilder creates title immediately in __init__ if enabled.
# These calls are kept for consistency with your original script.
doc.maketitle(True)
doc.toc(False)


# =============================================================================
# General helpers
# =============================================================================
def save_plot(filename, t, y, title, ysp=None, d=None, ylabel="Response"):
    filename = str(filename)
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, y, label="Output", linewidth=2)
    if ysp is not None:
        ysp_vec = np.ones_like(t) * ysp if np.isscalar(ysp) else np.asarray(ysp)
        plt.plot(t, ysp_vec, "--", label="Setpoint", linewidth=1.5)
    if d is not None:
        d_vec = np.ones_like(t) * d if np.isscalar(d) else np.asarray(d)
        plt.plot(t, d_vec, ":", label="Disturbance", linewidth=1.5)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=170)


def save_mimo_plot(filename, t, y1, y2, title):
    filename = str(filename)
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, y1, label="y1", linewidth=2)
    plt.plot(t, y2, label="y2", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Outputs")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()


def calculate_IAE(t, y, ysp):
    y = np.asarray(y).reshape(-1)
    if np.isscalar(ysp):
        ysp_vec = np.ones_like(t) * float(ysp)
    else:
        ysp_vec = np.asarray(ysp).reshape(-1)
    return float(np.trapezoid(np.abs(ysp_vec - y), t))


def interaction_area(t, y):
    return float(np.trapezoid(np.abs(np.asarray(y).reshape(-1)), t))


def rga(K):
    K = np.asarray(K, dtype=float)
    return K * np.linalg.inv(K).T


def matrix_rows(M):
    """Return rows as real lists, so doc.table() cannot lose them."""
    return [[float(v) for v in row] for row in np.asarray(M)]


def latex_matrix(M, fmt=".4f"):
    M = np.asarray(M)
    body = r" \\ ".join(
        " & ".join(format(float(v), fmt) for v in row) for row in M
    )
    return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"


def optional_figure(path, caption, label=None, width=r"0.85\textwidth"):
    """Include a figure only if it exists, avoiding LaTeX compile failures."""
    if Path(path).exists():
        figlog(path, caption=caption, label=label, width=width, position="H")


# =============================================================================
# Problems 1–3 data and cascade model
# =============================================================================
Kp1 = 5.0
taup1 = 5.0
Kp2 = 2.0
taup2 = 10.0


def ideal_pi(Kc, tauI):
    s = ct.tf("s")
    return Kc * (1 + 1 / (tauI * s))


def p_only_controller(Kc):
    # Important: P-only, but as a transfer function so ct.ss() works.
    return ct.tf([Kc], [1])


def build_closed_loop(Kc1, Kc2, tauI1, cascade=False):
    """
    Build the cascade-control system from Problems 1–3.

    Inputs:
        Ysp: setpoint
        D: disturbance added after Gp1 and before Gp2

    Output:
        Y

    Outer controller: ideal PI
    Inner controller: P-only
    """
    s = ct.tf("s")

    Gc1 = ideal_pi(Kc1, tauI1)
    Gc2 = ct.tf([Kc2],[1], name="Gc2", inputs="E2", outputs="Yc2")  # 
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
        # Inner feedback loop connected: E2 = Yc1 - P
        sum2 = ct.summing_junction(inputs=["Yc1", "-P"], output="E2", name="Sum2")
    else:
        # Inner feedback disconnected: E2 = Yc1.
        # If Kc2 = 1, the inner controller is a pass-through block.
        sum2 = ct.summing_junction(inputs=["Yc1"], output="E2", name="Sum2")

    sum3 = ct.summing_junction(inputs=["Yp1", "Yd"], output="P", name="Sum3")

    sys = ct.interconnect(
        [Gc1_blk, Gc2_blk, Gp1_blk, Gp2_blk, Gd_blk, sum1, sum2, sum3],
        inputs=["Ysp", "D"],
        outputs=["Y"],
    )
    return sys


def simulate_case(sys, t, ysp_input, d_input):
    U = np.vstack([ysp_input, d_input])
    resp = ct.forced_response(sys, T=t, U=U, squeeze=True, return_states=True)
    return resp.time, resp.outputs, resp.inputs, resp.states


def optimize_PI(cascade, Kc2, t, ysp_input, d_input, params):
    """Optimize positive Kc and tauI by minimizing IAE."""
    ysp_input = np.asarray(ysp_input)
    d_input = np.asarray(d_input)

    def objective(params):
        Kc = float(params[0])
        tauI = float(params[1])    
        tuned_sys = build_closed_loop(Kc, Kc2, tauI, cascade=cascade)
        t_sim, y_sim, _, _ = simulate_case(tuned_sys, t, ysp_input, d_input)
        return calculate_IAE(t_sim, y_sim, ysp_input)

    result = minimize(
        objective,
        params,
    )
    Kc_opt = float(result.x[0])
    tauI_opt = float(result.x[1])
    return Kc_opt, tauI_opt, result.fun


# =============================================================================
# Problems 5–10 MIMO model helpers
# =============================================================================
def fopdt(K, tau, theta, pade_order=1):
    s = ct.tf("s")
    numD, denD = ct.pade(theta, pade_order)
    return K * ct.tf(numD, denD) / (tau * s + 1)


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


def lambda_tuning(taup, thetad, Kp ,PI=True, PID=False):
    """
        Lambda tuning for FOPDT:
        Kc = tau / (K (lambda + theta))
        tauI = tau
    """
    if not PI and not PID:
        raise ValueError("At least one of PI or PID must be True.")
    if PI and PID:
        raise ValueError("Only one of PI or PID can be True.")
    
    if taup < 4*thetad:
        #Self regulating Process
        lam = thetad
        tauI = max(thetad/4, taup)
        P = taup / (Kp * (lam + thetad))
    else:
        #Integrating Process
        lam = max( taup/3, thetad)
        tauI = max(4*thetad,2*lam+thetad)
        Kprime = Kp / taup     
        P= tauI/(Kprime*(lam+thetad)**2)

    if PI:
        tauD = 0
    elif PID:
        tauD = min( tauI/4,thetad/2)

    return float(P), float(tauI), float(lam), float(tauD)


def build_two_loop_closed_system(Kc1, tauI1, Kc2, tauI2, include_cross_terms=True, decoupler="none"):
    """
    Two SISO PI loops around the 2x2 process.

    Inputs: r1, r2
    Outputs: y1, y2

    decoupler options:
        none, static, dynamic
    """
    s = ct.tf("s")
    blocks = [
        ct.ss(ideal_pi(Kc1, tauI1), name="C1", inputs="e1", outputs="v1"),
        ct.ss(ideal_pi(Kc2, tauI2), name="C2", inputs="e2", outputs="v2"),
        ct.summing_junction(inputs=["r1", "-y1"], output="e1", name="sum_e1"),
        ct.summing_junction(inputs=["r2", "-y2"], output="e2", name="sum_e2"),
    ]

    if decoupler == "none":
        blocks += [
            ct.ss(ct.tf([1], [1]), name="D11", inputs="v1", outputs="u1"),
            ct.ss(ct.tf([1], [1]), name="D22", inputs="v2", outputs="u2"),
        ]
    else:
        if decoupler == "static":
            D12 = ct.tf([-2 / 5], [1])
            D21 = ct.tf([-3 / 6], [1])
        elif decoupler == "dynamic":
            # Diagonal pairing decouplers:
            # D12 = -G12/G11 = -(2/5)(4s+1)/(8s+1) exp(+s)
            # The exp(+s) part is noncausal, so the realizable part is used.
            # D21 = -G21/G22 = -(3/6)(10s+1)/(12s+1)
            D12 = -(2 / 5) * (4 * s + 1) / (8 * s + 1)
            D21 = -(3 / 6) * (10 * s + 1) / (12 * s + 1)
        else:
            raise ValueError("decoupler must be 'none', 'static', or 'dynamic'")

        blocks += [
            ct.ss(ct.tf([1], [1]), name="D11", inputs="v1", outputs="u11"),
            ct.ss(D12, name="D12", inputs="v2", outputs="u12"),
            ct.ss(D21, name="D21", inputs="v1", outputs="u21"),
            ct.ss(ct.tf([1], [1]), name="D22", inputs="v2", outputs="u22"),
            ct.summing_junction(inputs=["u11", "u12"], output="u1", name="sum_u1"),
            ct.summing_junction(inputs=["u21", "u22"], output="u2", name="sum_u2"),
        ]

    blocks.append(build_mimo_process(include_cross_terms=include_cross_terms))
    return ct.interconnect(blocks, inputs=["r1", "r2"], outputs=["y1", "y2"])


# =============================================================================
# Document writeup
# =============================================================================
def write_intro():
    doc.section("Introduction")
    px(
        "This homework was completed using Python with the Control Systems Library. ",
        "The report uses the same block-diagram logic as the Simulink assignment, ",
        "but the systems are constructed programmatically using transfer functions, ",
        "state-space conversions, and ", im(r"\texttt{ct.interconnect}"), "."
    )
    p(
        "The cascade-control problems compare the response with and without the inner feedback loop. "
        "The first controller is treated as an ideal PI controller, while the inner controller is P-only. "
        "For the no-cascade case, the inner feedback is disconnected and the inner controller gain is set to 1."
    )
    eq(r"G_{c,1}(s) = K_{c,1}\left(1 + \frac{1}{\tau_{I,1}s}\right)")
    eq(r"G_{c,2}(s) = K_{c,2}")
    eq(r"IAE = \int_0^T |r(t)-y(t)|\,dt")

    lst([
        """def p_only_controller(Kc):
    # P-only, but represented as a transfer function so ct.ss() works.
    return ct.tf([Kc], [1])

Gc2 = p_only_controller(Kc2)
Gc2_blk = ct.ss(Gc2, name='Gc2', inputs='E2', outputs='Yc2')"""
    ])

    optional_figure(
        "HW5_CHE565_block_diagram.png",
        caption="Simulink block diagram for the cascade-control system.",
        label="fig:block_diagram",
        width=r"0.80\textwidth",
    )


def run_problems_1_to_3():
    tvals = np.linspace(0, 100, 1001)
    step_on = np.ones_like(tvals)
    step_off = np.zeros_like(tvals)
    thetad=1
    doc.section("Problems 1--3: Cascade Control")

    # Problem 1: no cascade, inner gain 1
    doc.subsection("Problem 1: System Without Cascade Control")
    Kc2_no = 1.0
    P_no, tauI_no = [1,1]
    I_no = 1 / tauI_no
    P_auto, I_auto = 0.183711730708738, 0.0816496580927727
    tauI_auto = 1 / I_auto

    sys_no = build_closed_loop(P_no, Kc2_no, tauI_no, cascade=False)

    t1, y1, _, _ = simulate_case(sys_no, tvals, step_off, step_on)
    iae1 = calculate_IAE(t1, y1, step_off)
    f1 = "closed_loop_no_cascade.png"
    save_plot(f1, t1, y1, "No cascade: unit step disturbance", ysp=step_off, d=step_on)
    
    sys_auto = build_closed_loop(P_auto, Kc2_no, tauI_auto, cascade=False)
    t2, y2, _, _ = simulate_case(sys_auto, tvals, step_off, step_off)
    iae2 = calculate_IAE(t2, y2, step_off)
    f2 = "closed_loop_no_cascade_autotuned.png"
    save_plot(f2, t2, y2, "No cascade: unit step disturbance autotuned", ysp=step_off)
    
    P_lambda, tauI_lambda, lam, _ = lambda_tuning(taup1, thetad, Kp1, PI=True)
    sys_lamda = build_closed_loop(P_lambda, Kc2_no, tauI_lambda, cascade=False)
    t3, y3, _, _ = simulate_case(sys_lamda, tvals, step_off, step_on)
    iae3 = calculate_IAE(t3, y3, step_off)
    f3 = "closed_loop_no_cascade_lambda.png"
    save_plot(f3, t3, y3, "No cascade: unit step disturbance lambda tuning", ysp=step_off, d=step_on)
    
    P_opt, tauI_opt, iae_opt = optimize_PI(cascade=False, Kc2=Kc2_no, t=tvals, ysp_input=step_off, d_input=step_on, params=[P_lambda, tauI_lambda])
    sys_opt = build_closed_loop(P_opt, Kc2_no, tauI_opt, cascade=False)
    t4, y4, _, _ = simulate_case(sys_opt, tvals, step_off, step_on)
    iae4 = calculate_IAE(t4, y4, step_off)
    f4 = "closed_loop_no_cascade_optimized.png"
    save_plot(f4, t4, y4, "No cascade: unit step disturbance IAE optimized", ysp=step_off, d=step_on)    

    t5, y5, _, _ = simulate_case(sys_no, tvals, step_on, step_off)
    iae5 = calculate_IAE(t5, y5, step_on)
    f5 = "closed_loop_no_cascade_setpoint.png"
    save_plot(f5, t5, y5, "No cascade: unit step setpoint", ysp=step_on, d=step_on)
    
    t6, y6, _, _ = simulate_case(sys_auto, tvals, step_on, step_off)
    iae6 = calculate_IAE(t6, y6, step_on)
    f6 = "closed_loop_no_cascade_autotuned_setpoint.png"
    save_plot(f6, t6, y6, "No cascade: unit step setpoint autotuned", ysp=step_on, d=step_off)
    
    P_lambda, tauI_lambda, lam, _ = lambda_tuning(taup1, thetad, Kp1, PI=True)
    sys_lamda = build_closed_loop(P_lambda, Kc2_no, tauI_lambda, cascade=False)
    t7, y7, _, _ = simulate_case(sys_lamda, tvals, step_on, step_off)
    iae7 = calculate_IAE(t7, y7, step_on)
    f7 = "closed_loop_no_cascade_lambda_setpoint.png"
    save_plot(f7, t7, y7, "No cascade: unit step setpoint lambda tuning", ysp=step_on, d=step_off)
    
    t8, y8, _, _ = simulate_case(sys_opt, tvals, step_on, step_off)
    f8 = "closed_loop_no_cascade_optimized_setpoint.png"
    iae8 = calculate_IAE(t8, y8, step_on)
    save_plot(f8, t8, y8, "No cascade: unit step setpoint IAE optimized", ysp=step_on, d=step_off)
    px(
        "For the no-cascade case, the inner feedback path was disconnected and the P-only inner controller gain was set to ",
        im(r"K_{c,2}=1"), ". The outer PI controller was then tuned using the autotuning from simulink and lambda tuning method and minimizing the IAE"
    )
    eq(rf"K_{{c,1}} = {P_no:.4f}, \qquad \tau_{{I,1}} = {tauI_no:.4f}, \qquad I_1 = {I_no:.4f}")
    table(
        headers=["Case", NoEscape(r"$K_{c,1}$"), NoEscape(r"$\tau_{I,1}$"), "IAE"],
        rows=[
            ["Disturbance", P_no, tauI_no, iae1],
            ["Autotune", P_auto, tauI_auto, iae2],
            ["Lambda", P_lambda, tauI_lambda, iae3],
            ["IAE optimized", P_opt, tauI_opt, iae4],
        ],
        caption="Problem 1 results without cascade control.",
        label="tab:problem1_no_cascade",
    )
    subfiglog(
        [(str(f1), "Disturbance response"), (str(f2), "Disturbance response (auto-tuned)"), (str(f3), "Disturbance response (lambda tuning)"), (str(f4), "Disturbance response (IAE optimized)")],
        caption="Problem 1 responses without cascade control.",
        label="fig:problem1_no_cascade",
        width=r"0.25\textwidth",
    )
    
    table(
        headers=["Case", NoEscape(r"$K_{c,1}$"), NoEscape(r"$\tau_{I,1}$"), "IAE"],
        rows=[
            ["Setpoint", P_no, tauI_no, iae5],
            ["Autotune", P_auto, tauI_auto, iae6],
            ["Lambda", P_lambda, tauI_lambda, iae7],
            ["IAE optimized", P_opt, tauI_opt, iae8],
        ],
        caption="Problem 1 setpoint results without cascade control.",
        label="tab:problem1_no_cascade_setpoint",
    
    )
    subfiglog(
        [(str(f5), "Setpoint response"), (str(f6), "Setpoint response (auto-tuned)"), (str(f7), "Setpoint response (lambda tuning)"), (str(f8), "Setpoint response (IAE optimized)")],
        caption="Problem 1 responses without cascade control.",
        label="fig:problem1_no_cascade",
        width=r"0.25\textwidth",
    )

    # Problem 2: cascade, inner gain 0.4
    doc.subsection("Problem 2: System With Cascade Control")
    Kc2_cas = 0.4
    P_cas, I_cas = [1, 1]
    tauI_cas = 1 / I_cas
    
    sys_cas = build_closed_loop(P_cas, Kc2_cas, tauI_cas, cascade=True)
    t1_cas, y1_cas, _, _ = simulate_case(sys_cas, tvals, step_off, step_on)
    iae1_cas = calculate_IAE(t1_cas, y1_cas, step_off)
    f1_cas ="closed_loop_cascade.png"
    save_plot(f1_cas, t1_cas, y1_cas, "Cascade: unit step disturbance", ysp=step_off, d=step_on)
    
    P_auto_cas, I_auto_cas = 0.750849555738588, 0.21996983950919
    tauI_auto_cas = 1 / I_auto_cas
    sys_auto_cas = build_closed_loop(P_auto_cas, Kc2_cas, tauI_auto_cas, cascade=True)
    t2_cas, y2_cas, _, _ = simulate_case(sys_auto_cas, tvals, step_off, step_on)
    iae2_cas = calculate_IAE(t2_cas, y2_cas, step_off)
    f2_cas = "closed_loop_cascade_autotuned_setpoint.png"
    save_plot(f2_cas, t2_cas, y2_cas, "Cascade: unit step disturbance (auto-tuned)", ysp=step_off, d=step_on)
    
    P_lambda_cas, tauI_lambda_cas, lam_cas, _ = lambda_tuning(taup1, thetad, Kp1, PI=True)
    sys_lambda_cas = build_closed_loop(P_lambda_cas, Kc2_cas, tauI_lambda_cas, cascade=True)
    t3_cas, y3_cas, _, _ = simulate_case(sys_lambda_cas, tvals, step_off, step_on)
    iae3_cas = calculate_IAE(t3_cas, y3_cas, step_off)
    f3_cas = "closed_loop_cascade_lambda_setpoint.png"
    save_plot(f3_cas, t3_cas, y3_cas, "Cascade: unit step disturbance (lambda tuning)", ysp=step_off, d=step_on)
    
    P_opt_cas, tauI_opt_cas, iae_opt_cas = optimize_PI(cascade=True, Kc2=Kc2_cas, t=tvals, ysp_input=step_off, d_input=step_on, params=[P_cas, tauI_cas])
    sys_opt_cas = build_closed_loop(P_opt_cas, Kc2_cas, tauI_opt_cas, cascade=True)
    t4_cas, y4_cas, _, _ = simulate_case(sys_opt_cas, tvals, step_off, step_on)
    iae4_cas = calculate_IAE(t4_cas, y4_cas, step_off)
    f4_cas = "closed_loop_cascade_optimized_setpoint.png"
    save_plot(f4_cas, t4_cas, y4_cas, "Cascade: unit step disturbance (IAE optimized)", ysp=step_off, d=step_on)
    
        
    # Problem 2: cascade, inner gain 0.4
    doc.subsection("Problem 2: System With Cascade Control")
    Kc2_cas = 0.4
    P_cas, I_cas = [1, 1]
    tauI_cas = 1 / I_cas
    
    t5_cas, y5_cas, _, _ = simulate_case(sys_cas, tvals, step_on, step_off)
    iae5_cas = calculate_IAE(t5_cas, y5_cas, step_on)
    f5_cas ="closed_loop_cascade.png"
    save_plot(f5_cas, t5_cas, y5_cas, "Cascade: unit step response", ysp=step_off, d=step_on)
    

    t6_cas, y6_cas, _, _ = simulate_case(sys_auto_cas, tvals, step_on, step_off)
    iae6_cas = calculate_IAE(t6_cas, y6_cas, step_on)
    f6_cas = "closed_loop_cascade_autotuned_setpoint.png"
    save_plot(f6_cas, t6_cas, y6_cas, "Cascade: unit step disturbance (auto-tuned)", ysp=step_off, d=step_on)
    
    t7_cas, y7_cas, _, _ = simulate_case(sys_lambda_cas, tvals, step_on, step_off)
    iae7_cas = calculate_IAE(t7_cas, y7_cas, step_on)
    f7_cas = "closed_loop_cascade_lambda_setpoint.png"
    save_plot(f7_cas, t7_cas, y7_cas, "Cascade: unit step disturbance (lambda tuning)", ysp=step_off, d=step_on)
    
    
    t8_cas, y8_cas, _, _ = simulate_case(sys_opt_cas, tvals, step_on, step_off)
    iae8_cas = calculate_IAE(t8_cas, y8_cas, step_on)
    f8_cas = "closed_loop_cascade_optimized_setpoint.png"
    save_plot(f8_cas, t8_cas, y8_cas, "Cascade: unit step disturbance (IAE optimized)", ysp=step_off, d=step_on)

    px(
        "For the cascade case, the inner feedback path was connected and the inner P-only controller gain was set to ",
        im(r"K_{c,2}=0.4"), ". The outer controller was tuned again because the outer loop sees a different equivalent process."
    )
    eq(rf"K_{{c,1}} = {P_cas:.4f}, \qquad \tau_{{I,1}} = {tauI_cas:.4f}, \qquad I_1 = {I_cas:.4f}")
    table(
        headers=["Case", NoEscape(r"$K_{c,2}$"), NoEscape(r"$K_{c,1}$"), NoEscape(r"$\tau_{I,1}$"), "IAE"],
        rows=[
            ["Disturbance", Kc2_cas, P_cas, tauI_cas, iae1_cas],
            ["Autotune", Kc2_cas, P_auto_cas, tauI_auto_cas, iae2_cas],
            ["Lambda", Kc2_cas, P_lambda_cas, tauI_lambda_cas, iae3_cas],
            ["IAE optimized", Kc2_cas, P_opt_cas, tauI_opt_cas, iae4_cas],
        ],
        caption="Problem 2 results with cascade control and unit step disturbance.",
        label="tab:problem2_cascade_disturbance",
    )
    
    subfiglog(
        [(str(f5_cas), "Setpoint response"), (str(f6_cas), "Setpoint response"), (str(f7_cas), "Setpoint response"), (str(f8_cas), "Setpoint response")],
        caption="Problem 2 responses with cascade control and unit step setpoint changes.",
        label="fig:problem2_cascade",
        width=r"0.25\textwidth",
    )
    
    table(
        headers=["Case", NoEscape(r"$K_{c,2}$"), NoEscape(r"$K_{c,1}$"), NoEscape(r"$\tau_{I,1}$"), "IAE"],
        rows=[
            ["Setpoint", Kc2_cas, P_cas, tauI_cas, iae5_cas],
            ["Autotune", Kc2_cas, P_auto_cas, tauI_auto_cas, iae6_cas],
            ["Lambda", Kc2_cas, P_lambda_cas, tauI_lambda_cas, iae7_cas],
            ["IAE optimized", Kc2_cas, P_opt_cas, tauI_opt_cas, iae8_cas],
        ],
        caption="Problem 2 results with cascade control and unit step setpoint changes.",
        label="tab:problem2_cascade_setpoint"
    )

    # Problem 3 comments
    doc.subsection("Problem 3: Comparison of Cascade and No-Cascade Results")
    table(
        headers=["Test", "No Cascade IAE", "Cascade IAE", "Change"],
        rows=[
            ["Disturbance", iae1, iae1_cas, iae1_cas - iae1],
            ["Setpoint", iae2, iae2_cas, iae2_cas - iae2],
        ],
        caption="Comparison of no-cascade and cascade-control performance.",
        label="tab:problem3_comparison",
    )
    
    p(
        "Cascade control is expected to make the largest difference for disturbance rejection. "
        "The disturbance enters at the intermediate process variable, so the inner loop can react before the effect fully propagates to the final output. "
        "For setpoint tracking, cascade control may still change the response, but the benefit is usually less direct because the setpoint must pass through both the outer and inner loops."
    )


# =============================================================================
# Problems 4–10
# =============================================================================
def run_problem_4():
    doc.section("Problem 4: Relative Gain Array for the 4 by 4 Process")
    K = np.array([
        [0.43, 0.43, 0.23, 0.22],
        [-0.33, 0.32, -0.20, 0.20],
        [0.22, 0.23, 0.42, 0.41],
        [-0.22, 0.22, -0.32, 0.32],
    ])
    Lam = rga(K)

    eq(r"\Lambda = K \circ \left(K^{-1}\right)^T")
    eq(r"K = " + latex_matrix(K))
    eq(r"\Lambda = " + latex_matrix(Lam))

    table(
        headers=[NoEscape(r"$y_i$"), NoEscape(r"$u_1$"), NoEscape(r"$u_2$"), NoEscape(r"$u_3$"), NoEscape(r"$u_4$")],
        rows=[[f"y{i+1}"] + [float(Lam[i, j]) for j in range(4)] for i in range(4)],
        caption="Relative gain array for Problem 4.",
        label="tab:problem4_rga",
    )

    pair_rows = [
        ["y1", "u2", Lam[0, 1]],
        ["y2", "u4", Lam[1, 3]],
        ["y3", "u1", Lam[2, 0]],
        ["y4", "u3", Lam[3, 2]],
    ]
    table(
        headers=["Output", "Manipulated Variable", NoEscape(r"$\lambda_{ij}$")],
        rows=pair_rows,
        caption="Suggested pairings for Problem 4.",
        label="tab:problem4_pairings",
    )
    p(
        "The suggested pairings are chosen from positive RGA values that are relatively close to one while avoiding negative pairings. "
        "Negative relative gains are avoided because they indicate that closing other loops can reverse the apparent gain direction."
    )


def run_problem_5():
    doc.section("Problem 5: Relative Gain Array for the 2 by 2 Process")
    K = np.array([[5.0, 2.0], [3.0, 6.0]])
    Lam = rga(K)
    eq(r"K = \begin{bmatrix} 5 & 2 \\ 3 & 6 \end{bmatrix}")
    eq(r"\Lambda = " + latex_matrix(Lam))
    table(
        headers=[NoEscape(r"$y_i$"), NoEscape(r"$u_1$"), NoEscape(r"$u_2$")],
        rows=[["y1", Lam[0, 0], Lam[0, 1]], ["y2", Lam[1, 0], Lam[1, 1]]],
        caption="Relative gain array for Problem 5.",
        label="tab:problem5_rga",
    )
    p(
        "The diagonal pairings are preferred because the diagonal RGA values are positive and greater than one, while the off-diagonal values are negative. "
        "Therefore, the best pairing is y1 with u1 and y2 with u2."
    )
    return Lam


def run_problems_6_to_10():
    doc.section("Problems 6--10: MIMO Control, Cross Terms, and Decoupling")
    t = np.linspace(0, 150, 1501)
    r1_step = np.vstack([np.ones_like(t), np.zeros_like(t)])
    r2_step = np.vstack([np.zeros_like(t), np.ones_like(t)])

    # Problem 6: neglect cross terms and tune diagonal loops.
    doc.subsection("Problem 6: Two Single-Loop Controllers with Cross Terms Neglected")
    Kc11, tauI11, lam11, _ = lambda_tuning(4, 5, 5)
    Kc22, tauI22, lam22, _ = lambda_tuning(10, 3, 6)
    eq(r"K_c = \frac{\tau_p}{K_p(\lambda + \theta)}, \qquad \tau_I = \tau_p")
    table(
        headers=["Loop", NoEscape(r"$K_p$"), NoEscape(r"$\tau_p$"), NoEscape(r"$\theta$"), NoEscape(r"$\lambda$"), NoEscape(r"$K_c$"), NoEscape(r"$\tau_I$")],
        rows=[
            ["G11", 5.0, 4.0, 5.0, lam11, Kc11, tauI11],
            ["G22", 6.0, 10.0, 3.0, lam22, Kc22, tauI22],
        ],
        caption="PI tuning values for the two diagonal loops.",
        label="tab:problem6_tuning",
    )

    sys_p6 = build_two_loop_closed_system(Kc11, tauI11, Kc22, tauI22, include_cross_terms=False, decoupler="none")
    resp = ct.forced_response(sys_p6, T=t, U=r1_step, squeeze=True)
    y = np.asarray(resp.outputs)
    f61 = PLOT_DIR / "P6_no_cross_r1.png"
    save_mimo_plot(f61, resp.time, y[0], y[1], "Problem 6: no cross terms, step in r1")
    p6_r1_y1 = calculate_IAE(resp.time, y[0], 1.0)
    p6_r1_y2 = interaction_area(resp.time, y[1])

    resp = ct.forced_response(sys_p6, T=t, U=r2_step, squeeze=True)
    y = np.asarray(resp.outputs)
    f62 = PLOT_DIR / "P6_no_cross_r2.png"
    save_mimo_plot(f62, resp.time, y[0], y[1], "Problem 6: no cross terms, step in r2")
    p6_r2_y1 = interaction_area(resp.time, y[0])
    p6_r2_y2 = calculate_IAE(resp.time, y[1], 1.0)

    subfiglog(
        [(str(f61), "Step in r1"), (str(f62), "Step in r2")],
        caption="Problem 6 responses with cross terms neglected.",
        label="fig:problem6_no_cross",
        width=r"0.48\textwidth",
    )

    # Problem 7: cross terms included
    doc.subsection("Problem 7: Cross Terms Included")
    sys_p7 = build_two_loop_closed_system(Kc11, tauI11, Kc22, tauI22, include_cross_terms=True, decoupler="none")
    resp = ct.forced_response(sys_p7, T=t, U=r1_step, squeeze=True)
    y = np.asarray(resp.outputs)
    f71 = PLOT_DIR / "P7_cross_r1.png"
    save_mimo_plot(f71, resp.time, y[0], y[1], "Problem 7: cross terms included, step in r1")
    p7_r1_y1 = calculate_IAE(resp.time, y[0], 1.0)
    p7_r1_y2 = interaction_area(resp.time, y[1])

    resp = ct.forced_response(sys_p7, T=t, U=r2_step, squeeze=True)
    y = np.asarray(resp.outputs)
    f72 = PLOT_DIR / "P7_cross_r2.png"
    save_mimo_plot(f72, resp.time, y[0], y[1], "Problem 7: cross terms included, step in r2")
    p7_r2_y1 = interaction_area(resp.time, y[0])
    p7_r2_y2 = calculate_IAE(resp.time, y[1], 1.0)

    table(
        headers=["Test", "Controlled-output IAE", "Interaction area"],
        rows=[
            ["Step in r1", p7_r1_y1, p7_r1_y2],
            ["Step in r2", p7_r2_y2, p7_r2_y1],
        ],
        caption="Problem 7 performance with cross terms included.",
        label="tab:problem7_cross_terms",
    )
    subfiglog(
        [(str(f71), "Step in r1"), (str(f72), "Step in r2")],
        caption="Problem 7 responses with cross terms included.",
        label="fig:problem7_cross_terms",
        width=r"0.48\textwidth",
    )

    # Problem 8: static decoupler
    doc.subsection("Problem 8: Static Decouplers")
    eq(r"D_{12} = -\frac{K_{12}}{K_{11}} = -\frac{2}{5}, \qquad D_{21} = -\frac{K_{21}}{K_{22}} = -\frac{3}{6}")
    sys_p8 = build_two_loop_closed_system(Kc11, tauI11, Kc22, tauI22, include_cross_terms=True, decoupler="static")
    resp = ct.forced_response(sys_p8, T=t, U=r1_step, squeeze=True)
    y = np.asarray(resp.outputs)
    f81 = PLOT_DIR / "P8_static_r1.png"
    save_mimo_plot(f81, resp.time, y[0], y[1], "Problem 8: static decoupler, step in r1")
    p8_r1_y1 = calculate_IAE(resp.time, y[0], 1.0)
    p8_r1_y2 = interaction_area(resp.time, y[1])

    resp = ct.forced_response(sys_p8, T=t, U=r2_step, squeeze=True)
    y = np.asarray(resp.outputs)
    f82 = PLOT_DIR / "P8_static_r2.png"
    save_mimo_plot(f82, resp.time, y[0], y[1], "Problem 8: static decoupler, step in r2")
    p8_r2_y1 = interaction_area(resp.time, y[0])
    p8_r2_y2 = calculate_IAE(resp.time, y[1], 1.0)

    table(
        headers=["Test", "Controlled-output IAE", "Interaction area"],
        rows=[
            ["Step in r1", p8_r1_y1, p8_r1_y2],
            ["Step in r2", p8_r2_y2, p8_r2_y1],
        ],
        caption="Problem 8 performance with static decouplers.",
        label="tab:problem8_static_decouplers",
    )
    subfiglog(
        [(str(f81), "Step in r1"), (str(f82), "Step in r2")],
        caption="Problem 8 responses with static decouplers.",
        label="fig:problem8_static_decouplers",
        width=r"0.48\textwidth",
    )

    # Problem 9: dynamic decoupler
    doc.subsection("Problem 9: Dynamic Decouplers")
    a(
        r"D_{12}(s) = -\frac{G_{12}}{G_{11}} = -\frac{2}{5}\frac{4s+1}{8s+1}e^{s}",
        r"D_{21}(s) = -\frac{G_{21}}{G_{22}} = -\frac{3}{6}\frac{10s+1}{12s+1}"
    )
    p(
        "The term e^{s} in D12 is noncausal because it is equivalent to a negative delay. "
        "Therefore, the realizable dynamic decoupler uses only the rational part of D12."
    )
    eq(r"D_{12,realizable}(s) = -\frac{2}{5}\frac{4s+1}{8s+1}")

    sys_p9 = build_two_loop_closed_system(Kc11, tauI11, Kc22, tauI22, include_cross_terms=True, decoupler="dynamic")
    resp = ct.forced_response(sys_p9, T=t, U=r1_step, squeeze=True)
    y = np.asarray(resp.outputs)
    f91 = PLOT_DIR / "P9_dynamic_r1.png"
    save_mimo_plot(f91, resp.time, y[0], y[1], "Problem 9: dynamic decoupler, step in r1")
    p9_r1_y1 = calculate_IAE(resp.time, y[0], 1.0)
    p9_r1_y2 = interaction_area(resp.time, y[1])

    resp = ct.forced_response(sys_p9, T=t, U=r2_step, squeeze=True)
    y = np.asarray(resp.outputs)
    f92 = PLOT_DIR / "P9_dynamic_r2.png"
    save_mimo_plot(f92, resp.time, y[0], y[1], "Problem 9: dynamic decoupler, step in r2")
    p9_r2_y1 = interaction_area(resp.time, y[0])
    p9_r2_y2 = calculate_IAE(resp.time, y[1], 1.0)

    table(
        headers=["Test", "Controlled-output IAE", "Interaction area"],
        rows=[
            ["Step in r1", p9_r1_y1, p9_r1_y2],
            ["Step in r2", p9_r2_y2, p9_r2_y1],
        ],
        caption="Problem 9 performance with dynamic decouplers.",
        label="tab:problem9_dynamic_decouplers",
    )
    subfiglog(
        [(str(f91), "Step in r1"), (str(f92), "Step in r2")],
        caption="Problem 9 responses with dynamic decouplers.",
        label="fig:problem9_dynamic_decouplers",
        width=r"0.48\textwidth",
    )

    # Problem 10 comments and summary table
    doc.subsection("Problem 10: Comments on MIMO Results")
    table(
        headers=["Case", "Step in r1: IAE y1", "Step in r1: y2 area", "Step in r2: y1 area", "Step in r2: IAE y2"],
        rows=[
            ["No cross terms", p6_r1_y1, p6_r1_y2, p6_r2_y1, p6_r2_y2],
            ["Cross terms", p7_r1_y1, p7_r1_y2, p7_r2_y1, p7_r2_y2],
            ["Static decoupler", p8_r1_y1, p8_r1_y2, p8_r2_y1, p8_r2_y2],
            ["Dynamic decoupler", p9_r1_y1, p9_r1_y2, p9_r2_y1, p9_r2_y2],
        ],
        caption="Summary of MIMO response metrics for Problems 6--10.",
        label="tab:problem10_summary",
    )
    p(
        "When the cross terms are added, a change in one loop causes movement in the other output, showing loop interaction. "
        "Static decouplers reduce the steady-state interaction because they cancel the cross-gains at zero frequency. "
        "However, static decoupling cannot remove all transient interaction because the cross terms have different delays and time constants. "
        "Dynamic decouplers account for those dynamics and should reduce interaction further, but any noncausal negative-delay terms must be removed or approximated with a realizable transfer function."
    )


# =============================================================================
# Main execution
# =============================================================================
def main():
    write_intro()
    run_problems_1_to_3()
    run_problem_4()
    run_problem_5()
    run_problems_6_to_10()

    txt_file, tex_file, pdf_file = doc.save_all(runs=2)
    print(f"Wrote text log: {txt_file}")
    print(f"Wrote LaTeX file: {tex_file}")
    print(f"Wrote PDF report: {pdf_file}")


if __name__ == "__main__":
    main()
