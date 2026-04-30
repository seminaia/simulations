from pathlib import Path

code = r'''"""
HW5_CHE565_full.py
==================
CHE 565 – Homework 5
Cascade Control, RGA, MIMO control, and decoupling.

This script is written to generate the main numerical results and plots for
Problems 1–10 of Homework 5.

It uses python-control to reproduce the Simulink-style block diagrams.

Outputs:
    - PNG plots for cascade-control and MIMO cases
    - HW5_results.txt summary file

Notes:
    - Inner cascade controller is P-only and represented as ct.tf([Kc2], [1])
      so it can be converted cleanly with ct.ss().
    - Outer controller is ideal PI: Kc * (1 + 1/(tauI*s)).
    - Delay terms for Problems 5–10 are approximated using Pade approximation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.optimize import minimize


# =============================================================================
# General utilities
# =============================================================================

OUT_TXT = "HW5_results.txt"


def iae(t: np.ndarray, y: np.ndarray, r: np.ndarray | float) -> float:
    """Integral absolute error."""
    if np.isscalar(r):
        r_vec = np.ones_like(t) * float(r)
    else:
        r_vec = np.asarray(r)
    return float(np.trapezoid(np.abs(r_vec - y), t))


def save_response_plot(
    filename: str,
    t: np.ndarray,
    y: np.ndarray,
    title: str,
    setpoint: np.ndarray | float | None = None,
    disturbance: np.ndarray | float | None = None,
    ylabel: str = "Response",
):
    """Save a single response plot."""
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, y, linewidth=2, label="output")

    if setpoint is not None:
        if np.isscalar(setpoint):
            sp = np.ones_like(t) * float(setpoint)
        else:
            sp = np.asarray(setpoint)
        plt.plot(t, sp, "--", linewidth=1.5, label="setpoint")

    if disturbance is not None:
        if np.isscalar(disturbance):
            d = np.ones_like(t) * float(disturbance)
        else:
            d = np.asarray(disturbance)
        plt.plot(t, d, ":", linewidth=1.5, label="disturbance")

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


def safe_forced_response(sys, t, u):
    """Return time, output, state for a forced response."""
    resp = ct.forced_response(sys, T=t, U=u, return_states=True, squeeze=True)
    return resp.time, np.asarray(resp.outputs), np.asarray(resp.states)


# =============================================================================
# Problems 1–3: Cascade control system
# =============================================================================

Kp1 = 5.0
taup1 = 5.0
Kp2 = 2.0
taup2 = 10.0


def ideal_pi(Kc: float, tauI: float):
    """Ideal PI controller Gc = Kc * (1 + 1/(tauI*s))."""
    s = ct.tf("s")
    return Kc * (1 + 1 / (tauI * s))


def p_controller(Kc: float):
    """P-only controller as transfer function, so ct.ss() works cleanly."""
    return ct.tf([Kc], [1])


def build_cascade_system(
    Kc_outer: float,
    tauI_outer: float,
    Kc_inner: float = 1.0,
    cascade: bool = False,
):
    """
    Build the homework cascade system.

    External inputs:
        Ysp = setpoint
        D   = disturbance added after Gp1 and before Gp2

    Output:
        Y

    Without cascade:
        E2 = Yc1, so the inner feedback is removed.
        If Kc_inner = 1, the inner controller is a pass-through.

    With cascade:
        E2 = Yc1 - P, where P is the measured intermediate variable.
    """
    s = ct.tf("s")

    Gc1 = ideal_pi(Kc_outer, tauI_outer)
    Gc2 = p_controller(Kc_inner)

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
        # Inner feedback is disconnected. With Kc_inner = 1, this is pass-through.
        sum2 = ct.summing_junction(inputs=["Yc1"], output="E2", name="Sum2")

    sum3 = ct.summing_junction(inputs=["Yp1", "Yd"], output="P", name="Sum3")

    sys = ct.interconnect(
        [Gc1_blk, Gc2_blk, Gp1_blk, Gp2_blk, Gd_blk, sum1, sum2, sum3],
        inputs=["Ysp", "D"],
        outputs=["Y"],
    )
    return sys


def simulate_hw5_cascade(sys, t, ysp, d):
    U = np.vstack([ysp, d])
    tout, y, x = safe_forced_response(sys, t, U)
    y = np.ravel(y)
    return tout, y, x


def optimize_outer_pi(
    cascade: bool,
    Kc_inner: float,
    t: np.ndarray,
    ysp: np.ndarray,
    d: np.ndarray,
    initial=(1.0, 1.0),
):
    """
    Tune outer PI controller by minimizing IAE.

    Parameters are optimized in log-space so Kc and tauI remain positive.
    """
    ysp = np.asarray(ysp)
    d = np.asarray(d)

    def objective(log_params):
        Kc = np.exp(log_params[0])
        tauI = np.exp(log_params[1])
        try:
            sys = build_cascade_system(Kc, tauI, Kc_inner=Kc_inner, cascade=cascade)
            tout, y, _ = simulate_hw5_cascade(sys, t, ysp, d)
            if np.any(~np.isfinite(y)) or np.max(np.abs(y)) > 1e8:
                return 1e12
            return iae(tout, y, ysp)
        except Exception:
            return 1e12

    result = minimize(
        objective,
        np.log(np.asarray(initial, dtype=float)),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )

    Kc_opt = float(np.exp(result.x[0]))
    tauI_opt = float(np.exp(result.x[1]))
    return Kc_opt, tauI_opt, result.fun


def run_problems_1_to_3(report):
    report.append("\n" + "=" * 80)
    report.append("Problems 1–3: Cascade control")
    report.append("=" * 80)

    t = np.linspace(0, 100, 1001)
    step_on = np.ones_like(t)
    step_off = np.zeros_like(t)

    # Problem 1: without cascade, inner controller gain = 1.
    Kc_inner_no_cascade = 1.0

    # Tune outer PI for disturbance case.
    Kc_nc_d, tauI_nc_d, _ = optimize_outer_pi(
        cascade=False,
        Kc_inner=Kc_inner_no_cascade,
        t=t,
        ysp=step_off,
        d=step_on,
        initial=(0.2, 10.0),
    )

    sys_nc_d = build_cascade_system(
        Kc_nc_d, tauI_nc_d, Kc_inner=Kc_inner_no_cascade, cascade=False
    )
    t_nc_d, y_nc_d, _ = simulate_hw5_cascade(sys_nc_d, t, step_off, step_on)
    iae_nc_d = iae(t_nc_d, y_nc_d, step_off)

    save_response_plot(
        "P1_no_cascade_disturbance.png",
        t_nc_d,
        y_nc_d,
        "Problem 1: No cascade, unit step disturbance",
        setpoint=step_off,
        disturbance=step_on,
    )

    # Use the same no-cascade tuned outer controller for setpoint test.
    sys_nc_sp = build_cascade_system(
        Kc_nc_d, tauI_nc_d, Kc_inner=Kc_inner_no_cascade, cascade=False
    )
    t_nc_sp, y_nc_sp, _ = simulate_hw5_cascade(sys_nc_sp, t, step_on, step_off)
    iae_nc_sp = iae(t_nc_sp, y_nc_sp, step_on)

    save_response_plot(
        "P1_no_cascade_setpoint.png",
        t_nc_sp,
        y_nc_sp,
        "Problem 1: No cascade, unit step setpoint change",
        setpoint=step_on,
        disturbance=step_off,
    )

    report.append("Problem 1: Without cascade")
    report.append(f"  Inner P gain = {Kc_inner_no_cascade:.4g}")
    report.append(f"  Tuned outer PI: Kc = {Kc_nc_d:.6g}, tauI = {tauI_nc_d:.6g}")
    report.append(f"  Disturbance IAE = {iae_nc_d:.6g}")
    report.append(f"  Setpoint IAE    = {iae_nc_sp:.6g}")

    # Problem 2: with cascade, inner controller gain = 0.4.
    Kc_inner_cascade = 0.4

    # Tune outer PI again because the outer loop sees a different equivalent process.
    Kc_c_d, tauI_c_d, _ = optimize_outer_pi(
        cascade=True,
        Kc_inner=Kc_inner_cascade,
        t=t,
        ysp=step_off,
        d=step_on,
        initial=(0.2, 10.0),
    )

    sys_c_d = build_cascade_system(
        Kc_c_d, tauI_c_d, Kc_inner=Kc_inner_cascade, cascade=True
    )
    t_c_d, y_c_d, _ = simulate_hw5_cascade(sys_c_d, t, step_off, step_on)
    iae_c_d = iae(t_c_d, y_c_d, step_off)

    save_response_plot(
        "P2_cascade_disturbance.png",
        t_c_d,
        y_c_d,
        "Problem 2: Cascade control, unit step disturbance",
        setpoint=step_off,
        disturbance=step_on,
    )

    # Use same cascade-tuned controller for setpoint test.
    sys_c_sp = build_cascade_system(
        Kc_c_d, tauI_c_d, Kc_inner=Kc_inner_cascade, cascade=True
    )
    t_c_sp, y_c_sp, _ = simulate_hw5_cascade(sys_c_sp, t, step_on, step_off)
    iae_c_sp = iae(t_c_sp, y_c_sp, step_on)

    save_response_plot(
        "P2_cascade_setpoint.png",
        t_c_sp,
        y_c_sp,
        "Problem 2: Cascade control, unit step setpoint change",
        setpoint=step_on,
        disturbance=step_off,
    )

    report.append("\nProblem 2: With cascade")
    report.append(f"  Inner P gain = {Kc_inner_cascade:.4g}")
    report.append(f"  Tuned outer PI: Kc = {Kc_c_d:.6g}, tauI = {tauI_c_d:.6g}")
    report.append(f"  Disturbance IAE = {iae_c_d:.6g}")
    report.append(f"  Setpoint IAE    = {iae_c_sp:.6g}")

    report.append("\nProblem 3: Comment")
    report.append(
        "  Cascade control should mainly improve disturbance rejection because the "
        "inner loop measures the intermediate variable P and reacts before the "
        "disturbance fully propagates through Gp2. The setpoint response may change, "
        "but the largest improvement is usually expected for disturbances entering "
        "inside the cascade structure."
    )

    return {
        "no_cascade_disturbance_iae": iae_nc_d,
        "no_cascade_setpoint_iae": iae_nc_sp,
        "cascade_disturbance_iae": iae_c_d,
        "cascade_setpoint_iae": iae_c_sp,
    }


# =============================================================================
# Problems 4–5: Relative Gain Array
# =============================================================================

def rga(K: np.ndarray) -> np.ndarray:
    """
    Relative gain array:
        Lambda = K .* (K^{-1})^T
    where .* is element-by-element multiplication.
    """
    K = np.asarray(K, dtype=float)
    return K * np.linalg.inv(K).T


def best_pairing_from_rga(Lambda: np.ndarray):
    """
    Simple greedy pairing using largest positive RGA value in each row/column.

    For homework explanation, also inspect signs and values close to 1.
    """
    L = np.asarray(Lambda)
    n_rows, n_cols = L.shape
    unused_rows = set(range(n_rows))
    unused_cols = set(range(n_cols))
    pairs = []

    while unused_rows and unused_cols:
        best = None
        best_score = -np.inf
        for i in unused_rows:
            for j in unused_cols:
                # Prefer values close to +1 and positive.
                if L[i, j] > 0:
                    score = -abs(L[i, j] - 1)
                else:
                    score = -1e6 - abs(L[i, j])
                if score > best_score:
                    best_score = score
                    best = (i, j)
        i, j = best
        pairs.append((i, j, L[i, j]))
        unused_rows.remove(i)
        unused_cols.remove(j)

    return pairs


def run_problems_4_and_5(report):
    report.append("\n" + "=" * 80)
    report.append("Problems 4–5: Relative Gain Array")
    report.append("=" * 80)

    K4 = np.array(
        [
            [0.43, 0.43, 0.23, 0.22],
            [-0.33, 0.32, -0.20, 0.20],
            [0.22, 0.23, 0.42, 0.41],
            [-0.22, 0.22, -0.32, 0.32],
        ],
        dtype=float,
    )

    Lambda4 = rga(K4)
    pairs4 = best_pairing_from_rga(Lambda4)

    report.append("Problem 4 gain matrix K:")
    report.append(str(K4))
    report.append("\nProblem 4 RGA:")
    report.append(np.array2string(Lambda4, precision=4, suppress_small=True))
    report.append("Suggested pairing, using row/output i with column/input j:")
    for i, j, val in pairs4:
        report.append(f"  y{i+1} with u{j+1}, lambda = {val:.4f}")

    # Problem 5 uses steady-state gains from the transfer functions.
    K5 = np.array(
        [
            [5.0, 2.0],
            [3.0, 6.0],
        ],
        dtype=float,
    )
    Lambda5 = rga(K5)
    pairs5 = best_pairing_from_rga(Lambda5)

    report.append("\nProblem 5 steady-state gain matrix K:")
    report.append(str(K5))
    report.append("\nProblem 5 RGA:")
    report.append(np.array2string(Lambda5, precision=4, suppress_small=True))
    report.append("Suggested pairing:")
    for i, j, val in pairs5:
        report.append(f"  y{i+1} with u{j+1}, lambda = {val:.4f}")

    return Lambda4, Lambda5


# =============================================================================
# Problems 6–10: 2x2 MIMO system with cross terms and decouplers
# =============================================================================

def fopdt(K: float, tau: float, theta: float, pade_order: int = 1):
    """First-order-plus-dead-time transfer function using Pade delay."""
    s = ct.tf("s")
    num_delay, den_delay = ct.pade(theta, pade_order)
    delay_tf = ct.tf(num_delay, den_delay)
    return K * delay_tf / (tau * s + 1)


def build_mimo_process(include_cross_terms=True, pade_order=1):
    """
    2x2 process:
        G11 = 5 e^-5s / (4s + 1)
        G12 = 2 e^-4s / (8s + 1)
        G21 = 3 e^-3s / (12s + 1)
        G22 = 6 e^-3s / (10s + 1)
    """
    G11 = fopdt(5, 4, 5, pade_order)
    G12 = fopdt(2, 8, 4, pade_order) if include_cross_terms else ct.tf([0], [1])
    G21 = fopdt(3, 12, 3, pade_order) if include_cross_terms else ct.tf([0], [1])
    G22 = fopdt(6, 10, 3, pade_order)

    # Use named I/O systems and summing junctions to avoid MIMO tf conversion issues.
    G11_blk = ct.ss(G11, name="G11", inputs="u1p", outputs="y11")
    G12_blk = ct.ss(G12, name="G12", inputs="u2p", outputs="y12")
    G21_blk = ct.ss(G21, name="G21", inputs="u1p", outputs="y21")
    G22_blk = ct.ss(G22, name="G22", inputs="u2p", outputs="y22")

    sum_y1 = ct.summing_junction(inputs=["y11", "y12"], output="y1", name="sum_y1")
    sum_y2 = ct.summing_junction(inputs=["y21", "y22"], output="y2", name="sum_y2")

    P = ct.interconnect(
        [G11_blk, G12_blk, G21_blk, G22_blk, sum_y1, sum_y2],
        inputs=["u1p", "u2p"],
        outputs=["y1", "y2"],
        name="P",
    )
    return P


def build_two_loop_closed_system(
    Kc1: float,
    tauI1: float,
    Kc2: float,
    tauI2: float,
    include_cross_terms=True,
    decoupler="none",
    pade_order=1,
):
    """
    Build a closed-loop 2x2 system with two PI controllers.

    Inputs:
        r1, r2

    Outputs:
        y1, y2

    decoupler:
        "none"
        "static"
        "dynamic"
    """
    s = ct.tf("s")

    C1 = ideal_pi(Kc1, tauI1)
    C2 = ideal_pi(Kc2, tauI2)

    C1_blk = ct.ss(C1, name="C1", inputs="e1", outputs="v1")
    C2_blk = ct.ss(C2, name="C2", inputs="e2", outputs="v2")

    sum_e1 = ct.summing_junction(inputs=["r1", "-y1"], output="e1", name="sum_e1")
    sum_e2 = ct.summing_junction(inputs=["r2", "-y2"], output="e2", name="sum_e2")

    blocks = [C1_blk, C2_blk, sum_e1, sum_e2]

    if decoupler == "none":
        # Controller outputs go directly to process inputs.
        pass1 = ct.ss(ct.tf([1], [1]), name="D11", inputs="v1", outputs="u1p")
        pass2 = ct.ss(ct.tf([1], [1]), name="D22", inputs="v2", outputs="u2p")
        blocks += [pass1, pass2]

    elif decoupler == "static":
        # For diagonal pairing, static decoupler:
        # u1p = v1 - (K12/K11) v2
        # u2p = v2 - (K21/K22) v1
        D11 = ct.tf([1], [1])
        D12 = ct.tf([-2 / 5], [1])
        D21 = ct.tf([-3 / 6], [1])
        D22 = ct.tf([1], [1])

        blocks += [
            ct.ss(D11, name="D11", inputs="v1", outputs="u11"),
            ct.ss(D12, name="D12", inputs="v2", outputs="u12"),
            ct.ss(D21, name="D21", inputs="v1", outputs="u21"),
            ct.ss(D22, name="D22", inputs="v2", outputs="u22"),
            ct.summing_junction(inputs=["u11", "u12"], output="u1p", name="sum_u1"),
            ct.summing_junction(inputs=["u21", "u22"], output="u2p", name="sum_u2"),
        ]

    elif decoupler == "dynamic":
        # Dynamic decouplers for diagonal pairing:
        # D12 = -G12/G11
        # D21 = -G21/G22
        #
        # G12/G11 = (2/5)*((4s+1)/(8s+1))*exp(+s)
        # This contains exp(+s), which is noncausal, so remove the positive delay
        # and use the realizable part:
        # D12 = -(2/5)*((4s+1)/(8s+1))
        #
        # G21/G22 = (3/6)*((10s+1)/(12s+1))*exp(0s)
        D11 = ct.tf([1], [1])
        D22 = ct.tf([1], [1])
        D12 = -(2 / 5) * (4 * s + 1) / (8 * s + 1)
        D21 = -(3 / 6) * (10 * s + 1) / (12 * s + 1)

        blocks += [
            ct.ss(D11, name="D11", inputs="v1", outputs="u11"),
            ct.ss(D12, name="D12", inputs="v2", outputs="u12"),
            ct.ss(D21, name="D21", inputs="v1", outputs="u21"),
            ct.ss(D22, name="D22", inputs="v2", outputs="u22"),
            ct.summing_junction(inputs=["u11", "u12"], output="u1p", name="sum_u1"),
            ct.summing_junction(inputs=["u21", "u22"], output="u2p", name="sum_u2"),
        ]

    else:
        raise ValueError("decoupler must be 'none', 'static', or 'dynamic'")

    P = build_mimo_process(include_cross_terms=include_cross_terms, pade_order=pade_order)
    blocks.append(P)

    sys = ct.interconnect(
        blocks,
        inputs=["r1", "r2"],
        outputs=["y1", "y2"],
        name=f"closed_{decoupler}",
    )
    return sys


def optimize_single_loop_pi(G, t, initial=(0.3, 10.0)):
    """
    Tune PI for a SISO process by minimizing setpoint IAE.
    """
    r = np.ones_like(t)

    def objective(log_params):
        Kc = np.exp(log_params[0])
        tauI = np.exp(log_params[1])
        C = ideal_pi(Kc, tauI)
        sys_cl = ct.feedback(C * G, 1)
        try:
            resp = ct.forced_response(sys_cl, T=t, U=r, squeeze=True)
            y = np.ravel(resp.outputs)
            if np.any(~np.isfinite(y)) or np.max(np.abs(y)) > 1e8:
                return 1e12
            return iae(resp.time, y, r)
        except Exception:
            return 1e12

    result = minimize(
        objective,
        np.log(np.asarray(initial, dtype=float)),
        method="Nelder-Mead",
        options={"maxiter": 1000},
    )
    return float(np.exp(result.x[0])), float(np.exp(result.x[1]))


def save_mimo_plot(filename, t, y1, y2, title):
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, y1, linewidth=2, label="y1")
    plt.plot(t, y2, linewidth=2, label="y2")
    plt.xlabel("Time")
    plt.ylabel("Outputs")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


def run_problems_6_to_10(report):
    report.append("\n" + "=" * 80)
    report.append("Problems 6–10: MIMO loops and decoupling")
    report.append("=" * 80)

    t = np.linspace(0, 150, 1501)

    # Problem 6: neglect cross terms and tune two independent loops.
    G11 = fopdt(5, 4, 5, pade_order=1)
    G22 = fopdt(6, 10, 3, pade_order=1)

    Kc1, tauI1 = optimize_single_loop_pi(G11, t, initial=(0.2, 8.0))
    Kc2, tauI2 = optimize_single_loop_pi(G22, t, initial=(0.2, 10.0))

    report.append("Problem 6: PI tuning with cross terms neglected")
    report.append(f"  Loop 1 controller: Kc1 = {Kc1:.6g}, tauI1 = {tauI1:.6g}")
    report.append(f"  Loop 2 controller: Kc2 = {Kc2:.6g}, tauI2 = {tauI2:.6g}")

    sys_no_cross = build_two_loop_closed_system(
        Kc1, tauI1, Kc2, tauI2, include_cross_terms=False, decoupler="none"
    )

    r1_step = np.vstack([np.ones_like(t), np.zeros_like(t)])
    r2_step = np.vstack([np.zeros_like(t), np.ones_like(t)])

    resp = ct.forced_response(sys_no_cross, T=t, U=r1_step, squeeze=True)
    y = np.asarray(resp.outputs)
    save_mimo_plot(
        "P6_no_cross_r1_step.png",
        resp.time,
        y[0],
        y[1],
        "Problem 6: No cross terms, step in r1",
    )

    resp = ct.forced_response(sys_no_cross, T=t, U=r2_step, squeeze=True)
    y = np.asarray(resp.outputs)
    save_mimo_plot(
        "P6_no_cross_r2_step.png",
        resp.time,
        y[0],
        y[1],
        "Problem 6: No cross terms, step in r2",
    )

    # Problem 7: add cross terms.
    sys_cross = build_two_loop_closed_system(
        Kc1, tauI1, Kc2, tauI2, include_cross_terms=True, decoupler="none"
    )

    resp = ct.forced_response(sys_cross, T=t, U=r1_step, squeeze=True)
    y = np.asarray(resp.outputs)
    save_mimo_plot(
        "P7_cross_terms_r1_step.png",
        resp.time,
        y[0],
        y[1],
        "Problem 7: Cross terms included, step in r1",
    )

    report.append("\nProblem 7: Cross terms included")
    report.append(f"  Step in r1: IAE y1 = {iae(resp.time, y[0], 1.0):.6g}, "
                  f"interaction area y2 = {np.trapezoid(np.abs(y[1]), resp.time):.6g}")

    resp = ct.forced_response(sys_cross, T=t, U=r2_step, squeeze=True)
    y = np.asarray(resp.outputs)
    save_mimo_plot(
        "P7_cross_terms_r2_step.png",
        resp.time,
        y[0],
        y[1],
        "Problem 7: Cross terms included, step in r2",
    )

    report.append(f"  Step in r2: interaction area y1 = {np.trapezoid(np.abs(y[0]), resp.time):.6g}, "
                  f"IAE y2 = {iae(resp.time, y[1], 1.0):.6g}")

    # Problem 8: static decouplers.
    sys_static = build_two_loop_closed_system(
        Kc1, tauI1, Kc2, tauI2, include_cross_terms=True, decoupler="static"
    )

    resp = ct.forced_response(sys_static, T=t, U=r1_step, squeeze=True)
    y = np.asarray(resp.outputs)
    save_mimo_plot(
        "P8_static_decoupler_r1_step.png",
        resp.time,
        y[0],
        y[1],
        "Problem 8: Static decoupler, step in r1",
    )

    report.append("\nProblem 8: Static decouplers")
    report.append(f"  Step in r1: IAE y1 = {iae(resp.time, y[0], 1.0):.6g}, "
                  f"interaction area y2 = {np.trapezoid(np.abs(y[1]), resp.time):.6g}")

    resp = ct.forced_response(sys_static, T=t, U=r2_step, squeeze=True)
    y = np.asarray(resp.outputs)
    save_mimo_plot(
        "P8_static_decoupler_r2_step.png",
        resp.time,
        y[0],
        y[1],
        "Problem 8: Static decoupler, step in r2",
    )

    report.append(f"  Step in r2: interaction area y1 = {np.trapezoid(np.abs(y[0]), resp.time):.6g}, "
                  f"IAE y2 = {iae(resp.time, y[1], 1.0):.6g}")

    # Problem 9: dynamic decouplers.
    sys_dynamic = build_two_loop_closed_system(
        Kc1, tauI1, Kc2, tauI2, include_cross_terms=True, decoupler="dynamic"
    )

    resp = ct.forced_response(sys_dynamic, T=t, U=r1_step, squeeze=True)
    y = np.asarray(resp.outputs)
    save_mimo_plot(
        "P9_dynamic_decoupler_r1_step.png",
        resp.time,
        y[0],
        y[1],
        "Problem 9: Dynamic decoupler, step in r1",
    )

    report.append("\nProblem 9: Dynamic decouplers")
    report.append(f"  Step in r1: IAE y1 = {iae(resp.time, y[0], 1.0):.6g}, "
                  f"interaction area y2 = {np.trapezoid(np.abs(y[1]), resp.time):.6g}")

    resp = ct.forced_response(sys_dynamic, T=t, U=r2_step, squeeze=True)
    y = np.asarray(resp.outputs)
    save_mimo_plot(
        "P9_dynamic_decoupler_r2_step.png",
        resp.time,
        y[0],
        y[1],
        "Problem 9: Dynamic decoupler, step in r2",
    )

    report.append(f"  Step in r2: interaction area y1 = {np.trapezoid(np.abs(y[0]), resp.time):.6g}, "
                  f"IAE y2 = {iae(resp.time, y[1], 1.0):.6g}")

    report.append("\nProblem 10: Comment")
    report.append(
        "  Adding cross terms produces loop interaction: a step in one setpoint also "
        "moves the other output. Static decouplers reduce steady-state interaction, "
        "but they do not fully correct dynamic mismatch because each transfer function "
        "has a different time constant and delay. Dynamic decouplers should reduce the "
        "interaction more effectively, but any noncausal positive-delay term must be "
        "removed or approximated by a realizable transfer function."
    )


# =============================================================================
# Main
# =============================================================================

def main():
    report = []
    run_problems_1_to_3(report)
    run_problems_4_and_5(report)
    run_problems_6_to_10(report)

    text = "\n".join(report)
    print(text)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/HW5_CHE565_full.py")
path.write_text(code, encoding="utf-8")
print(f"Created {path}")
