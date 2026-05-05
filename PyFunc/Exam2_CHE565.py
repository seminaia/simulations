import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import control as ct
from pathlib import Path

from doc_builder import DocumentBuilder


# =============================================================================
# Global output settings
# =============================================================================
OUTPUT_FILE = "Exam2_CHE565"
PLOT_DIR = Path("Exam2_plots")
PLOT_DIR.mkdir(exist_ok=True)

T_FINAL = 500
NPTS = 2000
T = np.linspace(0, T_FINAL, NPTS)

s_sym = sp.symbols("s")


# =============================================================================
# Report setup
# =============================================================================
doc = DocumentBuilder(
    OUTPUT_FILE,
    title="CHE 565 -- Exam 2",
    author="Soki Sem",
)

p = doc.p
line = doc.line
eq = doc.eq
a = doc.align
figlog = doc.figure
subfiglog = doc.subfigures
px = doc.px
im = doc.im


# =============================================================================
# Utility functions
# =============================================================================
def save_plot(filename, t, y, title, ysp=None, ylabel="Response"):
    filename = str(filename)
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, y, label="Output", linewidth=2)

    if ysp is not None:
        ysp_vec = np.ones_like(t) * ysp if np.isscalar(ysp) else np.asarray(ysp)
        plt.plot(t, ysp_vec, "--", label="Reference", linewidth=1.5)

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()


def calculate_IAE(t, y, target):
    y = np.asarray(y).reshape(-1)

    if np.isscalar(target):
        target_vec = np.ones_like(t) * float(target)
    else:
        target_vec = np.asarray(target).reshape(-1)

    return float(np.trapezoid(np.abs(target_vec - y), t))


def pade_delay(theta, order=1):
    num, den = ct.delay.pade(theta, order)
    return ct.tf(num, den)


def latex_tf(tf_expr):
    return sp.latex(sp.simplify(tf_expr))


def forced_two_input_response(sys, ysp_value, d_value):
    """
    System input order is always [Ysp, D].
    """
    U = np.vstack([
        np.ones_like(T) * ysp_value,
        np.ones_like(T) * d_value,
    ])

    resp = ct.forced_response(sys, T=T, U=U)
    y = np.asarray(resp.outputs).squeeze()

    return resp.time, y


# =============================================================================
# Simple block-diagram drawing helpers
# =============================================================================
def draw_box(ax, xy, w, h, text):
    x, y = xy
    rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, start, end, text=None, yoff=0.08):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + yoff,
            text,
            ha="center",
            va="center",
            fontsize=8,
        )


def save_basic_feedback_diagram(filename):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    ax.text(1.0, 2.1, r"$+$", fontsize=13)
    ax.text(1.0, 1.65, r"$-$", fontsize=13)
    ax.text(1.2, 1.85, r"$\Sigma$", fontsize=12)

    draw_box(ax, (2.0, 1.4), 1.2, 0.9, r"$G_c$")
    draw_box(ax, (4.0, 1.4), 1.5, 0.9, r"$G_p$")
    draw_box(ax, (3.6, 2.7), 1.4, 0.7, r"$G_d$")

    arrow(ax, (0.2, 1.9), (1.0, 1.9), r"$Y_{sp}$")
    arrow(ax, (1.35, 1.9), (2.0, 1.9), r"$E$")
    arrow(ax, (3.2, 1.9), (4.0, 1.9), r"$M$")
    arrow(ax, (5.5, 1.9), (7.0, 1.9), r"$Y$")

    arrow(ax, (2.8, 3.05), (3.6, 3.05), r"$D$")
    arrow(ax, (5.0, 3.05), (5.0, 2.35))
    arrow(ax, (6.7, 1.9), (6.7, 0.65))
    arrow(ax, (6.7, 0.65), (1.1, 0.65))
    arrow(ax, (1.1, 0.65), (1.1, 1.65))

    ax.set_title("Basic feedback control with input-side disturbance")
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()


def save_cascade_diagram(filename):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)

    ax.text(1.0, 2.5, r"$\Sigma$", fontsize=12)
    ax.text(1.0, 2.75, r"$+$", fontsize=12)
    ax.text(1.0, 2.25, r"$-$", fontsize=12)

    ax.text(3.0, 2.5, r"$\Sigma$", fontsize=12)
    ax.text(3.0, 2.75, r"$+$", fontsize=12)
    ax.text(3.0, 2.25, r"$-$", fontsize=12)

    ax.text(6.0, 3.8, r"$\Sigma$", fontsize=12)

    draw_box(ax, (1.6, 2.1), 1.1, 0.8, r"$G_{c,o}$")
    draw_box(ax, (3.6, 2.1), 1.1, 0.8, r"$G_{c,i}$")
    draw_box(ax, (5.2, 2.1), 1.3, 0.8, r"$G_{p1}$")
    draw_box(ax, (7.1, 2.1), 1.3, 0.8, r"$G_{p2}$")
    draw_box(ax, (4.9, 3.6), 1.0, 0.6, r"$G_d$")

    arrow(ax, (0.2, 2.5), (1.0, 2.5), r"$Y_{sp}$")
    arrow(ax, (1.2, 2.5), (1.6, 2.5), r"$E_o$")
    arrow(ax, (2.7, 2.5), (3.0, 2.5), r"$V_{sp}$")
    arrow(ax, (3.2, 2.5), (3.6, 2.5), r"$E_i$")
    arrow(ax, (4.7, 2.5), (5.2, 2.5), r"$M$")
    arrow(ax, (6.5, 2.5), (7.1, 2.5), r"$V$")
    arrow(ax, (8.4, 2.5), (10.2, 2.5), r"$Y$")

    arrow(ax, (4.2, 3.9), (4.9, 3.9), r"$D$")
    arrow(ax, (5.9, 3.9), (6.0, 3.0))

    arrow(ax, (6.8, 2.5), (6.8, 1.2))
    arrow(ax, (6.8, 1.2), (3.05, 1.2))
    arrow(ax, (3.05, 1.2), (3.05, 2.25), r"$V$")

    arrow(ax, (9.8, 2.5), (9.8, 0.55))
    arrow(ax, (9.8, 0.55), (1.05, 0.55))
    arrow(ax, (1.05, 0.55), (1.05, 2.25), r"$Y$")

    ax.set_title("Cascade control structure")
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()


def save_feedforward_diagram(filename):
    fig, ax = plt.subplots(figsize=(10, 3.7))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    ax.text(1.0, 1.9, r"$\Sigma$", fontsize=12)
    ax.text(1.0, 2.15, r"$+$", fontsize=12)
    ax.text(1.0, 1.65, r"$-$", fontsize=12)

    ax.text(4.3, 1.9, r"$\Sigma$", fontsize=12)
    ax.text(4.3, 2.15, r"$+$", fontsize=12)
    ax.text(4.3, 1.65, r"$+$", fontsize=12)

    draw_box(ax, (1.7, 1.5), 1.2, 0.8, r"$G_c$")
    draw_box(ax, (3.0, 2.7), 1.2, 0.7, r"$G_{ff}$")
    draw_box(ax, (5.0, 1.5), 1.4, 0.8, r"$G_p$")
    draw_box(ax, (5.0, 2.7), 1.2, 0.7, r"$G_d$")

    arrow(ax, (0.2, 1.9), (1.0, 1.9), r"$Y_{sp}$")
    arrow(ax, (1.2, 1.9), (1.7, 1.9), r"$E$")
    arrow(ax, (2.9, 1.9), (4.3, 1.9), r"$M_{fb}$")
    arrow(ax, (4.5, 1.9), (5.0, 1.9), r"$M$")
    arrow(ax, (6.4, 1.9), (8.2, 1.9), r"$Y$")

    arrow(ax, (2.1, 3.05), (3.0, 3.05), r"$D$")
    arrow(ax, (4.2, 3.05), (4.35, 2.05), r"$M_{ff}$")

    arrow(ax, (4.3, 3.05), (5.0, 3.05))
    arrow(ax, (6.2, 3.05), (6.2, 2.3))

    arrow(ax, (7.9, 1.9), (7.9, 0.7))
    arrow(ax, (7.9, 0.7), (1.05, 0.7))
    arrow(ax, (1.05, 0.7), (1.05, 1.65))

    ax.set_title("Feedback plus feedforward disturbance compensation")
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()


def save_mimo_diagram(filename, decoupled=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    draw_box(ax, (4.2, 2.4), 1.7, 2.0, "MIMO\nProcess")

    draw_box(ax, (1.2, 4.8), 1.2, 0.7, r"$G_{c1}$")
    draw_box(ax, (1.2, 1.5), 1.2, 0.7, r"$G_{c2}$")

    if decoupled:
        draw_box(ax, (2.8, 4.8), 1.0, 0.7, r"$D_{12}$")
        draw_box(ax, (2.8, 1.5), 1.0, 0.7, r"$D_{21}$")
        ax.text(5.05, 4.05, r"$G_{11},G_{12}$", ha="center", fontsize=9)
        ax.text(5.05, 2.75, r"$G_{21},G_{22}$", ha="center", fontsize=9)
    else:
        ax.text(5.05, 4.05, r"$G_{11},G_{12}$", ha="center", fontsize=9)
        ax.text(5.05, 2.75, r"$G_{21},G_{22}$", ha="center", fontsize=9)

    arrow(ax, (0.1, 5.15), (1.2, 5.15), r"$Y_{1,sp}$")
    arrow(ax, (2.4, 5.15), (4.2, 4.0), r"$M_1$")
    arrow(ax, (0.1, 1.85), (1.2, 1.85), r"$Y_{2,sp}$")
    arrow(ax, (2.4, 1.85), (4.2, 2.8), r"$M_2$")

    arrow(ax, (5.9, 4.0), (8.6, 5.15), r"$Y_1$")
    arrow(ax, (5.9, 2.8), (8.6, 1.85), r"$Y_2$")

    if decoupled:
        arrow(ax, (2.4, 5.15), (2.8, 5.15))
        arrow(ax, (3.8, 5.15), (4.0, 2.95), r"cross")
        arrow(ax, (2.4, 1.85), (2.8, 1.85))
        arrow(ax, (3.8, 1.85), (4.0, 3.85), r"cross")

    ax.set_title("MIMO feedback structure" + (" with decouplers" if decoupled else ""))
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()


# =============================================================================
# Controller tuning
# =============================================================================
def lambda_tuning_integrating_plus_delay(Kp, taup, theta):
    """
    Lambda tuning for integrating-plus-delay model:

        G_model = Kp / s * exp(-theta s)

    Written in the form used in the student's original script:

        lambda = 8 theta
        tauI = max(4 theta, 2 lambda + theta)
        Kprime = Kp / taup
        Kc = tauI / [Kprime (lambda + theta)^2]

    Here taup is retained for consistency with the original notation.
    """
    lam = 8.0 * theta
    tauI = max(4.0 * theta, 2.0 * lam + theta)
    Kprime = Kp / taup
    Kc = tauI / (Kprime * (lam + theta) ** 2)
    tauD = 0.0

    return float(Kc), float(tauI), float(lam), float(tauD)


# =============================================================================
# Transfer functions
# =============================================================================
def actual_process_tf():
    s = ct.tf("s")
    return 0.8 / (s * (2 * s + 1) * (5 * s + 1))


def model_process_tf():
    s = ct.tf("s")
    return 0.75 / s * pade_delay(4, order=1)


def disturbance_tf():
    """
    Disturbance transfer function used from the original student script:
        Gd = e^(-s)/(s + 1)

    First-order Pade approximation is used for simulation.
    """
    s = ct.tf("s")
    return pade_delay(1, order=1) / (s + 1)


def pi_controller_tf(Kc, tauI):
    s = ct.tf("s")
    return Kc * (1 + 1 / (tauI * s))


def p_controller_tf(Kc):
    return ct.tf([Kc], [1])


# =============================================================================
# Problem 1A/B: basic feedback loop
# =============================================================================
def build_basic_feedback_loop(Kc, tauI):
    """
    Input order:
        [Ysp, D]

    Output:
        Y

    Disturbance is modeled as entering on the plant input side:
        Y = Gp * (M + Gd D)

    This is consistent with a feed-water flow disturbance entering before
    the boiler-level process dynamics.
    """
    Gc = pi_controller_tf(Kc, tauI)
    Gp = actual_process_tf()
    Gd = disturbance_tf()

    Gc_blk = ct.ss(Gc, name="Gc", inputs="E", outputs="Mfb")
    Gp_blk = ct.ss(Gp, name="Gp", inputs="Min", outputs="Y")
    Gd_blk = ct.ss(Gd, name="Gd", inputs="D", outputs="Md")

    sum_error = ct.summing_junction(
        inputs=["Ysp", "-Y"],
        output="E",
        name="sum_error",
    )

    sum_input = ct.summing_junction(
        inputs=["Mfb", "Md"],
        output="Min",
        name="sum_input",
    )

    sys = ct.interconnect(
        [Gc_blk, Gp_blk, Gd_blk, sum_error, sum_input],
        inputs=["Ysp", "D"],
        outputs=["Y"],
    )

    return sys


# =============================================================================
# Problem 1C/D: cascade loop
# =============================================================================
def build_cascade_loop(Kc_outer, tauI_outer, Kc_inner):
    """
    Cascade structure:

        outer PI controller gives Vsp
        inner P controller manipulates M
        inner process: Gp1
        final process: Gp2

    The actual process is split as:

        Gp_actual = Gp1 * Gp2
        Gp1 = 1 / (2s + 1)
        Gp2 = 0.8 / [s(5s + 1)]

    Disturbance enters the inner-loop process variable V:

        V = Gp1 M + Gd D
        Y = Gp2 V
    """
    s = ct.tf("s")

    Gco = pi_controller_tf(Kc_outer, tauI_outer)
    Gci = p_controller_tf(Kc_inner)

    Gp1 = 1 / (2 * s + 1)
    Gp2 = 0.8 / (s * (5 * s + 1))
    Gd = disturbance_tf()

    Gco_blk = ct.ss(Gco, name="Gco", inputs="Eo", outputs="Vsp")
    Gci_blk = ct.ss(Gci, name="Gci", inputs="Ei", outputs="M")
    Gp1_blk = ct.ss(Gp1, name="Gp1", inputs="M", outputs="Vproc")
    Gp2_blk = ct.ss(Gp2, name="Gp2", inputs="V", outputs="Y")
    Gd_blk = ct.ss(Gd, name="Gd", inputs="D", outputs="Vd")

    sum_outer = ct.summing_junction(
        inputs=["Ysp", "-Y"],
        output="Eo",
        name="sum_outer",
    )

    sum_inner = ct.summing_junction(
        inputs=["Vsp", "-V"],
        output="Ei",
        name="sum_inner",
    )

    sum_v = ct.summing_junction(
        inputs=["Vproc", "Vd"],
        output="V",
        name="sum_v",
    )

    sys = ct.interconnect(
        [Gco_blk, Gci_blk, Gp1_blk, Gp2_blk, Gd_blk, sum_outer, sum_inner, sum_v],
        inputs=["Ysp", "D"],
        outputs=["Y", "V"],
    )

    return sys


# =============================================================================
# Problem 1 report
# =============================================================================
def problem_1():
    doc.section("Problem 1")

    # Given model parameters
    Kp_model = 0.75
    taup_model = 1.0
    theta_model = 4.0

    Kc, tauI, lam, tauD = lambda_tuning_integrating_plus_delay(
        Kp=Kp_model,
        taup=taup_model,
        theta=theta_model,
    )

    # -------------------------------------------------------------------------
    # Problem 1A/B
    # -------------------------------------------------------------------------
    doc.subsection("Parts A and B: Basic PI Feedback Control")

    save_basic_feedback_diagram("problem1_basic_feedback_diagram.png")
    figlog("problem1_basic_feedback_diagram.png", "Basic feedback control diagram.")

    p(
        "The actual boiler liquid-level process is used for the closed-loop simulation, "
        "while the integrating-plus-delay model is used only to obtain the controller settings."
    )

    a(
        r"G_{actual}(s)=\frac{0.8}{s(2s+1)(5s+1)}",
        r"G_{model}(s)=\frac{0.75}{s}e^{-4s}",
        r"G_d(s)=\frac{1}{s+1}e^{-s}",
    )

    p("Using the lambda-tuning rule with an integrating-plus-delay model:")

    a(
        r"\lambda = 8\theta",
        r"\tau_I = \max(4\theta,\ 2\lambda+\theta)",
        r"K' = \frac{K_p}{\tau_p}",
        r"K_c = \frac{\tau_I}{K'(\lambda+\theta)^2}",
    )

    a(
        rf"\theta = {theta_model}",
        rf"\lambda = {lam:.4f}",
        rf"\tau_I = {tauI:.4f}",
        rf"K_c = {Kc:.4f}",
    )

    basic_sys = build_basic_feedback_loop(Kc, tauI)

    # Setpoint step
    t_sp, y_sp = forced_two_input_response(basic_sys, ysp_value=1, d_value=0)
    iae_sp = calculate_IAE(t_sp, y_sp, target=1)

    save_plot(
        "problem1_basic_setpoint.png",
        t_sp,
        y_sp,
        "Problem 1A: Basic PI Response to Setpoint Step",
        ysp=1,
        ylabel="Liquid Level",
    )

    # Disturbance step
    t_d, y_d = forced_two_input_response(basic_sys, ysp_value=0, d_value=1)
    iae_d = calculate_IAE(t_d, y_d, target=0)

    save_plot(
        "problem1_basic_disturbance.png",
        t_d,
        y_d,
        "Problem 1B: Basic PI Response to Disturbance Step",
        ysp=0,
        ylabel="Liquid Level",
    )

    subfiglog(
        [
            ("problem1_basic_setpoint.png", "Setpoint step"),
            ("problem1_basic_disturbance.png", "Disturbance step"),
        ],
        "Basic PI closed-loop responses.",
    )

    p("The IAE values for the basic PI controller are:")

    a(
        rf"IAE_{{setpoint,basic}} = {iae_sp:.4f}",
        rf"IAE_{{disturbance,basic}} = {iae_d:.4f}",
    )

    p(
        "For the setpoint response, the error is measured relative to a final desired value of 1. "
        "For the disturbance-rejection response, the desired value is 0 because the controller should reject "
        "the disturbance and return the output to its original operating point."
    )

    # -------------------------------------------------------------------------
    # Problem 1C
    # -------------------------------------------------------------------------
    doc.subsection("Part C: Cascade Control Structure")

    save_cascade_diagram("problem1_cascade_diagram.png")
    figlog("problem1_cascade_diagram.png", "Cascade control diagram.")

    p(
        "A cascade loop is added by splitting the process into an inner process and an outer process. "
        "The inner controller is P-only, while the outer controller remains PI. The disturbance is assumed "
        "to enter the inner-loop process variable, allowing the inner loop to reject the disturbance before "
        "it propagates through the slower boiler-level dynamics."
    )

    a(
        r"G_{p1}(s)=\frac{1}{2s+1}",
        r"G_{p2}(s)=\frac{0.8}{s(5s+1)}",
        r"G_{actual}(s)=G_{p1}(s)G_{p2}(s)",
    )

    # -------------------------------------------------------------------------
    # Problem 1D
    # -------------------------------------------------------------------------
    doc.subsection("Part D: Cascade Control with Inner Gain of 50")

    Kc_inner = 50.0
    cascade_sys = build_cascade_loop(Kc_outer=Kc, tauI_outer=tauI, Kc_inner=Kc_inner)

    # Output 0 is Y. Output 1 is V.
    t_csp, y_csp_all = forced_two_input_response(cascade_sys, ysp_value=1, d_value=0)
    y_csp = np.asarray(y_csp_all)[0, :] if np.asarray(y_csp_all).ndim == 2 else y_csp_all
    iae_csp = calculate_IAE(t_csp, y_csp, target=1)

    save_plot(
        "problem1_cascade_setpoint.png",
        t_csp,
        y_csp,
        "Problem 1D: Cascade Response to Setpoint Step",
        ysp=1,
        ylabel="Liquid Level",
    )

    t_cd, y_cd_all = forced_two_input_response(cascade_sys, ysp_value=0, d_value=1)
    y_cd = np.asarray(y_cd_all)[0, :] if np.asarray(y_cd_all).ndim == 2 else y_cd_all
    iae_cd = calculate_IAE(t_cd, y_cd, target=0)

    save_plot(
        "problem1_cascade_disturbance.png",
        t_cd,
        y_cd,
        "Problem 1D: Cascade Response to Disturbance Step",
        ysp=0,
        ylabel="Liquid Level",
    )

    subfiglog(
        [
            ("problem1_cascade_setpoint.png", "Cascade setpoint step"),
            ("problem1_cascade_disturbance.png", "Cascade disturbance step"),
        ],
        "Cascade closed-loop responses with inner controller gain equal to 50.",
    )

    p("The IAE values for the cascade controller are:")

    a(
        rf"K_{{c,inner}} = {Kc_inner:.4f}",
        rf"IAE_{{setpoint,cascade}} = {iae_csp:.4f}",
        rf"IAE_{{disturbance,cascade}} = {iae_cd:.4f}",
    )

    # -------------------------------------------------------------------------
    # Problem 1E
    # -------------------------------------------------------------------------
    doc.subsection("Part E: Discussion")

    p(
        "The effectiveness of the cascade controller can be evaluated by comparing the IAE values for the "
        "basic feedback controller and the cascade controller. A lower IAE indicates better overall control "
        "performance because the total accumulated error is smaller."
    )

    a(
        rf"IAE_{{setpoint,basic}} = {iae_sp:.4f}",
        rf"IAE_{{setpoint,cascade}} = {iae_csp:.4f}",
        rf"IAE_{{disturbance,basic}} = {iae_d:.4f}",
        rf"IAE_{{disturbance,cascade}} = {iae_cd:.4f}",
    )

    if iae_cd < iae_d:
        p(
            "The cascade controller improves disturbance rejection because the inner loop reacts directly "
            "to the disturbance before the disturbance fully propagates through the outer liquid-level dynamics."
        )
    else:
        p(
            "For the assumptions used in this simulation, the cascade controller does not reduce the disturbance "
            "IAE. This suggests that either the inner-loop process split, the disturbance location, or the chosen "
            "inner gain should be examined carefully against the intended block diagram."
        )

    return {
        "Kc": Kc,
        "tauI": tauI,
        "lambda": lam,
        "IAE_basic_sp": iae_sp,
        "IAE_basic_d": iae_d,
        "IAE_cascade_sp": iae_csp,
        "IAE_cascade_d": iae_cd,
    }


# =============================================================================
# Problem 2: Feedforward controller
# =============================================================================
def problem_2():
    doc.section("Problem 2")

    save_feedforward_diagram("problem2_feedforward_diagram.png")
    figlog("problem2_feedforward_diagram.png", "Feedforward plus feedback control diagram.")

    p(
        "For the feedforward design, the disturbance is assumed to enter on the plant-input side. "
        "The manipulated variable and the disturbance therefore both pass through the same boiler-level "
        "process dynamics."
    )

    a(
        r"Y(s)=G_p(s)\left[M(s)+G_d(s)D(s)\right]",
        r"M(s)=M_{fb}(s)+M_{ff}(s)",
        r"M_{ff}(s)=G_{ff}(s)D(s)",
    )

    p("For perfect disturbance cancellation, the disturbance contribution inside the plant input must be zero:")

    a(
        r"G_{ff}(s)D(s)+G_d(s)D(s)=0",
        r"G_{ff}(s)=-G_d(s)",
    )

    p("Using the disturbance transfer function:")

    a(
        r"G_d(s)=\frac{1}{s+1}e^{-s}",
        r"\boxed{G_{ff}(s)=-\frac{1}{s+1}e^{-s}}",
    )

    p(
        "This result follows because the disturbance enters at the same summing point as the manipulated "
        "variable before the plant. Therefore, the plant transfer function cancels out in the feedforward "
        "design."
    )


# =============================================================================
# Problem 3: RGA and decouplers
# =============================================================================
def problem_3():
    doc.section("Problem 3")

    # -------------------------------------------------------------------------
    # Given transfer functions
    # -------------------------------------------------------------------------
    doc.subsection("Part A: RGA and Pairing")

    p("The steady-state gain matrix is found by evaluating each transfer function at s = 0.")

    K = np.array([
        [2.3, 0.47],
        [-1.2, 0.58],
    ], dtype=float)

    K_inv_T = np.linalg.inv(K).T
    RGA = K * K_inv_T

    a(
        r"K=\begin{bmatrix}2.3 & 0.47\\-1.2 & 0.58\end{bmatrix}",
        rf"\Lambda = K \circ (K^{{-1}})^T"
    )

    a(
        rf"\Lambda = "
        rf"\begin{{bmatrix}}"
        rf"{RGA[0,0]:.4f} & {RGA[0,1]:.4f}\\"
        rf"{RGA[1,0]:.4f} & {RGA[1,1]:.4f}"
        rf"\end{{bmatrix}}"
    )

    p(
        "The diagonal RGA values are closer to 1 than the off-diagonal values, so the appropriate pairings are:"
    )

    a(
        r"Y_1 \leftrightarrow M_1",
        r"Y_2 \leftrightarrow M_2",
    )

    # -------------------------------------------------------------------------
    # Part B: MIMO block diagram
    # -------------------------------------------------------------------------
    doc.subsection("Part B: MIMO Feedback Block Diagram")

    save_mimo_diagram("problem3_mimo_diagram.png", decoupled=False)
    figlog("problem3_mimo_diagram.png", "MIMO feedback control diagram without decouplers.")

    p(
        "The MIMO process contains four transfer functions. Each manipulated variable affects both output "
        "variables, which creates loop interaction."
    )

    a(
        r"G_{11}(s)=\frac{2.3}{4.6s+1}e^{-0.2s}",
        r"G_{12}(s)=\frac{0.47}{2.2s+1}e^{-0.3s}",
        r"G_{21}(s)=\frac{-1.2}{18s+1}e^{-0.4s}",
        r"G_{22}(s)=\frac{0.58}{1.8s+1}e^{-0.5s}",
    )

    # -------------------------------------------------------------------------
    # Part C: Decouplers
    # -------------------------------------------------------------------------
    doc.subsection("Part C: Decoupler Design")

    p(
        "For diagonal pairing, the decouplers are selected to cancel the off-diagonal process interactions."
    )

    a(
        r"D_{12}(s)=-\frac{G_{12}(s)}{G_{11}(s)}",
        r"D_{21}(s)=-\frac{G_{21}(s)}{G_{22}(s)}",
    )

    D12_gain = -0.47 / 2.3
    D21_gain = -(-1.2 / 0.58)

    p("Therefore:")

    a(
        rf"D_{{12}}(s)=({D12_gain:.4f})\frac{{4.6s+1}}{{2.2s+1}}e^{{-0.1s}}",
        rf"D_{{21}}(s)=({D21_gain:.4f})\frac{{1.8s+1}}{{18s+1}}e^{{0.1s}}",
    )

    p(
        "The first decoupler contains an additional delay and is causal. The second decoupler contains "
        "a time advance, represented by the positive exponential term. A pure time advance is noncausal, "
        "so an exact implementation of this decoupler is not physically realizable. In practice, this "
        "decoupler would need to be approximated or modified."
    )

    save_mimo_diagram("problem3_decoupled_mimo_diagram.png", decoupled=True)
    figlog("problem3_decoupled_mimo_diagram.png", "MIMO feedback control diagram with decouplers.")


# =============================================================================
# Main
# =============================================================================
def main():
    doc.maketitle(True)
    doc.toc(False)

    results = problem_1()
    problem_2()
    problem_3()

    txt_file, tex_file, pdf_file = doc.save_all(runs=2)

    print("Finished.")
    print(f"Wrote text log: {txt_file}")
    print(f"Wrote LaTeX file: {tex_file}")
    print(f"Wrote PDF file: {pdf_file}")
    print()
    print("Important numerical results:")
    for key, val in results.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()