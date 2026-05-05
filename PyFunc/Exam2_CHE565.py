from math import e
import numpy as np
import sympy as sp
from sympy import Add, Mul 
import matplotlib
from pathlib import Path
from sympy.functions.combinatorial.factorials import rf
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
from scipy.stats import t as t_dist
import pandas as pd
from NRroots import newton_raphson
from regression_analysis import RegressionAnalysis
from doc_builder import DocumentBuilder
import control as ct
from pylatex import NoEscape

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
    plt.close()


def calculate_IAE(t, y, ysp):
    y = np.asarray(y).reshape(-1)
    if np.isscalar(ysp):
        ysp_vec = np.ones_like(t) * float(ysp)
    else:
        ysp_vec = np.asarray(ysp).reshape(-1)
    return float(np.trapezoid(np.abs(ysp_vec - y), t))

# =============================================================================
# Output names
# =============================================================================
OUTPUT_FILE = "Exam2_CHE565"
PLOT_DIR = Path("Exam2_plots")
PLOT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Document setup
# =============================================================================
doc = DocumentBuilder(
    OUTPUT_FILE,
    title="CHE 565 -- Exam 2",
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

doc.maketitle(True)
doc.toc(False)
s = sp.symbols("s")

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
    
        #Integrating Process
    print("Integrating process.")
    lam = 8*thetad
    tauI = max(4*thetad,2*lam+thetad)
    Kprime = Kp / taup     
    P= tauI/(Kprime*(lam+thetad)**2)

    if PI:
        tauD = 0
    elif PID:
        tauD = min( tauI/4,thetad/2)

    return float(P), float(tauI), float(lam), float(tauD)

def build_closed_loop(Kc1, tauI1, disturbance=True):
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
    numD1, denD1 = ct.delay.pade(1, 1)
    numD2, denD2 = ct.delay.pade(4, 1)
    
    G_delay1 = ct.tf(numD1, denD1)
    G_delay2 = ct.tf(numD2, denD2)
    
    Gc = ct.tf([Kc1, Kc1 / tauI1], [1, 0])
    Gp = 0.8/(s*(2*s+1)*(5*s+1))
    Gp_mod = ct.tf([0.75],[1, 0])
    Gp_mod_delayed = Gp_mod*G_delay2
    G_step = ct.tf([1], [1, 0])
    Gd = ct.tf([1], [1, 1])

    Gc_blk = ct.ss(Gc, name="Gc", inputs="E1", outputs="Yc")
    Gp_blk = ct.ss(Gp, name="Gp", inputs="E2", outputs="Yp")
    Gp_mod_blk = ct.ss(Gp_mod_delayed, name="Gp_mod", inputs="E2", outputs="Yp_mod")
    if disturbance:
        Gd_step = Gd*G_step*G_delay1
        Gd_blk = ct.ss(Gd_step, name="Gd", inputs="D", outputs="Yd")
        sum1 = ct.summing_junction(inputs=["-Yp_mod"], output="E1", name="Sum")
        sum2 = ct.summing_junction(inputs=["Yc","Yd"], output="E2", name="Sum2")
        sys = ct.interconnect(
        [Gc_blk,Gp_mod_blk, Gd_blk ,sum1, sum2],
        inputs=["D"] ,
        outputs=["Yp_mod"],
        )
    else:
        
        Gstep_blk = ct.ss(G_step, name="Gd", inputs="Ysp", outputs="Yd")
        sum1 = ct.summing_junction(inputs=["Yd","-Yp_mod"], output="E1", name="Sum")
        sum2 = ct.summing_junction(inputs=["Yc"], output="E2", name="Sum2")
        sys = ct.interconnect(
            [Gc_blk, Gp_mod_blk, Gstep_blk ,sum1, sum2],
            inputs=["Ysp"],
            outputs=["Yp_mod"],
        )
    print(sys)
    return sys
def problem_1():
    taup1 = 1
    thetad1 = 1
    thetad2 = 4
    s =sp.symbols("s")
    numD1, denD1 = ct.delay.pade(1, 1)
    numD2, denD2 = ct.delay.pade(4, 1)
    Gd= 1/(s+1)
    G_delay1 = sp.exp(-thetad1*s)
    Kp = 0.75
    Gp= 0.8/(s*(2*s+1)*(5*s+1))
    Gp_mod= 0.75/(s)
    Gd_delay = Gd*G_delay1
    G_delay2 = sp.exp(-thetad2*s)
    Gp_mod_delayed = Gp_mod*G_delay2
    Kc1, tauI1, lam, tauD = lambda_tuning(taup1, thetad2, Kp, PI=True)
    sys1 = build_closed_loop(Kc1, tauI1,disturbance=True)
    resp1 = ct.step_response(sys1, T=np.linspace(0, 500, 500))
    save_plot("problem1_disturbance.png", resp1.time, resp1.outputs, "Step Response of the Closed-LoopSystem")
    Kc2, tauI2, lam, tauD = lambda_tuning(taup1, thetad2, Kp, PI=True)
    sys2 = build_closed_loop(Kc2, tauI2, disturbance=False)
    resp2 = ct.step_response(sys2, T=np.linspace(0, 500, 500))
    save_plot("problem1_setpoint.png", resp2.time, resp2.outputs, "Step Response of the Closed-Loop System to a Setpoint Change")
    print(f"Calculated controller parameters: Kc = {Kc2:.4f}, tauI = {tauI2:.4f}, lambda = {lam:.4f}, tauD = {tauD:.4f}")

    doc.section("Problem 1")
    doc.subsection("A")
    subfiglog([("Exam2_block_diagram_disturbance.png", "Disturbance Diagram"),
                        ("Exam2_block_diagram_setpoint.png", "Setpoint Diagram")], "Block Diagram Control System")
    a(rf"\text{{Process Transfer Function:}} {sp.latex(Gp)}",
      rf"\text{{Modeled Process Transfer Function:}} {sp.latex(Gp_mod_delayed)}",
      rf"\text{{Disturbance Transfer Function:}} {sp.latex(Gd_delay)}",
    )
    p(rf"Using lambda tuning for PI control, we find the controller parameters as follows:")
    a(rf"K' = \frac{{K_p}}{{\tau_p}},Kc = \frac{{\tau_p}}{{K' (\lambda + \theta_d)^2}}", rf"\tau_{{I}} = \tau_p", rf"\lambda = 8 \theta_d", rf"\tau_D = 0")
    px("Where ", im(r"\tau_p"), f"= {taup1}", im(r"\quad"), im(r"\theta_d"), f"= {thetad2}", im(r"\quad"), im(r"K_p"), f"= {Kp}")
    px(im(rf"K_c:"), f"= {Kc1:.4f} ", im(r"\tau_I:"), f"= {tauI1:.4f} ", im(r"\lambda:"), f"= {lam:.4f} ", im(latex=r"\tau_D:"), f"= {tauD:.4f} ", im(r"\quad"), "And an IAE of:", im(r"\quad"), f"{calculate_IAE(resp1.time, resp1.outputs, 0):.4f}", im(r"\quad"), "for the setpoint change and", im(r"\quad"), f"{calculate_IAE(resp2.time, resp2.outputs, 0):.4f}", im(r"\quad"), "for the disturbance.")
    subfiglog([("problem1_disturbance.png", "Disturbance"),
               ("problem1_setpoint.png", "Setpoint Change")], "Closed-Loop Step Responses")
    
    
def __main__():
    problem_1()
    txt_file, tex_file, pdf_file = doc.save_all(runs=2)
    print(f"Wrote text log: {txt_file}")
    print(f"Wrote LaTeX file: {tex_file}")
    print(f"Wrote PDF report: {pdf_file}")

    

if __name__ == "__main__":
    __main__()