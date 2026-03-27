"""
HW2_CHE565.py
=============
CHE 565 – Homework 2
All five problems solved with full work shown.
Results are written to HW2_CHE565_results.txt and mirrored to the console.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import t as t_dist
from NRroots import newton_raphson
from regression_analysis import RegressionAnalysis
from matplotlib.backends.backend_pdf import PdfPages
# ══════════════════════════════════════════════════════════════════════════════
#  Output file setup
#  All problems write through the same RegressionAnalysis writer so the
#  complete solution ends up in one tidy file.
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_FILE = "HW2_CHE565_results.txt"
PLOT_FILE = "HW2_CHE565_plot.png"

report_lines = []


def w(text=""):
    report_lines.append(str(text))
    print(text)

w("  CHE 565 - Homework 2")
w("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM 1
# ══════════════════════════════════════════════════════════════════════════════

w()
w("-" * 80)
w("  PROBLEM 1")
w("-" * 80)
w()
w("  SETUP")
w("  -----")
w("  Option A: $3.8e6 ")
w("             F=$1.1e6/yr")
w("  Option B: $5.0e6")
w("             F=1.41e6/yr")
w("   a.) 10 year lifetime, no salvage value, and 10\\% interest rate.")
w("   NPV = F*[((1+r)^n - 1)/(r*(1+r)^n)]")

def NPV(F, r, n, C0=0):
    """
    F = annual cash flow, 
    r = interest rate,
    n = number of years,
    C0 = initial cost (default 0)
    """
    return F * (((1 + r)**n - 1) / (r * (1 + r)**n)) + C0

def future_value(P0, r, n, k):
    return P0 * ((1 + r)**(n - k + 1))
    
lifetime_interest = 0.10 # 10% interest rate
loan_interest = 0.05     # 5% loan interest rate
start=0
stop=10
n = np.linspace(start, stop, 10) # 10 year lifetime
F_A = 1.1e6
F_B = 1.41e6
C0_A = -3.8e6
C0_B = -5.0e6
P0_A = -C0_A
P0_B = -C0_B
for i in n:
    npv_A = NPV(F_A, lifetime_interest, i, C0_A)
    npv_B = NPV(F_B, lifetime_interest, i, C0_B)
    w(f"  n = {i:.0f} year(s): NPV(A) = ${npv_A:,.2f}   NPV(B) = ${npv_B:,.2f}")
w("NPV is higher for option B at 10% interest, so B is preferred under these assumptions.")
for i in n:
    fv_A = future_value(P0_A, loan_interest, i, start)
    fv_B = future_value(P0_B, loan_interest, i, start)
    w(f"  n = {i:.0f} year(s): FV(A) = ${fv_A:,.2f}   FV(B) = ${fv_B:,.2f}")
for i in n:
    total_A = future_value(P0_A, loan_interest, i, stop) + NPV(F_A, lifetime_interest, i, C0_A)
    total_B = future_value(P0_B, loan_interest, i, stop) + NPV(F_B, lifetime_interest, i, C0_B)
    w(f"  n = {i:.0f} year(s): T(A) = ${total_A:,.2f}   T(B) = ${total_B:,.2f}")
from pathlib import Path
import shutil
import subprocess


Path(OUTPUT_FILE).write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def build_latex_pdf():
    tex_file = "CHE565_HW2.tex"

    latex = rf"""
\documentclass[12pt]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{fancyvrb}}
\usepackage{{graphicx}}
\usepackage{{float}}

\title{{CHE 565 -- Homework 2}}
\date{{}}

\begin{{document}}

\maketitle

\section*{{Full Solution}}
\subsection*{{Text Report}}

\VerbatimInput[
    fontsize=\scriptsize,
    frame=single,
    framesep=2mm
]{{{OUTPUT_FILE}}}

\end{{document}}
"""

    Path(tex_file).write_text(latex, encoding="utf-8")

    engine = shutil.which("pdflatex")
    if engine is None:
        print("pdflatex not found; wrote the .tex file only")
        return

    subprocess.run([engine, "-interaction=nonstopmode", tex_file], check=True)

    print("PDF generated")

build_latex_pdf()
