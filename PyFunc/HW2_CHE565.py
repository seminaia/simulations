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

def NPV(PV, C0=0):
    """
    PV = present value, 
    C0 = initial cost (default 0)
    """
    return PV + C0

def future_value(P0, r, n, k):
    return P0 * ((1 + r)**n*r)/((1+r)**n-1)

def present_value(F, r, n):
    return F * ((1+r)**n - 1) / (r * (1+r)**n)

i_npv = 0.10 # 10% interest rate
i_loan = 0.05     # 5% loan interest rate
n=10
F_A = 1.1e6
F_B = 1.41e6
C0_A = -3.8e6
C0_B = -5.0e6
P0_A = -C0_A
P0_B = -C0_B

for i in range(1, n + 1):
    PV_A = present_value(F_A, i_npv, i)
    PV_B = present_value(F_B, i_npv, i)
    w(f"  n = {i:.0f} year(s): PV(A) = ${PV_A:,.2f}   PV(B) = ${PV_B:,.2f}")
NPV_A = present_value(F_A, i_npv, n)
NPV_B = present_value(F_B, i_npv, n)
w(f"\nNPV(A) = ${NPV_A:,.2f}   NPV(B) = ${NPV_B:,.2f}")

if NPV_A + C0_A > NPV_B + C0_B:
    w(f"\nNPV is higher for option A at 10% interest, so A is preferred under these assumptions.")
else:
    w(f"\nNPV is higher for option B at 10% interest, so B is preferred under these assumptions.")

for i in range(1, n + 1):
    fv_A = future_value(P0_A, i_loan, i, 1)
    fv_B = future_value(P0_B, i_loan, i, 1)
    PV_A = present_value(F_A, i_npv, i)
    PV_B = present_value(F_B, i_npv, i)
    w(f"  n = {i:.0f} year(s): FV(A) = ${fv_A:,.2f}   FV(B) = ${fv_B:,.2f}")
    total_A = fv_A + PV_A
    total_B = fv_B + PV_B
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
