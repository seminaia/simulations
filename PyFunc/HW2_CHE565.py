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

def tex_escape(s):
    return (str(s)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("$", r"\$")
            .replace("#", r"\#")
            .replace("_", r"\_")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("~", r"\textasciitilde{}")
            .replace("^", r"\textasciicircum{}"))

def w(text=""):
    s = str(text)
    report_lines.append(tex_escape(s) + r"\\")
    print(s)

def wm(tex=""):
    report_lines.append(tex + r"\\")
    print(tex)
    
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
w("  Option A:  C0_A = $3,800,000")
w("             FV_A = $1,100,000/yr")
w("  Option B:  C0_B = $5,000,000")
w("             FV_B = $1,410,000/yr")
w("   a.) 10 year lifetime, no salvage value, and 10 % yearly interest rate. What is the NPV of each option?,")
w("       and which is preferred under these assumptions?")
w()
wm(r"$NPV = PV + C_0$")
wm(r"$PV = F\left[\frac{(1+r)^n - 1}{r(1+r)^n}\right]$")
w("       PV = present value, FV = future value (annual cash flow), r = yearly interest rate, n = number of years")
w("       n= 10 years, r = 0.10")

def NPV(PV, C0=0):
    """
    PV = present value, 
    C0 = initial cost (default 0)
    """
    return PV + C0

def annual_payment(C0, r, n, k):
    return C0 * ((1 + r)**n*r)/((1+r)**n-1)

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

PV_A = present_value(F_A, i_npv, n)
PV_B = present_value(F_B, i_npv, n)
NPV_A = NPV(PV_A, C0_A)
NPV_B = NPV(PV_B, C0_B)
P_A = annual_payment(C0_A, i_loan, n, 1)
P_B = annual_payment(C0_B, i_loan, n, 1)
wm(rf"       $NPV(A) = {NPV_A:,.2f},\quad NPV(B) = {NPV_B:,.2f}$")

if NPV_A > NPV_B:
    w(f"    NPV is higher for option A at 10% yearly interest, so A is preferred under these assumptions.")
else:
    w(f"    NPV is higher for option B at 10% yearly interest, so B is preferred under these assumptions.")

w("\n  b.) 10 year lifetime, no salvage value, and 5 % yearly interest rate. What will be the yearly payment?")
w()
wm(rf"        $P =  C0 * \frac{{(1 + r)^n r}}{{(1+r)^n-1}}$")
w(f"        P = annual payment, C0 = initial cost, r = yearly interest rate, n = number of years")
wm(rf"        $P(A) = {P_A:,.2f}/\text{{year}},\quad P(B) = {P_B:,.2f}/\text{{year}}$")
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
{"\n".join(report_lines)}

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

! Undefined control sequence.
l.43         $P(A) = -492,117.38/\text
                                      {year},\quad P(B) = -647,522.87/\text{...

! Undefined control sequence.
l.43 ...\text{year},\quad P(B) = -647,522.87/\text
                                                  {year}$\\

Underfull \hbox (badness 10000) in paragraph at lines 39--44


Underfull \hbox (badness 10000) in paragraph at lines 39--44

[1{/var/lib/texmf/fonts/map/pdftex/updmap/pdftex.map}{/usr/share/texmf/fonts/en
c/dvips/cm-super/cm-super-t1.enc}] (./CHE565_HW2.aux) )
(see the transcript file for additional information)</usr/share/texlive/texmf-d
ist/fonts/type1/public/amsfonts/cm/cmex10.pfb></usr/share/texlive/texmf-dist/fo
nts/type1/public/amsfonts/cm/cmmi12.pfb></usr/share/texlive/texmf-dist/fonts/ty
pe1/public/amsfonts/cm/cmmi6.pfb></usr/share/texlive/texmf-dist/fonts/type1/pub
lic/amsfonts/cm/cmmi8.pfb></usr/share/texlive/texmf-dist/fonts/type1/public/ams
fonts/cm/cmr12.pfb></usr/share/texlive/texmf-dist/fonts/type1/public/amsfonts/c
m/cmr8.pfb></usr/share/texlive/texmf-dist/fonts/type1/public/amsfonts/cm/cmsy10
.pfb></usr/share/texlive/texmf-dist/fonts/type1/public/amsfonts/cm/cmsy8.pfb></
usr/share/texmf/fonts/type1/public/cm-super/sfrm1200.pfb></usr/share/texmf/font
s/type1/public/cm-super/sfrm2074.pfb>
Output written on CHE565_HW2.pdf (1 page, 104804 bytes).
Transcript written on CHE565_HW2.log.
Traceback (most recent call last):
  File "/home/soki/simulations/PyFunc/HW2_CHE565.py", line 157, in <module>
    build_latex_pdf()
    ~~~~~~~~~~~~~~~^^
  File "/home/soki/simulations/PyFunc/HW2_CHE565.py", line 153, in build_latex_pdf
    subprocess.run([engine, "-interaction=nonstopmode", tex_file], check=True)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/soki/miniconda3/envs/myGPAWenv/lib/python3.14/subprocess.py", line 577, in run
    raise CalledProcessError(retcode, process.args,
                             output=stdout, stderr=stderr)
subprocess.CalledProcessError: Command '['/usr/bin/pdflatex', '-interaction=nonstopmode', 'CHE565_HW2.tex']' returned non-zero exit status 1.