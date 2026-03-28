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
from doc_builder import DocumentBuilder
# ══════════════════════════════════════════════════════════════════════════════
#  Output file setup
#  All problems write through the same RegressionAnalysis writer so the
#  complete solution ends up in one tidy file.
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_FILE = "HW2_CHE565"
PLOT_FILE = "HW2_CHE565_plot.png"
report_lines = []

doc = DocumentBuilder(
    OUTPUT_FILE,
    title="CHE 565 -- Homework 2",
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

doc.section("Problem 1")
doc.subsection("Setup")

# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM 1
# ══════════════════════════════════════════════════════════════════════════════
a(
    r"\text{Option A:}\quad C_{0,A} &= 3{,}800{,}000",
    r"\quad\quad\quad\;\;\; FV_A &= 1{,}100{,}000/\text{yr}",
    r"\text{Option B:}\quad C_{0,B} &= 5{,}000{,}000",
    r"\quad\quad\quad\;\;\; FV_B &= 1{,}410{,}000/\text{yr}",
)
w()
m(r"NPV = PV + C_0")
m(r"PV = F\left[\frac{(1+r)^n - 1}{r(1+r)^n}\right]")
w("PV = present value, FV = future value (annual cash flow), r = yearly interest rate, n = number of years")
w("n= 10 years, r = 0.10")

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
m(rf"\text{{NPV(A)}} = \${NPV_A:,.2f},\quad \text{{NPV(B)}} = \${NPV_B:,.2f}")

if NPV_A > NPV_B:
    w(f"NPV is higher for option A at 10% yearly interest, so A is preferred under these assumptions.")
else:
    w(f"NPV is higher for option B at 10% yearly interest, so B is preferred under these assumptions.")

w("b.) 10 year lifetime, no salvage value, and 5 % yearly interest rate. What will be the yearly payment?")
w()
m(r"P =  C0 * \frac{{(1 + r)^n r}}{{(1+r)^n-1}}")
w(f"P = annual payment, C0 = initial cost, r = yearly interest rate, n = number of years")
m(r"P(A) = {P_A:,.2f}/\text{{year}},\quad P(B) = {P_B:,.2f}/\text{{year}}")

# save outputs
txt_file, tex_file, pdf_file = doc.save_all()

print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX source: {tex_file}")
print(f"Wrote PDF: {pdf_file}")

