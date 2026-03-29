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
m(rf"P(A) = \${P_A:,.2f}/\text{{year}},\quad P(B) = \${P_B:,.2f}/\text{{year}}")
doc.section("Problem 2")
doc.subsection("Setup")
w(f"$0.94/lbs is the price of powdered detergent.")
w(f"0.1 % of 4e6 lbs of detergent per year is carried through the exhaust")
w(f"Adding a second cyclone separator costs $10,000, with $300/year in maintenance costs will reduce carryover to 0.0%, with a 8% yearly interest rate.")
interest_rate = 0.08
detergent_price = 0.94 # $/lb
total_detergent = 4e6 # lbs/year
carryover_fraction = 0.001 # 0.1% carryover
carryover_amount = total_detergent * carryover_fraction # lbs/year
carryover_cost = carryover_amount * detergent_price # $/year
separator_cost = 10000 # $ initial cost
separator_maintenance = 300 # $/year
w(f"Cost of carryover detergent: ${carryover_cost:,.2f}/year\nCost of second separator: ${separator_cost:,.2f} initial + ${2*separator_maintenance:,.2f}/year maintenance = ${separator_cost + 2*separator_maintenance:,.2f} total")
doc.section("Problem 2")
doc.subsection("Setup")

w(f"$0.94/lb is the price of powdered detergent.")
w(f"0.1% of 4e6 lb/year is carried through the exhaust.")
w(f"Adding a second cyclone separator costs $10,000 with $300/year maintenance and reduces carryover to 0.0%.")

interest_rate = 0.08
detergent_price = 0.94  # $/lb
total_detergent = 4e6   # lb/year
carryover_fraction = 0.001

carryover_amount = total_detergent * carryover_fraction
carryover_cost = carryover_amount * detergent_price  # lost value

separator_cost = 10000
separator_maintenance = 300

w(f"Lost detergent value: ${carryover_cost:,.2f}/year")

# Part (a)
w("a.) Find the additional yearly income")

gross_additional_income = carryover_cost
net_additional_income = carryover_cost - separator_maintenance

w(f"Gross additional income (recovered detergent): ${gross_additional_income:,.2f}/year")
w(f"Net additional income after maintenance: ${net_additional_income:,.2f}/year")

# Part (b)
w("b.) Find the payback period")

payback_years = separator_cost / net_additional_income
payback_months = payback_years * 12

w(f"Payback period: {payback_years:.2f} years ({payback_months:.1f} months)")

doc.section("Problem 3")
doc.subsection("Setup")
w("Determine if each function is convex")

m(r"\text{A}: f(x) = (x_1-x_2)^2 + x_2^2")
m(r"\text{B}: f(x) = x_1^2 + x_2^2+x_3^2")
m(r"\text{C}: f(x) = \exp(x_1)+\exp(x_2)")
def f_A(x):
    return (x[0] - x[1])**2 + x[1]**2
def f_B(x):
    return x[0]**2 + x[1]**2 + x[2]**2
def f_C(x):
    return np.exp(x[0]) + np.exp(x[1])


gamma_A = gradient_A = lambda x: np.array([2*(x[0]-x[1]), 2*(x[1]-x[0]) + 2*x[1]])  
gamma_B = gradient_B = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])
gamma_C = gradient_C = lambda x: np.array([np.exp(x[0]), np.exp(x[1])])
hessian_A = lambda x: np.array([[2, -2], [-2, 4]])
hessian_B = lambda x: np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
hessian_C = lambda x: np.array([[np.exp(x[0]), 0], [0, np.exp(x[1])]])
x1 = np.array([1, 1])
rhs = f_A(x1)*gamma_A(x1)+(1-gamma_A(x1))
lhs = f_A(gamma_A(x1)*x1+(1-gamma_A(x1))*x1)
px(f"A: {lhs} <=", im{rf"\text{f(x1)=},{rhs}"},)

# save outputs
txt_file, tex_file, pdf_file = doc.save_all()

print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX source: {tex_file}")
print(f"Wrote PDF: {pdf_file}")

