"""
HW2_CHE565.py
=============
CHE 565 – Homework 2
All five problems solved with full work shown.
Results are written to HW2_CHE565_results.txt and mirrored to the console.
"""

from math import e

import numpy as np
import matplotlib
from sympy.functions.combinatorial.factorials import rf
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import t as t_dist
import pandas as pd
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

from scipy.optimize import fsolve
import sympy as sp
# Define NPV equation
def NPV_eq(n):
    return net_additional_income * ((1 + interest_rate)**n - 1) / (interest_rate * (1 + interest_rate)**n) - separator_cost

# Solve for n
n_guess = 3
n_solution = fsolve(NPV_eq, n_guess)[0]

w(f"Discounted payback period (NPV = 0): {n_solution:.2f} years")
doc.section("Problem 3")
doc.subsection("Setup")
w("Determine if each function is convex")

a(
    r"\text{A}: f(x) &= (x_1-x_2)^2 + x_2^2",
    r"\text{B}: f(x) &= x_1^2 + x_2^2+x_3^2",
    r"\text{C}: f(x) &= e^{x_1}+e^{x_2}"
)

x1, x2, x3 = sp.symbols('x1 x2 x3')

# A
fa = (x1 - x2)**2 + x2**2
fa_latex = sp.latex(fa)
HA = sp.hessian(fa, (x1, x2))
HA_latex = sp.latex(HA)
eig_A = list(HA.eigenvals().keys())
eig_A_latex = sp.latex(sp.Matrix(eig_A))

# B
fb = x1**2 + x2**2 + x3**2
fb_latex = sp.latex(fb)
HB = sp.hessian(fb, (x1, x2, x3))
HB_latex = sp.latex(HB)
eig_B = list(HB.eigenvals().keys())
eig_B_latex = sp.latex(sp.Matrix(eig_B))

# C
fc = sp.exp(x1) + sp.exp(x2)
fc_latex = sp.latex(fc)
HC = sp.hessian(fc, (x1, x2))
HC_latex = sp.latex(HC)
eig_C = list(HC.eigenvals().keys())
eig_C_latex = sp.latex(sp.Matrix(eig_C))

doc.subsection("Results")


t(['Function','Hessian','Eigenvalues','Convex?','Reason'],
 [
      [im(fa_latex),im(latex=HA_latex), im(eig_A_latex), "Yes", "Eigenvalues are non-negative"],
      [im(fb_latex),im(HB_latex), im(eig_B_latex), "Yes", "Eigenvalues are positive"],
      [im(fc_latex),im(HC_latex), im(eig_C_latex), "Yes", "Eigenvalues are positive"]
  ],
 alignment='c|c|c|c|c'
)

doc.section("Problem 4")

w("Is the region defined by the constraints convex?")
constraints = [
    (x2 >= 1-x1)&
    (x2 <= 1+sp.Rational(1,2)*x1)&
    (x1 <= 2)&
    (x2 >= 0)
]

feasible_region =sp.And(*constraints)
feasible_region_latex = sp.latex(feasible_region)
px(rf"The feasible region is defined by the following constraints: " , im(feasible_region_latex))
PLOT_FILE="HW2_CHE565_plot.png"
sp.plot_implicit(feasible_region,(x1,0,3),(x2,0,3),show=False, title="CHE 565: Problem 4 Feasible Region").save(PLOT_FILE)
figlog(PLOT_FILE,"Feasible Region")
w("The feasible region is convex because it is defined by a set of linear inequalities, which form a convex set.")
doc.section(("Problem 5"))
doc.subsection("Setup")
w("Minimize the objective function:")

def func(x):
    return 1 + 8*x + 2*x**2 - 10/3*x**3 - 1/4*x**4 + 4/5*x**5 - 1/6*x**6

def dfunc(x):
    return (1 + x)**2 * (2 - x)**3

x = sp.symbols('x')
f = 1 + 8*x + 2*x**2 - sp.Rational(10,3)*x**3 - sp.Rational(1,4)*x**4 + sp.Rational(4,5)*x**5 - sp.Rational(1,6)*x**6
dfun = (1 + x)**2 * (2 - x)**3
d2fun = sp.diff(dfun,x)
df = sp.diff(f,x)
d2f = sp.diff(df,x)
m(rf"f(x) = {sp.latex(f)}")
w(text="Given that:")
m(rf"\frac{{df}}{{dx}}= {sp.latex(dfun)}")
m(rf"\frac{{d^2f}}{{dx^2}}= {sp.latex(d2fun)}")
doc.subsection("a.) Analytical Solution")
w("Expanded form:")
m(rf"\frac{{df}}{{dx}} = {sp.latex(df)}")
m(rf"\frac{{d^2f}}{{dx^2}} = {sp.latex(d2f)}")
sol = sp.solve(dfun,x)
f_sol = [func(s) for s in sol]
w(f"The solutions to df/dx = 0 are x* = {sol[0]}, {sol[1]}")
w(f"The corresponding function values are f(x*) = {f_sol[0]:.2f}, {f_sol[1]:.2f}")
doc.subsection("b.) Excel Solution")
df = pd.read_csv("HW2_CHE565.csv")

df.drop('Unnamed: 6',axis=1,inplace=True)

df_A = df.iloc[:,:6].copy()
df_B = df.iloc[:,6:].copy()
headers_A = df_A.columns.tolist()
row_A = df_A.values.tolist()
headers_B = df_B.columns.tolist()
row_B = df_B.values.tolist()
w("The Excel data used for the numerical solution is shown below:")
t(headers_A,row_A,alignment='c'*len(headers_A),caption="Optimization Results from Excel starting at x0=5")
t(headers_B,row_B,alignment='c'*len(headers_B),caption="Optimization Results from Excel starting at x0=-5")

x_guess = [-5,5]
roots = fsolve(func=dfunc,x0=x_guess)
x1_min = roots[0]
x2_min = roots[1]
a(
    r"x^* &= -1, 2",
    rf"f(x^*) &= {func(x1_min):.2f},\ {func(x2_min):.2f}"
)

w(f"The roots of df/dx are at x* = {x1_min:.2f} with multiplicity 2, {x2_min:.2f} with multiplicity 3")
w(f"The minimum value of the function is f(x*) = {func(x1_min):.2f}, {func(x2_min):.2f}")


txt_file, tex_file, pdf_file = doc.save_all()

print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX source: {tex_file}")
print(f"Wrote PDF: {pdf_file}")

