"""
HW2_CHE565.py
=============
CHE 565 – Homework 2
All five problems solved with full work shown.
Results are written to HW2_CHE565_results.txt and mirrored to the console.
"""

from math import e
import numpy as np
import sympy as sp 
import matplotlib
from sympy.functions.combinatorial.factorials import rf
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
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
    r"\text{Option A:}\quad C_{0,A} &= \$3{,}800{,}000",
    r"\quad\quad\quad\;\;\; FV_A &= \$1{,}100{,}000/\text{yr}",
    r"\text{Option B:}\quad C_{0,B} &= \$5{,}000{,}000",
    r"\quad\quad\quad\;\;\; FV_B &= \$1{,}410{,}000/\text{yr}",
)

w("PV = present value, FV = future value (annual cash flow), r = yearly interest rate, n = number of years")
w("n= 10 years, r = 0.10")

def present_value(F, r, n):
    """
    Compute the present value of an annual cash flow F over n years at interest rate r.

    Args:
        F : annual cash flow ($/year)
        r : yearly interest rate (decimal)
        n : number of years

    Returns:
        present value of the cash flows ($)
    """
    pv_factor = 1/(1+r)**n
    return F * pv_factor

def NPV(F,r,n, C0):
    """
    Compute the net present value of an annual cash flow F over n years at interest rate r,
    given an initial cost C0.

    Args:
        F : annual cash flow ($/year)
        r : yearly interest rate (decimal)
        n : number of years
        C0 : initial cost ($)

    Returns:
        net present value ($)
    """
    return present_value(F,r,n) + C0

def annual_payment(C0, r, n):
    """
    Compute the annual payment required to amortize a loan of amount C0 over n years
    at yearly interest rate r.

    Args:
        C0 : loan principal amount ($)
        r : yearly interest rate (decimal)
        n : number of years

    Returns:
        annual payment ($/year)
    """
    return C0 * ((r*(1 + r)**n) / ((1+r)**n-1))

i_npv = 0.10 # 10% interest rate
i_loan = 0.05     # 5% loan interest rate
n=10
F_A = 1.1e6
F_B = 1.41e6
C0_A = -3.8e6
C0_B = -5.0e6
P0_A = -C0_A
P0_B = -C0_B
F, C0, n1, r= sp.symbols(names='F C_0 n r')

NPV_A = NPV(F_A, i_npv, n, C0_A)
NPV_B = NPV(F_B, i_npv, n, C0_B)
P_A = annual_payment(P0_A, i_loan, n)
P_B = annual_payment(P0_B, i_loan, n)
PV_symp = sp.sympify(present_value(F, r, n=n1))
NPV_symp = sp.sympify(NPV(F, r, n=n1, C0=C0),)
P_symp = sp.sympify(annual_payment(C0, r, n=n1))
print(PV_symp)
print(NPV_symp)
print(P_symp)

PV_latex = sp.latex(PV_symp,mul_symbol = 'dot')
NPV_latex = sp.latex(NPV_symp,mul_symbol = 'dot')
P_latex = sp.latex(P_symp, mul_symbol = 'dot')
print(PV_latex)
print(NPV_latex)
print(P_latex)


doc.subsection("A.) Compute Net Present Value for Options A and B")
a(
    rf"\text{{Present Value formula:}}\ PV &= {PV_latex}",
    rf"\text{{Net Present Value formula:}}\ \text{{NPV}} &= {NPV_latex}",
    rf"\text{{Annual payment formula:}}\ P &= {P_latex}",
)
m(rf"\text{{Annual payment for option A:}}\ P_A = \${-P_A:,.2f}/\text{{year}}") 
m(rf"\text{{Annual payment for option B:}}\ P_B = \${-P_B:,.2f}/\text{{year}}")
m(rf"\text{{NPV}}_A = \${NPV_A:,.2f},\quad \text{{NPV}}_B = \${NPV_B:,.2f}")

if NPV_A > NPV_B:
    w(f"NPV is higher for option A at 10% yearly interest, so A is preferred under these assumptions.")
else:
    w(f"NPV is higher for option B at 10% yearly interest, so B is preferred under these assumptions.")

doc.subsection("B.) Annual Payment for 10 Year Lifetime, No Salvage Value, 5% Interest Rate")
w()
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
NPV_eq = NPV(F=net_additional_income, r=interest_rate, n=n1, C0=-separator_cost)
dNPV_eq = sp.diff(NPV_eq,n1)
dNPV_eq = sp.lambdify(n1, dNPV_eq)
NPV_latex = sp.latex(NPV_eq)
print(NPV_eq)
print(NPV_latex)

# Solve for n
n_guess = 3
n_solution = sp.solve(NPV_eq, n)
m(rf"\text{{The NPV equation for the payback period is:}} {NPV_latex}")
w(f"Discounted payback period (NPV = 0): {n_solution} years")
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
    
x = sp.symbols('x')
f = 1 + 8*x + 2*x**2 - sp.Rational(10,3)*x**3 - sp.Rational(1,4)*x**4 + sp.Rational(4,5)*x**5 - sp.Rational(1,6)*x**6
func = sp.lambdify(x,-f)
df = sp.diff(f,x)
dfunc = sp.lambdify(x,-df)
d2f = sp.diff(df,x)
d2func = sp.lambdify(x,-d2f)
dfun = (1 + x)**2 * (2 - x)**3
d2fun = sp.diff(dfun,x)

m(rf"f(x) = {sp.latex(f)}")
w(text="Given that:")
m(rf"\frac{{df}}{{dx}}= {sp.latex(dfun)}")
m(rf"\frac{{d^2f}}{{dx^2}}= {sp.latex(d2fun)}")
doc.subsection("a.) Analytical Solution")
w("Expanded form:")
m(rf"\frac{{df}}{{dx}} = {sp.latex(df)}")
m(rf"\frac{{d^2f}}{{dx^2}} = {sp.latex(d2f)}")

sol = sp.solve(df,x)
f_sol = [f.subs(x,s) for s in sol]
w(f"The roots to df/dx are x* = {sol[0]}, {sol[1]}")
w(f"The corresponding function values are f(x*) = {f_sol[0]:.2f}, {f_sol[1]:.2f}")
px(rf"The Hessian evaluated at the critical points are",im(r"\frac{d^2f}{dx^2}|_{x^*}"),f" = {d2func(sol[0])}, {d2func(sol[1])}, inconclusive 2nd derivative test")

doc.subsection("b.) Excel Solution")
df = pd.read_csv("HW2_CHE565.csv")
df.drop('Unnamed: 6',axis=1,inplace=True)

df_A = df.iloc[:,:6].copy()
df_B = df.iloc[:,6:].copy()
df_A.columns = ["Iteration","x_n","f(x_n)","f'(x_n)","f''(x_n)","x_n+1"]
df_B.columns = ["Iteration","x_n","f(x_n)","f'(x_n)","f''(x_n)","x_n+1"]
df_A_latex = sp.latex(df_A)
df_B_latex = sp.latex(df_B)
headers_A = df_A.columns.tolist()
row_A = df_A.values.tolist()
headers_B = df_B.columns.tolist()
row_B = df_B.values.tolist()
summary = pd.DataFrame([
                    [5,df_A.iloc[-1]["x_n+1"], df_A.iloc[-1]["f(x_n)"]],
                    [-5, df_B.iloc[-1]["x_n+1"], df_B.iloc[-1]["f(x_n)"]]],
                    columns=["x0","x*","f(x*)"])
summary_headers = summary.columns.tolist()
summary_rows = summary.values.tolist()

w("The Excel data was used to perform root-finding using the Newton-Raphson method:")
m(rf"x_{{n+1}} = x_n - \frac{{f'(x_n)}}{{f''(x_n)}}")
t(headers_A,row_A,alignment='c'*len(headers_A),caption="Optimization Results from Excel starting at x0=5")
t(headers_B,row_B,alignment='c'*len(headers_B),caption="Optimization Results from Excel starting at x0=-5")
t(summary_headers,summary_rows,alignment='c'*len(summary_headers),caption="Summary of Optimization Results from Excel")

doc.subsection(title="c.) Numerical Solution Using Nelder-Mead Simplex algorithm which is the same as fminsearch in matlab")

x_guess = 5
xmin = minimize(fun=func,x0=x_guess, method='Nelder-Mead')
xmin1 = xmin.x
funmin = xmin.fun
w(f"Using Nelder-Mead Simplex algorithm with initial guesses x0 = {x_guess}: ")
w(f"One of the roots of df/dx are found to be x* = {xmin1}")
w(f"The corresponding optimum function values are f(x*) = {-funmin}")
w(f"Actual maximum of the function:")
a(
    rf"x^* &=  {xmin.x}",
    rf"f(x^*) &= {xmin.fun}",
    rf"\text{{iterations}} &= {xmin.nit}",
    rf"\text{{function calls}} &= {xmin.nfev}"
)

doc.subsection("d.) Numerical Solution Using Newton Conjugate Gradient Method which is the same as fmincon in matlab except required to give jacobian and hessian information")
x_guess2 = -5
xmin1_ncg = minimize(func,x_guess, method='Newton-CG', jac=dfunc, hess=d2func)
xmin2_ncg = minimize(func,x_guess2, method='Newton-CG', jac=dfunc, hess=d2func)
w(f"Using the Newton Conjugate Gradient method with initial guesses x0 = {x_guess}: ")
w(f"One of the roots of df/dx are found to be x* = {xmin1_ncg.x}")
w(f"The corresponding optimum function values are f(x*) = {xmin1_ncg.fun}")
a(
    rf"x^* &= {xmin1_ncg.x}",
    rf"f(x^*) &= {xmin1_ncg.fun}",
    rf"\text{{iterations}} &= {xmin1_ncg.nit}",
    rf"\text{{function calls}} &= {xmin1_ncg.nfev}"

)

w(f"Using the Newton Conjugate Gradient method with initial guesses x0 = {x_guess2}: ")
w(f"One of the roots of df/dx are found to be x* = {xmin2_ncg.x}")
w(f"The corresponding optimum function values are f(x*) = {xmin2_ncg.fun}")
a(
    rf"x^* &= {xmin2_ncg.x}",
    rf"f(x^*) &= {xmin2_ncg.fun}",
    rf"\text{{iterations}} &= {xmin2_ncg.nit}",
    rf"\text{{function calls}} &= {xmin2_ncg.nfev}"
)
t(
    ["Method","Initial Guess","x*","f(x*)","Iterations","Function Calls"],
    [
        ["Nelder-Mead", x_guess, xmin.x, xmin.fun, xmin.nit, xmin.nfev],
        ["Excel", x_guess, df_A.iloc[-1]["x_n+1"], df_A.iloc[-1]["f(x_n)"],im(f"{len(df_A)}"),"-"],
        ["Excel", x_guess2, df_B.iloc[-1]["x_n+1"], df_B.iloc[-1]["f(x_n)"], im(f"{len(df_B)}"), "-"],
        ["Newton-CG", x_guess, xmin1_ncg.x, xmin1_ncg.fun, xmin1_ncg.nit, xmin1_ncg.nfev],
        ["Newton-CG", x_guess2, xmin2_ncg.x, xmin2_ncg.fun,xmin2_ncg.nit, xmin2_ncg.nfev],
        ]

)

txt_file, tex_file, pdf_file = doc.save_all()

print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX source: {tex_file}")
print(f"Wrote PDF: {pdf_file}")

