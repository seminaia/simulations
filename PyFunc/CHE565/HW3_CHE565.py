"""
HW2_CHE565.py
=============
CHE 565 – Homework 2 All five problems solved with full work shown. Results are
written to HW2_CHE565_results.txtseminaia401@gmail..com and mirrored to the console.
"""

from math import e
import numpy as np
import sympy as sp
from sympy import Add, Function, LessThan, Mul 
import matplotlib
from sympy.abc import alpha, phi
from sympy.functions.combinatorial.factorials import rf
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
from scipy.stats import t as t_dist
import pandas as pd
from NRroots import newton_raphson
from regression_analysis import RegressionAnalysis
from doc_builder import DocumentBuilder
from sympy.vector import CoordSys3D, Del
OUTPUT_FILE = "HW3_CHE565"
PLOT_FILE = "HW3_CHE565_plot.png"
report_lines = []

doc = DocumentBuilder(
    OUTPUT_FILE,
    title="CHE 565 -- Homework 3",
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
doc.subsection("Steepest Descent ")
x1, x2, x3, x4= sp.symbols('x1 x2 x3 x4')
f = sp.Function('f')(x1, x2)
expr= -x1-x2+1/2*(x1**2+2*x1*x2+2*x2**2)
f_eq = sp.Eq(f, expr)
f0_eq = sp.lambdify((x1, x2), expr)
x0 = sp.Matrix([1,1])
f0 = f0_eq(*x0)
Df = sp.Matrix([expr]).jacobian((x1, x2)).T
Df_eq = sp.lambdify((x1, x2), Df)
Df0 = sp.Matrix(Df_eq(*x0))
x_old = sp.Matrix([x1, x2])
x_new = sp.Matrix([x3, x4])
g1 = sp.Function('g1')(alpha)
g2 = sp.Function('g2')(alpha)
g = sp.Matrix(2, 1, [g1, g2])

expr_g = x0 - alpha*Df0
g_eq = sp.Eq(g,expr_g).subs({x1: x0[0], x2: x0[1]})

phi = Function('phi')(alpha)
grad_phi = sp.diff(phi, alpha)
phi_k = expr.subs({x1: expr_g[0], x2: expr_g[1]}).subs({x1: x0[0], x2: x0[1]})
phi_k_collected = sp.collect(phi_k, alpha)
phi_k_simplified = sp.simplify(expr=phi_k_collected)
dphi_k = sp.diff(phi_k_simplified, alpha)
alpha_opt = sp.Matrix(sp.solve(dphi_k, alpha))
x_next = expr_g.subs(alpha, alpha_opt[0]).subs({x1: x0[0], x2: x0[1]})
f_next = expr.subs({x1: x_next[0], x2: x_next[1]})

doc.p("Function :")
m(sp.latex(f_eq))
w(rf"At x0 ={x0}, f(x0) = {f0}")
w("Gradient of f:")
m(r"\nabla f = " + sp.latex(sp.simplify(Df), mat_str='pmatrix'))
w(rf"At x0 ={x0}, Df(x0) = {Df0[0]:.2f}, {Df0[1]:.2f}")
m(sp.latex(g_eq))

w("Function phi_k:")
m(sp.latex(sp.Eq(phi, phi_k_simplified)))
w("Derivative of phi_k:")
m(sp.latex(sp.Eq(grad_phi, dphi_k)))
m(sp.latex(sp.Eq(0, dphi_k)))
w(rf"Optimal alpha: {alpha_opt[0]:.2f}")
w(rf"Next iterate x1: {x_next[0]:.2f}, x2: {x_next[1]:.2f}")
w(rf"Function value at next iterate: {f_next:.2f}")

doc.section("Problem 2")
doc.subsection("Conjugate Gradient Method")
s0 = -Df0
s1= sp.Function('s1')(alpha)
s2= sp.Function('s2')(alpha)
s = sp.Matrix(2, 1, [s1, s2])
expr_s = -Df_eq(*x_new) + s0*Df_eq(*x_new).T*Df_eq(*x_new)/Df_eq(*x_old).T*Df_eq(*x_old)
s_eq = sp.Eq(s, expr_s).subs({x1: x0[0], x2: x0[1], x3: x_next[0], x4: x_next[1]})
m(sp.latex(s_eq))

doc.section(title="Problem 3")
doc.section(title="Problem 3")
doc.subsection("Blending LP Formulation")

# Tables (unchanged)
rows_A = [
    [1, 3000, 26.00],      
    [2, 2000, 30.60],
    [3, 4000, 29.20],
    [4, 1000, 29.80]
]
headings_A = ["Constituent", "Max quantity (bbl/day)", "Production Cost ($/bbl)"]
t(headings_A, rows_A, float_fmt=".2f")

rows_B = [
    ["A", "Not more than 15% of 1; Not less than 40% of 2", 32.40],
    ["B", "Not more than 50% of 3; Not more than 10% of 1", 31.50],
    ["C", "Not less than 10% of 2; Not more than 20% of 1", 30.60]
]
headings_B = ["Grade", "Specifications", "Selling Price ($/bbl)"]
t(headings_B, rows_B, float_fmt=".2f", alignment="c p{4.5cm} c")

# Decision variables
a1, a2, a3, a4 = sp.symbols('a_1 a_2 a_3 a_4', nonnegative=True)
b1, b2, b3, b4 = sp.symbols('b_1 b_2 b_3 b_4', nonnegative=True)
c1, c2, c3, c4 = sp.symbols('c_1 c_2 c_3 c_4', nonnegative=True)

# Total production of each grade
A= sp.MatrixSymbol('A',1,4)
B= sp.MatrixSymbol('B',1,4)
C= sp.MatrixSymbol('C',1,4)
A_mat = [a1, a2, a3, a4]
B_mat = [b1, b2, b3, b4]
C_mat = [c1, c2, c3, c4]
A_sum = sum(A_mat)
B_sum = sum(B_mat)
C_sum = sum(C_mat)
tot_prod = sp.MatAdd(A,B,C)
comp_tot = sp.MatAdd(sp.Matrix(A_mat), sp.Matrix(B_mat), sp.Matrix(C_mat))
tot_prod_eq = sp.latex(sp.Eq(tot_prod,)))
m(sp.latex(sp.Eq(A.apart(), A_sum)))
m(sp.latex(sp.Eq(B.apart(), B_sum)))
m(sp.latex(sp.Eq(C.apart(), C_sum)))
m(tot_prod_eq)
price_mat = sp.Matrix([32.40,31.50,30.60])
grade_mat = sp.Matrix([A_sum, B_sum, C_sum])
# Objective: Maximize Profit
revenue = price_mat.dot(grade_mat)
cost_mat = sp.Matrix([26.00, 30.60, 29.20, 29.80])
cost_1 = a1+b1+c1
cost_2 = a2+b2+c2
cost_3 = a3+b3+c3
cost_4 = a4+b4+c4
cost = cost_mat.dot(sp.Matrix([cost_1, cost_2, cost_3, cost_4]))

profit_expr = revenue - cost
P = sp.symbols('P', nonnegative=True)
profit_eq = sp.Eq(P, profit_expr)

# Constraints list (for display and later solving)
constraints = [
    # Grade A specs
    (a1 <= 0.15*A_sum),
    (a2 >= 0.40*A_sum),
    # Grade B specs
    (b3 <= 0.50*B_sum),
    (b1 <= 0.10*B_sum),
    # Grade C specs
    (c2 >= 0.10*C_sum),
    (c1 <= 0.20*C_sum),
    # Availability limits
    (a1 + b1 + c1 <= 3000),
    (a2 + b2 + c2 <= 2000),
    (a3 + b3 + c3 <= 4000),
    (a4 + b4 + c4 <= 1000),
]

# Display using doc.align() for clean multi-line equations
w("Decision Variables:")
px(im(sp.latex(A)), im(sp.latex(B)), im(sp.latex(C)))
w("Total production:")
a(
  rf"\text{{A=}}"+sp.latex(A),
  rf"\text{{B=}}"+sp.latex(B), 
  rf"\text{{C=}}"+sp.latex(C)
)
prod_tot = A + B + C 
prod = sp.Matrix([3000, 2000, 4000, 1000])
w("Objective function (Profit):")
m(sp.latex(profit_eq))
w("Constraints:")
for c in constraints:
    m(sp.latex(c))
txt_file, tex_file, pdf_file = doc.save_all()

print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX source: {tex_file}")
print(f"Wrote PDF: {pdf_file}")
