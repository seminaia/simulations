"""
HW2_CHE565.py
=============
CHE 565 – Homework 2 All five problems solved with full work shown. Results are
written to HW2_CHE565_results.txtseminaia401@gmail..com and mirrored to the console.
"""

from math import e
import numpy as np
import sympy as sp
from sympy import Add, Function, Mul 
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
rows_A = [
    [1, 3000, 26.00],      
    [2, 2000, 30.60],
    [3, 4000, 29.20],
    [4, 1000, 29.80]
]
headings_A = ["Constituents","Maximum Quantity (bbl/day)"," Production Cost ($/bbl)"]
t(headings_A, rows_A,float_fmt=".2f")

rows_B = [["A","No more than 15% of 1\n No more than 40% of 2",32.40],["B","No more than 50% of 3\\ No more than 10% of 1",31.50],["C","No less than 10% of 2\\ No more than 20% of 1", 30.60]]
headings_B = ["Grade","Specifications","Selling Price ($/bbl)"]
t(headings_B, rows_B,float_fmt=".2f")

txt_file, tex_file, pdf_file = doc.save_all()

print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX source: {tex_file}")
print(f"Wrote PDF: {pdf_file}")
