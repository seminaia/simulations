"""
HW2_CHE565.py
=============
CHE 565 – Homework 2 All five problems solved with full work shown. Results are
written to HW2_CHE565_results.txt and mirrored to the console.
"""

from math import e
import numpy as np
import sympy as sp
from sympy import Add, Mul 
import matplotlib
from sympy.abc import alpha
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
doc.subsection("Setup")
x1, x2= sp.symbols('x1 x2')
f = sp.Function('f')(x1, x2)
expr= -x1-x2+1/2*(x1**2+2*x1*x2+2*x2**2)
f_eq = sp.Eq(f, expr)
f0_eq = sp.lambdify((x1, x2), expr)
x0 = [1,1]
f0 = f0_eq(*x0)
Df = sp.Matrix([expr]).jacobian((x1, x2)).T
Df_eq = sp.lambdify((x1, x2), Df)
Df0 = Df_eq(*x0)
x = sp.Matrix([x1, x2])
g1 = sp.Function('g1')(alpha)
g2 = sp.Function('g2')(alpha)
g = sp.FunctionMatrix(2, 1, g1)
g_vec = sp.Matrix([g1,g2])

expr_g = x - alpha*Df
g_eq = sp.Eq(g,expr_g)
sp.pprint(g_eq)
g0_eq = sp.lambdify(alpha, expr_g)

#phi_k_eq = sp.Eq(phi_k, phi_k)
#phi_k_lam = sp.lambdify(alpha, phi_k)
#alpha_opt = minimize(phi_k_lam, 0.1).x[0]
doc.p("Function :")
m(sp.latex(f_eq))
w(rf"At x0 ={x0}, f(x0) = {f0}")
w("Gradient of f:")
m(sp.latex(Df))
w(rf"At x0 ={x0}, Df(x0) = {Df0}")
w("Function g:")
m(sp.latex(g_eq))
#w("Function phi_k:")
#m(sp.latex(phi_k_eq))
#w(rf"Optimal alpha: {alpha_opt:.4f}")

txt_file, tex_file, pdf_file = doc.save_all()

print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX source: {tex_file}")
print(f"Wrote PDF: {pdf_file}")
