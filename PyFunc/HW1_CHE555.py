"""
HW1_CHE555.py
=============
CHE 555 – Homework 1
All five problems solved with full work shown.
Results are written to HW1_CHE555_results.txt and mirrored to the console.

Requires:  numpy, scipy, matplotlib, regression_analysis.py (same directory)
Run:       python HW1_CHE555.py
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

OUTPUT_FILE = "HW1_CHE555_results.txt"
ra = RegressionAnalysis(output_file=OUTPUT_FILE, verbose=True)
w  = ra._w.write      # shortcut to the writer

w()
w("  CHE 555 — Homework 1")
w("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM 1
#  Find the point on y = 2x² + 3x + 1 nearest to the origin.
# ══════════════════════════════════════════════════════════════════════════════

w()
w("━" * 80)
w("  PROBLEM 1")
w("━" * 80)
w()
w("  SETUP")
w("  ─────")
w("  Curve: y = 2x² + 3x + 1")
w("  The distance from a point (x, y(x)) to the origin is")
w("      D(x) = sqrt[ x² + y(x)² ]")
w("  Minimising D² :")
w("      D²(x) = x² + (2x² + 3x + 1)²")
w("      dD²(x*)/dx = 2x + 2(2x² + 3x + 1)(4x + 3) = 0")
w("      2x + 2(2x² + 3x + 1)(4x + 3) = 0 -> 8x³ + 22x² + 19x + 3 = 0")
w("      The roots of this cubic equations were solved numerically using the Newton-Raphson method.")

def dist_sq_p1(x):
    y = 2*x**2 + 3*x + 1
    return x**2 + y**2
def dx_dist_sq_p1(x):
    return 2*x + 2*(2*x**2 + 3*x + 1)*(4*x + 3)
def dx2_dist_sq_p1(x):
    return 2 + 2*(4*x + 3)**2 + 16*(2*x**2 + 3*x + 1)

x0 = 5
x_opt, es, iter = newton_raphson(dx_dist_sq_p1, dx2_dist_sq_p1 ,x0)
x1   = x_opt
y1   = 2*x1**2 + 3*x1 + 1
d1   = np.sqrt(dist_sq_p1(x1))

w("  ────")
w(f"  Optimal x      = {x1:.8f}")
w(f"  y(x*)          = 2({x1:.6f})² + 3({x1:.6f}) + 1")
w(f"                 = {y1:.8f}")
w(f"  D²(x*)         = ({x1:.6f})² + ({y1:.6f})²")
w(f"                 = {dist_sq_p1(x1):.8f}")
w(f"  D(x*)          = √{dist_sq_p1(x1):.8f} = {d1:.8f}")
w()
w("  ANSWER")
w("  ──────")
w(f"  Nearest point : ({x1:.6f},  {y1:.6f})")
w(f"  Distance      : {d1:.6f}")
w(f"  error estimate : {es:.2e}")
w(f"  iterations     : {iter}")
# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM 2
#  Minimise material (surface area) for a square-base, open-top box with
#  V = 1000 cm³.
# ══════════════════════════════════════════════════════════════════════════════

w()
w("━" * 80)
w("  PROBLEM 2")
w("━" * 80)
w()
w("  Variables : x = side length of square base (cm)")
w("              h = height (cm)")
w()
w("  Volume constraint : x² · h = 1000  ->  h = 1000 / x²")
w()
w("  Surface area (no top lid) :")
w("      S(x) = x²  +  4·x·h")
w("           = x²  +  4·x·(1000/x²)")
w("           = x²  +  4000/x")
w()
w("  ANALYTICAL SOLUTION")
w("  ───────────────────")
w("  dS/dx = 2x − 4000/x²  = 0")
w("  2x³ = 4000")
w("  x³  = 2000")
w("  x*  = 2000^(1/3)")
w()
w("  d²S/dx² = 2 + 8000/x³  > 0  for all x > 0")
w()

x2   = 2000 ** (1/3)
h2   = 1000 / x2**2
S2   = x2**2 + 4000 / x2
d2S  = 2 + 8000 / x2**3    # second derivative check

w("  NUMERICAL VALUES")
w("  ─────────────────")
w(f"  x* = 2000^(1/3)  = {x2:.6f} cm")
w(f"  h* = 1000 / x*²  = 1000 / {x2:.6f}²  = {h2:.6f} cm")
w(f"  S*  = ({x2:.4f})² + 4000/{x2:.4f} = {S2:.6f} cm²")
w(f"  d²S/dx²|x* = {d2S:.4f} > 0  ")
w()
w("  ANSWER")
w("  ──────")
w(f"  Optimal base side length : {x2:.4f} cm")
w(f"  Optimal height           : {h2:.4f} cm")
w(f"  Minimum surface area     : {S2:.4f} cm²")


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM 3
#  Variable / degree-of-freedom analysis for a constrained optimisation.
# ══════════════════════════════════════════════════════════════════════════════

w()
w("━" * 80)
w("  PROBLEM 3")
w("━" * 80)
w()
w("  ─────────────────")
w("  Minimise:   f(x₁, x₂) = 4x₁ − x₁² − 12")
w()
w("  Subject to:")
w("    g₁ : 25 − x₁² − x₂²  = 0")
w("    g₂ : 10x₁ − x₁² + 10x₂ − x₂² − 34 ≥ 0")
w("    g₃ : (x₁−3)² + (x₂−1)² ≥ 0")
w("    x₁, x₂ ≥ 0")
w()
w("  VARIABLE COUNT")
w("  ──────────────")
w("  Total Variables                  : 2   (x₁, x₂)")
w()
w("  DOF ANALYSIS")
w("  ───────────────────────────")
w("  Each independent equality constraint reduces DOF by 1.")
w("  Equality constraints        : 1")
w("  DOF = total variables − equality constraints = 2 − 1 = 1")
w()
w("  Number of independent variables  : 1")
w("  One valid choice                 : { x₁ }")
w()
w("  EXPLANATION")
w("  ───────────")
w("  E1 defines x₂ as a function of x₁:  x₂ = √(25 − x₁²)  (with x₂ ≥ 0).")
w("  Once x₁ is chosen (satisfying 0 ≤ x₁ ≤ 5), x₂ is fully determined.")
w("  The inequality constraints further restrict the feasible range of x₁")
w("  but do not reduce the number of independent variables.")
w("  g₃ = (x₁−3)² + (x₂−1)² ≥ 0 is always satisfied (sum of squares).")
w()
w("  ANSWER")
w("  ──────")
w("  Total variables            : 2")
w("  Independent variables      : 1")
w("  One possible choice        : { x₁ }")


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM 4
#  Oil refinery: minimise cost per barrel and maximise daily profit.
# ══════════════════════════════════════════════════════════════════════════════

w()
w("━" * 80)
w("  PROBLEM 4")
w("━" * 80)
w()
w("  GIVEN")
w("  ─────")
w("  Cost per barrel : C(P) = 50 + 0.1P + 9000/P   ($/barrel)")
w("  Selling price   : $300 / barrel")
w("  P = production rate (barrels/day)")
w()

# ── Part A: minimise cost per barrel ────────────────────────────────────────

w("  PART A – Minimise cost per barrel")
w("  ──────────────────────────────────")
w("  dC/dP = 0.1 − 9000/P²  = 0")
w("  P²  = 9000 / 0.1 = 90000")
w("  P*  = sqrt(90000) = 300 barrels/day")
w()
w("  d²C/dP² = 18000/P³  > 0  for P > 0  ")
w()

P4a  = 300.0
C4a  = 50 + 0.1*P4a + 9000/P4a
d2C  = 18000 / P4a**3

w(f"  Numerical check:")
w(f"    P*     = {P4a:.1f} barrels/day")
w(f"    C(P*)  = 50 + 0.1({P4a}) + 9000/{P4a} = {C4a:.2f} $/barrel")
w(f"    d²C/dP²|P* = {d2C:.6f} > 0 ")

# ── Part B: maximise daily profit ───────────────────────────────────────────

w()
w("  PART B – Maximise daily profit")
w("  ──────────────────────────────")
w("  Revenue per day  : R(P) = 300 · P")
w("  Total cost / day : TC(P) = C(P) · P = 50P + 0.1P² + 9000")
w("  Profit / day     : Π(P) = R − TC = 300P − 50P − 0.1P² − 9000")
w("                           = 250P − 0.1P² − 9000")
w()
w("  dΠ/dP = 250 − 0.2P  = 0")
w("  P*  = 250 / 0.2 = 1250 barrels/day")
w()
w("  d²Π/dP² = −0.2  < 0")
w()

P4b  = 1250.0
Rev4 = 300*P4b
TC4  = 50*P4b + 0.1*P4b**2 + 9000
Pi4  = Rev4 - TC4

w(f"  Numerical check:")
w(f"    P*           = {P4b:.0f} barrels/day")
w(f"    Revenue      = 300 × {P4b:.0f}             = ${Rev4:,.2f}/day")
w(f"    Total cost   = 50({P4b:.0f}) + 0.1({P4b:.0f})² + 9000 = ${TC4:,.2f}/day")
w(f"    Profit Π(P*) = ${Rev4:,.2f} − ${TC4:,.2f}   = ${Pi4:,.2f}/day")
w()
w("  ANSWERS")
w("  ───────")
w(f"  (A) Production minimising cost/barrel : P = {P4a:.0f} barrel/day  -> C = ${C4a:.2f}/barrel")
w(f"  (B) Production maximising daily profit: P = {P4b:.0f} barrel/day  -> Π = ${Pi4:,.2f}/day")


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM 5
#  Curve fitting: compare three models using regression statistics.
# ══════════════════════════════════════════════════════════════════════════════

w()
w("━" * 80)
w("  PROBLEM 5")
w("━" * 80)
w()

x_data = np.array([10, 20, 30, 40, 50], dtype=float)
y_data = np.array([1.00, 1.26, 1.86, 3.31, 7.08])

w("  DATA")
w("  ────")
w(f"  {'x':>6}  {'y':>8}")
w(f"  {'──────':>6}  {'──────':>8}")
for xi, yi in zip(x_data, y_data):
    w(f"  {xi:>6.0f}  {yi:>8.2f}")

w()
w("  MODELS TO FIT")
w("  ─────────────")
w("  Model 1 : y = exp(a + b·x)            [exp-linear]")
w("  Model 2 : y = exp(a + b·x + c·x²)     [exp-quadratic]")
w("  Model 3 : y = a · x^b                 [power law]")
w()
w("  METHOD")
w("  ──────")
w("  Models 1 & 2 are linearised by taking ln y and fitting via OLS.")
w("  Model 3 is linearised as ln y = ln a + b·ln x and fitted via OLS.")
w("  Full regression statistics (R², adj-R², SE, RMSE, F-test, t-tests,")
w("  ANOVA table) are reported for each model in the sections below.")
w()
w("─" * 40)
w("Regression details:")
w("─" * 40)
w()

# Fit all three models (full stats printed & written by RegressionAnalysis)
model_results = ra.fit_all_models(x_data, y_data)

# ── Pull best-model info for the conclusion ───────────────────────────────
models_summary = {
    "exp-linear":    model_results["exp_linear"].get("R2_adj", np.nan),
    "exp-quadratic": model_results["exp_quadratic"].get("R2_adj", np.nan),
    "power-law":     model_results["power_law"].get("R2_adj_orig_scale",
                     model_results["power_law"].get("R2_adj", np.nan)),
}
best5 = max(models_summary, key=lambda k: models_summary[k])

# ── Fitted-value comparison table ────────────────────────────────────────────
el_pred = model_results["exp_linear"]["y_pred_original"]
ep_pred = model_results["exp_quadratic"]["y_pred_original"]
pw_pred = model_results["power_law"]["y_pred_original"]

w()
w("  FITTED VALUES vs. DATA")
w("  ─────────────────────────────────────────────────────────────────")
w(f"  {'x':>4}  {'y_obs':>8}  {'exp-lin':>10}  {'exp-quad':>10}  {'power':>10}")
w(f"  {'────':>4}  {'──────':>8}  {'────────':>10}  {'────────':>10}  {'────────':>10}")
for i in range(len(x_data)):
    w(f"  {x_data[i]:>4.0f}  {y_data[i]:>8.4f}  {el_pred[i]:>10.4f}  "
      f"{ep_pred[i]:>10.4f}  {pw_pred[i]:>10.4f}")

w()
w("  RESIDUALS")
w("  ─────────────────────────────────────────────────────────────────")
w(f"  {'x':>4}  {'exp-lin':>10}  {'exp-quad':>10}  {'power':>10}")
w(f"  {'────':>4}  {'────────':>10}  {'────────':>10}  {'────────':>10}")
for i in range(len(x_data)):
    w(f"  {x_data[i]:>4.0f}  {y_data[i]-el_pred[i]:>10.4f}  "
      f"{y_data[i]-ep_pred[i]:>10.4f}  {y_data[i]-pw_pred[i]:>10.4f}")

w()
w("  CONCLUSION")
w("  ──────────")
w(f"  The exp-quadratic model has the highest adjusted R²")
w(f"  among the three candidates.")
w(f"  Best model: {best5}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT  (saved as PNG so it works without a display)
# ══════════════════════════════════════════════════════════════════════════════

x_plot = np.linspace(10, 50, 300)

# Reconstruct model predictions for the plot
el_a, el_b = model_results["exp_linear"]["params"]
ep_a, ep_b, ep_c = model_results["exp_quadratic"]["params"]
pw_lna, pw_b = model_results["power_law"]["params"]
pw_a = np.exp(pw_lna)

y_el = np.exp(el_a + el_b * x_plot)
y_ep = np.exp(ep_a + ep_b * x_plot + ep_c * x_plot**2)
y_pw = pw_a * x_plot ** pw_b
    
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("CHE 555 HW1 – Problem 5: Curve Fitting", fontsize=13, fontweight="bold")

# Left: all three fits
ax = axes[0]
ax.plot(x_data, y_data, 'ko', ms=7, zorder=5, label="Data")
ax.plot(x_plot, y_el, '-',  lw=2, label=f"exp-linear  (adj R²={models_summary['exp-linear']:.4f})")
ax.plot(x_plot, y_ep, '--', lw=2, label=f"exp-quad    (adj R²={models_summary['exp-quadratic']:.4f})")
ax.plot(x_plot, y_pw, '-.', lw=2, label=f"power law   (adj R²={models_summary['power-law']:.4f})")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Model fits"); ax.legend(fontsize=8); ax.grid(alpha=0.4)

# Right: residuals
ax2 = axes[1]
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.plot(x_data, y_data - el_pred, 'o-',  ms=6, label="exp-linear")
ax2.plot(x_data, y_data - ep_pred, 's--', ms=6, label="exp-quad")
ax2.plot(x_data, y_data - pw_pred, '^:',  ms=6, label="power law")
ax2.set_xlabel("x"); ax2.set_ylabel("Residual  (y_obs − y_fit)")
ax2.set_title("Residuals"); ax2.legend(fontsize=8); ax2.grid(alpha=0.4)

plt.tight_layout()
PLOT_FILE = "HW1_CHE555_plot.png"
plt.savefig(PLOT_FILE, dpi=150)
plt.close()
ra.close()

print(f"Full solution written to  {OUTPUT_FILE}")
print(f"Plot saved to            {PLOT_FILE}")
from pathlib import Path
import shutil
import subprocess

def build_latex_pdf():
    tex_file = "CHE555_HW1.tex"

    latex = rf"""
\documentclass[12pt]{{article}}
\usepackage{{fontspec}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{fancyvrb}}
\usepackage{{graphicx}}
\usepackage{{float}}

\setmonofont{{DejaVu Sans Mono}}

\title{{CHE 555 -- Homework 1}}
\date{{}}

\begin{{document}}

\maketitle

\section*{{Full Solution}}

\VerbatimInput[
    fontsize=\scriptsize,
    frame=single,
    framesep=2mm
]{{{OUTPUT_FILE}}}

\newpage
\section*{{Plot}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{{PLOT_FILE}}}
\caption{{Curve fitting comparison for Problem 5.}}
\end{{figure}}

\end{{document}}
"""

    Path(tex_file).write_text(latex, encoding="utf-8")

    engine = shutil.which("lualatex")
    if engine is None:
        raise RuntimeError("lualatex not found")

    subprocess.run([engine, "-interaction=nonstopmode", tex_file], check=True)

    print("PDF generated")

build_latex_pdf()
