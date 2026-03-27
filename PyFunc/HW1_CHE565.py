"""
HW1_CHE565.py
=============
CHE 565 – Homework 1
All five problems solved with full work shown.
Results are written to HW1_CHE565_results.txt and mirrored to the console.
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
from pathlib import Path
from pylatex import Document, Section, Subsection, Figure, NoEscape, Command
from worklog import WorkLog
# ══════════════════════════════════════════════════════════════════════════════
#  Output file setup
#  All problems write through the same RegressionAnalysis writer so the
#  complete solution ends up in one tidy file.
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_FILE = "HW1_CHE565_results.txt"
log = WorkLog("HW1_CHE565","CHE 565 Homework 1")
log.title("CHE 565 - Homework 1")
log.section("Problem 1")
log.text("")

report_lines = []

ra = RegressionAnalysis(output_file=OUTPUT_FILE, verbose=False)


def w(text=""):
    log.text(text)
def wm(tex=""):
    log.math(tex)
w()
w("  CHE 565 — Homework 1")
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
wm(r"  Curve: y = 2x^2 + 3x + 1")
w("  The distance from a point (x, y(x)) to the origin is")
wm(r"      D(x) = \sqrt{ x^2 + y(x)^2 }")
wm(r"  Minimising D^2 :")
wm(r"      D^2(x) = x^2 + (2x^2 + 3x + 1)^2")
wm(r"      \frac{dD^2(x*)}{dx} = 2x + 2(2x^2 + 3x + 1)(4x + 3) = 0")
wm(r"      2x + 2(2x^2 + 3x + 1)(4x + 3) = 0 ")
wm(r"      2x + (4x^2 + 6x + 2)(4x + 3) = 0 ")
wm(r"      2x + 16x^3 + 24x^2 + 8x + 12x^2 + 18x + 6 = 0")
wm(r"      8x^3 + 18x^2 + 14x + 3 = 0")
w("The roots of this cubic equations were solved numerically using the Newton-Raphson method.")

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

w()
wm(rf"  x^*     = {x1:.8f}")
wm(rf"  y(x^*)  = 2({x1:.6f})^2 + 3({x1:.6f}) + 1")
wm(rf"          = {y1:.8f}")
wm(rf"  D^2(x^*) = ({x1:.6f})^2 + ({y1:.6f})^2")
wm(rf"          = {dist_sq_p1(x1):.8f}")
wm(rf"  D(x^*)  = \sqrt{{{dist_sq_p1(x1):.8f}}} = {d1:.8f}")
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
wm(r"  Volume constraint : x^2 · h = 1000  ->  h = 1000 / x^2")
w()
w("  Surface area (no top lid) :")
wm(r"      S(x) = x^2  +  4·x·h")
wm(r"           = x^2  +  4·x·(1000/x^2)")
wm(r"           = x^2  +  4000/x")
w()
w("  ANALYTICAL SOLUTION")
w("  ───────────────────")
wm(r"  \frac{dS}{dx} = 2x − 4000/x^2  = 0")
wm(r"  2x^3 = 4000")
wm(r"  x^3  = 2000")
wm(r"  x*  = 2000^(1/3)")
w()
wm(r"  \frac{d^2S}{dx^2} = 2 + 8000/x^3  > 0  for all x > 0")
w()

x2   = 2000 ** (1/3)
h2   = 1000 / x2**2
S2   = x2**2 + 4000 / x2
d2S  = 2 + 8000 / x2**3    # second derivative check

w("  NUMERICAL VALUES")
w("  ─────────────────")
wm(rf"  x* = 2000^(1/3)  = {x2:.6f} cm")
wm(rf"  h* = 1000 / x*²  = 1000 / {x2:.6f}²  = {h2:.6f} cm")
wm(rf"  S*  = ({x2:.4f})² + 4000/{x2:.4f} = {S2:.6f} cm²")
wm(rf"  d²S/dx²|x* = {d2S:.4f} > 0  ")
w()
w("  ANSWER")
w("  ──────")
wm(rf"  Optimal base side length : {x2:.4f} cm")
wm(rf"  Optimal height           : {h2:.4f} cm")
wm(rf"  Minimum surface area     : {S2:.4f} cm²")


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
wm(r"  Minimise:   f(x_1, x_2) = 4x_1 − x_1^2 − 12")
w()
w("  Subject to:")
wm(r"    g_1 : 25 − x_1^2 − x_2^2  = 0")
wm(r"    g_2 : 10x_1 − x_1^2 + 10x_2 − x_2^2 − 34 ≥ 0")
wm(r"    g_3 : (x_1−3)^2 + (x_2−1)^2 ≥ 0")
wm(r"    x_1, x_2 ≥ 0")
w()
w("  VARIABLE COUNT")
w("  ──────────────")
wm(r"  Total Variables                  : 1 (x_1)")
w()
w("  DOF ANALYSIS")
w("  ───────────────────────────")
w("  Each independent equality constraint reduces DOF by 1.")
w("  Equality constraints        : 1")
w("  DOF = total variables − equality constraints = 2 − 1 = 1")
w()
w("  Number of independent variables  : 1")
wm(r"  One valid choice                 : { x_1 }")
w()
w("  EXPLANATION")
w("  ───────────")
wm(r"  E1 defines x_2 as a function of x_1:  x_2 = \sqrt{25 − x_1^2}  (with x_2 \ge 0).")
wm(r"  Once x_1 is chosen (satisfying 0 \le x_1 \le 5), x_2 is fully determined.")
wm(r"  The inequality constraints further restrict the feasible range of x_1")
wm(r"  but do not reduce the number of independent variables.")
wm(r"  g_3 = (x_1−3)^2 + (x_2−1)^2  \ge 0 is always satisfied (sum of squares).")
w()
w("  ANSWER")
w("  ──────")
w("  Total variables            : 2")
w("  Independent variables      : 1")
w("  One possible choice        : { x_1 }")


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
wm(r"  \frac{dC}{dP} = 0.1 − 9000/P^2  = 0")
wm(r"  P^2  = 9000 / 0.1 = 90000")
wm(r"  P^*  = \sqrt{90000} = 300 barrels/day")
w()
wm(r"  \frac{d^2C}{dP^2} = 18000/P^3  > 0  for P > 0  ")
w()

P4a  = 300.0
C4a  = 50 + 0.1*P4a + 9000/P4a
d2C  = 18000 / P4a**3

w("  Numerical check:")
wm(rf"    P*     = {P4a:.1f} barrels/day")
wm(rf"    C(P*)  = 50 + 0.1({P4a}) + 9000/{P4a} = {C4a:.2f} $/barrel")
wm(rf"    d²C/dP²|P* = {d2C:.6f} > 0 ")

# ── Part B: maximise daily profit ───────────────────────────────────────────

w()
w("  PART B – Maximise daily profit")
w("  ──────────────────────────────")
w("  Revenue per day  : R(P) = 300 · P")
wm(r"  Total cost / day : TC(P) = C(P) · P = 50P + 0.1P^2 + 9000")
wm(r"  Profit / day     : \Pi(P) = R − TC = 300P − 50P − 0.1P^2 − 9000")
wm(r"                           = 250P − 0.1P^2 − 9000")
w()
wm(r"  d\Pi/dP = 250 − 0.2P  = 0")
wm(r"  P*  = 250 / 0.2 = 1250 barrels/day")
w()
wm(r"  d²\Pi/dP² = −0.2  < 0")
w()

P4b  = 1250.0
Rev4 = 300*P4b
TC4  = 50*P4b + 0.1*P4b**2 + 9000
Pi4  = Rev4 - TC4

w("  Numerical check:")
wm(rf"    P*           = {P4b:.0f} barrels/day")
wm(rf"    Revenue      = 300 × {P4b:.0f}             = ${Rev4:,.2f}/day")
wm(rf"    Total cost   = 50({P4b:.0f}) + 0.1({P4b:.0f})² + 9000 = ${TC4:,.2f}/day")
wm(rf"    Profit \Pi(P*) = ${Rev4:,.2f} − ${TC4:,.2f}   = ${Pi4:,.2f}/day")
w()
w("  ANSWERS")
w("  ───────")
wm(rf"  (A) Production minimising cost/barrel : P = {P4a:.0f} barrel/day  -> C = ${C4a:.2f}/barrel")
wm(rf"  (B) Production maximising daily profit: P = {P4b:.0f} barrel/day  -> \Pi = ${Pi4:,.2f}/day")


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
wm(r"  Model 1 : y = \exp(a + b·x)            [exp-linear]")
wm(r"  Model 2 : y = \exp(a + b·x + c·x²)     [exp-quadratic]")
wm(r"  Model 3 : y = a · x^b                 [power law]")
w()
w("  METHOD")
w("  ──────")
wm(r"  Models 1 & 2 are linearised by taking \ln y and fitting via OLS.")
wm(r"  Model 3 is linearised as \ln y = \ln a + b·\ln x and fitted via OLS.")
wm(r"  Full regression statistics (R^2, adj-R^2, SE, RMSE, F-test, t-tests,")
wm(r"  (ANOVA table) are reported for each model in the sections below.")
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
PLOT_FILE = "HW1_CHE565_plot.png"
plt.savefig(PLOT_FILE, dpi=150)
plt.close()
ra.close()
log.save_all()
print(f"Full solution written to  {OUTPUT_FILE}")
print(f"Plot saved to            {PLOT_FILE}")
