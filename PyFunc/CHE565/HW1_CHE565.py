"""
HW1_CHE565.py
=============
CHE 565 – Homework 1
All five problems solved with full work shown.
Results are written to HW1_CHE565.txt, HW1_CHE565.tex, and HW1_CHE565.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from NRroots import newton_raphson
from regression_analysis import RegressionAnalysis
from doc_builder import DocumentBuilder


# ============================================================================
# Output setup
# ============================================================================

doc = DocumentBuilder(
    "HW1_CHE565",
    title="CHE 565 -- Homework 1",
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

# ============================================================================
# PROBLEM 1
# Find the point on y = 2x² + 3x + 1 nearest to the origin.
# ============================================================================

doc.section("Problem 1")
doc.subsection("Setup")

w("Find the point nearest to the origin.")
w("The squared distance to the origin is minimized instead of the distance itself.")

m(r"y(x) = 2x^2 + 3x + 1")
m(r"D(x) = \sqrt{x^2 + y(x)^2}")
m(r"D^2(x) = x^2 + (2x^2 + 3x + 1)^2")

a(
    r"\frac{dD^2}{dx} &= 2x + 2(2x^2 + 3x + 1)(4x + 3)",
    r"&= 0"
)

a(
    r"2x + 2(2x^2 + 3x + 1)(4x + 3) &= 0",
    r"2x + (4x^2 + 6x + 2)(4x + 3) &= 0",
    r"2x + 16x^3 + 36x^2 + 26x + 6 &= 0",
    r"16x^3 + 36x^2 + 28x + 6 &= 0",
    r"8x^3 + 18x^2 + 14x + 3 &= 0"
)

w("The derivative equation was solved numerically using the Newton-Raphson method.")

def dist_sq_p1(x):
    y = 2*x**2 + 3*x + 1
    return x**2 + y**2

def dx_dist_sq_p1(x):
    return 2*x + 2*(2*x**2 + 3*x + 1)*(4*x + 3)

def dx2_dist_sq_p1(x):
    return 2 + 2*(4*x + 3)**2 + 16*(2*x**2 + 3*x + 1)

x0 = 5
x_opt, es, iters = newton_raphson(dx_dist_sq_p1, dx2_dist_sq_p1, x0)
x1 = x_opt
y1 = 2*x1**2 + 3*x1 + 1
d1 = np.sqrt(dist_sq_p1(x1))

doc.subsection("Numerical Results")

a(
    rf"x^* &= {x1:.8f}",
    rf"y(x^*) &= 2({x1:.6f})^2 + 3({x1:.6f}) + 1",
    rf"&= {y1:.8f}",
    rf"D^2(x^*) &= ({x1:.6f})^2 + ({y1:.6f})^2",
    rf"&= {dist_sq_p1(x1):.8f}",
    rf"D(x^*) &= \sqrt{{{dist_sq_p1(x1):.8f}}}",
    rf"&= {d1:.8f}"
)

doc.subsection("Answer")

w(f"Nearest point: ({x1:.6f}, {y1:.6f})")
w(f"Distance: {d1:.6f}")
w(f"Error estimate: {es:.2e}")
w(f"Iterations: {iters}")


# ============================================================================
# PROBLEM 2
# Minimise material (surface area) for a square-base, open-top box with
# V = 1000 cm³.
# ============================================================================

doc.section("Problem 2")

w("Minimize material usage for a square-base, open-top box with volume 1000 cm^3.")
w("Let x be the side length of the square base and h be the box height.")

m(r"V = x^2 h = 1000")
m(r"h = \frac{1000}{x^2}")
m(r"S(x) = x^2 + 4xh")
m(r"S(x) = x^2 + 4x\left(\frac{1000}{x^2}\right) = x^2 + \frac{4000}{x}")

doc.subsection("Analytical Solution")

a(
    r"\frac{dS}{dx} &= 2x - \frac{4000}{x^2}",
    r"&= 0"
)

a(
    r"2x^3 &= 4000",
    r"x^3 &= 2000",
    r"x^* &= 2000^{1/3}"
)

m(r"\frac{d^2S}{dx^2} = 2 + \frac{8000}{x^3} > 0 \quad \text{for } x>0")

x2 = 2000 ** (1/3)
h2 = 1000 / x2**2
S2 = x2**2 + 4000 / x2
d2S = 2 + 8000 / x2**3

doc.subsection("Numerical Results")

a(
    rf"x^* &= 2000^{{1/3}} = {x2:.6f}\ \text{{cm}}",
    rf"h^* &= \frac{{1000}}{{(x^*)^2}} = {h2:.6f}\ \text{{cm}}",
    rf"S^* &= (x^*)^2 + \frac{{4000}}{{x^*}} = {S2:.6f}\ \text{{cm}}^2",
    rf"\frac{{d^2S}}{{dx^2}}\bigg|_{{x^*}} &= {d2S:.4f} > 0"
)

doc.subsection("Answer")

w(f"Optimal base side length: {x2:.4f} cm")
w(f"Optimal height: {h2:.4f} cm")
w(f"Minimum surface area: {S2:.4f} cm^2")


# ============================================================================
# PROBLEM 3
# Variable / degree-of-freedom analysis for a constrained optimisation.
# ============================================================================

doc.section("Problem 3")
doc.subsection("Problem Statement")
a(
    r"\min_{x_1, x_2} \quad & f(x_1,x_2) = 4x_1 - x_1^2 - 12",
    r"\text{subject to} \quad & g_1: 25 - x_1^2 - x_2^2 = 0",
    r"& g_2: 10x_1 - x_1^2 + 10x_2 - x_2^2 - 34 \ge 0",
    r"& g_3: (x_1 - 3)^2 + (x_2 - 1)^2 \ge 0",
    r"& x_1, x_2 \ge 0"
)
doc.subsection("Degree-of-Freedom Analysis")

w("There are 2 variables: x1 and x2.")
w("There is 1 equality constraint, which reduces the degrees of freedom by 1.")
w("Inequality constraints restrict the feasible region but do not reduce the number of independent variables.")

a(
    r"\text{DOF} &= \text{number of variables} - \text{number of equality constraints}",
    r"&= 2 - 1",
    r"&= 1"
)

w("One possible independent variable is x1.")
m(r"x_2 = \sqrt{25 - x_1^2} \quad \text{with } x_2 \ge 0")

w("The third inequality is always satisfied because it is a sum of squares.")

doc.subsection("Answer")

w("Total variables: 2")
w("Independent variables: 1")
w("One possible choice: {x1}")


# ============================================================================
# PROBLEM 4
# Oil refinery: minimise cost per barrel and maximise daily profit.
# ============================================================================

doc.section("Problem 4")
doc.subsection("Given")

w("The cost per barrel is given by C(P) = 50 + 0.1P + 9000/P, where P is the production rate in barrels/day.")
w("The selling price is $300 per barrel.")

m(r"C(P) = 50 + 0.1P + \frac{9000}{P}")
m(r"\text{Selling price} = 300\ \$/\text{barrel}")

# Part A
doc.subsection("Part A: Minimize Cost per Barrel")

a(
    r"\frac{dC}{dP} &= 0.1 - \frac{9000}{P^2}",
    r"&= 0"
)

a(
    r"P^2 &= \frac{9000}{0.1} = 90000",
    r"P^* &= \sqrt{90000} = 300\ \text{barrels/day}"
)

m(r"\frac{d^2C}{dP^2} = \frac{18000}{P^3} > 0 \quad \text{for } P>0")

P4a = 300.0
C4a = 50 + 0.1 * P4a + 9000 / P4a
d2C = 18000 / P4a**3

a(
    rf"P^* &= {P4a:.1f}\ \text{{barrels/day}}",
    rf"C(P^*) &= 50 + 0.1({P4a:.1f}) + \frac{{9000}}{{{P4a:.1f}}} = {C4a:.2f}\ \$/\text{{barrel}}",
    rf"\frac{{d^2C}}{{dP^2}}\bigg|_{{P^*}} &= {d2C:.6f} > 0"
)

# Part B
doc.subsection("Part B: Maximize Daily Profit")

m(r"R(P) = 300P")
m(r"TC(P) = C(P)\,P = 50P + 0.1P^2 + 9000")

a(
    r"\Pi(P) &= R(P) - TC(P)",
    r"&= 300P - (50P + 0.1P^2 + 9000)",
    r"&= 250P - 0.1P^2 - 9000"
)

a(
    r"\frac{d\Pi}{dP} &= 250 - 0.2P",
    r"&= 0"
)

a(
    r"P^* &= \frac{250}{0.2}",
    r"&= 1250\ \text{barrels/day}"
)

m(r"\frac{d^2\Pi}{dP^2} = -0.2 < 0")

P4b = 1250.0
Rev4 = 300 * P4b
TC4 = 50 * P4b + 0.1 * P4b**2 + 9000
Pi4 = Rev4 - TC4

a(
    rf"P^* &= {P4b:.0f}\ \text{{barrels/day}}",
    rf"R(P^*) &= {Rev4:,.2f}\ \$/\text{{day}}",
    rf"TC(P^*) &= {TC4:,.2f}\ \$/\text{{day}}",
    rf"\Pi(P^*) &= {Pi4:,.2f}\ \$/\text{{day}}"
)

doc.subsection("Answers")

w(f"(A) Production minimizing cost/barrel: P = {P4a:.0f} barrels/day, C = ${C4a:.2f}/barrel")
px(f"(B) Production maximizing daily profit: P = {P4b:.0f} barrels/day ", im(r'\Pi'),f" = ${Pi4:,.2f}/day")


# ============================================================================
# PROBLEM 5
# Curve fitting: compare three models using regression statistics.
# ============================================================================

doc.section("Problem 5")
doc.subsection("Data")

x_data = np.array([10, 20, 30, 40, 50], dtype=float)
y_data = np.array([1.00, 1.26, 1.86, 3.31, 7.08])

t(
    headers=["x", "y"],
    rows=[[x_data[i], y_data[i]] for i in range(len(x_data))],
    caption="Raw data for curve fitting",
    label="tab:raw-data",
    float_fmt=".2f",
)

doc.subsection("Models")

w("Three candidate models are fit to the data.")
a(  r"\text{Model 1: } & y = \exp(a + bx)",
    r"\text{Model 2: } & y = \exp(a + bx + cx^2)",
    r"\text{Model 3: } & y = ax^b")

w("Models 1 and 2 are linearized using ln(y).")
w("Model 3 is linearized as ln(y) = ln(a) + b ln(x).")

# Fit models
ra = RegressionAnalysis(output_file="HW1_CHE565_regression_details.txt", verbose=False)
model_results = ra.fit_all_models(x_data, y_data)
ra.close()

models_summary = {
    "exp-linear": model_results["exp_linear"].get("R2_adj", np.nan),
    "exp-quadratic": model_results["exp_quadratic"].get("R2_adj", np.nan),
    "power-law": model_results["power_law"].get(
        "R2_adj_orig_scale", model_results["power_law"].get("R2_adj", np.nan)
    ),
}
best5 = max(models_summary, key=lambda k: models_summary[k])

el_pred = model_results["exp_linear"]["y_pred_original"]
ep_pred = model_results["exp_quadratic"]["y_pred_original"]
pw_pred = model_results["power_law"]["y_pred_original"]

doc.subsection("Fitted Values vs. Data")

t(
    headers=["x", "y_obs", "exp-lin", "exp-quad", "power"],
    rows=[
        [x_data[i], y_data[i], el_pred[i], ep_pred[i], pw_pred[i]]
        for i in range(len(x_data))
    ],
    caption="Fitted values versus observed data",
    label="tab:fitted-values",
    float_fmt=".4f",
)

doc.subsection("Residuals")

t(
    headers=["x", "exp-lin", "exp-quad", "power"],
    rows=[
        [x_data[i], y_data[i] - el_pred[i], y_data[i] - ep_pred[i], y_data[i] - pw_pred[i]]
        for i in range(len(x_data))
    ],
    caption="Residuals for each fitted model",
    label="tab:residuals",
    float_fmt=".4f",
)

doc.subsection("Conclusion")

w("The exp-quadratic model has the highest adjusted R^2 among the three candidate models.")
w(f"Best model: {best5}")

# Plot
x_plot = np.linspace(10, 50, 300)

el_a, el_b = model_results["exp_linear"]["params"]
ep_a, ep_b, ep_c = model_results["exp_quadratic"]["params"]
pw_lna, pw_b = model_results["power_law"]["params"]
pw_a = np.exp(pw_lna)

y_el = np.exp(el_a + el_b * x_plot)
y_ep = np.exp(ep_a + ep_b * x_plot + ep_c * x_plot**2)
y_pw = pw_a * x_plot ** pw_b

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("CHE 565 HW1 – Problem 5: Curve Fitting", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(x_data, y_data, 'ko', ms=7, zorder=5, label="Data")
ax.plot(x_plot, y_el, '-',  lw=2, label=f"exp-linear (adj R²={models_summary['exp-linear']:.4f})")
ax.plot(x_plot, y_ep, '--', lw=2, label=f"exp-quad (adj R²={models_summary['exp-quadratic']:.4f})")
ax.plot(x_plot, y_pw, '-.', lw=2, label=f"power law (adj R²={models_summary['power-law']:.4f})")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Model fits")
ax.legend(fontsize=8)
ax.grid(alpha=0.4)

ax2 = axes[1]
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.plot(x_data, y_data - el_pred, 'o-',  ms=6, label="exp-linear")
ax2.plot(x_data, y_data - ep_pred, 's--', ms=6, label="exp-quad")
ax2.plot(x_data, y_data - pw_pred, '^:',  ms=6, label="power law")
ax2.set_xlabel("x")
ax2.set_ylabel("Residual (y_obs - y_fit)")
ax2.set_title("Residuals")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.4)

plt.tight_layout()
PLOT_FILE = "HW1_CHE565_plot.png"
plt.savefig(PLOT_FILE, dpi=150)
plt.close()

doc.subsection("Figure")
figlog(
    PLOT_FILE,
    caption="Curve fitting comparison for Problem 5.",
    label="fig:curve-fit-comparison",
)

# save outputs
txt_file, tex_file, pdf_file = doc.save_all()

print(f"Wrote text log: {txt_file}")
print(f"Wrote LaTeX source: {tex_file}")
print(f"Wrote PDF: {pdf_file}")

