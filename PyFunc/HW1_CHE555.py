import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist, t as t_dist
import NRroots as NR

def regression_stats(x, y, y_pred, n_params, pcov):
    """Compute Excel-style regression statistics."""
    n = len(y)
    p = n_params  # number of fitted parameters
    residuals = y - y_pred
    SSE = np.sum(residuals**2)
    SST = np.sum((y - np.mean(y))**2)
    SSR = SST - SSE
    R2 = 1 - SSE / SST
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - p)
    dof = n - p  # degrees of freedom for error
    MSE = SSE / dof if dof > 0 else np.inf
    MSR = SSR / (p - 1) if p > 1 else SSR
    SE = np.sqrt(MSE)  # standard error of the regression
    RMSE = np.sqrt(SSE / n)
    F_stat = MSR / MSE if MSE > 0 else np.inf
    F_pval = 1 - f_dist.cdf(F_stat, p - 1, dof) if dof > 0 else np.nan
    # Parameter standard errors and p-values (t-test: H0: param = 0)
    param_SE = np.sqrt(np.diag(pcov))
    return {
        'R2': R2, 'R2_adj': R2_adj, 'SE': SE, 'RMSE': RMSE,
        'SSE': SSE, 'SSR': SSR, 'SST': SST,
        'F': F_stat, 'F_pval': F_pval,
        'param_SE': param_SE, 'dof': dof
    }

def print_model_stats(name, popt, param_names, stats):
    """Print regression statistics for a model."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  R²          = {stats['R2']:.6f}")
    print(f"  R² (adj)    = {stats['R2_adj']:.6f}")
    print(f"  SE (regr)   = {stats['SE']:.6f}")
    print(f"  RMSE        = {stats['RMSE']:.6f}")
    print(f"  SSE         = {stats['SSE']:.6f}")
    print(f"  SSR         = {stats['SSR']:.6f}")
    print(f"  SST         = {stats['SST']:.6f}")
    print(f"  F-statistic = {stats['F']:.4f}")
    print(f"  F p-value   = {stats['F_pval']:.6g}")
    print(f"  DOF (error) = {stats['dof']}")
    print(f"  {'Param':<6} {'Value':>12} {'Std Err':>12} {'t-stat':>12} {'p-value':>12}")
    print(f"  {'-'*54}")
    for i, pname in enumerate(param_names):
        se = stats['param_SE'][i]
        t_stat = popt[i] / se if se > 0 else np.inf
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), stats['dof'])) if stats['dof'] > 0 else np.nan
        print(f"  {pname:<6} {popt[i]:>12.6f} {se:>12.6f} {t_stat:>12.4f} {p_val:>12.6g}")

def func(x):
    return 2*x**2 + 3*x + 1

def dfunc(x):
    return 4*x + 3

def SA_func(x):
    """SA of open-top box with square base x and V=1000: x^2 + 4000/x"""
    return x**2 + 4000/x

def dSA_func(x):
    """dSA/dx = 2x - 4000/x^2 (set to 0 to minimize SA)"""
    return 2*x - 4000/x**2

def d2SA_func(x):
    """d^2SA/dx^2 = 2 + 8000/x^3"""
    return 2 + 8000/x**3
def obj_func(x):
    return 4*x - x**2 - 12

def exp_lin_func(x,a,b):
    return np.exp(a+b*x)

def exp_poly_func(x,a,b,c):
    return np.exp(a+b*x+c*x**2)

def power_func(x,a,b):
    return a*x**b

if __name__ == "__main__":
    x = np.linspace(-10, 10, 10000)
    x_fit = [10, 20, 30, 40, 50]
    y_fit = [1, 1.26, 1.86, 3.31, 7.08]
    y = func(x)
    r, ea, iter = NR.newton_raphson(func, dfunc, x0=5)
    # Minimize SA of open-top box with V=1000
    # Find root of dSA/dx = 0 using Newton-Raphson (with d2SA/dx2 as derivative)
    # Volume Constraint: V = x**2 *h = 1000 => h = 1000/x**2 
    x_opt, ea_sa, iter_sa = NR.newton_raphson(dSA_func, d2SA_func, x0=5)
    h_opt = 1000 / x_opt**2
    SA_min = SA_func(x_opt)
    print(f"Optimal base side: {x_opt:.4f} cm")
    print(f"Optimal height: {h_opt:.4f} cm")
    print(f"Minimum surface area: {SA_min:.4f} cm^2")

    # Fit the three models to x_fit, y_fit
    x_fit = np.array(x_fit)
    y_fit = np.array(y_fit)

    popt_el, eps_el = curve_fit(exp_lin_func, x_fit, y_fit)
    popt_ep, eps_ep = curve_fit(exp_poly_func, x_fit, y_fit, p0=[0, 0, 0.001])
    popt_pw, eps_pw = curve_fit(power_func, x_fit, y_fit)

    # Compute and print full regression statistics for each model
    stats_el = regression_stats(x_fit, y_fit, exp_lin_func(x_fit, *popt_el), len(popt_el), eps_el)
    stats_ep = regression_stats(x_fit, y_fit, exp_poly_func(x_fit, *popt_ep), len(popt_ep), eps_ep)
    stats_pw = regression_stats(x_fit, y_fit, power_func(x_fit, *popt_pw), len(popt_pw), eps_pw)

    print_model_stats("Exp-Linear: exp(a + b*x)", popt_el, ['a', 'b'], stats_el)
    print_model_stats("Exp-Poly: exp(a + b*x + c*x²)", popt_ep, ['a', 'b', 'c'], stats_ep)
    print_model_stats("Power: a*x^b", popt_pw, ['a', 'b'], stats_pw)

    x_plot = np.linspace(min(x_fit), max(x_fit), 200)
    plt.figure()
    plt.plot(x_fit, y_fit, 'ko', label='Data')
    plt.plot(x_plot, exp_lin_func(x_plot, *popt_el), '-', label='Exp-Linear')
    plt.plot(x_plot, exp_poly_func(x_plot, *popt_ep), '--', label='Exp-Poly')
    plt.plot(x_plot, power_func(x_plot, *popt_pw), '-.', label='Power')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.title('Curve Fitting')
    plt.show()

    # Find point closest to origin
    dist = np.sqrt(x**2 + y**2)
    idx = np.argmin(dist)
    x_closest, y_closest = x[idx], y[idx]

    plt.plot(x, y)
    plt.plot(r, func(r), 'ro', label=f'Root ({r:.2f}, {func(r):.2f})')
    plt.plot(x_closest, y_closest, 'gs', markersize=10, label=f'Closest to origin ({x_closest:.2f}, {y_closest:.2f})')
    plt.title("Plot of the function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()