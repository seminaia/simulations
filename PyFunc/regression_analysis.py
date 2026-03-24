"""
regression_analysis.py
======================
Comprehensive regression & hypothesis-testing toolkit for CHE 555 / general use.

Capabilities
------------
* Ordinary least-squares (OLS) linear regression
* Nonlinear curve fitting via scipy.optimize.curve_fit
    - exp-linear :  y = exp(a + b·x)
    - exp-quadratic: y = exp(a + b·x + c·x²)
    - power law :   y = a · x^b
* Regression statistics  (R², adj-R², SE, RMSE, SSE/SSR/SST)
* ANOVA table            (Regression / Residual / Total)
* Parameter t-tests      (estimate, SE, t-stat, p-value, 95 % CI)
* F-test (overall model significance)
* All results written to a plain-text report file AND optionally printed
  to the console.

Quick-start example
-------------------
    from regression_analysis import RegressionAnalysis

    x = np.array([10, 20, 30, 40, 50])
    y = np.array([1.00, 1.26, 1.86, 3.31, 7.08])

    ra = RegressionAnalysis(output_file="results.txt", verbose=True)

    # Fit all three nonlinear models and pick the best
    results = ra.fit_all_models(x, y)

    # Or do a plain OLS fit
    X = np.column_stack([np.ones_like(x), x])
    ra.linear_regression(X, np.log(y), param_names=["a", "b"],
                         model_label="Log-linear fit")

    ra.close()   # flush & close the output file
"""

import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist, t as t_dist
from scipy.linalg import lstsq


# ──────────────────────────────────────────────────────────────────────────────
# Output helper
# ──────────────────────────────────────────────────────────────────────────────

class _Writer:
    """Writes to a file and optionally to stdout simultaneously."""

    def __init__(self, filepath: str, verbose: bool = True):
        self._fh = open(filepath, "w", encoding="utf-8")
        self._verbose = verbose

    def write(self, text: str = ""):
        self._fh.write(text + "\n")
        if self._verbose:
            print(text)

    def close(self):
        self._fh.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class RegressionAnalysis:
    """
    Fit models, compute statistics, and write a formatted report.

    Parameters
    ----------
    output_file : str
        Path of the text file that will receive the report.
    verbose : bool
        If True (default), mirror output to stdout as well.
    """

    def __init__(self, output_file: str = "regression_results.txt",
                 verbose: bool = True):
        self._w = _Writer(output_file, verbose)
        self._w.write("=" * 80)
        self._w.write("  REGRESSION ANALYSIS REPORT")
    # ── public interface ──────────────────────────────────────────────────────

    def linear_regression(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          param_names: list[str] | None = None,
                          model_label: str = "OLS Regression") -> dict:
        """
        Ordinary least-squares regression.

        Parameters
        ----------
        X : (n, p) design matrix — include a column of ones for the intercept.
        y : (n,)  response vector.
        param_names : list of p strings naming each column of X.
        model_label : heading printed in the report.

        Returns
        -------
        dict  — all statistics (see keys listed at the top of this file).
        """
        n, p = X.shape
        if param_names is None:
            param_names = [f"β{i}" for i in range(p)]

        coeffs, *_ = lstsq(X, y, cond=None)
        y_pred    = X @ coeffs
        residuals = y - y_pred

        SST = float(np.sum((y - y.mean()) ** 2))
        SSR = float(np.sum((y_pred - y.mean()) ** 2))
        SSE = float(np.sum(residuals ** 2))

        dof   = n - p
        R2    = 1.0 - SSE / SST if SST else np.nan
        R2adj = 1.0 - (1.0 - R2) * (n - 1) / dof if dof > 0 else np.nan
        MSE   = SSE / dof if dof > 0 else np.inf
        MSR   = SSR / (p - 1) if p > 1 else SSR
        SE    = float(np.sqrt(MSE))
        RMSE  = float(np.sqrt(SSE / n))

        # Covariance matrix → parameter standard errors
        XtX = X.T @ X
        try:
            inv_XtX = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            inv_XtX = np.linalg.pinv(XtX)
        cov = inv_XtX * MSE
        param_se = np.sqrt(np.diag(cov))

        # t-tests for each parameter
        t_stats = coeffs / param_se
        p_vals  = 2.0 * (1.0 - t_dist.cdf(np.abs(t_stats), dof))
        t_crit  = t_dist.ppf(0.975, dof) if dof > 0 else np.inf
        ci_lo   = coeffs - t_crit * param_se
        ci_hi   = coeffs + t_crit * param_se

        # Overall F-test
        F_stat = MSR / MSE if MSE > 0 else np.inf
        F_pval = float(1.0 - f_dist.cdf(F_stat, p - 1, dof)) if dof > 0 else np.nan

        stats = {
            "model_label": model_label,
            "params":      coeffs,
            "param_names": param_names,
            "stderr":      param_se,
            "t_stats":     t_stats,
            "p_values":    p_vals,
            "ci_lower":    ci_lo,
            "ci_upper":    ci_hi,
            "R2":          R2,
            "R2_adj":      R2adj,
            "SE":          SE,
            "RMSE":        RMSE,
            "SSE":         SSE,
            "SSR":         SSR,
            "SST":         SST,
            "n":           n,
            "p":           p,
            "dof":         dof,
            "MSE":         MSE,
            "MSR":         MSR,
            "F":           F_stat,
            "F_pval":      F_pval,
            "cov_matrix":  cov,
            "y_pred":      y_pred,
            "residuals":   residuals,
        }
        self._print_stats(stats)
        return stats

    # ── nonlinear helpers ─────────────────────────────────────────────────────

    def fit_exp_linear(self, x: np.ndarray, y: np.ndarray,
                       label: str = "Model: y = exp(a + b·x)") -> dict:
        """Fit y = exp(a + b·x)  via OLS on ln y."""
        X    = np.column_stack([np.ones_like(x), x])
        logy = np.log(y)
        stats = self.linear_regression(X, logy, ["a", "b"], label)
        stats["y_pred_original"] = np.exp(X @ stats["params"])
        return stats

    def fit_exp_quadratic(self, x: np.ndarray, y: np.ndarray,
                          label: str = "Model: y = exp(a + b·x + c·x²)") -> dict:
        """Fit y = exp(a + b·x + c·x²)  via OLS on ln y."""
        X    = np.column_stack([np.ones_like(x), x, x ** 2])
        logy = np.log(y)
        stats = self.linear_regression(X, logy, ["a", "b", "c"], label)
        stats["y_pred_original"] = np.exp(X @ stats["params"])
        return stats

    def fit_power_law(self, x: np.ndarray, y: np.ndarray,
                      label: str = "Model: y = a·x^b") -> dict:
        """Fit y = a · x^b  via OLS on ln y = ln a + b·ln x."""
        X        = np.column_stack([np.ones_like(x), np.log(x)])
        logy     = np.log(y)
        log_stats = self.linear_regression(X, logy, ["ln a", "b"],
                                           label + " [log-log space]")
        a = float(np.exp(log_stats["params"][0]))
        b = float(log_stats["params"][1])
        y_pred = a * x ** b

        # Re-compute R² on the *original* scale for fair comparison
        n    = len(y)
        SST  = float(np.sum((y - y.mean()) ** 2))
        SSE  = float(np.sum((y - y_pred) ** 2))
        dof  = n - 2
        R2   = 1.0 - SSE / SST
        R2a  = 1.0 - (1.0 - R2) * (n - 1) / dof

        self._w.write()
        self._w.write(f"Original-scale R² for {label}: {R2:.6f}  "
                      f"(adj R²: {R2a:.6f})")
        log_stats.update({
            "a_orig":            a,
            "b_orig":            b,
            "R2_orig_scale":     R2,
            "R2_adj_orig_scale": R2a,
            "y_pred_original":   y_pred,
        })
        return log_stats

    # ── nonlinear via scipy.optimize.curve_fit ─────────────────────────────────

    def fit_nonlinear(self,
                      func,
                      x: np.ndarray,
                      y: np.ndarray,
                      param_names: list[str],
                      p0=None,
                      label: str = "Nonlinear model") -> dict:
        """
        Fit an arbitrary model using scipy curve_fit and report full stats.

        Parameters
        ----------
        func        : callable  f(x, *params) → y
        x, y        : data arrays
        param_names : names for each parameter
        p0          : initial guess (optional)
        label       : heading in the report
        """
        popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10_000)
        y_pred    = func(x, *popt)
        residuals = y - y_pred
        n  = len(y)
        p  = len(popt)
        dof = n - p

        SST = float(np.sum((y - y.mean()) ** 2))
        SSE = float(np.sum(residuals ** 2))
        SSR = SST - SSE

        R2   = 1.0 - SSE / SST if SST else np.nan
        R2a  = 1.0 - (1.0 - R2) * (n - 1) / dof if dof > 0 else np.nan
        MSE  = SSE / dof if dof > 0 else np.inf
        MSR  = SSR / (p - 1) if p > 1 else SSR
        SE   = float(np.sqrt(MSE))
        RMSE = float(np.sqrt(SSE / n))

        param_se = np.sqrt(np.diag(pcov))
        t_stats  = popt / param_se
        p_vals   = 2.0 * (1.0 - t_dist.cdf(np.abs(t_stats), dof))
        t_crit   = t_dist.ppf(0.975, dof) if dof > 0 else np.inf
        ci_lo    = popt - t_crit * param_se
        ci_hi    = popt + t_crit * param_se

        F_stat = MSR / MSE if MSE > 0 else np.inf
        F_pval = float(1.0 - f_dist.cdf(F_stat, p - 1, dof)) if dof > 0 else np.nan

        stats = {
            "model_label": label,
            "params":      popt,
            "param_names": param_names,
            "stderr":      param_se,
            "t_stats":     t_stats,
            "p_values":    p_vals,
            "ci_lower":    ci_lo,
            "ci_upper":    ci_hi,
            "R2":          R2,
            "R2_adj":      R2a,
            "SE":          SE,
            "RMSE":        RMSE,
            "SSE":         SSE,
            "SSR":         SSR,
            "SST":         SST,
            "n":           n,
            "p":           p,
            "dof":         dof,
            "MSE":         MSE,
            "MSR":         MSR,
            "F":           F_stat,
            "F_pval":      F_pval,
            "cov_matrix":  pcov,
            "y_pred":      y_pred,
            "residuals":   residuals,
        }
        self._print_stats(stats)
        return stats

    # ── convenience: fit all three standard models and pick the best ───────────

    def fit_all_models(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Fit exp-linear, exp-quadratic, and power-law models.
        Returns a dict keyed by model name; prints a comparison table.
        """
        results = {}
        results["exp_linear"]    = self.fit_exp_linear(x, y)
        results["exp_quadratic"] = self.fit_exp_quadratic(x, y)
        results["power_law"]     = self.fit_power_law(x, y)
        self._comparison_table(results)
        return results

    # ── close / flush ─────────────────────────────────────────────────────────

    def close(self):
        """Flush and close the output file."""
        self._w.write()
        self._w.write("=" * 80)
        self._w.write("  END OF REPORT")
        self._w.write("=" * 80)
        self._w.close()

    # ── private printing helpers ───────────────────────────────────────────────

    def _print_stats(self, s: dict):
        w = self._w.write
        w()
        w("─" * 80)
        w(f"  {s['model_label']}")
        w("─" * 80)
        w(f"  Observations     : {s['n']}")
        w(f"  Parameters (p)   : {s['p']}")
        w(f"  DOF (error)      : {s['dof']}")
        w()
        w("  ── Goodness of Fit ──────────────────────────────")
        w(f"  R²               : {s['R2']:.6f}")
        w(f"  R² (adjusted)    : {s['R2_adj']:.6f}")
        w(f"  SE of regression : {s['SE']:.6f}")
        w(f"  RMSE             : {s['RMSE']:.6f}")
        w(f"  SSE              : {s['SSE']:.6f}")
        w(f"  SSR              : {s['SSR']:.6f}")
        w(f"  SST              : {s['SST']:.6f}")
        w()
        w("  ── Overall F-Test ───────────────────────────────")
        w(f"  F-statistic      : {s['F']:.4f}")
        w(f"  F p-value        : {s['F_pval']:.6g}")
        w()
        w("  ── Parameter t-Tests  (H₀: param = 0, α = 0.05) ─")
        hdr = (f"  {'Param':<8} {'Estimate':>12} {'Std Err':>12} "
               f"{'t-stat':>10} {'p-value':>12} {'95% CI lower':>14} {'95% CI upper':>14}")
        w(hdr)
        w("  " + "─" * 84)
        for i, name in enumerate(s["param_names"]):
            se    = s["stderr"][i]
            tstat = s["t_stats"][i]
            pval  = s["p_values"][i]
            ci_lo = s["ci_lower"][i]
            ci_hi = s["ci_upper"][i]
            sig   = "  *" if pval < 0.05 else ""
            w(f"  {name:<8} {s['params'][i]:>12.6f} {se:>12.6f} "
              f"{tstat:>10.4f} {pval:>12.6g} {ci_lo:>14.6f} {ci_hi:>14.6f}{sig}")
        w("  (* p < 0.05)")
        w()
        w("  ── ANOVA Table ──────────────────────────────────")
        w(f"  {'Source':<12} {'df':>6} {'SS':>16} {'MS':>16} {'F':>12} {'p-value':>12}")
        w("  " + "─" * 66)
        p   = s["p"]
        dof = s["dof"]
        SSR = s["SSR"]
        SSE = s["SSE"]
        SST = s["SST"]
        MSR = s["MSR"]
        MSE = s["MSE"]
        w(f"  {'Regression':<12} {p - 1:>6d} {SSR:>16.6f} {MSR:>16.6f} "
          f"{s['F']:>12.4f} {s['F_pval']:>12.6g}")
        w(f"  {'Residual':<12} {dof:>6d} {SSE:>16.6f} {MSE:>16.6f}")
        w(f"  {'Total':<12} {s['n'] - 1:>6d} {SST:>16.6f}")

    def _comparison_table(self, results: dict):
        w = self._w.write
        w()
        w("=" * 80)
        w("  MODEL COMPARISON SUMMARY")
        w("=" * 80)
        w(f"  {'Model':<20} {'R²':>10} {'adj R²':>10} {'RMSE':>12} {'F-stat':>12} {'F p-val':>12}")
        w("  " + "─" * 76)
        best_name, best_r2 = "", -np.inf
        for name, s in results.items():
            # use original-scale adj-R² for power law if available
            r2a = s.get("R2_adj_orig_scale", s["R2_adj"])
            r2  = s.get("R2_orig_scale",     s["R2"])
            rmse = s["RMSE"]
            w(f"  {name:<20} {r2:>10.6f} {r2a:>10.6f} {rmse:>12.6f} "
              f"{s['F']:>12.4f} {s['F_pval']:>12.6g}")
            if r2a > best_r2:
                best_r2, best_name = r2a, name
        w()
        w(f"  Best model (highest adj R²): {best_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo / self-test  (runs when you execute this file directly)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nRunning demo with CHE 555 HW1 Problem 5 data...\n")

    x = np.array([10, 20, 30, 40, 50], dtype=float)
    y = np.array([1.00, 1.26, 1.86, 3.31, 7.08])

    ra = RegressionAnalysis(output_file="regression_results.txt", verbose=True)
    ra.fit_all_models(x, y)
    ra.close()

    print("\nDone. Results written to regression_results.txt")