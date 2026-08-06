"""
CUPED: Controlled-experiment Using Pre-Experiment Data (Deng et al., 2013;
still the industry-standard variance reduction technique as of 2025-2026,
used at Microsoft, Netflix, Booking, Etsy, and others).

Idea: some of a candidate's outcome is predictable from things we already
knew about them BEFORE the experiment (their pre-experiment match-score
history / skill strength) -- that predictable part is noise, not signal
from our treatment. We subtract it out to shrink variance, which lets us
detect a real effect with a smaller sample or detect it more confidently
with the sample we have.

theta is chosen to minimize variance of the adjusted metric:
    theta = Cov(Y, X) / Var(X)
    Y_cuped = Y - theta * (X - mean(X))

This does NOT change the expected value of the metric (still unbiased),
it only reduces its variance.
"""

import numpy as np
import pandas as pd


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    y: post-experiment outcome (e.g. offer = 0/1) per unit
    x: pre-experiment covariate per unit (must be same length as y,
       correlated with y but NOT affected by treatment)
    Returns CUPED-adjusted y with reduced variance, same mean as y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    theta = cov_xy / var_x if var_x > 0 else 0.0

    y_adjusted = y - theta * (x - x.mean())
    return y_adjusted, theta


def variance_reduction_report(y: np.ndarray, x: np.ndarray) -> dict:
    y_adj, theta = cuped_adjust(y, x)
    var_before = np.var(y, ddof=1)
    var_after = np.var(y_adj, ddof=1)
    reduction_pct = (1 - var_after / var_before) * 100 if var_before > 0 else 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return {
        "theta": theta,
        "correlation_x_y": corr,
        "var_before": var_before,
        "var_after": var_after,
        "variance_reduction_pct": reduction_pct,
        "y_adjusted": y_adj,
    }


if __name__ == "__main__":
    # Smoke test with synthetic correlated data
    np.random.seed(0)
    n = 5000
    x = np.random.normal(0.5, 0.15, n)  # pre-experiment covariate
    noise = np.random.normal(0, 0.1, n)
    y = 0.3 * x + noise + 0.05  # y correlated with x

    report = variance_reduction_report(y, x)
    print(f"Correlation(X, Y): {report['correlation_x_y']:.3f}")
    print(f"Theta: {report['theta']:.4f}")
    print(f"Variance before: {report['var_before']:.6f}")
    print(f"Variance after:  {report['var_after']:.6f}")
    print(f"Variance reduction: {report['variance_reduction_pct']:.1f}%")
