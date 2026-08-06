"""
Always-valid sequential testing (Johari et al., 2015/2022 -- "Always Valid
Inference: Bringing Sequential Analysis to A/B Testing"; implemented in
production at Optimizely, Wish, and other experimentation platforms).

Problem this solves: our previous analyze.py computed one p-value after
the experiment "finished." In reality, teams want to monitor results as
data comes in and stop early if there's a clear winner (or clear loser).
Doing that with an ordinary t-test/z-test inflates the false-positive rate
-- this is called "peeking" and it's one of the most common real-world
A/B testing mistakes.

This module implements a simplified mixture Sequential Probability Ratio
Test (mSPRT) for a difference in proportions, which stays valid (Type I
error controlled at alpha) no matter when you stop looking.
"""

import numpy as np


def mixture_sprt_sequence(control_outcomes: np.ndarray, treatment_outcomes: np.ndarray,
                           tau: float = 0.03, alpha: float = 0.05) -> dict:
    """
    control_outcomes, treatment_outcomes: 1D arrays of 0/1 outcomes, assumed
    to arrive in time order (e.g. sorted by when the candidate applied).

    tau: prior std dev on the plausible effect size (on the raw proportion
    scale). Set this to roughly the smallest effect you'd care about
    detecting -- e.g. tau=0.03 says "I expect real effects to be on the
    order of a few percentage points." This is the standard mSPRT tuning
    parameter (Johari et al., 2015/2022): Lambda_n = sqrt(1/(1+tau^2*V_n))
    * exp(tau^2 * V_n^2 * D_n^2 / (2*(1+tau^2*V_n))), where D_n is the
    running difference in proportions and V_n = 1/Var(D_n) is the running
    Fisher information. Always-valid p-value = min(1, 1/Lambda_n).
    """
    n = min(len(control_outcomes), len(treatment_outcomes))
    c = np.asarray(control_outcomes[:n], dtype=float)
    t = np.asarray(treatment_outcomes[:n], dtype=float)

    idx = np.arange(1, n + 1)
    p_c = np.cumsum(c) / idx
    p_t = np.cumsum(t) / idx

    var_d = (p_c * (1 - p_c) + p_t * (1 - p_t)) / idx
    var_d = np.maximum(var_d, 1e-12)
    V = 1.0 / var_d  # running Fisher information of the difference

    D = p_t - p_c
    tau2 = tau ** 2

    denom = 1 + tau2 * V
    lambda_n = np.sqrt(1.0 / denom) * np.exp((tau2 * V ** 2 * D ** 2) / (2 * denom))
    always_valid_p = np.minimum(1.0, 1.0 / np.maximum(lambda_n, 1e-300))

    below = np.where(always_valid_p < alpha)[0]
    stop_idx = int(below[0]) if len(below) > 0 else None

    return {
        "n": idx,
        "diff": D,
        "always_valid_p": always_valid_p,
        "stop_index": stop_idx,
        "stop_n": int(idx[stop_idx]) if stop_idx is not None else None,
        "final_p": always_valid_p[-1],
    }


if __name__ == "__main__":
    np.random.seed(1)
    n = 13000
    # simulate a real ~1.4pp lift like our experiment found
    control = np.random.binomial(1, 0.24, n)
    treatment = np.random.binomial(1, 0.2542, n)

    result = mixture_sprt_sequence(control, treatment, tau=0.03)
    print(f"Final always-valid p-value at n={n}: {result['final_p']:.4f}")
    if result["stop_n"]:
        print(f"Could have safely stopped at n={result['stop_n']:,} per arm "
              f"(vs. waiting for the full {n:,}) without inflating Type I error.")
    else:
        print("Never crossed the significance threshold in this run -- with "
              "always-valid testing that's an honest 'not there yet,' unlike "
              "a plain t-test which would give a false sense of certainty if "
              "peeked at repeatedly.")

    # sanity check against a standard (non-sequential) z-test at the same n
    from statsmodels.stats.proportion import proportions_ztest
    z, p = proportions_ztest([control.sum(), treatment.sum()], [n, n])
    print(f"\nFor comparison, a single fixed-horizon z-test at n={n}: p={p:.4f}")
    print("(mSPRT is expected to need somewhat more data than a fixed-horizon "
          "test to reach the same confidence -- that's the known cost of "
          "buying the ability to peek at any time without inflating errors.)")
