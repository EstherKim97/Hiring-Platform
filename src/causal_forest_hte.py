"""
Heterogeneous treatment effect (HTE) estimation via causal forest
(Wager & Athey; implemented here via EconML's CausalForestDML -- the same
family of method used in the 2024/2025 tutorials and applied papers on HTE
estimation, e.g. Sverdrup, Petukhova & Wager 2025, "Estimating Treatment
Effect Heterogeneity ... With Causal Forests").

Why this upgrades our earlier equity analysis: instead of manually testing
one segment variable at a time (background: traditional / career-switcher /
non-traditional), a causal forest estimates each candidate's individual
Conditional Average Treatment Effect (CATE) as a function of ALL their
covariates simultaneously, then lets us rank candidates by how much they're
predicted to be helped or hurt by the treatment. This surfaces
non-obvious interactions (e.g. "low-experience non-traditional candidates"
specifically) that a single-variable segment table would miss entirely.
"""

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def estimate_cate(df: pd.DataFrame, outcome_col: str, treatment_col: str,
                   covariate_cols: list) -> pd.DataFrame:
    """
    df: one row per unit (e.g. per candidate-job pair or per candidate)
    outcome_col: binary or continuous outcome (e.g. 'applied' or 'offer')
    treatment_col: 0/1 treatment indicator
    covariate_cols: list of column names to use as effect modifiers
                     (numeric; encode categoricals beforehand)

    Returns df with an added 'cate' column: each row's estimated
    individual treatment effect.
    """
    X = df[covariate_cols].values
    Y = df[outcome_col].values.astype(float)
    T = df[treatment_col].values.astype(float)

    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=20, random_state=42),
        model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=42),
        n_estimators=400,
        min_samples_leaf=20,
        random_state=42,
        discrete_treatment=True,
    )
    model.fit(Y, T, X=X)

    cate = model.effect(X)
    df = df.copy()
    df["cate"] = cate
    return df, model


def summarize_cate_by_segment(df_with_cate: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Quick readable summary: average estimated individual effect by segment."""
    return (
        df_with_cate.groupby(segment_col)["cate"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "avg_cate", "std": "cate_std", "count": "n"})
        .sort_values("avg_cate")
    )


if __name__ == "__main__":
    # Smoke test with synthetic data: treatment effect depends on covariate X1
    np.random.seed(3)
    n = 3000
    X1 = np.random.normal(0, 1, n)       # e.g. years_experience (standardized)
    X2 = np.random.binomial(1, 0.3, n)   # e.g. non_traditional flag
    T = np.random.binomial(1, 0.5, n)
    true_cate = 0.05 - 0.08 * X2         # non-traditional candidates hurt more
    Y = 0.3 + true_cate * T + 0.1 * X1 + np.random.normal(0, 0.2, n)

    df = pd.DataFrame({"applied": Y, "treated": T, "years_exp_std": X1, "non_traditional": X2})
    result_df, model = estimate_cate(df, "applied", "treated", ["years_exp_std", "non_traditional"])

    print("Estimated CATE by non_traditional flag (should show more negative for 1):")
    print(summarize_cate_by_segment(result_df, "non_traditional"))
