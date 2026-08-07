import pandas as pd
import numpy as np

from src.causal_forest_hte import estimate_cate


def test_estimate_cate_fallback():
    # Small synthetic dataset where econml may be unavailable; ensure
    # estimate_cate returns a dataframe with a 'cate' column and same length.
    n = 100
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "applied": rng.binomial(1, 0.2, size=n),
        "treated": rng.binomial(1, 0.5, size=n),
        "years_exp_std": rng.normal(0, 1, size=n),
        "non_traditional": rng.binomial(1, 0.3, size=n),
    })
    out_df, model = estimate_cate(df, "applied", "treated", ["years_exp_std", "non_traditional"])
    assert "cate" in out_df.columns
    assert len(out_df) == len(df)
    # If econml is not installed, model is None and cate should be constant (ATE fallback)
    if model is None:
        assert np.allclose(out_df["cate"].values, out_df["cate"].iloc[0])
