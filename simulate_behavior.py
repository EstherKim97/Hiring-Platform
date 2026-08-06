"""
SIMULATED COMPONENT -- documented explicitly, not hidden.

No public dataset captures "candidate saw their match score before applying."
This module simulates that one behavioral link, using two real, defensible
mechanisms from nudge/information-disclosure literature:

1. Self-selection: showing people a probability/fit score makes them act on
   it -- people are more likely to pursue things they're told they're likely
   to succeed at, and less likely to pursue things they're told they're not
   (well documented in behavioral econ, e.g. work on how displaying
   "match" or "compatibility" scores on platforms shifts user choices).
2. Heterogeneous discouragement: the effect is not uniform. Candidates with
   less traditional backgrounds/lower baseline confidence are more sensitive
   to a low score -- a known equity risk with any AI-disclosure feature,
   which is exactly why we track apply-rate-by-segment as a guardrail metric.

Every parameter below is a named, adjustable assumption -- change them and
rerun to stress-test how sensitive the experiment's conclusion is to your
behavioral model. That sensitivity-analysis step is itself worth mentioning
in your writeup.
"""

import numpy as np
import pandas as pd

np.random.seed(7)

# ---- Adjustable behavioral parameters ----
CONTROL_APPLY_BASE_RATE = 0.35       # "spray and pray" baseline apply rate,
                                      # weakly tied to true match quality
CONTROL_APPLY_SCORE_SENSITIVITY = 0.3

TREATMENT_APPLY_BASE_RATE = 0.20     # lower baseline: seeing a bad score
                                      # suppresses low-fit applications
TREATMENT_APPLY_SCORE_SENSITIVITY = 1.8  # much more responsive to score

# extra discouragement penalty for non_traditional candidates in treatment
# when their score is below this threshold (equity risk we're testing for)
DISCOURAGEMENT_THRESHOLD = 0.55
NON_TRADITIONAL_EXTRA_PENALTY = 0.6

# Real-outcome calibration: interview/offer probability as a function of
# TRUE match quality -- this does not change between arms, because the
# algorithm and actual candidate quality are identical. Only who applies
# changes. Rates loosely calibrated to typical tech-hiring funnel benchmarks
# (~15-25% interview rate, ~20-30% interview->offer rate for well-matched
# candidates).
INTERVIEW_INTERCEPT = -3.0
INTERVIEW_SCORE_COEF = 5.0

OFFER_INTERCEPT = -1.5
OFFER_SCORE_COEF = 2.5


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate_apply_decision(match_df: pd.DataFrame, candidates: pd.DataFrame,
                             assignment: pd.Series) -> pd.DataFrame:
    """
    assignment: Series indexed by candidate_id, values in {"control","treatment"}
    Returns match_df with an added `applied` column (0/1).
    """
    df = match_df.merge(candidates[["candidate_id", "background"]], on="candidate_id")
    df["arm"] = df["candidate_id"].map(assignment)

    apply_prob = np.where(
        df["arm"] == "control",
        _sigmoid(CONTROL_APPLY_SCORE_SENSITIVITY * (df["match_score"] - 0.5)
                 + np.log(CONTROL_APPLY_BASE_RATE / (1 - CONTROL_APPLY_BASE_RATE))),
        _sigmoid(TREATMENT_APPLY_SCORE_SENSITIVITY * (df["match_score"] - 0.5)
                 + np.log(TREATMENT_APPLY_BASE_RATE / (1 - TREATMENT_APPLY_BASE_RATE))),
    )

    # equity effect: extra discouragement for non-traditional candidates
    # in treatment arm when score is below threshold
    discourage_mask = (
        (df["arm"] == "treatment")
        & (df["background"] == "non_traditional")
        & (df["match_score"] < DISCOURAGEMENT_THRESHOLD)
    )
    apply_prob = np.where(discourage_mask, apply_prob * (1 - NON_TRADITIONAL_EXTRA_PENALTY), apply_prob)

    df["apply_prob"] = apply_prob
    df["applied"] = np.random.binomial(1, apply_prob)
    return df


def simulate_hire_funnel(applied_df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where applied == 1, simulate interview and offer outcomes based
    on TRUE match quality only (identical function in both arms).
    """
    df = applied_df[applied_df["applied"] == 1].copy()

    interview_prob = _sigmoid(INTERVIEW_INTERCEPT + INTERVIEW_SCORE_COEF * df["match_score"])
    df["interviewed"] = np.random.binomial(1, interview_prob)

    offer_prob = np.where(
        df["interviewed"] == 1,
        _sigmoid(OFFER_INTERCEPT + OFFER_SCORE_COEF * df["match_score"]),
        0.0,
    )
    df["offer"] = np.where(df["interviewed"] == 1, np.random.binomial(1, offer_prob), 0)

    return df


if __name__ == "__main__":
    from load_data import load_candidates, load_jobs
    from match_score import compute_match_scores

    candidates = load_candidates()
    jobs = load_jobs()
    match_df = compute_match_scores(candidates, jobs)

    # quick smoke test: 50/50 random assignment
    assignment = pd.Series(
        np.random.choice(["control", "treatment"], size=len(candidates)),
        index=candidates["candidate_id"],
    )
    applied_df = simulate_apply_decision(match_df, candidates, assignment)
    funnel_df = simulate_hire_funnel(applied_df)

    print("Apply rate by arm:")
    print(applied_df.groupby("arm")["applied"].mean())
    print("\nAvg match score of submitted applications, by arm:")
    print(applied_df[applied_df["applied"] == 1].groupby("arm")["match_score"].mean())
    print("\nInterview->offer conversion by arm:")
    print(funnel_df.groupby("arm")["offer"].mean())
    print("\nApply rate by arm x background (equity check):")
    print(applied_df.groupby(["arm", "background"])["applied"].mean())
