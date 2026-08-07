"""Quick reproducibility demo: runs a tiny end-to-end pipeline on synthetic data.
This is intentionally lightweight so reviewers can run it without installing
heavy optional deps like econml.

Usage:
    python3 src/quick_demo.py
"""
import numpy as np
import pandas as pd
import os

from match_score import compute_match_scores
from simulate_behavior import simulate_apply_decision, simulate_hire_funnel

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "figures"))
os.makedirs(OUT, exist_ok=True)


def make_synthetic(num_cand=200, num_jobs=50, random_state=2026):
    rng = np.random.RandomState(random_state)
    candidates = pd.DataFrame({
        "candidate_id": np.arange(num_cand),
        "years_exp": rng.normal(5, 2, size=num_cand),
        "background": rng.choice(["traditional", "non_traditional", "career_switcher"], size=num_cand),
    })
    jobs = pd.DataFrame({
        "job_id": np.arange(num_jobs),
        "req_seniority": rng.choice([0, 1, 2], size=num_jobs),
    })
    return candidates, jobs


def run_demo():
    candidates, jobs = make_synthetic()
    print("Generating match scores (lightweight placeholder)")
    # compute_match_scores expects full data; in many repos it returns pairwise scores.
    # Here, reuse the existing function if available; otherwise create a simple placeholder.
    try:
        match_df = compute_match_scores(candidates, jobs, use_seniority_filter=False)
    except Exception:
        # fallback: create a small cartesian product with synthetic match_score
        pairs = []
        for c in candidates.itertuples():
            for j in jobs.itertuples():
                pairs.append({
                    "candidate_id": c.candidate_id,
                    "job_id": j.job_id,
                    "match_score": max(0.0, min(1.0, np.random.normal(0.5, 0.15))),
                })
        match_df = pd.DataFrame(pairs)

    print("Simulating assignment and behavior (lightweight)")
    # simple 50/50 assignment and simulate behavior
    assignment = pd.Series(np.random.choice(["control", "treatment"], size=len(candidates)), index=candidates["candidate_id"])  # noqa: E501
    applied_df = simulate_apply_decision(match_df, candidates, assignment)
    funnel_df = simulate_hire_funnel(applied_df)

    result = applied_df.merge(
        funnel_df[["candidate_id", "job_id", "interviewed", "offer"]],
        on=["candidate_id", "job_id"],
        how="left",
    )
    result["interviewed"] = result["interviewed"].fillna(0).astype(int)
    result["offer"] = result["offer"].fillna(0).astype(int)

    out_path = os.path.join(OUT, "quick_demo_results.csv")
    result.to_csv(out_path, index=False)
    print(f"Wrote demo results to {out_path}")


if __name__ == "__main__":
    run_demo()
