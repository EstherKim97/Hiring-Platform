"""
Runs the full simulated experiment once:
1. Randomly assign candidates to control/treatment (50/50)
2. Simulate apply decisions
3. Simulate hire funnel outcomes
4. Save the resulting dataset for analysis
"""

import os
import numpy as np
import pandas as pd

from load_data import load_candidates, load_jobs, DATA_DIR
from match_score import compute_match_scores
from simulate_behavior import simulate_apply_decision, simulate_hire_funnel

np.random.seed(2024)


def run(seed=2024, out_path=None):
    if out_path is None:
        out_path = os.path.join(DATA_DIR, "experiment_results.csv")
    np.random.seed(seed)

    candidates = load_candidates()
    jobs = load_jobs()

    # Safety net: randomization requires unique candidate_id. Should already
    # be deduplicated in load_data.py, but guard here too in case a future
    # data source doesn't do this upstream.
    if candidates["candidate_id"].duplicated().any():
        n_dupes = candidates["candidate_id"].duplicated().sum()
        print(f"WARNING: found {n_dupes:,} duplicate candidate_id rows -- deduplicating.")
        candidates = candidates.drop_duplicates(subset="candidate_id", keep="first").reset_index(drop=True)

    # Real data is large -- subsample to keep this runnable on a laptop.
    # Match match_score.py's defaults so results are comparable.
    MAX_CANDIDATES = 8000
    MAX_JOBS = 5000
    if len(candidates) > MAX_CANDIDATES:
        print(f"Subsampling candidates: {len(candidates):,} -> {MAX_CANDIDATES:,}")
        candidates = candidates.sample(n=MAX_CANDIDATES, random_state=seed).reset_index(drop=True)
    if len(jobs) > MAX_JOBS:
        print(f"Subsampling jobs: {len(jobs):,} -> {MAX_JOBS:,}")
        jobs = jobs.sample(n=MAX_JOBS, random_state=seed).reset_index(drop=True)

    match_df = compute_match_scores(candidates, jobs, use_seniority_filter=False)

    # --- Random assignment: candidate-level, 50/50 ---
    assignment = pd.Series(
        np.random.choice(["control", "treatment"], size=len(candidates), p=[0.5, 0.5]),
        index=candidates["candidate_id"],
    )

    # Sample Ratio Mismatch check
    counts = assignment.value_counts()
    print("Assignment counts (SRM check):")
    print(counts)
    ratio = counts.min() / counts.max()
    if ratio < 0.98:
        print(f"  WARNING: assignment ratio {ratio:.3f} suggests possible SRM issue.")
    else:
        print(f"  OK: assignment ratio {ratio:.3f}, no SRM concern.")

    applied_df = simulate_apply_decision(match_df, candidates, assignment)
    funnel_df = simulate_hire_funnel(applied_df)

    # merge outcome columns back so we keep non-applied rows for volume metrics
    result = applied_df.merge(
        funnel_df[["candidate_id", "job_id", "interviewed", "offer"]],
        on=["candidate_id", "job_id"],
        how="left",
    )
    result["interviewed"] = result["interviewed"].fillna(0).astype(int)
    result["offer"] = result["offer"].fillna(0).astype(int)

    result.to_csv(out_path, index=False)
    print(f"\nSaved experiment results: {out_path}")
    print(f"Total rows (candidate-job pairs considered): {len(result):,}")
    print(f"Total applications submitted: {result['applied'].sum():,}")
    return result


if __name__ == "__main__":
    run()
