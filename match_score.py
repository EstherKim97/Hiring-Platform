"""
Computes a real match score between every candidate and every job they
apply to (or are considered for), based on skill-text similarity.

This is NOT simulated -- it's a genuine TF-IDF cosine-similarity computation
over the candidate `skills` field and the job `required_skills` field.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from load_data import load_candidates, load_jobs, bucket_years_experience, DATA_DIR


def compute_match_scores(candidates: pd.DataFrame, jobs: pd.DataFrame, use_seniority_filter: bool = True) -> pd.DataFrame:
    """
    Returns a long-form dataframe: one row per (candidate, job) pair the
    candidate is "eligible" to see. Columns: candidate_id, job_id, match_score (0-1)

    use_seniority_filter: restrict to same-seniority-tier pairs when the
    job source has a clean, matchable seniority taxonomy (e.g. LinkedIn's
    formatted_experience_level). Set to False when jobs' "seniority" field
    is free text (e.g. Russian experienceRequirements sentences) that
    won't match our candidate-side buckets -- in that case we just take
    top-K by similarity across ALL provided jobs directly.
    """
    if "experience_level" in candidates.columns:
        cand_seniority = candidates["experience_level"].values
    else:
        cand_seniority = bucket_years_experience(candidates["years_experience"]).values

    corpus = list(candidates["skills"].fillna("")) + list(jobs["required_skills"].fillna(""))
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    n_cand = len(candidates)
    cand_vecs = tfidf[:n_cand]
    job_vecs = tfidf[n_cand:]

    sim_matrix = cosine_similarity(cand_vecs, job_vecs)  # (n_cand, n_jobs)
    seniority = jobs["seniority"].values

    records = []
    for i in range(n_cand):
        if use_seniority_filter:
            eligible_job_idx = np.where(seniority == cand_seniority[i])[0]
            if len(eligible_job_idx) == 0:
                eligible_job_idx = np.arange(len(jobs))  # fall back to all jobs
        else:
            eligible_job_idx = np.arange(len(jobs))

        # cap at 15 recommended jobs per candidate, like a real feed would
        top_idx = eligible_job_idx[
            np.argsort(-sim_matrix[i, eligible_job_idx])[:15]
        ]
        for j in top_idx:
            records.append({
                "candidate_id": candidates.iloc[i]["candidate_id"],
                "job_id": jobs.iloc[j]["job_id"],
                "match_score": sim_matrix[i, j],
            })

    match_df = pd.DataFrame(records)
    return match_df


if __name__ == "__main__":
    candidates = load_candidates()
    jobs = load_jobs()

    # Real data is large -- full TF-IDF + cosine similarity across all of
    # it would exceed a laptop's memory. Subsample to a manageable size.
    MAX_CANDIDATES = 8000
    MAX_JOBS = 5000
    if len(candidates) > MAX_CANDIDATES:
        print(f"[match_score] Subsampling candidates: {len(candidates):,} -> {MAX_CANDIDATES:,}")
        candidates = candidates.sample(n=MAX_CANDIDATES, random_state=42).reset_index(drop=True)
    if len(jobs) > MAX_JOBS:
        print(f"[match_score] Subsampling jobs: {len(jobs):,} -> {MAX_JOBS:,}")
        jobs = jobs.sample(n=MAX_JOBS, random_state=42).reset_index(drop=True)

    # Jobs now come from the same Russian-language dataset as candidates,
    # with a free-text seniority field -- skip the seniority filter and
    # match directly on skill-text similarity across all sampled jobs.
    match_df = compute_match_scores(candidates, jobs, use_seniority_filter=False)
    print(match_df.head(10))
    print(f"\nTotal candidate-job pairs: {len(match_df)}")
    print(f"Match score distribution:\n{match_df['match_score'].describe()}")

    out_path = os.path.join(DATA_DIR, "match_scores.csv")
    match_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
