"""
Loads real data:
- LinkedIn Job Postings dataset (postings.csv + job_skills.csv + skills.csv)
- Resume-Job Matching Dataset (train.csv, pipe-delimited)

Falls back to synthetic data only if real files aren't found. Prints
diagnostics on every run so schema issues surface immediately instead of
failing silently downstream.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

RESUME_MATCH_FILE = "Resume-Job Matching Dataset train.csv"


def _derive_background(out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    No usable 'background' field in this dataset (academicDegree is entirely
    null). Try a few candidate fields in priority order and use whichever
    actually has variety; falls back to a skill-overlap heuristic (does the
    candidate's stated desired position textually match their hard skills?
    low overlap => 'non_traditional' proxy, i.e. someone whose skills don't
    obviously match their stated target role -- e.g. a career switcher).
    This is a documented proxy, not ground truth -- flagged clearly here
    and in the report.
    """
    for col in ["education", "educationList", "typicalPosition_cv"]:
        if col in df.columns and df[col].notna().sum() > len(df) * 0.1:
            print(f"[load_data] Using '{col}' for background proxy "
                  f"(non-null: {df[col].notna().sum():,} / {len(df):,})")
            print(f"[load_data] Sample values: {df[col].dropna().unique()[:10].tolist()}")
            val = df[col].astype(str)
            # This dataset's education field is in Russian. "Высшее" (any
            # variant) = has a higher-education degree = 'traditional' proxy.
            # Everything else (Среднее/secondary, Основное/basic-general,
            # Нет/none, etc.) = 'non_traditional'. Documented assumption --
            # not a judgment on candidate quality, just the split we defined
            # for this experiment's equity check.
            has_higher_ed = val.str.contains("Высшее", case=False, na=False)
            out["background"] = np.where(has_higher_ed, "traditional", "non_traditional")
            return out

    print("[load_data] No usable education field found -- falling back to "
          "skill/desired-position text-overlap heuristic for background proxy.")
    desired = out["desired_position"].astype(str).str.lower()
    skills = out["skills"].astype(str).str.lower()
    overlap = [
        any(word in skills.iloc[i] for word in desired.iloc[i].split() if len(word) > 3)
        for i in range(len(out))
    ]
    out["background"] = np.where(overlap, "traditional", "non_traditional")
    return out


def bucket_years_experience(years: pd.Series) -> pd.Series:
    """
    Maps continuous years_experience onto the same seniority buckets used
    by the LinkedIn job postings (formatted_experience_level), so
    match_score.py can compare candidates and jobs on a shared scale.
    """
    years = pd.to_numeric(years, errors="coerce").fillna(0)
    bins = [-1, 0.5, 2, 5, 10, 100]
    labels = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Executive"]
    return pd.cut(years, bins=bins, labels=labels).astype(str)


# ---------------------------------------------------------------------
# Candidates: from Resume-Job Matching Dataset
# ---------------------------------------------------------------------
def load_candidates():
    real_path = os.path.join(DATA_DIR, RESUME_MATCH_FILE)
    if os.path.exists(real_path):
        print(f"[load_data] Using REAL candidate data: {real_path}")
        df = pd.read_csv(real_path, sep="|")

        print(f"[load_data] Raw shape: {df.shape}")
        print(f"[load_data] cv_status value counts:\n{df['cv_status'].value_counts()}")

        out = pd.DataFrame()
        out["candidate_id"] = df["idCv"]
        out["job_id_applied_to"] = df["idVacancy"]  # keep the real pairing info
        out["cv_status_raw"] = df["cv_status"]

        # cv_status is in Russian: "Отказ" = Rejection, "Приглашение" = Invitation.
        # This dataset only has these two states -- no separate offer stage --
        # so our "invited" flag is the real outcome, and we treat it as our
        # primary metric going forward (redefining "offer" as an alias for
        # "invited" rather than a separate funnel stage).
        out["interviewed"] = (df["cv_status"] == "Приглашение").astype(int)
        out["offer"] = out["interviewed"]  # no distinct offer stage in this data

        # Candidate skills -- combine hard + soft skills text
        out["skills"] = (
            df.get("hardSkills_cv", "").fillna("").astype(str)
            + ", "
            + df.get("softSkills_cv", "").fillna("").astype(str)
            + ", "
            + df.get("skills_cv", "").fillna("").astype(str)
        ).str.strip(", ")

        out["years_experience"] = pd.to_numeric(df.get("experience"), errors="coerce")
        out["age"] = pd.to_numeric(df.get("age"), errors="coerce")
        out["desired_position"] = df.get("positionName", "")
        out["academic_degree_raw"] = df.get("academicDegree", "")

        out = out.dropna(subset=["skills"])
        out = out[out["skills"].str.len() > 3]  # drop empty/near-empty skills rows

        print(f"[load_data] Invitation rate (real): {out['interviewed'].mean():.2%}")
        print(f"[load_data] academicDegree unique values: {df['academicDegree'].unique().tolist()}")

        out = _derive_background(out, df)

        # IMPORTANT: each row in this dataset is one APPLICATION EVENT, so
        # the same candidate can appear multiple times (once per job they
        # applied to). Our experiment's unit of randomization is the
        # candidate, so we need exactly one row per unique candidate_id
        # before assignment -- otherwise a candidate could get randomly
        # assigned to both control AND treatment across their different
        # rows, which breaks the whole design. Collapse to one row per
        # candidate (first occurrence); we don't need the per-application
        # historical outcome for the simulation anyway, since our
        # simulated hire funnel generates its own outcomes.
        before_dedup = len(out)
        out = out.drop_duplicates(subset="candidate_id", keep="first").reset_index(drop=True)
        print(f"[load_data] Deduplicated to one row per candidate: "
              f"{before_dedup:,} application rows -> {len(out):,} unique candidates")

        print(f"[load_data] Cleaned candidates shape: {out.shape}")
        print(f"[load_data] Background distribution:\n{out['background'].value_counts()}")
        print(f"[load_data] Sample:\n{out.head(3)}")
        return out

    print("[load_data] Real resume-matching file not found, falling back to synthetic.")
    return _load_synthetic_candidates()


def _load_synthetic_candidates(n=5000):
    synth_path = os.path.join(DATA_DIR, "candidates_synthetic.csv")
    SKILL_POOL = [
        "python", "sql", "excel", "java", "javascript", "react", "aws",
        "machine learning", "data analysis", "project management",
        "communication", "sales", "customer service", "accounting",
        "marketing", "c++", "docker", "kubernetes", "figma", "product management",
    ]
    EXPERIENCE_LEVELS = ["entry", "mid", "senior"]
    if os.path.exists(synth_path):
        return pd.read_csv(synth_path)
    rows = []
    for i in range(n):
        n_skills = np.random.randint(3, 8)
        skills = np.random.choice(SKILL_POOL, size=n_skills, replace=False)
        rows.append({
            "candidate_id": f"C{i:06d}",
            "experience_level": np.random.choice(EXPERIENCE_LEVELS, p=[0.4, 0.4, 0.2]),
            "years_experience": max(0, np.random.normal(4, 3)),
            "skills": ", ".join(skills),
            "background": np.random.choice(
                ["traditional", "career_switcher", "non_traditional"], p=[0.6, 0.25, 0.15]
            ),
        })
    df = pd.DataFrame(rows)
    df.to_csv(synth_path, index=False)
    return df


# ---------------------------------------------------------------------
# Jobs: from LinkedIn Job Postings dataset
# ---------------------------------------------------------------------
def load_jobs():
    """
    Jobs are now derived from the SAME Resume-Job Matching Dataset as
    candidates (vacancy-side fields), not the LinkedIn dataset. Combining
    Russian-language candidate skills with English-language LinkedIn job
    skills produced near-zero TF-IDF similarity for every pair (a language/
    market mismatch, not a real finding) -- this file's vacancy fields are
    the same language and genuinely paired with real candidates.
    """
    real_path = os.path.join(DATA_DIR, RESUME_MATCH_FILE)
    if os.path.exists(real_path):
        print(f"[load_data] Using REAL job data (vacancy side of same file): {real_path}")
        df = pd.read_csv(real_path, sep="|")

        out = pd.DataFrame()
        out["job_id"] = df["idVacancy"]
        out["title"] = df.get("vacancyName", "")
        out["required_skills"] = (
            df.get("hardSkills_vacancy", "").fillna("").astype(str)
            + ", "
            + df.get("softSkills_vacancy", "").fillna("").astype(str)
            + ", "
            + df.get("skills_vacancy", "").fillna("").astype(str)
        ).str.strip(", ")
        out["seniority"] = df.get("experienceRequirements", "unspecified").fillna("unspecified")

        out = out.dropna(subset=["required_skills"])
        out = out[out["required_skills"].str.len() > 3]
        out = out.drop_duplicates(subset="job_id", keep="first").reset_index(drop=True)

        print(f"[load_data] Cleaned jobs shape: {out.shape}")
        print(f"[load_data] Sample:\n{out.head(3)}")
        return out

    print("[load_data] Real job file not found, falling back to synthetic.")
    return _load_synthetic_jobs()


def _load_synthetic_jobs(n=800):
    synth_path = os.path.join(DATA_DIR, "jobs_synthetic.csv")
    SKILL_POOL = [
        "python", "sql", "excel", "java", "javascript", "react", "aws",
        "machine learning", "data analysis", "project management",
        "communication", "sales", "customer service", "accounting",
        "marketing", "c++", "docker", "kubernetes", "figma", "product management",
    ]
    ROLES = ["Software Engineer", "Data Analyst", "Product Manager", "Sales Rep",
              "Customer Success Manager", "Marketing Specialist", "Accountant"]
    EXPERIENCE_LEVELS = ["entry", "mid", "senior"]
    if os.path.exists(synth_path):
        return pd.read_csv(synth_path)
    rows = []
    for i in range(n):
        role = np.random.choice(ROLES)
        n_skills = np.random.randint(3, 7)
        req_skills = np.random.choice(SKILL_POOL, size=n_skills, replace=False)
        rows.append({
            "job_id": f"J{i:05d}",
            "title": role,
            "required_skills": ", ".join(req_skills),
            "seniority": np.random.choice(EXPERIENCE_LEVELS, p=[0.35, 0.45, 0.2]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(synth_path, index=False)
    return df


if __name__ == "__main__":
    cands = load_candidates()
    jobs = load_jobs()
    print("\n=== FINAL CANDIDATES ===")
    print(cands.shape)
    print(cands.columns.tolist())
    print("\n=== FINAL JOBS ===")
    print(jobs.shape)
    print(jobs.columns.tolist())
