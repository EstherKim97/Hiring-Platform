import pandas as pd

# ---- Need to adjust to the location of the data ----
POSTINGS_PATH = "data/postings.csv"
JOB_SKILLS_PATH = "data/job_skills.csv"
SKILLS_MAP_PATH = "data/skills.csv"
OUT_PATH = "data/postings_sample.csv"

SAMPLE_SIZE = 20000

# Only read the columns we actually need (huge memory/size savings)
postings_cols = ["job_id", "title", "formatted_experience_level", "location"]
postings = pd.read_csv(POSTINGS_PATH, usecols=postings_cols)

# Drop rows with no experience level (we need this for seniority matching)
postings = postings.dropna(subset=["formatted_experience_level"])

# Sample down to a manageable size
postings = postings.sample(n=min(SAMPLE_SIZE, len(postings)), random_state=42)

# Join skills onto the sampled jobs only (much cheaper than joining first)
job_skills = pd.read_csv(JOB_SKILLS_PATH)
skills_map = pd.read_csv(SKILLS_MAP_PATH)

job_skills = job_skills.merge(skills_map, on="skill_abr", how="left")
job_skills_agg = (
    job_skills.groupby("job_id")["skill_name"]
    .apply(lambda x: ", ".join(x.dropna().unique()))
    .reset_index()
    .rename(columns={"skill_name": "required_skills"})
)

postings = postings.merge(job_skills_agg, on="job_id", how="left")
postings = postings.dropna(subset=["required_skills"])  # need skills to compute match scores

postings.to_csv(OUT_PATH, index=False)
print(f"Saved {len(postings):,} rows to {OUT_PATH}")
print(f"File size check: run `ls -lh {OUT_PATH}` -- should be just a few MB")
