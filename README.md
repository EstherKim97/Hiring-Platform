# Does Showing Candidates Their AI Match Score Actually Help Them?

I've been job hunting recently, and like a lot of people I kept running
into those AI "match score" badges — LinkedIn Premium puts one on almost
every posting, and a few other job sites do too. At first I trusted it. A
high score gave me hope, and I'd prioritize applying there. But the more
jobs I applied to and the more carefully I actually read the job
descriptions, the less the score seemed to line up with reality. I'd see
a high match score on postings where I clearly didn't meet the core
requirements, and a low score on ones where my experience was honestly a
pretty good fit.

That disconnect is what got me curious: does showing people a match score
actually help them apply smarter, or could it be doing more harm than
good? So I built an experiment to find out, using real hiring data instead
of just my own anecdotal feeling. The answer surprised me — it makes
hiring outcomes *worse*, not better. Here's the experiment, and how a
naive read of the results would have gotten the conclusion backwards.

![Naive vs real effect](output/charts_real/02_naive_vs_real_effect.png)

## The short version

Candidates who saw their match score before applying ended up
significantly *less* likely to land any offer at all — 11.6% vs. 19.1% in
the group that didn't see a score. If you only look at people who actually
applied, it looks like the opposite happened: offer rate went up, from
6.4% to 14.4%. But that comparison is misleading, because seeing the score
changes who bothers to apply in the first place. You're not comparing the
same group of people anymore, so the "improvement" isn't real — it's an
illusion caused by who dropped out.

Once you control for that (I used CUPED, a variance-reduction technique
that came out of Microsoft's experimentation team), the quality
improvement basically vanishes. What's actually happening is that seeing a
mediocre score just scares people off — applications dropped 70% in the
group that saw their score — and that drop in volume costs more than any
quality gain from the people who stuck around. It also hit some people
harder than others: candidates with less traditional backgrounds were
discouraged from applying even more than everyone else, so on top of the
overall harm, it's arguably making the process less fair too.

**Full write-up:** [`output/REPORT.md`](output/REPORT.md)

## Why I built it this way

Most A/B testing portfolio projects stop at "I ran a t-test and got a
p-value." I wanted this one to actually reflect the mistake that trips up
real data scientists, which is measuring the wrong population. The naive
metric said ship it. The real metric said don't. Catching that gap, and
being able to explain why it happens, is really the point of this project
— not the t-test itself.

## What's real and what I had to simulate

The candidate resumes, job postings, and match scores are all real — over
150,000 actual application records, and the match scores come from genuine
text similarity between candidate skills and job requirements, not made
up. What I couldn't get real data for is how someone actually reacts to
*seeing* their match score before applying — no public dataset captures
that, since it would require a live product experiment that only the
company running the platform could actually run. I simulated that one
piece, and I tried to be upfront about exactly how (it's all documented in
`src/simulate_behavior.py`), rather than quietly hiding it.

## Methods used

| Technique | What it's for |
|---|---|
| Randomized experiment design | Candidate-level randomization, SRM checks |
| Power analysis | Sample size before running anything |
| CUPED (Deng et al. 2013) | Variance reduction — this is what caught the selection bias |
| Always-valid sequential testing (Johari et al.) | Peeking-safe continuous monitoring |
| Causal forest CATE (Wager & Athey) | Individual-level heterogeneous treatment effects |
| Full-population analysis | The de-biased answer to the actual question I cared about |

## Repo structure

```
src/                  pipeline code (run in order: load_data -> match_score
                       -> run_experiment -> analyze -> charts)
output/REPORT.md       full write-up with methodology and results
output/charts_real/    charts generated from the real experiment run
data/                  (not included — see data/README.md for where to
                        get the real datasets and how to set them up)
```

## Reproducing this

```bash
pip install pandas numpy scipy scikit-learn statsmodels econml matplotlib
python3 src/load_data.py
python3 src/match_score.py
python3 src/run_experiment.py
python3 src/analyze.py
python3 src/charts.py
```

`data/README.md` has the real dataset sources and setup instructions.
