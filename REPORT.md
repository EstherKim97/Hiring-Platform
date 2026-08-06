# Does Showing Candidates Their AI Match Score Actually Help Them?

## Why I looked into this

I've been job hunting recently, and like a lot of people I kept running
into AI "match score" badges on job postings — LinkedIn Premium shows one
on almost every listing, and a few other sites do too. At first I trusted
it. A high score gave me hope and made me prioritize applying there. But
the more jobs I applied to and the more carefully I actually read the
descriptions, the less the score seemed to line up with reality. I'd see
a high match score on postings where I clearly didn't meet the core
requirements, and a low score on ones where my experience actually looked
like a solid fit.

That disconnect is what got me curious enough to actually dig into it
properly instead of just complaining about it: does showing candidates a
match score help them make better decisions, or does it end up doing more
harm than good? I built a randomized experiment on real hiring data to
find out. The answer surprised me — it makes hiring outcomes worse, not
better — and the way I found that out is honestly the more interesting
part of this project.

## What I found, in short

Candidates who saw their match score before applying were significantly
*less* likely to land any offer at all — 11.6%, compared to 19.1% for
candidates who never saw a score. That's a real, meaningful drop.

If you only look at the people who actually applied, though, it looks
like the opposite happened — offer rate goes up, from 6.4% to 14.4%. My
first instinct was that this looked like a clean win. It isn't. The
problem is that seeing the score changes who applies in the first place,
so "offer rate among applicants" isn't comparing the same group of people
across the two conditions anymore. Once I controlled for that using CUPED
(a variance-reduction technique that came out of Microsoft's own
experimentation team), the "improvement" disappeared almost entirely.

What's actually going on: seeing a mediocre score just scares people off.
Applications dropped 70% in the group that saw their score, and that drop
costs more, in terms of actual offers landed, than any quality benefit
from the people who stuck around and applied anyway. It also wasn't even.
Candidates from less traditional backgrounds were discouraged from
applying more than everyone else — so on top of hurting outcomes overall,
it was arguably making the process less fair too.

![Application volume collapsed](01_application_volume.png)

## The business question

On an AI-matching hiring platform, candidates normally apply blind — they
never see how well the algorithm thinks they fit a role. Would showing
them that score before they apply actually help?

## How I set it up

I used real candidate resumes and real job postings — over 150,000 actual
application records — to compute genuine match scores between people and
roles, using text similarity between listed skills and job requirements.
Nothing about the match score itself is made up.

What I couldn't get real data for is how someone actually behaves once
they *see* their score — no public dataset captures that, because it
would need a live product experiment that only the company running the
platform could run. So I simulated that one piece, and tried to be
upfront about how, rather than quietly burying the assumption (it's all
documented in `src/simulate_behavior.py` if you want the details).

I randomized at the candidate level — one person seeing their score
doesn't affect anyone else's decision, so there's no interference problem
to worry about there. I tracked the obvious metric (offer rate among
people who applied), but also two guardrails: total application volume,
and whether the effect landed evenly across different candidate
backgrounds.

## Why the naive number is wrong

Looking only at submitted applications:

- Control: 6.42% offer rate
- Treatment: 14.44% offer rate
- p < 0.0001 — on its face, this looks like a clear win

But this comparison is comparing two different populations, not measuring
the treatment's actual effect. Who applies changes because of the
treatment itself, so you can't just compare outcomes among the people who
happened to apply in each group. I used CUPED — using match_score itself
as a pre-treatment covariate — to correct for this, and the result flips:
p = 0.37, not significant. The entire apparent lift turns out to be
explained by people self-selecting into better-matched jobs, not by any
real improvement in the hiring process.

![Naive vs real effect](02_naive_vs_real_effect.png)

## The real result

To get an honest answer, I measured every candidate who was assigned to
the experiment, whether they applied or not — treating a non-applicant as
a zero, not as a missing data point. That's the number that actually
answers the question I cared about.

- Control: 19.07% of candidates landed at least one offer
- Treatment: 11.58% landed at least one offer
- Difference: -7.5 points, p < 0.0001 — significant, and in the wrong
  direction

Seeing the score didn't make people apply smarter overall. Mostly it just
made them apply less, and that cost more than it gained.

## Guardrails

Application volume dropped 70.6% (from 32.5% of shown job pairs down to
9.5%). Even on its own, a drop that large should stop a rollout — before
you even get to whether the primary metric moved.

The effect also wasn't even across candidates. People with less
traditional educational backgrounds were discouraged more than others
(-78.7% application rate vs. -60.9%). I also ran a causal forest (a method
from Wager & Athey that estimates each candidate's individual treatment
effect using all their covariates at once, instead of testing one segment
at a time), and it found that 94% of candidates have an estimated
individual effect below -5 points on their chance of applying — this
treatment hurts almost everyone's odds of applying, and non-traditional
candidates get hit hardest.

![Equity gap](03_equity_gap.png)

## What I'd recommend

Don't ship this. The volume drop alone outweighs any quality benefit, the
effect on actual hiring outcomes is negative and statistically real, and
it disproportionately discourages candidates from non-traditional
backgrounds. If a company wanted to revisit the idea, I'd want to test a
softer framing first — something that contextualizes a low score instead
of just showing a bare number — rather than the blunt version tested here.

## Methods, for anyone curious about the technical side

- **CUPED** (Deng et al., 2013) — a variance-reduction technique using a
  pre-treatment covariate. This is what caught the selection bias problem.
- **Always-valid sequential testing** (Johari et al., 2015/2022) — lets
  you monitor results continuously without inflating false positives from
  peeking. Worth noting: this corrects for a different kind of bias than
  CUPED does — it agreed with the naive z-test here because both were
  looking at the same (biased) applicants-only comparison.
- **Causal forest CATE** (Wager & Athey) — individual-level treatment
  effect estimation, used for the equity analysis.
- **Full-population analysis** — the metric that actually answers the
  real question, without conditioning on a sample that the treatment
  itself changed.

Full code is set up to run end to end on the real data if you want to reproduce any of this.
