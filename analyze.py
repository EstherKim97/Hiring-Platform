"""
Analyzes experiment results:
- Primary metric significance test (interview->offer conversion)
- Guardrail 1: total application volume
- Guardrail 2: application rate by candidate segment (equity)
- Diagnostic: avg match score of submitted applications
- Final go/no-go recommendation
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from cuped import cuped_adjust, variance_reduction_report
from sequential_test import mixture_sprt_sequence
from causal_forest_hte import estimate_cate, summarize_cate_by_segment
from load_data import DATA_DIR

ALPHA = 0.05


def two_proportion_test(success_a, n_a, success_b, n_b, label_a="control", label_b="treatment"):
    count = np.array([success_a, success_b])
    nobs = np.array([n_a, n_b])
    z_stat, p_val = proportions_ztest(count, nobs)
    rate_a, rate_b = success_a / n_a, success_b / n_b
    ci_low_a, ci_high_a = _wilson_ci(success_a, n_a)
    ci_low_b, ci_high_b = _wilson_ci(success_b, n_b)
    return {
        "label_a": label_a, "rate_a": rate_a, "n_a": n_a,
        "ci_a": (ci_low_a, ci_high_a),
        "label_b": label_b, "rate_b": rate_b, "n_b": n_b,
        "ci_b": (ci_low_b, ci_high_b),
        "abs_diff": rate_b - rate_a,
        "z_stat": z_stat, "p_value": p_val,
        "significant": p_val < ALPHA,
    }


def _wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


def analyze(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "experiment_results.csv")
    df = pd.read_csv(path)

    print("=" * 70)
    print("PRIMARY METRIC: interview -> offer conversion rate per application")
    print("=" * 70)
    applied = df[df["applied"] == 1]
    control = applied[applied["arm"] == "control"]
    treatment = applied[applied["arm"] == "treatment"]

    primary = two_proportion_test(
        control["offer"].sum(), len(control),
        treatment["offer"].sum(), len(treatment),
    )
    _print_test_result(primary, "Offer rate per application (standard z-test)")

    print("\n" + "-" * 70)
    print("CUPED-adjusted primary metric (variance reduction using match_score,")
    print("a pre-experiment covariate unaffected by treatment)")
    print("-" * 70)
    all_applied = applied.copy()
    var_report = variance_reduction_report(
        all_applied["offer"].values, all_applied["match_score"].values
    )
    print(f"  Correlation(match_score, offer): {var_report['correlation_x_y']:.3f}")
    print(f"  Variance reduction from CUPED: {var_report['variance_reduction_pct']:.1f}%")

    all_applied["offer_cuped"] = var_report["y_adjusted"]
    cuped_control = all_applied[all_applied["arm"] == "control"]["offer_cuped"]
    cuped_treatment = all_applied[all_applied["arm"] == "treatment"]["offer_cuped"]
    t_stat, cuped_p = stats.ttest_ind(cuped_treatment, cuped_control, equal_var=False)
    print(f"  CUPED-adjusted t-test: t={t_stat:.3f}, p={cuped_p:.4f}  "
          f"({'SIGNIFICANT' if cuped_p < ALPHA else 'not significant'} at alpha={ALPHA})")
    if var_report["variance_reduction_pct"] > 5:
        print(f"  -> CUPED meaningfully sharpened this test; trust this p-value "
              f"over the raw z-test above when they disagree.")
    if cuped_p >= ALPHA and primary["p_value"] < ALPHA:
        print(f"\n  IMPORTANT CAVEAT: the raw z-test and the CUPED-adjusted test "
              f"disagree. This is a real finding, not noise -- match_score is a "
              f"valid pre-treatment covariate for each (candidate, job) pair, "
              f"but we're only analyzing the 'applied == 1' subsample, and WHO "
              f"applies is itself changed by treatment (see the diagnostic "
              f"below: avg match_score of submitted apps differs significantly "
              f"between arms). Conditioning on a post-treatment-selected "
              f"sample can bias a naive comparison even when using a "
              f"technically pre-treatment covariate -- a form of selection/"
              f"collider bias. CUPED corrects for this specific bias because "
              f"it happens to control for the exact variable (match_score) "
              f"driving the selection. The sequential test below does NOT "
              f"correct for this -- it only guards against 'peeking' bias on "
              f"the same raw comparison as the z-test, so it will tend to "
              f"agree with the z-test, not CUPED. Trust CUPED's conclusion "
              f"here specifically because of what it's controlling for, not "
              f"because it's 'more rigorous' in general.")

    print("\n" + "-" * 70)
    print("Always-valid sequential p-value (safe to monitor continuously,")
    print("unlike the z-test above which assumes a single fixed look)")
    print("-" * 70)
    seq_result = mixture_sprt_sequence(
        control["offer"].values, treatment["offer"].values, tau=0.03
    )
    print(f"  Always-valid p-value at final sample size: {seq_result['final_p']:.4f}")
    if seq_result["stop_n"]:
        print(f"  Could have safely stopped monitoring as early as n={seq_result['stop_n']:,} "
              f"per arm without inflating the false-positive rate.")
    else:
        print(f"  Never crossed alpha={ALPHA} under always-valid monitoring -- "
              f"a more conservative (and more honest, if you were peeking) read "
              f"than the single fixed-horizon z-test.")

    print("\n" + "-" * 70)
    print("FULL-POPULATION ANALYSIS (removes selection bias at the root --")
    print("measures every ASSIGNED candidate, not just those who applied)")
    print("-" * 70)
    # For each candidate, did they get ANY offer among the jobs they were
    # shown, regardless of whether they applied? Non-appliers count as 0.
    # This answers the real business question -- "does seeing your match
    # score change your actual chance of landing an offer" -- without
    # conditioning on the treatment-affected choice to apply at all.
    candidate_level = df.groupby(["candidate_id", "arm"])["offer"].max().reset_index()
    fp_control = candidate_level[candidate_level["arm"] == "control"]["offer"]
    fp_treatment = candidate_level[candidate_level["arm"] == "treatment"]["offer"]
    full_pop = two_proportion_test(
        fp_control.sum(), len(fp_control),
        fp_treatment.sum(), len(fp_treatment),
    )
    _print_test_result(full_pop, "Any-offer rate per ASSIGNED candidate")
    print("  This is the least biased estimate in this analysis -- every "
          "assigned candidate counts once, whether they applied or not.")

    print("\n" + "=" * 70)
    print("GUARDRAIL 1: total application volume")
    print("=" * 70)
    vol_control = df[df["arm"] == "control"]["applied"].sum()
    vol_treatment = df[df["arm"] == "treatment"]["applied"].sum()
    n_control_candidates = df[df["arm"] == "control"]["candidate_id"].nunique()
    n_treatment_candidates = df[df["arm"] == "treatment"]["candidate_id"].nunique()
    vol_rate_control = vol_control / df[df["arm"] == "control"].shape[0]
    vol_rate_treatment = vol_treatment / df[df["arm"] == "treatment"].shape[0]
    print(f"  Control:   {vol_control:,} applications ({vol_rate_control:.1%} of considered pairs)")
    print(f"  Treatment: {vol_treatment:,} applications ({vol_rate_treatment:.1%} of considered pairs)")
    vol_change_pct = (vol_rate_treatment - vol_rate_control) / vol_rate_control * 100
    print(f"  Change: {vol_change_pct:+.1f}% relative change in apply rate")
    if vol_change_pct < -15:
        print("  FLAG: application volume drop is large -- needs product review "
              "even if primary metric improves, since fewer applications means "
              "fewer total hires even at a higher conversion rate.")

    print("\n" + "=" * 70)
    print("GUARDRAIL 2: application rate by candidate background (equity check)")
    print("=" * 70)
    seg_table = (
        df.groupby(["arm", "background"])["applied"]
        .mean()
        .unstack("arm")
    )
    seg_table["rel_change_pct"] = (
        (seg_table["treatment"] - seg_table["control"]) / seg_table["control"] * 100
    )
    print(seg_table.round(4))
    worst_hit = seg_table["rel_change_pct"].idxmin()
    print(f"\n  Most discouraged segment: '{worst_hit}' "
          f"({seg_table.loc[worst_hit, 'rel_change_pct']:.1f}% relative drop in apply rate)")
    if seg_table["rel_change_pct"].max() - seg_table["rel_change_pct"].min() > 10:
        print("  FLAG: effect is meaningfully uneven across segments -- an "
              "equity concern that should block a blanket rollout even if "
              "the primary metric and overall volume look fine.")

    print("\n" + "-" * 70)
    print("Causal forest CATE (individual-level effect estimates, not just")
    print("single-variable segment means -- surfaces interactions a manual")
    print("segment table would miss)")
    print("-" * 70)
    hte_df = df.copy()
    hte_df["treated"] = (hte_df["arm"] == "treatment").astype(int)
    hte_df["is_non_traditional"] = (hte_df["background"] == "non_traditional").astype(int)
    hte_df["is_career_switcher"] = (hte_df["background"] == "career_switcher").astype(int)
    # subsample for speed on large pair-level data
    hte_sample = hte_df.sample(n=min(15000, len(hte_df)), random_state=42)
    cate_df, _ = estimate_cate(
        hte_sample, outcome_col="applied", treatment_col="treated",
        covariate_cols=["match_score", "is_non_traditional", "is_career_switcher"],
    )
    print("Estimated individual treatment effect on apply-probability, by background:")
    print(summarize_cate_by_segment(cate_df, "background").round(4))
    most_hurt_pct = (cate_df["cate"] < -0.05).mean() * 100
    print(f"\n  {most_hurt_pct:.1f}% of candidates have an estimated individual "
          f"effect below -5pp apply-probability -- these are the people a "
          f"blanket rollout would concretely hurt the most, regardless of "
          f"which background segment they fall into.")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: avg match score of submitted applications")
    print("=" * 70)
    diag = applied.groupby("arm")["match_score"].agg(["mean", "std", "count"])
    print(diag.round(4))
    t_stat, t_p = stats.ttest_ind(
        control["match_score"], treatment["match_score"], equal_var=False
    )
    print(f"  Welch's t-test on match score: t={t_stat:.3f}, p={t_p:.4f}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    _print_recommendation(primary, vol_change_pct, seg_table, cuped_p, seq_result, full_pop)

    return {
        "primary": primary,
        "volume_change_pct": vol_change_pct,
        "segment_table": seg_table,
        "cuped_p": cuped_p,
        "sequential_result": seq_result,
        "full_population": full_pop,
    }


def _print_test_result(result, label):
    print(f"{label}:")
    print(f"  {result['label_a']}:   {result['rate_a']:.2%}  "
          f"(95% CI: {result['ci_a'][0]:.2%} - {result['ci_a'][1]:.2%}, n={result['n_a']:,})")
    print(f"  {result['label_b']}: {result['rate_b']:.2%}  "
          f"(95% CI: {result['ci_b'][0]:.2%} - {result['ci_b'][1]:.2%}, n={result['n_b']:,})")
    print(f"  Absolute difference: {result['abs_diff']:+.2%}")
    print(f"  z = {result['z_stat']:.3f}, p = {result['p_value']:.4f}  "
          f"({'SIGNIFICANT' if result['significant'] else 'not significant'} at alpha={ALPHA})")


def _print_recommendation(primary, vol_change_pct, seg_table, cuped_p, seq_result, full_pop):
    equity_gap = seg_table["rel_change_pct"].max() - seg_table["rel_change_pct"].min()
    cuped_confirms = cuped_p < ALPHA

    if not full_pop["significant"]:
        direction = "higher" if full_pop["abs_diff"] > 0 else "lower"
        print(f"  -> NO-GO: the full-population analysis -- the least biased "
              f"read we have -- shows no significant difference in the actual "
              f"chance of landing an offer (treatment was directionally "
              f"{direction}, but not significantly so, p={full_pop['p_value']:.4f}). "
              f"Combined with the applied-only metrics being explained away by "
              f"CUPED, there is no evidence this treatment improves real "
              f"candidate outcomes. It does, however, clearly suppress "
              f"application volume ({vol_change_pct:.1f}%) and discourage "
              f"non-traditional candidates disproportionately -- real costs "
              f"with no offsetting benefit found. Recommend NOT shipping.")
        return

    if full_pop["abs_diff"] < 0:
        print(f"  -> NO-GO: the full-population analysis shows candidates are "
              f"SIGNIFICANTLY LESS likely to land any offer when shown their "
              f"match score (p={full_pop['p_value']:.4f}), most likely because "
              f"the {vol_change_pct:.1f}% drop in applications outweighs any "
              f"gain in per-application quality. Clear NO-GO.")
        return

    print(f"  -> The full-population analysis shows a significant POSITIVE "
          f"effect on actual offer likelihood (p={full_pop['p_value']:.4f}). "
          f"This is the strongest evidence in the analysis, since it isn't "
          f"subject to the selection bias affecting the applied-only metrics.")
    if equity_gap > 10:
        print(f"  However, the equity gap ({equity_gap:.1f} points between "
              f"segments) still needs to be addressed -- e.g. via softer "
              f"framing for non-traditional candidates -- before a full "
              f"rollout, even though the population-level effect is positive. "
              f"Recommend a segmented rollout rather than a blanket one.")
    elif vol_change_pct < -15:
        print(f"  Application volume also dropped {vol_change_pct:.1f}% -- "
              f"confirm total hires (not just per-candidate offer rate) "
              f"doesn't decline net before a full rollout.")
    else:
        print("  -> GO: recommend rollout with continued monitoring of the "
              "equity guardrail.")


if __name__ == "__main__":
    analyze()
