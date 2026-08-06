"""
Sample size / power calculation -- done BEFORE looking at "experiment"
results, as a real DS would. Uses the primary metric (interview->offer
conversion rate per application) as the basis.
"""

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def required_sample_size(baseline_rate: float, mde_abs: float,
                          alpha: float = 0.05, power: float = 0.8) -> dict:
    """
    baseline_rate: expected control conversion rate (e.g. 0.24 = 24%)
    mde_abs: minimum detectable *absolute* effect you care about (e.g. 0.03 = 3pp)
    Returns required sample size PER ARM (of applications, not candidates).
    """
    treatment_rate = baseline_rate + mde_abs
    effect_size = proportion_effectsize(treatment_rate, baseline_rate)

    analysis = NormalIndPower()
    n_per_arm = analysis.solve_power(
        effect_size=effect_size, alpha=alpha, power=power, ratio=1.0
    )
    return {
        "baseline_rate": baseline_rate,
        "treatment_rate": treatment_rate,
        "mde_abs": mde_abs,
        "alpha": alpha,
        "power": power,
        "effect_size_h": effect_size,
        "required_n_per_arm": int(round(n_per_arm)),
    }


if __name__ == "__main__":
    # Baseline conversion ~24% (from our funnel simulation / typical tech
    # hiring benchmarks). We care about detecting at least a 3-point
    # absolute lift (24% -> 27%) -- below that, the change isn't worth the
    # product/eng investment to ship.
    scenarios = [
        {"mde_abs": 0.02, "label": "2pp lift (24% -> 26%)"},
        {"mde_abs": 0.03, "label": "3pp lift (24% -> 27%)"},
        {"mde_abs": 0.05, "label": "5pp lift (24% -> 29%)"},
    ]

    print("Sample size required PER ARM (unit = applications), baseline=24%:\n")
    for s in scenarios:
        result = required_sample_size(baseline_rate=0.24, mde_abs=s["mde_abs"])
        print(f"  {s['label']:30s} -> {result['required_n_per_arm']:,} applications/arm")

    print(
        "\nNote: our unit of RANDOMIZATION is candidates, but this metric is "
        "measured per APPLICATION. Since each candidate can submit multiple "
        "applications, we need candidate-level counts too -- see "
        "run_experiment.py for how many candidates that implies given our "
        "simulated ~0.2-0.35 applications-per-candidate-per-job-seen rate."
    )
