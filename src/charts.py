"""
Generates the 3 headline charts for the report, from the real
experiment_results.csv (or synthetic fallback if that's what's present).
Run this after analyze.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data import DATA_DIR

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_CONTROL = "#6B7280"
COLOR_TREATMENT = "#DC2626"
COLOR_GOOD = "#059669"


def chart_application_volume(df):
    vol = df.groupby("arm")["applied"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(vol.index, vol.values, color=[COLOR_CONTROL, COLOR_TREATMENT], width=0.5)
    for bar, val in zip(bars, vol.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%",
                ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("Applications submitted (% of pairs shown)", fontsize=11)
    ax.set_title("Application volume collapsed when candidates saw their match score",
                  fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_application_volume.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def chart_naive_vs_real(df):
    applied = df[df["applied"] == 1]
    naive = applied.groupby("arm")["offer"].mean() * 100

    candidate_level = df.groupby(["candidate_id", "arm"])["offer"].max().reset_index()
    full_pop = candidate_level.groupby("arm")["offer"].mean() * 100

    labels = ["Naive: offer rate\n(applicants only)", "Real: offer rate\n(all assigned candidates)"]
    control_vals = [naive.get("control", 0), full_pop.get("control", 0)]
    treatment_vals = [naive.get("treatment", 0), full_pop.get("treatment", 0)]

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    b1 = ax.bar(x - width / 2, control_vals, width, label="Control (no score shown)", color=COLOR_CONTROL)
    b2 = ax.bar(x + width / 2, treatment_vals, width, label="Treatment (score shown)", color=COLOR_TREATMENT)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}%",
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Offer rate", fontsize=11)
    ax.set_title("The naive comparison flips once you measure the full population",
                  fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_naive_vs_real_effect.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def chart_equity_gap(df):
    seg = df.groupby(["arm", "background"])["applied"].mean().unstack("arm") * 100
    seg = seg.sort_values("treatment")

    x = np.arange(len(seg.index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    b1 = ax.bar(x - width / 2, seg["control"], width, label="Control", color=COLOR_CONTROL)
    b2 = ax.bar(x + width / 2, seg["treatment"], width, label="Treatment", color=COLOR_TREATMENT)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}%",
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in seg.index], fontsize=11)
    ax.set_ylabel("Application rate", fontsize=11)
    ax.set_title("Non-traditional candidates were discouraged the most",
                  fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_equity_gap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    path = os.path.join(DATA_DIR, "experiment_results.csv")
    df = pd.read_csv(path)
    chart_application_volume(df)
    chart_naive_vs_real(df)
    chart_equity_gap(df)
    print("\nAll charts saved to output/charts/")
