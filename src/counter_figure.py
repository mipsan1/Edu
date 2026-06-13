"""
counter_figure.py — Render the counter-experiment figure.

Reads results/counter_experiment.csv and produces a 2-panel figure:
  (a) Pearson r as a function of noise sigma for Shapley (noisy v(S) input)
  (b) Comparison of Shapley (clean), Shapley (noisy, sigma=8), and the four
      baselines (Equal, Peer, Instructor, CATME)

Output: figures/fig5_counter_experiment.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS_OF_INTEREST = [
    ("Equal",            "Equal",            "#888"),
    ("Peer",             "Peer ($\\sigma=8$)",  "#5aa"),
    ("Instructor",       "Instructor ($\\sigma=12$)", "#5aa"),
    ("CATME",            "CATME ($\\sigma=5$)", "#5aa"),
    ("Shapley (clean)",  "Shapley (clean)",  "#c33"),
    ("Shapley (noisy)",  "Shapley (noisy $v(S)$)", "#c33"),
]


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["noise_sigma"] = pd.to_numeric(df["noise_sigma"], errors="coerce")
    df["fdr"] = pd.to_numeric(df["fdr"], errors="coerce")
    return df


def fig5_counter_experiment(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    # Panel (a): Shapley (noisy) Pearson r vs. noise sigma
    ax = axes[0]
    noisy = df[df["method"] == "Shapley (noisy)"]
    grp = noisy.groupby("noise_sigma")["pearson_r"].agg(["mean", "sem"])
    ax.errorbar(grp.index, grp["mean"], yerr=grp["sem"],
                marker="o", color="#c33", linewidth=2, capsize=3,
                label="Shapley (noisy $v(S)$)")
    # Add horizontal reference lines for the four baselines + clean Shapley
    baselines = {
        "Peer (\\sigma=8)":       df[df["method"] == "Peer"]["pearson_r"].mean(),
        "Instructor (\\sigma=12)": df[df["method"] == "Instructor"]["pearson_r"].mean(),
        "CATME (\\sigma=5)":      df[df["method"] == "CATME"]["pearson_r"].mean(),
        "Shapley (clean)":       df[df["method"] == "Shapley (clean)"]["pearson_r"].mean(),
    }
    for label, val in baselines.items():
        ax.axhline(val, linestyle="--", linewidth=1, alpha=0.6,
                   label=f"{label} (mean $r = {val:.3f}$)")
    ax.set_xlabel(r"Noise $\sigma$ applied to $v(S)$ inputs $\hat{p}_i$")
    ax.set_ylabel("Pearson $r$ (alignment with clean $p_i$)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(a) SVFGS robustness to $v(S)$ input noise", fontsize=10)

    # Panel (b): Bar chart of grand-mean Pearson r
    ax = axes[1]
    method_means = {}
    for method, label, color in METHODS_OF_INTEREST:
        sub = df[df["method"] == method]["pearson_r"]
        method_means[label] = (sub.mean(), sub.sem(), color)
    labels = list(method_means.keys())
    means = [v[0] for v in method_means.values()]
    sems  = [v[1] for v in method_means.values()]
    colors = [v[2] for v in method_means.values()]
    bars = ax.bar(range(len(labels)), means, yerr=sems, capsize=3, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Pearson $r$ (grand mean, $N = 1{,}400$ runs)")
    ax.set_ylim(0, 1.05)
    # Annotate bar values
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_title("(b) Grand-mean comparison", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    df = load_results(repo / "results" / "counter_experiment.csv")
    out = repo / "figures" / "fig5_counter_experiment.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig5_counter_experiment(df, out)
    print(f"Wrote {out}")

    # Print a concise numerical summary to stdout
    print()
    print("Counter-experiment grand-mean Pearson r:")
    print("-" * 60)
    for method, label, _ in METHODS_OF_INTEREST:
        sub = df[df["method"] == method]["pearson_r"]
        print(f"  {label:<35s}  r = {sub.mean():.4f} ± {sub.sem():.4f}")
    print()
    print("Noisy Shapley r by sigma:")
    print("-" * 60)
    noisy = df[df["method"] == "Shapley (noisy)"]
    for sigma, sub in noisy.groupby("noise_sigma"):
        print(f"  sigma = {sigma:5.1f}    r = {sub['pearson_r'].mean():.4f} ± {sub['pearson_r'].sem():.4f}")


if __name__ == "__main__":
    main()
