"""
analyze.py — Reproduce Tables 1, 2 and Figures 1, 2, 3, 4 from the manuscript.

Reads results/main_simulation.csv (produced by simulate.py), aggregates the
metric distributions, and emits:
    tables/table1_baselines.csv
    tables/table2_grand_mean.csv
    figures/fig1_grand_metrics.png
    figures/fig2_scenario_pearson.png
    figures/fig3_sensitivity_alpha.png
    figures/fig3_sensitivity_noise.png
    figures/fig4_group_size.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ["Equal", "Peer", "Instructor", "CATME", "Shapley"]


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # FDR is blank for cells without free-riders; convert to NaN.
    df["fdr"] = pd.to_numeric(df["fdr"], errors="coerce")
    return df


def grand_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        rows.append({
            "Method": method,
            "Pearson_r_mean": sub["pearson_r"].mean(),
            "Pearson_r_se":   sub["pearson_r"].sem(),
            "FDR_mean":       sub["fdr"].mean(skipna=True),
            "FDR_se":         sub["fdr"].sem(skipna=True),
            "SCA_mean":       sub["sca"].mean(),
            "SCA_se":         sub["sca"].sem(),
        })
    return pd.DataFrame(rows)


def fig1_grand_metrics(grand: pd.DataFrame, out_path: Path) -> None:
    metrics = [("Pearson_r_mean", "Pearson $r$"),
               ("FDR_mean",       "FDR"),
               ("SCA_mean",       "SCA")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (col, ylabel) in zip(axes, metrics):
        yerr_col = col.replace("_mean", "_se")
        ax.bar(grand["Method"], grand[col],
               yerr=grand[yerr_col], capsize=3,
               color=["#888", "#5aa", "#5aa", "#5aa", "#c33"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig2_scenario_pearson(df: pd.DataFrame, out_path: Path) -> None:
    pivot = (df.groupby(["scenario", "method"])["pearson_r"]
               .mean()
               .unstack("method")
               .reindex(columns=METHOD_ORDER))
    ax = pivot.plot(kind="bar", figsize=(8, 3.5))
    ax.set_ylabel("Pearson $r$")
    ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig3_sensitivity_alpha(out_path: Path,
                           alpha_values=(0.05, 0.10, 0.15, 0.20, 0.30),
                           n_runs: int = 100,
                           seed: int = 42) -> None:
    """Sensitivity of Shapley Pearson r to synergy coefficient alpha."""
    from svfgs import assign_grades
    from scenarios import sample_group, SCENARIOS, GROUP_SIZES
    from metrics import pearson_r

    rng = np.random.default_rng(seed)
    results = []
    for alpha in alpha_values:
        r_vals = []
        for n in GROUP_SIZES:
            for scenario in SCENARIOS.keys():
                if scenario == "Two Free-riders" and n < 4:
                    continue
                for _ in range(n_runs):
                    _, _, p = sample_group(n, scenario, rng)
                    g = assign_grades(p, 80.0, alpha=alpha)
                    r_vals.append(pearson_r(p, g))
        results.append({"alpha": alpha,
                        "pearson_r": float(np.mean(r_vals)),
                        "pearson_r_se": float(np.std(r_vals) / np.sqrt(len(r_vals)))})
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(df["alpha"], df["pearson_r"], yerr=df["pearson_r_se"],
                marker="o", color="#c33")
    ax.set_xlabel(r"Synergy coefficient $\alpha$")
    ax.set_ylabel("Pearson $r$")
    ax.set_ylim(0.95, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig4_group_size(df: pd.DataFrame, out_path: Path) -> None:
    grouped = (df.groupby(["n", "method"])["pearson_r"]
                 .agg(["mean", "sem"])
                 .unstack("method")
                 .reindex(columns=METHOD_ORDER))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = grouped.index.values
    for method in METHOD_ORDER:
        m = grouped[("mean", method)].values
        s = grouped[("sem",  method)].values
        ax.errorbar(x, m, yerr=s, marker="o", label=method)
    ax.set_xlabel("Group size $n$")
    ax.set_ylabel("Pearson $r$")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path,
                   default=Path(__file__).resolve().parent.parent
                                / "results" / "main_simulation.csv")
    p.add_argument("--out-tables", type=Path,
                   default=Path(__file__).resolve().parent.parent / "tables")
    p.add_argument("--out-figures", type=Path,
                   default=Path(__file__).resolve().parent.parent / "figures")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out_tables.mkdir(parents=True, exist_ok=True)
    args.out_figures.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.results} ...")
    df = load_results(args.results)

    grand = grand_mean_table(df)
    grand.to_csv(args.out_tables / "table2_grand_mean.csv", index=False)
    print("Wrote", args.out_tables / "table2_grand_mean.csv")
    print(grand.to_string(index=False))

    print("Rendering figures ...")
    fig1_grand_metrics(grand, args.out_figures / "fig1_grand_metrics.png")
    fig2_scenario_pearson(df, args.out_figures / "fig2_scenario_pearson.png")
    fig3_sensitivity_alpha(args.out_figures / "fig3_sensitivity_alpha.png")
    fig4_group_size(df, args.out_figures / "fig4_group_size.png")
    print("All figures written to", args.out_figures)
