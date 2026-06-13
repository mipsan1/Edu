"""
worked_example.py — Tutorial: SVFGS in a hypothetical 25-student course.

Setup (hypothetical)
--------------------
"CS401 Software Engineering Studio" (Spring 2027, 25 students, 5 teams of 5).
Four-week group project: design + implement a database-backed web app.
Peer rating: 1-5 scale, four dimensions (code, design, collaboration, on-time),
averaged over weeks 2-4. Instructor assigns a single group score G per team
on a 0-100 scale at project close.

Goal
----
Walk through the SVFGS computation step-by-step on this concrete dataset.
Demonstrate:
  1. How p_i is computed from peer ratings.
  2. How v(S) is computed for each coalition of one team.
  3. How the Shapley value decomposes the group score.
  4. How individual grades compare to the equal-distribution baseline.
  5. The counter-experiment: noisy peer ratings collapse alignment.

Output
------
results/worked_example.csv  (student-level input data + grades)
figures/fig6_worked_example.png  (Shapley value decomposition)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from svfgs import assign_grades
from baselines import equal_distribution, peer_assessment
from characteristic import v
from metrics import pearson_r


# Hypothetical 25-student course
TEAMS = {
    "T1 (Mixed)":   [0.85, 0.70, 0.60, 0.50, 0.40],   # one star, one struggling
    "T2 (Strong)":  [0.90, 0.85, 0.80, 0.78, 0.75],   # uniformly strong
    "T3 (One free-rider)": [0.80, 0.70, 0.65, 0.55, 0.15],  # last is free-rider
    "T4 (Two free-riders)": [0.80, 0.75, 0.25, 0.20, 0.20],  # two free-riders
    "T5 (Dominant)": [0.95, 0.40, 0.35, 0.30, 0.25],   # one dominant
}
GROUP_SCORES = {
    "T1 (Mixed)":   78,
    "T2 (Strong)":  85,
    "T3 (One free-rider)": 72,
    "T4 (Two free-riders)": 65,
    "T5 (Dominant)": 80,
}
ALPHA = 0.15


def shapley_values(p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Shapley values via explicit coalition enumeration (n <= 8)."""
    from math import factorial
    n = p.size
    values = np.zeros(2 ** n)
    for mask in range(1, 2 ** n):
        coalition = [j for j in range(n) if (mask >> j) & 1]
        values[mask] = v(coalition, p, alpha=alpha)
    phi = np.zeros(n)
    for i in range(n):
        bit_i = 1 << i
        for mask in range(2 ** n):
            if mask & bit_i:
                continue
            s_size = bin(mask).count("1")
            weight = factorial(s_size) * factorial(n - s_size - 1) / factorial(n)
            phi[i] += weight * (values[mask | bit_i] - values[mask])
    return phi


def run_team(team_name: str, p: np.ndarray, G: float) -> dict:
    """Run SVFGS + baseline on one team, return summary dict."""
    shapley = shapley_values(p)
    g_shapley = assign_grades(p, G, alpha=ALPHA)
    g_equal = equal_distribution(p, G)
    g_noisy_peer = peer_assessment(p, G, sigma=8.0,
                                    rng=np.random.default_rng(0))
    return {
        "team": team_name,
        "p": p,
        "G": G,
        "shapley_phi": shapley,
        "g_shapley": g_shapley,
        "g_equal": g_equal,
        "g_noisy_peer": g_noisy_peer,
    }


def write_csv(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "student_idx", "p_i", "shapley_phi",
                    "g_shapley", "g_equal", "g_noisy_peer"])
        for r in results:
            for i in range(r["p"].size):
                w.writerow([r["team"], i + 1,
                            f"{r['p'][i]:.3f}",
                            f"{r['shapley_phi'][i]:.4f}",
                            f"{r['g_shapley'][i]:.2f}",
                            f"{r['g_equal'][i]:.2f}",
                            f"{r['g_noisy_peer'][i]:.2f}"])


def render_figure(results: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.0), sharey=True)
    for ax, r in zip(axes, results):
        n = r["p"].size
        x = np.arange(n)
        width = 0.27
        ax.bar(x - width, r["g_equal"],    width, label="Equal",   color="#888")
        ax.bar(x,         r["g_shapley"],  width, label="SVFGS",   color="#c33")
        ax.bar(x + width, r["g_noisy_peer"], width, label="Peer (noisy)", color="#5aa")
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i+1}" for i in range(n)], fontsize=8)
        ax.set_title(r["team"], fontsize=9)
        ax.set_ylim(0, 100)
        if r is results[0]:
            ax.set_ylabel("Assigned grade")
    fig.suptitle("Worked example: CS401 Software Engineering Studio (hypothetical, 25 students, 5 teams)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    results = []
    print("Worked example: CS401 Software Engineering Studio (hypothetical)")
    print("=" * 72)
    for name, p_list in TEAMS.items():
        p = np.array(p_list, dtype=float)
        G = GROUP_SCORES[name]
        r = run_team(name, p, G)
        results.append(r)
        # Quick per-team summary
        r_shap = pearson_r(p, r["g_shapley"])
        r_equal = pearson_r(p, r["g_equal"])
        r_peer = pearson_r(p, r["g_noisy_peer"])
        print(f"\n{name}  (G = {G})")
        print(f"  Pearson r:  SVFGS = {r_shap:.3f}   "
              f"Equal = {r_equal:.3f}   Peer(noisy) = {r_peer:.3f}")
        for i in range(p.size):
            print(f"  S{i+1}:  p = {p[i]:.2f}  "
                  f"g_SVFGS = {r['g_shapley'][i]:5.1f}  "
                  f"g_Equal = {r['g_equal'][i]:5.1f}  "
                  f"g_Peer(noisy) = {r['g_noisy_peer'][i]:5.1f}")

    write_csv(results, repo / "results" / "worked_example.csv")
    render_figure(results, repo / "figures" / "fig6_worked_example.png")
    print(f"\nWrote {repo / 'results' / 'worked_example.csv'}")
    print(f"Wrote {repo / 'figures' / 'fig6_worked_example.png'}")


if __name__ == "__main__":
    main()
