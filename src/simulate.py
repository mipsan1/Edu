"""
simulate.py — Main Monte Carlo simulation driver.

Runs the full 5 (scenario) x 3 (group size) x N_runs grid and produces a
results table saved as results/main_simulation.csv. Group score G is
fixed at 80 for all runs (representative of a strong but not perfect
group project outcome).

The default N_runs=300 reproduces the main analysis of the manuscript
(4,500 runs total). The diagnostic simulation uses N_runs=50.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from svfgs import assign_grades
from baselines import (
    equal_distribution,
    peer_assessment,
    instructor_assessment,
    catme,
)
from scenarios import sample_group, SCENARIOS, GROUP_SIZES
from metrics import pearson_r, sca, fdr


METHODS = {
    "Equal":          ("equal",  None),
    "Peer":           ("peer",   8.0),
    "Instructor":     ("instr",  12.0),
    "CATME":          ("catme",  5.0),
    "Shapley":        ("shap",   None),
}


def run_one(n: int,
            scenario: str,
            group_score: float,
            alpha: float,
            rng: np.random.Generator) -> dict:
    """Run a single (n, scenario, method) cell across all 5 methods."""
    _, effort, p = sample_group(n, scenario, rng)

    # Generate per-method grades.
    grades_by_method: dict[str, np.ndarray] = {}
    for name, (kind, sigma) in METHODS.items():
        if kind == "equal":
            grades_by_method[name] = equal_distribution(p, group_score)
        elif kind == "peer":
            grades_by_method[name] = peer_assessment(p, group_score, sigma, rng)
        elif kind == "instr":
            grades_by_method[name] = instructor_assessment(p, group_score, sigma, rng)
        elif kind == "catme":
            grades_by_method[name] = catme(p, group_score, sigma, rng)
        elif kind == "shap":
            grades_by_method[name] = assign_grades(p, group_score, alpha=alpha)

    row = {
        "n": n,
        "scenario": scenario,
        "effort": effort,
        "p": p,
        "grades": grades_by_method,
    }
    return row


def main_simulation(n_runs: int = 300,
                    group_score: float = 80.0,
                    alpha: float = 0.15,
                    seed: int = 42,
                    out_path: Path | None = None) -> list[dict]:
    """
    Run the full main simulation and return a list of run records.

    A run record contains:
        {n, scenario, run_id, method, pearson_r, fdr, sca, p, grades}
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for n in GROUP_SIZES:
        for scenario in SCENARIOS.keys():
            if scenario == "Two Free-riders" and n < 4:
                continue
            t0 = time.time()
            for run_id in range(n_runs):
                cell = run_one(n, scenario, group_score, alpha, rng)
                effort = cell["effort"]
                p = cell["p"]
                for method, g in cell["grades"].items():
                    records.append({
                        "n": n,
                        "scenario": scenario,
                        "run_id": run_id,
                        "method": method,
                        "p": p,
                        "effort": effort,
                        "grades": g,
                        "pearson_r": pearson_r(p, g),
                        "fdr": fdr(effort, g),
                        "sca": sca(p, g),
                    })
            elapsed = time.time() - t0
            print(f"  n={n}  {scenario:<16}  {n_runs} runs  ({elapsed:.1f}s)")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "n", "scenario", "run_id", "method",
                "pearson_r", "fdr", "sca",
            ])
            for r in records:
                fdr_val = r["fdr"]
                writer.writerow([
                    r["n"], r["scenario"], r["run_id"], r["method"],
                    f"{r['pearson_r']:.6f}",
                    "" if math.isnan(fdr_val) else f"{fdr_val:.6f}",
                    f"{r['sca']:.6f}",
                ])

    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=300,
                   help="Monte Carlo runs per (n, scenario) cell.")
    p.add_argument("--alpha", type=float, default=0.15,
                   help="Synergy coefficient (default 0.15).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default 42).")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent
                                / "results" / "main_simulation.csv",
                   help="Output CSV path.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Running main simulation: {args.n_runs} runs per cell, "
          f"alpha={args.alpha}, seed={args.seed}")
    main_simulation(n_runs=args.n_runs,
                    alpha=args.alpha,
                    seed=args.seed,
                    out_path=args.out)
    print(f"Done. Results written to {args.out}")
