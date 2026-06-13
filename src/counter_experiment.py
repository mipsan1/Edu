"""
counter_experiment.py — Robustness of SVFGS to noisy v(S) inputs.

Addresses reviewer concern: "Shapley cannot fail to recover a construct
built from its own inputs."

Setup
-----
In the main analysis, v(S) is computed from the clean performance scores
p_i that *also* define the ground-truth alignment metric. This is the
self-fulfilling alignment concern.

In this counter-experiment, we reconstruct v(S) from *noisy* peer ratings
\hat{p}_i = p_i + eps, with eps ~ N(0, sigma^2), clipped to [0, 1]. The
ground-truth metric (Pearson r against clean p) is unchanged. If Shapley
alignment collapses when v(S) is reconstructed from noisy inputs, the
high r in the main analysis is partly an artefact. If Shapley alignment
remains high, the main result is robust.

Comparison
----------
We compare three conditions:
  (a) Shapley with clean v(S) inputs (main-analysis baseline)
  (b) Shapley with noisy v(S) inputs, sigma in {4, 8, 12, 16, 20}
  (c) Four baseline methods (Equal, Peer, Instructor, CATME)

Metrics
-------
For each condition and each (n, scenario) cell, we report the mean and
standard error of Pearson r, FDR, and SCA over N=100 runs.

Output
------
results/counter_experiment.csv
figures/fig5_counter_experiment.png
"""

from __future__ import annotations

import argparse
import csv
import math
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
from characteristic import v


# ----------------------------------------------------------------------------
# Noisy Shapley: v(S) reconstructed from noisy peer ratings
# ----------------------------------------------------------------------------

def shapley_values_noisy(p: np.ndarray,
                         noise_sigma: float,
                         alpha: float = 0.15,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Compute Shapley values using v(S) reconstructed from noisy inputs.

    Step 1: Sample noisy peer ratings \hat{p}_i = p_i + N(0, sigma^2).
    Step 2: Compute v(S) from \hat{p}_i (not from clean p).
    Step 3: Shapley value phi_i is computed against the noisy v(S).
    Step 4: Per-student grade g_i is computed against the noisy phi.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=float)

    # Noisy observations clipped to [0, 1]
    p_obs = np.clip(p + rng.normal(0.0, noise_sigma, size=p.size), 0.0, 1.0)

    # v(S) for all subsets using the NOISY inputs
    fact = math.factorial
    values = np.zeros(2 ** n, dtype=float)
    for mask in range(1, 2 ** n):
        coalition = [j for j in range(n) if (mask >> j) & 1]
        values[mask] = v(coalition, p_obs, alpha=alpha)

    phi = np.zeros(n, dtype=float)
    for i in range(n):
        bit_i = 1 << i
        acc = 0.0
        for mask in range(2 ** n):
            if mask & bit_i:
                continue
            s_size = bin(mask).count("1")
            weight = fact(s_size) * fact(n - s_size - 1) / fact(n)
            marginal = values[mask | bit_i] - values[mask]
            acc += weight * marginal
        phi[i] = acc

    return phi


def assign_grades_noisy(p: np.ndarray,
                        group_score: float,
                        noise_sigma: float,
                        alpha: float = 0.15,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    phi = shapley_values_noisy(p, noise_sigma, alpha=alpha, rng=rng)
    total = phi.sum()
    if total <= 0.0:
        return np.full(p.size, group_score, dtype=float)
    grades = (phi / total) * group_score * p.size
    return np.clip(grades, 0.0, 100.0)


# ----------------------------------------------------------------------------
# Main counter-experiment driver
# ----------------------------------------------------------------------------

NOISE_LEVELS = [4.0, 8.0, 12.0, 16.0, 20.0]


def run_cell(n: int,
             scenario: str,
             group_score: float,
             alpha: float,
             n_runs: int,
             rng: np.random.Generator) -> list[dict]:
    """
    Run all (method, sigma) combinations for a single (n, scenario) cell.
    """
    records: list[dict] = []

    for run_id in range(n_runs):
        _, effort, p = sample_group(n, scenario, rng)
        truth = p  # Ground truth is the CLEAN p_i.

        # 1. Equal distribution (sigma-independent)
        g = equal_distribution(p, group_score)
        records.append({
            "n": n, "scenario": scenario, "run_id": run_id,
            "method": "Equal", "noise_sigma": None,
            "pearson_r": pearson_r(truth, g),
            "fdr": fdr(effort, g),
            "sca": sca(truth, g),
        })

        # 2-4. Rating-based baselines (each uses its own sigma)
        for name, fn, sig in [
            ("Peer",       peer_assessment,      8.0),
            ("Instructor", instructor_assessment, 12.0),
            ("CATME",      catme,                5.0),
        ]:
            g = fn(p, group_score, sig, rng)
            records.append({
                "n": n, "scenario": scenario, "run_id": run_id,
                "method": name, "noise_sigma": sig,
                "pearson_r": pearson_r(truth, g),
                "fdr": fdr(effort, g),
                "sca": sca(truth, g),
            })

        # 5. Clean Shapley (sigma-independent)
        g = assign_grades(p, group_score, alpha=alpha)
        records.append({
            "n": n, "scenario": scenario, "run_id": run_id,
            "method": "Shapley (clean)", "noise_sigma": 0.0,
            "pearson_r": pearson_r(truth, g),
            "fdr": fdr(effort, g),
            "sca": sca(truth, g),
        })

        # 6. Noisy Shapley (sigma-dependent)
        for sigma in NOISE_LEVELS:
            g = assign_grades_noisy(p, group_score, sigma,
                                    alpha=alpha, rng=rng)
            records.append({
                "n": n, "scenario": scenario, "run_id": run_id,
                "method": f"Shapley (noisy)", "noise_sigma": sigma,
                "pearson_r": pearson_r(truth, g),
                "fdr": fdr(effort, g),
                "sca": sca(truth, g),
            })

    return records


def main(n_runs: int = 100,
         group_score: float = 80.0,
         alpha: float = 0.15,
         seed: int = 42,
         out_csv: Path | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for n in GROUP_SIZES:
        for scenario in SCENARIOS.keys():
            if scenario == "Two Free-riders" and n < 4:
                continue
            cell = run_cell(n, scenario, group_score, alpha, n_runs, rng)
            records.extend(cell)
            print(f"  n={n}  {scenario:<16}  {n_runs} runs done")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "n", "scenario", "run_id", "method", "noise_sigma",
                "pearson_r", "fdr", "sca",
            ])
            for r in records:
                fdr_val = r["fdr"]
                writer.writerow([
                    r["n"], r["scenario"], r["run_id"], r["method"],
                    "" if r["noise_sigma"] is None else f"{r['noise_sigma']:.1f}",
                    f"{r['pearson_r']:.6f}",
                    "" if math.isnan(fdr_val) else f"{fdr_val:.6f}",
                    f"{r['sca']:.6f}",
                ])

    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent
                                / "results" / "counter_experiment.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Running counter-experiment: {args.n_runs} runs per cell, "
          f"alpha={args.alpha}, seed={args.seed}")
    main(n_runs=args.n_runs, alpha=args.alpha, seed=args.seed,
         out_csv=args.out)
    print(f"Done. Results written to {args.out}")
