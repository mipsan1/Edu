"""
table1_baselines.py — Reproduce Table 1 of the manuscript.

Table 1 summarizes the five assessment methods and their noise
parameterization. This is a static documentation table (no simulation
output), emitted as a CSV for reproducibility.
"""

from pathlib import Path

import pandas as pd


def build_table1() -> pd.DataFrame:
    rows = [
        {
            "Method":         "Equal Distribution",
            "Grade Formula":  "g_i = G",
            "Noise sigma":    "---",
        },
        {
            "Method":         "Peer Assessment",
            "Grade Formula":  "g_i ∝ p_hat_i^{peer} + eps,  eps ~ N(0, 8**2)",
            "Noise sigma":    8,
        },
        {
            "Method":         "Instructor Assessment",
            "Grade Formula":  "g_i ∝ p_hat_i^{instr} + eps,  eps ~ N(0, 12**2)",
            "Noise sigma":    12,
        },
        {
            "Method":         "CATME",
            "Grade Formula":  "g_i ∝ p_hat_i^{CATME} + eps,  eps ~ N(0, 5**2)",
            "Noise sigma":    5,
        },
        {
            "Method":         "Shapley (Proposed)",
            "Grade Formula":  "g_i via Eqs. (1)-(3) of the manuscript",
            "Noise sigma":    "---",
        },
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_table1()
    df.to_csv(out_dir / "table1_baselines.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out_dir / 'table1_baselines.csv'}")
