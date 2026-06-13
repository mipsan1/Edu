#!/usr/bin/env bash
# reproduce.sh — One-shot reproduction of all results reported in the
# manuscript. Run from the repository root.
#
# Outputs:
#   results/main_simulation.csv
#   tables/table1_baselines.csv
#   tables/table2_grand_mean.csv
#   figures/fig1_grand_metrics.png
#   figures/fig2_scenario_pearson.png
#   figures/fig3_sensitivity_alpha.png
#   figures/fig4_group_size.png
#
# Wall time: ~10 s on a 2020-era laptop.
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/3] Running main simulation (4,500 Monte Carlo runs)..."
python3 src/simulate.py --n-runs 300

echo "[2/3] Emitting Table 1 (assessment methods)..."
python3 src/table1_baselines.py

echo "[3/3] Aggregating results and rendering figures..."
python3 src/analyze.py

echo "Done. See results/, tables/, and figures/ for outputs."
