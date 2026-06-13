#!/usr/bin/env bash
# counter_experiment.sh — Reproduce the noisy v(S) counter-experiment
# reported in Section 4.5 of the manuscript.
#
# Outputs:
#   results/counter_experiment.csv (~840 KB, 14,000 rows)
#   figures/fig5_counter_experiment.png
#
# Wall time: ~30 s on a 2020-era laptop.
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/2] Running counter-experiment (14 cells x 100 runs x 10 conditions)..."
python3 src/counter_experiment.py --n-runs 100

echo "[2/2] Rendering Figure 5..."
python3 src/counter_figure.py

echo "Done. See results/counter_experiment.csv and figures/fig5_counter_experiment.png."
