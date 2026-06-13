#!/usr/bin/env bash
# worked_example.sh — Reproduce the worked example in Section 7
# of the manuscript.
#
# Outputs:
#   results/worked_example.csv
#   figures/fig6_worked_example.png
#
# Wall time: ~5 s on a 2020-era laptop.
set -euo pipefail

cd "$(dirname "$0")"

echo "Running worked example: CS401 Software Engineering Studio (hypothetical)..."
python3 src/worked_example.py

echo "Done. See results/worked_example.csv and figures/fig6_worked_example.png."
