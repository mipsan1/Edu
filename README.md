# svfgs-simulation

Simulation code and data for the manuscript:

> **A Shapley Value-Based Fair Grading System for Group Project Assessment in
> Higher Education: A Simulation Study with Non-Linear Characteristic
> Function** (submitted to *Applied Sciences*, MDPI, 2026).

The repository implements the non-linear characteristic function (Eq. 1),
the Shapley value computation (Eq. 2), the per-student grade normalization
(Eq. 3), four baseline methods (equal distribution, peer assessment,
instructor assessment, CATME), and the Monte Carlo simulation design
described in Section 3 of the manuscript.

## Repository layout

```
svfgs-simulation/
├── src/
│   ├── characteristic.py   # Non-linear v(S), Eq. (1)
│   ├── svfgs.py            # Shapley value (Eq. 2) + grade normalization (Eq. 3)
│   ├── baselines.py        # Four baseline methods
│   ├── scenarios.py        # Five collaboration scenarios × three group sizes
│   ├── metrics.py          # Pearson r, FDR, SCA (Gini-based)
│   ├── simulate.py         # Main Monte Carlo driver (4,500 runs)
│   ├── analyze.py          # Reproduces Tables 1–2 and Figures 1–4
│   └── table1_baselines.py # Static Table 1 emitter
├── results/                # CSV output of the main simulation
├── figures/                # PNG output of the analysis step
├── tables/                 # CSV output of the analysis step
├── reproduce.sh            # One-shot reproduction script
├── requirements.txt
├── LICENSE                 # MIT License
└── README.md
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main simulation (4,500 Monte Carlo runs, ~5 s on a laptop)
python3 src/simulate.py --n-runs 300

# 3. Reproduce Tables 1–2 and Figures 1–4
python3 src/table1_baselines.py
python3 src/analyze.py
```

For a one-shot reproduction of all outputs, run:

```bash
bash reproduce.sh
```

## Reproducing the manuscript's numerical results

With the default settings (`--n-runs 300`, `--seed 42`, `--alpha 0.15`),
the script produces the following grand-mean Pearson $r$ values
(reproducible to within Monte Carlo noise):

| Method        | Pearson $r$ (mean ± SE) |
|---------------|--------------------------|
| Equal         | 0.000 ± 0.000            |
| Peer          | ~0.77 ± 0.01             |
| Instructor    | ~0.74 ± 0.01             |
| CATME         | ~0.79 ± 0.01             |
| Shapley       | 0.998 ± 0.000            |

The full results table is written to
`results/main_simulation.csv` (4,200 runs × 5 methods = 21,000 rows;
the "Two Free-riders" scenario is excluded for $n=3$ per the
manuscript design, yielding 14 cells × 300 runs × 5 methods = 21,000
records).

## Software versions

The code was developed and tested with:

- Python 3.11
- NumPy 1.24
- SciPy 1.10 (not required by default but available)
- pandas 2.0
- matplotlib 3.7

The exact versions used to produce the published results are pinned
in `requirements.txt`.

## License

The code is released under the MIT License. See `LICENSE`.

## Contact

Anonymous for review. Correspondence: anonymous.review@example.org.
