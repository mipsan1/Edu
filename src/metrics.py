"""
metrics.py — Evaluation metrics for SVFGS comparison.

    pearson_r(p, g)        : Pearson correlation (alignment)
    fdr(effort, g)         : Free-rider detection rate (label: e_i < 0.35)
    sca(p, g)              : Score-contribution alignment via Gini
    gini(x)                : Gini coefficient of a non-negative vector
"""

from __future__ import annotations

import numpy as np


def pearson_r(p: np.ndarray, g: np.ndarray) -> float:
    """Pearson correlation between true contributions p and grades g."""
    if p.std() == 0.0 or g.std() == 0.0:
        # No variance -> alignment undefined; return 0 (worst case).
        return 0.0
    return float(np.corrcoef(p, g)[0, 1])


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def sca(p: np.ndarray, g: np.ndarray) -> float:
    """Score-contribution alignment: 1 - |Gini(p) - Gini(g)|."""
    return 1.0 - abs(gini(p) - gini(g))


def fdr(effort: np.ndarray, g: np.ndarray, tau_quantile: float = 0.35,
        free_rider_threshold: float = 0.35) -> float:
    """
    Free-rider detection rate.

    Free-riders are members with e_i < free_rider_threshold. A free-rider
    is correctly detected if their grade falls below the tau_quantile
    percentile of the group's assigned grades.
    """
    fr_mask = effort < free_rider_threshold
    n_fr = int(fr_mask.sum())
    if n_fr == 0:
        return float("nan")  # Not defined for this group.
    if g.size == 0:
        return 0.0
    tau = np.quantile(g, tau_quantile)
    detected = int(np.sum((g < tau) & fr_mask))
    return detected / n_fr
