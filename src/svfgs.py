"""
svfgs.py — Shapley Value-based Fair Grading System.

Exact Shapley value computation via coalition enumeration (Eq. 2 in the
manuscript). For a group of size n, this requires 2**n coalition evaluations.

Also implements the per-student grade normalization (Eq. 3 in the manuscript):

    g_i = (phi_i / sum_j phi_j) * G * n

where G is the group raw score and n is the group size. Final individual
grades are clipped to [0, 100].
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from characteristic import v_all_subsets


def shapley_values(p: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    Compute the Shapley value phi_i(v) for every player i in N.

    Parameters
    ----------
    p : (n,) array
        Performance scores p_i in [0, 1].
    alpha : float
        Synergy coefficient passed to the characteristic function.

    Returns
    -------
    (n,) array
        Shapley values phi_i(v) for each player.
    """
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=float)

    values = v_all_subsets(p, alpha=alpha)
    fact = math.factorial

    phi = np.zeros(n, dtype=float)
    for i in range(n):
        bit_i = 1 << i
        acc = 0.0
        # Sum over all subsets S that do NOT contain i.
        for mask in range(2 ** n):
            if mask & bit_i:
                continue
            s_size = bin(mask).count("1")
            weight = fact(s_size) * fact(n - s_size - 1) / fact(n)
            marginal = values[mask | bit_i] - values[mask]
            acc += weight * marginal
        phi[i] = acc

    return phi


def individual_grades(phi: np.ndarray, group_score: float) -> np.ndarray:
    """
    Convert Shapley values into per-student grades g_i (Eq. 3).

    Parameters
    ----------
    phi : (n,) array
        Shapley values (sum phi_i equals v(N), the group raw value).
    group_score : float
        Group raw score G in [0, 100] assigned by the instructor.

    Returns
    -------
    (n,) array
        Per-student grades g_i, clipped to [0, 100].
    """
    n = phi.size
    total = phi.sum()
    if total <= 0.0:
        # Degenerate: every member contributed nothing. Equal split as fallback.
        return np.full(n, group_score, dtype=float)
    grades = (phi / total) * group_score * n
    return np.clip(grades, 0.0, 100.0)


def assign_grades(p: np.ndarray,
                  group_score: float,
                  alpha: float = 0.15) -> np.ndarray:
    """Convenience: Shapley -> per-student grades in one call."""
    phi = shapley_values(p, alpha=alpha)
    return individual_grades(phi, group_score)
