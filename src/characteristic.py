"""
characteristic.py — Non-linear characteristic function for SVFGS.

Implements Equation (1) of the manuscript:

    v(S) = min[ (0.6 * GM(S) + 0.4 * MaxEff(S) + 0.1 * SD(S))
                 * (1 + alpha * ln|S|) * 100 , 100 ]

where
    GM(S)     = (prod_{i in S} p_i)^{1/|S|}
    MaxEff(S) = max_{i in S} p_i
    SD(S)     = sqrt( (1/|S|) * sum_{i in S} (p_i - mean(p_S))^2 )
    p_i       = s_i * e_i  (skill * effort)
    alpha     = synergy coefficient (default 0.15)

All scores are clipped to [0, 100] at the coalition level.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def gm(coalition_p: np.ndarray) -> float:
    """Geometric mean of performance scores in a coalition."""
    if coalition_p.size == 0:
        return 0.0
    # Guard against zero(s) — geometric mean with any zero is exactly zero.
    p = np.clip(coalition_p, 1e-12, 1.0)
    return float(np.exp(np.mean(np.log(p))))


def max_eff(coalition_p: np.ndarray) -> float:
    """Maximum performance score in a coalition."""
    if coalition_p.size == 0:
        return 0.0
    return float(np.max(coalition_p))


def std_dev(coalition_p: np.ndarray) -> float:
    """Standard deviation of performance scores in a coalition."""
    if coalition_p.size <= 1:
        return 0.0
    return float(np.std(coalition_p, ddof=0))


def v(coalition: Iterable[int],
      p: np.ndarray,
      alpha: float = 0.15) -> float:
    """
    Characteristic function value v(S) for a coalition.

    Parameters
    ----------
    coalition : iterable of int
        Indices of group members in the coalition.
    p : (n,) array
        Performance scores p_i in [0, 1] for all n group members.
    alpha : float
        Synergy coefficient (default 0.15, as in the manuscript).

    Returns
    -------
    float
        Coalition value in [0, 100].
    """
    idx = np.asarray(list(coalition), dtype=int)
    if idx.size == 0:
        return 0.0

    p_s = p[idx]
    g = gm(p_s)
    m = max_eff(p_s)
    sd = std_dev(p_s)
    size = idx.size

    base = 0.6 * g + 0.4 * m + 0.1 * sd
    log_factor = 1.0 + alpha * math.log(size)
    raw = base * log_factor * 100.0
    return float(min(raw, 100.0))


def v_all_subsets(p: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    Compute v(S) for every subset S of N = {0, ..., n-1}.

    Returns an array of length 2**n indexed by integer bitmask.
    The empty set (mask 0) is assigned value 0.0.
    """
    n = p.size
    out = np.zeros(2 ** n, dtype=float)
    for mask in range(1, 2 ** n):
        coalition = [i for i in range(n) if (mask >> i) & 1]
        out[mask] = v(coalition, p, alpha=alpha)
    return out
