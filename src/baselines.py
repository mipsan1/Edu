"""
baselines.py — Four baseline grading methods.

Implements:
    1. Equal distribution
    2. Peer assessment (with additive Gaussian noise)
    3. Instructor assessment (with additive Gaussian noise)
    4. CATME-style peer rating (with additive Gaussian noise)

All noisy baselines generate observed ratings \hat{p}_i = p_i + eps with
eps ~ N(0, sigma**2), clipped to [0, 100], then map to grades proportional
to \hat{p}_i while preserving the group average G.
"""

from __future__ import annotations

import numpy as np

from svfgs import individual_grades


def _normalize(weights: np.ndarray, group_score: float) -> np.ndarray:
    """Map non-negative weights to per-student grades preserving group mean G."""
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0.0:
        return np.full(weights.size, group_score, dtype=float)
    # Use individual_grades math (same normalization as Shapley).
    return individual_grades(weights, group_score)


def equal_distribution(p: np.ndarray, group_score: float) -> np.ndarray:
    """All members receive the same grade G."""
    return np.full(p.size, group_score, dtype=float)


def peer_assessment(p: np.ndarray,
                    group_score: float,
                    sigma: float = 8.0,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Peer ratings: p_i + N(0, sigma), clipped to [0, 100]."""
    if rng is None:
        rng = np.random.default_rng()
    observed = np.clip(p * 100.0 + rng.normal(0.0, sigma, size=p.size),
                       0.0, 100.0)
    return _normalize(observed, group_score)


def instructor_assessment(p: np.ndarray,
                           group_score: float,
                           sigma: float = 12.0,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """Instructor ratings: p_i + N(0, sigma), clipped to [0, 100]."""
    if rng is None:
        rng = np.random.default_rng()
    observed = np.clip(p * 100.0 + rng.normal(0.0, sigma, size=p.size),
                       0.0, 100.0)
    return _normalize(observed, group_score)


def catme(p: np.ndarray,
          group_score: float,
          sigma: float = 5.0,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """CATME-style peer rating with calibrated noise (sigma=5)."""
    if rng is None:
        rng = np.random.default_rng()
    observed = np.clip(p * 100.0 + rng.normal(0.0, sigma, size=p.size),
                       0.0, 100.0)
    return _normalize(observed, group_score)
