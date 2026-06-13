"""
scenarios.py — Five collaboration scenarios and group size grid.

Scenario construction (Sec. 3.3 of the manuscript):

    S1 Balanced        : all members s,e ~ Beta(5, 2)
    S2 One Free-rider  : one member e ~ Beta(1, 5), others s,e ~ Beta(5, 2)
    S3 Two Free-riders : two members e ~ Beta(1, 5), others s,e ~ Beta(5, 2)
    S4 Dominant Member : one member s,e ~ Beta(8, 1), others s,e ~ Beta(2, 4)
    S5 Highly Unequal  : all s,e ~ Beta(1, 1)

Performance scores are p_i = s_i * e_i in [0, 1].
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# (n, scenario_name) -> (n_free_riders, is_dominant)
SCENARIOS = {
    "Balanced":        {"free_riders": 0, "dominant": False},
    "One Free-rider":  {"free_riders": 1, "dominant": False},
    "Two Free-riders": {"free_riders": 2, "dominant": False},
    "Dominant Member": {"free_riders": 0, "dominant": True},
    "Highly Unequal":  {"free_riders": 0, "dominant": False},
}

GROUP_SIZES = [3, 4, 5]


def _draw(rng: np.random.Generator, a: float, b: float, size: int) -> np.ndarray:
    return rng.beta(a, b, size=size)


def sample_group(n: int,
                 scenario: str,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample skill, effort, and performance for one group of size n.

    Returns
    -------
    skill : (n,) array
    effort : (n,) array
    performance : (n,) array of s_i * e_i in [0, 1]
    """
    cfg = SCENARIOS[scenario]
    skill = _draw(rng, 5.0, 2.0, n)
    effort = _draw(rng, 5.0, 2.0, n)

    fr = cfg["free_riders"]
    if fr > 0:
        # Replace fr members' effort with Beta(1, 5) (low effort).
        idx = rng.choice(n, size=fr, replace=False)
        effort[idx] = _draw(rng, 1.0, 5.0, fr)

    if cfg["dominant"]:
        # Replace one member's (skill, effort) with Beta(8, 1) (very high).
        idx = int(rng.integers(0, n))
        skill[idx] = _draw(rng, 8.0, 1.0, 1)[0]
        effort[idx] = _draw(rng, 8.0, 1.0, 1)[0]

    if scenario == "Highly Unequal":
        # Override with uniform draws.
        skill = _draw(rng, 1.0, 1.0, n)
        effort = _draw(rng, 1.0, 1.0, n)

    performance = skill * effort
    return skill, effort, performance
