"""Epsilon controller for the observation-intervention tradeoff.

Implements the theoretically grounded epsilon controller from Causal Bayesian
Optimization (Aglietti et al.) that decides whether to observe (trust the
surrogate model) or intervene (run an actual experiment).

The key idea: as the observed data covers more of the search space, we should
trust the surrogate more and intervene less. Epsilon measures the fraction of
the domain covered by observations, rescaled by experiment budget progress.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import ConvexHull, QhullError

logger = logging.getLogger(__name__)


def compute_epsilon(
    observed_data: np.ndarray,
    domain_bounds: list[tuple[float, float]],
    n_current: int,
    n_max: int,
) -> float:
    """Compute the epsilon value for the observation-intervention tradeoff.

    Uses the convex hull volume of observed data relative to the total domain
    volume, rescaled by budget progress (n_current / n_max), to determine
    how much to trust the surrogate model.

    The formula ``epsilon = coverage_ratio / (n_current / n_max)`` requires
    that coverage grows at least proportionally to budget consumption. Early
    in the budget (small n_current/n_max), the denominator is small so high
    coverage is needed to reach epsilon=1. As the budget is consumed, the
    bar for trusting the surrogate lowers proportionally.

    Args:
        observed_data: Array of shape (n_samples, n_dims) with observed points.
        domain_bounds: List of (lower, upper) bounds for each dimension.
        n_current: Number of experiments run so far.
        n_max: Maximum number of experiments in the budget.

    Returns:
        Epsilon in [0, 1]. Higher epsilon means more trust in the surrogate
        (more likely to observe rather than intervene).

    Note:
        The convex hull provides an **upper bound** on actual coverage: it
        includes regions enclosed by observed points but not yet sampled.
        For example, 4 corner points of a unit square give coverage=1.0
        despite an unobserved interior.

        ``ConvexHull`` complexity grows exponentially with dimensionality.
        For search spaces with more than ~10 continuous variables, consider
        reducing dimensionality or using a simpler coverage estimate.
    """
    if n_current <= 0 or n_max <= 0:
        return 0.0

    if observed_data.ndim != 2 or observed_data.shape[0] < 3:
        return 0.0

    n_dims = observed_data.shape[1]
    if n_dims < 2 or len(domain_bounds) != n_dims:
        return 0.0

    # Compute total domain volume
    total_volume = 1.0
    for lower, upper in domain_bounds:
        side = upper - lower
        if side <= 0:
            return 0.0
        total_volume *= side

    if total_volume <= 0:
        return 0.0

    # Need at least n_dims + 1 points for a convex hull in n_dims dimensions
    if observed_data.shape[0] < n_dims + 1:
        return 0.0

    # Compute convex hull volume of observed data
    try:
        hull = ConvexHull(observed_data)
        hull_volume = hull.volume
    except QhullError:
        # Degenerate cases: collinear points, duplicate points, etc.
        logger.debug("QhullError computing convex hull; returning epsilon=0.0")
        return 0.0

    if hull_volume <= 0:
        return 0.0

    coverage_ratio = hull_volume / total_volume
    rescale = n_current / n_max
    epsilon = coverage_ratio / rescale
    return float(min(epsilon, 1.0))


def should_observe(
    observed_data: np.ndarray,
    domain_bounds: list[tuple[float, float]],
    n_current: int,
    n_max: int,
    rng: np.random.Generator | None = None,
) -> bool:
    """Decide whether to observe (trust surrogate) or intervene (run experiment).

    With probability epsilon, we observe (return True — skip the experiment
    and trust the surrogate prediction). With probability 1-epsilon, we
    intervene (return False — run the actual experiment).

    Args:
        observed_data: Array of shape (n_samples, n_dims) with observed points.
        domain_bounds: List of (lower, upper) bounds for each dimension.
        n_current: Number of experiments run so far.
        n_max: Maximum number of experiments in the budget.
        rng: Random number generator for reproducibility.

    Returns:
        True if we should observe (skip experiment), False if we should intervene.
    """
    if rng is None:
        rng = np.random.default_rng()

    epsilon = compute_epsilon(observed_data, domain_bounds, n_current, n_max)
    return bool(rng.random() < epsilon)
