"""Ax/BoTorch Bayesian optimizer wrapper.

Wraps Ax ServiceAPI for a suggest/update loop with optional causal guidance
via focus variables (POMIS/screening integration) and POMIS priors.
"""

from __future__ import annotations

from typing import Any

try:
    from ax.service.ax_client import AxClient as _AxClient  # noqa: F401

    _AX_AVAILABLE = True
except ImportError:
    _AX_AVAILABLE = False

from causal_optimizer.types import SearchSpace


class AxBayesianOptimizer:
    """Wraps Ax ServiceAPI for suggest/update loop.

    Parameters
    ----------
    search_space:
        The optimization search space.
    objective_name:
        Name of the metric to optimize.
    minimize:
        Whether to minimize (True) or maximize (False) the objective.
    focus_variables:
        If provided, only these variables are optimized by Ax; others are fixed at midpoint.
    pomis_prior:
        If provided, candidates that only touch POMIS variables get a soft bonus.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective_name: str,
        minimize: bool = True,
        focus_variables: list[str] | None = None,
        pomis_prior: list[frozenset[str]] | None = None,
        seed: int | None = None,
    ) -> None:
        raise NotImplementedError

    def suggest(self) -> dict[str, Any]:
        """Generate the next candidate parameter dict."""
        raise NotImplementedError

    def update(self, params: dict[str, Any], value: float) -> None:
        """Feed an observed (params, value) pair back to the optimizer."""
        raise NotImplementedError

    def best(self) -> dict[str, Any] | None:
        """Return the best observed parameter dict, or None if no observations yet."""
        raise NotImplementedError
