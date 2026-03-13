"""Ax/BoTorch Bayesian optimizer wrapper.

Wraps Ax ServiceAPI for a suggest/update loop with optional causal guidance
via focus variables (POMIS/screening integration) and POMIS priors.

Graceful degradation: if ``ax`` is not installed, instantiating
``AxBayesianOptimizer`` raises an ``ImportError`` with a clear install hint.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from causal_optimizer.types import SearchSpace, Variable, VariableType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability check — runs at module import time so the error is surfaced
# as early as possible (constructor call) rather than at first suggest().
# ---------------------------------------------------------------------------

try:
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties

    _AX_AVAILABLE = True
except ImportError:
    _AX_AVAILABLE = False


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
        If provided, only these variables are optimized by Ax; others are
        fixed at their midpoint value.
    pomis_prior:
        When set, 80% of suggestions fix non-POMIS variables at their
        midpoints so the optimizer only varies the POMIS intervention set;
        the remaining 20% explore the full space.  This is a post-hoc
        hard constraint applied after ``get_next_trial()``, not a soft
        bonus in the acquisition function.
    seed:
        Random seed forwarded to ``AxClient`` for reproducibility.

    Raises
    ------
    ImportError
        If ``ax-platform`` is not installed.  Install with
        ``uv sync --extra bayesian``.
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
        if not _AX_AVAILABLE:
            raise ImportError(
                "ax-platform is required for AxBayesianOptimizer. "
                "Install it with: uv sync --extra bayesian"
            )

        self._search_space = search_space
        self._objective_name = objective_name
        self._minimize = minimize
        self._pomis_prior = pomis_prior or []

        # Determine which variables are "active" (varied by Ax) vs "fixed" (held at midpoint)
        focus_set = set(focus_variables) if focus_variables else None

        self._active_vars = [
            v for v in search_space.variables if focus_set is None or v.name in focus_set
        ]
        self._fixed_vars = [
            v for v in search_space.variables if focus_set is not None and v.name not in focus_set
        ]

        # Pre-compute midpoints for fixed variables and POMIS non-focus variables.
        # Also precompute the POMIS union once so suggest() doesn't recompute it each call.
        self._midpoints: dict[str, Any] = {}
        for var in self._fixed_vars:
            self._midpoints[var.name] = self._midpoint_for(var)

        # Precompute the set of variables that the POMIS prior should clamp.
        # These are variables NOT in the POMIS union.
        # When focus_variables is set, also exclude explicitly-focused variables:
        # clamping a variable that Ax is actively optimizing would waste model
        # capacity — Ax fits a GP over its range but the output is discarded 80%
        # of the time.  When focus_variables is None (all vars active), the POMIS
        # prior clamps non-POMIS variables normally since no subset was requested.
        self._pomis_non_focus: set[str] = set()
        if self._pomis_prior:
            pomis_union: set[str] = set()
            for s in self._pomis_prior:
                pomis_union |= s
            for var in search_space.variables:
                if var.name not in pomis_union:
                    # Only skip clamping if focus_set explicitly includes this var
                    if focus_set is not None and var.name in focus_set:
                        continue
                    self._pomis_non_focus.add(var.name)
                    if var.name not in self._midpoints:
                        self._midpoints[var.name] = self._midpoint_for(var)

        # Build the Ax parameter defs for active variables only
        ax_params = self._build_ax_params()
        if not ax_params:
            # Nothing to optimise — reset to all variables
            self._active_vars = list(search_space.variables)
            self._fixed_vars = []
            ax_params = self._build_ax_params()

        # Initialise AxClient
        # enforce_sequential_optimization=False: allows get_next_trial() even when
        # there are attach_trial()-sourced observations that weren't generated by Ax
        # (historical data pre-loading).  Without this flag, Ax raises DataRequiredError
        # when it tries to transition from Sobol to BoTorch but finds that the "generated"
        # trial count is below the MinTrials threshold (because the historical trials
        # were attached, not generated).
        client_kwargs: dict[str, Any] = {
            "verbose_logging": False,
            "enforce_sequential_optimization": False,
        }
        if seed is not None:
            client_kwargs["random_seed"] = seed

        self._client: AxClient = AxClient(**client_kwargs)
        self._client.create_experiment(
            name="causal_optimizer",
            parameters=ax_params,
            objectives={objective_name: ObjectiveProperties(minimize=minimize)},
        )

        # Precompute active-variable name set for fast filtering in update()
        self._active_names: frozenset[str] = frozenset(v.name for v in self._active_vars)

        # Pending trial state: when suggest() is called, a trial is left RUNNING
        # until update() completes it or the next suggest() abandons it.
        self._pending_trial_idx: int | None = None

        # Track observations for best() lookup
        self._observations: list[tuple[dict[str, Any], float]] = []

        # Seeded RNG for reproducible POMIS prior sampling.
        # Uses seed+1 to avoid collision with the AxClient random_seed so the
        # two independent sources of randomness don't share state.
        self._rng = np.random.default_rng(seed + 1 if seed is not None else None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self) -> dict[str, Any]:
        """Generate the next candidate parameter dict.

        If a pending trial exists from a previous ``suggest()`` call that was
        never updated, it is abandoned before generating a new trial.

        If ``pomis_prior`` is set, with 80% probability the non-POMIS variables
        are clamped to their midpoints so the suggestion targets POMIS variables
        only.
        """
        # Abandon any un-updated pending trial from a previous suggest() call
        if self._pending_trial_idx is not None:
            self._client.abandon_trial(
                trial_index=self._pending_trial_idx,
                reason="superseded by new suggest() call",
            )
            self._pending_trial_idx = None

        active_params, trial_idx = self._client.get_next_trial()
        self._pending_trial_idx = trial_idx

        result: dict[str, Any] = dict(active_params)

        # Fill fixed variables at midpoint
        for var in self._fixed_vars:
            result[var.name] = self._midpoints[var.name]

        # Apply POMIS prior with 80% probability.
        # _pomis_non_focus was precomputed in __init__ from the POMIS union.
        if self._pomis_non_focus and self._rng.random() < 0.8:
            for var_name in self._pomis_non_focus:
                if var_name in self._midpoints:
                    result[var_name] = self._midpoints[var_name]

        # Round boolean variables: Ax returns floats from the [0, 1] range;
        # convert back to True/False so callers always receive actual booleans.
        for var in self._search_space.variables:
            if var.variable_type == VariableType.BOOLEAN and var.name in result:
                result[var.name] = bool(result[var.name] > 0.5)

        return result

    def update(self, params: dict[str, Any], value: float) -> None:
        """Feed an observed (params, value) pair back to the optimizer.

        If the last ``suggest()`` trial is still pending, it is completed with
        the given value.  Otherwise, the observation is attached as a new trial
        and immediately completed (historical data loading).

        Only the active-variable subset of ``params`` is forwarded to Ax.

        .. warning::
            When a pending trial exists, ``params`` may differ from the
            parameters that ``suggest()`` proposed (e.g. if the caller modified
            them before running the experiment).  Ax always records the
            *generated* parameters for the pending trial; the caller-supplied
            ``params`` are used only for the ``_observations`` list (which
            powers ``best()``).  If you need Ax's model to reflect the actual
            run parameters, call ``update()`` without a preceding ``suggest()``
            so the observation is attached as a fresh trial with the correct
            parameter values.
        """
        if self._pending_trial_idx is not None:
            # Complete the pending trial that was generated by suggest().
            # Ax uses the parameters it generated for this trial, not `params`.
            self._client.complete_trial(
                trial_index=self._pending_trial_idx,
                raw_data={self._objective_name: value},
            )
            self._pending_trial_idx = None
        else:
            # Pre-load historical observation: attach a new trial and complete immediately.
            # Only forward active-variable values to Ax.
            active_params = {k: v for k, v in params.items() if k in self._active_names}
            _, trial_idx = self._client.attach_trial(active_params)
            self._client.complete_trial(
                trial_index=trial_idx,
                raw_data={self._objective_name: value},
            )

        self._observations.append((dict(params), value))

    def best(self) -> dict[str, Any] | None:
        """Return the best observed parameter dict, or None if no observations yet."""
        if not self._observations:
            return None

        if self._minimize:
            best_params, _ = min(self._observations, key=lambda x: x[1])
        else:
            best_params, _ = max(self._observations, key=lambda x: x[1])
        return dict(best_params)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_ax_params(self) -> list[dict[str, Any]]:
        """Build the Ax parameter specification list from active variables."""
        ax_params: list[dict[str, Any]] = []
        for var in self._active_vars:
            if var.variable_type == VariableType.CONTINUOUS:
                ax_params.append(
                    {
                        "name": var.name,
                        "type": "range",
                        "bounds": [
                            var.lower if var.lower is not None else 0.0,
                            var.upper if var.upper is not None else 1.0,
                        ],
                        "value_type": "float",
                    }
                )
            elif var.variable_type == VariableType.INTEGER:
                ax_params.append(
                    {
                        "name": var.name,
                        "type": "range",
                        "bounds": [
                            int(var.lower) if var.lower is not None else 0,
                            int(var.upper) if var.upper is not None else 10,
                        ],
                        "value_type": "int",
                    }
                )
            elif var.variable_type == VariableType.CATEGORICAL and var.choices:
                ax_params.append(
                    {
                        "name": var.name,
                        "type": "choice",
                        "values": list(var.choices),
                        "is_ordered": False,
                    }
                )
            elif var.variable_type == VariableType.BOOLEAN:
                # Represent booleans as a FLOAT [0, 1] range; suggest() rounds the
                # raw float back to True/False after get_next_trial() returns.
                ax_params.append(
                    {
                        "name": var.name,
                        "type": "range",
                        "bounds": [0.0, 1.0],
                        "value_type": "float",
                    }
                )
        return ax_params

    @staticmethod
    def _midpoint_for(var: Variable) -> Any:
        """Compute the midpoint (default value) for a variable."""
        vt = var.variable_type
        if vt == VariableType.CONTINUOUS:
            lo = var.lower if var.lower is not None else 0.0
            hi = var.upper if var.upper is not None else 1.0
            return (lo + hi) / 2.0
        if vt == VariableType.INTEGER:
            lo = int(var.lower) if var.lower is not None else 0
            hi = int(var.upper) if var.upper is not None else 10
            return (lo + hi) // 2
        if vt == VariableType.BOOLEAN:
            return False
        if vt == VariableType.CATEGORICAL and var.choices:
            return var.choices[len(var.choices) // 2]
        return None
