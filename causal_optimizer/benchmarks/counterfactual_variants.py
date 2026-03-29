"""Harder counterfactual benchmark variants for causal differentiation.

Provides two variants of the base DemandResponseScenario that create
stronger positive-control pressure for causal guidance:

1. **HighNoiseDemandResponse** -- Adds 10 irrelevant nuisance dimensions
   to the search space.  The causal graph excludes these nuisance vars
   as ancestors of the objective, so causal guidance focuses on 3 real
   dimensions while surrogate_only must search 13+.

2. **ConfoundedDemandResponse** -- Introduces a hidden confounder ("grid
   stress") that affects both treatment assignment probability and the
   base load outcome.  This creates Simpson's paradox: naive estimation
   overestimates treatment benefit because treated hours systematically
   have higher load.  The causal graph marks the confounding with a
   bidirected edge, enabling POMIS-aware search.

Both variants reuse the same treatment-effect function as the base
benchmark (sigmoid * Gaussian on temperature and hour) so oracle
statistics are directly comparable.  Both inherit ``run_benchmark``
from the base class -- only ``generate``, ``search_space``, and
``causal_graph`` are overridden (the minimum needed for polymorphism).

Public API
----------
- :class:`HighNoiseDemandResponse`
- :class:`ConfoundedDemandResponse`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from causal_optimizer.benchmarks.counterfactual_energy import (
    DemandResponseScenario,
    _propensity,
    _treatment_effect,
)
from causal_optimizer.types import (
    CausalGraph,
    SearchSpace,
    Variable,
    VariableType,
)

# Number of nuisance dimensions added by the high-noise variant.
_NUM_NUISANCE_VARS: int = 10


# ── High-noise variant ─────────────────────────────────────────────


class HighNoiseDemandResponse(DemandResponseScenario):
    """High-dimensional noise variant of the demand-response benchmark.

    Adds 10 irrelevant nuisance dimensions (``noise_var_0`` through
    ``noise_var_9``) to the search space.  These are continuous [0, 1]
    variables with ZERO effect on the treatment outcome.  The causal
    graph does NOT include edges from nuisance vars to the objective,
    providing the structural advantage that causal knowledge enables.

    Surrogate_only must search a 15-dimensional space (3 real + 2
    original noise + 10 new nuisance); causal can focus on the 3 real
    parents.

    Inherits ``run_benchmark`` from :class:`DemandResponseScenario`
    unchanged -- polymorphism through overridden ``generate``,
    ``search_space``, and ``causal_graph`` is sufficient.
    """

    def generate(self) -> pd.DataFrame:
        """Generate data with nuisance columns appended.

        Calls the base ``generate()`` to produce counterfactual outcomes,
        then adds ``noise_var_0`` through ``noise_var_9`` as independent
        uniform [0, 1] columns with no relationship to any outcome or
        treatment variable.
        """
        df = super().generate()

        rng = np.random.default_rng(self._seed + 9999)
        for i in range(_NUM_NUISANCE_VARS):
            df[f"noise_var_{i}"] = rng.random(len(df))

        return df

    @staticmethod
    def search_space() -> SearchSpace:
        """Return the expanded search space with nuisance dimensions.

        The 3 real causal parents + 2 original noise dims from the base
        benchmark, plus 10 new nuisance dimensions (noise_var_0 through
        noise_var_9), for 15 total.
        """
        base_vars = DemandResponseScenario.search_space().variables

        nuisance_vars = [
            Variable(
                name=f"noise_var_{i}",
                variable_type=VariableType.CONTINUOUS,
                lower=0.0,
                upper=1.0,
            )
            for i in range(_NUM_NUISANCE_VARS)
        ]

        return SearchSpace(variables=base_vars + nuisance_vars)

    @staticmethod
    def causal_graph() -> CausalGraph:
        """Return the causal graph with nuisance vars disconnected from objective.

        The 3 true parents (treat_temp_threshold, treat_hour_start,
        treat_hour_end) are direct parents of objective.  The nuisance
        variables connect to an isolated ``nuisance_sink`` node, ensuring
        they are NOT ancestors of ``objective``.
        """
        base_edges = [
            ("treat_temp_threshold", "objective"),
            ("treat_hour_start", "objective"),
            ("treat_hour_end", "objective"),
            ("treat_humidity_threshold", "base_load"),
            ("treat_day_filter", "base_load"),
        ]

        nuisance_edges = [
            (f"noise_var_{i}", "nuisance_sink") for i in range(_NUM_NUISANCE_VARS)
        ]

        return CausalGraph(edges=base_edges + nuisance_edges)


# ── Confounded variant ─────────────────────────────────────────────


def _confounded_propensity(
    temperature: np.ndarray,
    hour_of_day: np.ndarray,
    grid_stress: np.ndarray,
) -> np.ndarray:
    """Treatment propensity with confounding from grid stress.

    In addition to the base propensity (temperature + hour), grid stress
    increases treatment probability.  High-stress hours are more likely
    to trigger demand response, AND they have higher base load -- this
    is the confounding mechanism.

    Args:
        temperature: Array of temperatures (Celsius).
        hour_of_day: Array of hour values (0-23).
        grid_stress: Array of grid-stress values in [0, 1].

    Returns:
        Array of propensities in [0.05, 0.90].
    """
    base_prop = _propensity(temperature, hour_of_day)

    # Shift logit by grid stress: high stress -> more likely to treat.
    logit_base = np.log(base_prop / (1.0 - base_prop))
    logit_confounded = logit_base + 1.5 * (grid_stress - 0.5)
    confounded_prop = 1.0 / (1.0 + np.exp(-logit_confounded))

    return np.clip(confounded_prop, 0.05, 0.90)


class ConfoundedDemandResponse(DemandResponseScenario):
    """Confounded treatment assignment variant of the demand-response benchmark.

    Introduces a hidden confounder ("grid stress") that affects BOTH:

    1. Treatment assignment probability (high stress -> more likely to treat)
    2. Base load outcome (high stress -> higher load, making treatment
       *appear* more effective because load reduction is larger)

    This creates Simpson's paradox: naive estimation sees treated hours
    with higher load (due to grid stress, not treatment) and overestimates
    treatment benefit.  The oracle policy (based on true causal effect)
    differs from the naive "best predictor" policy.

    **Confounding mechanism:**

    Grid stress U ~ Beta(2, 5) is unobserved.  It affects:

    - Y(0): base_load += 500 * U (higher stress -> higher load)
    - Propensity: logit(p) += 1.5 * (U - 0.5) (higher stress -> more treatment)

    The treatment effect itself is NOT affected by grid stress -- it is
    still a deterministic function of temperature and hour only.  But
    because treated units tend to have higher U (and thus higher Y(0)),
    the observed Y(0) - Y(1) difference is biased upward for treated
    units.

    Inherits ``run_benchmark`` from :class:`DemandResponseScenario`
    unchanged -- the causal graph's bidirected edge alerts POMIS
    computation to be conservative about intervention sets.
    """

    def generate(self) -> pd.DataFrame:
        """Generate confounded counterfactual data.

        Overrides the base ``generate()`` to introduce grid stress as a
        hidden confounder that shifts both base load and treatment
        propensity.
        """
        rng = np.random.default_rng(self._seed)
        df = self._covariates.copy()

        temp = df["temperature"].values
        hour = df["hour_of_day"].values

        # Hidden confounder: grid stress ~ Beta(2, 5), mean ~0.286
        grid_stress = rng.beta(2.0, 5.0, size=len(df))

        # Y(0) includes confounding shift: high stress -> higher base load
        y0_base = df["target_load"].values.astype(np.float64)
        y0 = y0_base + 500.0 * grid_stress

        # Treatment effect: deterministic given covariates only (not grid stress)
        effect = _treatment_effect(temp, hour)
        y1 = y0 - effect

        # Treatment assignment: confounded propensity
        propensity = _confounded_propensity(temp, hour, grid_stress)
        treatment = (rng.random(len(df)) < propensity).astype(int)

        observed = np.where(treatment == 1, y1, y0)

        df["demand_response_event"] = treatment
        df["y0"] = y0
        df["y1"] = y1
        df["observed_outcome"] = observed
        df["true_treatment_effect"] = effect
        df["propensity"] = propensity
        df["grid_stress"] = grid_stress

        return df

    @staticmethod
    def search_space() -> SearchSpace:
        """Return the same search space as the base benchmark.

        The confounded variant does not add dimensions -- the difficulty
        comes from biased estimation, not dimensionality.
        """
        return DemandResponseScenario.search_space()

    @staticmethod
    def causal_graph() -> CausalGraph:
        """Return causal graph with bidirected edge marking confounding.

        The directed edge structure matches the base benchmark.  A
        bidirected edge between ``treat_temp_threshold`` and ``objective``
        signals that an unobserved confounder (grid stress) affects both
        the treatment decision context and the outcome.
        """
        return CausalGraph(
            edges=[
                ("treat_temp_threshold", "objective"),
                ("treat_hour_start", "objective"),
                ("treat_hour_end", "objective"),
                ("treat_humidity_threshold", "base_load"),
                ("treat_day_filter", "base_load"),
            ],
            bidirected_edges=[
                ("treat_temp_threshold", "objective"),
            ],
        )

    def naive_policy(self, data: pd.DataFrame, cost: float) -> np.ndarray:
        """Compute the naive "best predictor" policy from biased observed data.

        Estimates treatment effect as the difference in mean outcome
        between treated and untreated groups, stratified by temperature
        bin.  Because grid stress confounds the treatment assignment,
        this estimate is biased upward -- treated units have
        systematically higher base load.

        Args:
            data: DataFrame with ``demand_response_event`` and
                ``observed_outcome`` columns.
            cost: Treatment cost threshold.

        Returns:
            Boolean array of naive treatment decisions.
        """
        import pandas as pd_rt  # runtime import; module-level is TYPE_CHECKING only

        temp_bins = pd_rt.cut(data["temperature"], bins=5, labels=False)

        naive_effects = np.zeros(len(data))
        for bin_val in range(5):
            mask = temp_bins == bin_val
            if mask.sum() == 0:
                continue
            bin_data = data[mask]
            treated = bin_data[bin_data["demand_response_event"] == 1]
            untreated = bin_data[bin_data["demand_response_event"] == 0]
            if len(treated) < 2 or len(untreated) < 2:
                continue
            naive_eff = float(
                untreated["observed_outcome"].mean() - treated["observed_outcome"].mean()
            )
            naive_effects[mask] = naive_eff

        return naive_effects > cost
