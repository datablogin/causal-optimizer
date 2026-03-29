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
statistics are directly comparable.  ``HighNoiseDemandResponse``
inherits ``run_benchmark`` from the base class (only ``generate``,
``search_space``, and ``causal_graph`` are overridden).
``ConfoundedDemandResponse`` overrides ``run_benchmark`` to
deconfound the evaluation metric -- the optimizer trains on biased
data but is evaluated on the true causal benefit.

Public API
----------
- :class:`HighNoiseDemandResponse`
- :class:`ConfoundedDemandResponse`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from causal_optimizer.benchmarks.counterfactual_energy import (
    CounterfactualBenchmarkResult,
    DemandResponseScenario,
    _PolicyRunner,
    _propensity,
    _treatment_effect,
    evaluate_policy,
)
from causal_optimizer.benchmarks.runner import sample_random_params
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

        nuisance_edges = [(f"noise_var_{i}", "nuisance_sink") for i in range(_NUM_NUISANCE_VARS)]

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
    # Epsilon guard: _propensity clips to [0.05, 0.90], so logit is bounded,
    # but guard against future changes that widen the clip range.
    safe_prop = np.clip(base_prop, 1e-6, 1.0 - 1e-6)
    logit_base = np.log(safe_prop / (1.0 - safe_prop))
    logit_confounded = logit_base + 1.5 * (grid_stress - 0.5)
    confounded_prop = 1.0 / (1.0 + np.exp(-logit_confounded))

    return np.clip(confounded_prop, 0.05, 0.90)


class ConfoundedDemandResponse(DemandResponseScenario):
    """Confounded treatment assignment variant of the demand-response benchmark.

    Introduces a hidden confounder ("grid stress") that affects BOTH:

    1. Treatment assignment probability (high stress -> more likely to treat)
    2. Base load outcome (high stress -> higher Y(0), inflating the
       *apparent* treatment benefit y0 - y1)

    **Confounding mechanism:**

    Grid stress U ~ Beta(2, 5) + 0.3 * normalized_temperature.  U is
    unobserved but correlates with temperature, creating a back-door
    path from policy parameters to the outcome.  It affects:

    - Y(0): base_load += 500 * U (higher stress -> higher base load)
    - Y(1): base_load - effect (NO grid stress shift -- treatment
      outcome is purely causal)
    - Propensity: logit(p) += 1.5 * (U - 0.5)

    Because Y(0) includes grid stress but Y(1) does not, the apparent
    treatment benefit ``y0 - y1 = effect + 500 * U`` is inflated for
    high-stress hours.  Since U correlates with temperature, policies
    that set a low temperature threshold (treating many hot hours) will
    appear better than they truly are.

    The oracle uses ``true_treatment_effect`` (= ``effect``, independent
    of U) and is therefore unbiased.  A naive estimator that uses
    ``y0 - y1`` will overestimate the benefit in high-temperature
    windows and recommend over-treating.

    Overrides ``run_benchmark`` to deconfound the evaluation metric:
    the optimizer trains on confounded data (y0 includes grid stress)
    but is evaluated on the true causal benefit (y0 swapped to baseline).
    The causal graph's bidirected edges alert POMIS computation to be
    conservative about intervention sets.
    """

    def generate(self) -> pd.DataFrame:
        """Generate confounded counterfactual data.

        Overrides the base ``generate()`` to introduce grid stress as a
        hidden confounder that inflates Y(0) but not Y(1), so the
        apparent benefit ``y0 - y1`` is confounded while the true
        treatment effect remains causal.
        """
        rng = np.random.default_rng(self._seed)
        df = self._covariates.copy()

        temp = df["temperature"].values
        hour = df["hour_of_day"].values

        # Hidden confounder: grid stress ~ Beta(2, 5) + temperature correlation.
        # Correlating with temperature creates a back-door path so that
        # policies selecting high-temperature hours see inflated benefits.
        temp_norm = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)
        grid_stress = np.clip(
            rng.beta(2.0, 5.0, size=len(df)) + 0.3 * temp_norm,
            0.0,
            1.0,
        )

        y0_base = df["target_load"].values.astype(np.float64)

        # Y(0) includes confounding: high stress -> higher base load
        y0 = y0_base + 500.0 * grid_stress

        # Treatment effect: deterministic given covariates only
        effect = _treatment_effect(temp, hour)

        # Y(1) does NOT carry grid stress -- treatment outcome is causal.
        # This makes y0 - y1 = effect + 500 * grid_stress, so the
        # apparent benefit is inflated for high-stress (high-temp) hours.
        y1 = y0_base - effect

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
        df["_debug_grid_stress"] = grid_stress
        # Deconfounded Y(0) for causal evaluation (no grid stress).
        df["_deconfounded_y0"] = y0_base

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
        """Return causal graph with bidirected edges marking confounding.

        The directed edge structure matches the base benchmark.  Bidirected
        edges on the three treatment-parameter nodes signal that an
        unobserved confounder (grid stress) affects both the treatment
        decision context and the outcome.  Grid stress correlates with
        temperature and shifts propensity via all three parameters, so all
        three carry confounding.
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
                ("treat_hour_start", "objective"),
                ("treat_hour_end", "objective"),
            ],
        )

    def naive_policy(self, data: pd.DataFrame, cost: float) -> np.ndarray:
        """Compute the naive "best predictor" policy from biased observed data.

        Estimates treatment effect as the difference in mean outcome
        between treated and untreated groups, stratified by temperature
        bin.  Because grid stress confounds the treatment assignment,
        this estimate is biased -- the direction depends on the
        interplay between the confounded Y(0) and treatment selection.

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

    def oracle_policy_value(self, data: pd.DataFrame) -> float:
        """Oracle value anchored to the true causal effect, not confounded y0-y1.

        In the confounded variant, ``y0 - y1 = effect + 500 * grid_stress``,
        so the inherited oracle (which uses ``_net_benefit(y0, y1, ...)``)
        would NOT be the ceiling of the causal metric.  We override to
        compute oracle value from ``true_treatment_effect`` directly.
        """
        effect = data["true_treatment_effect"].values
        oracle_treat = effect > self.treatment_cost
        unit_benefit = np.where(oracle_treat, effect - self.treatment_cost, 0.0)
        return float(unit_benefit.mean())

    def run_benchmark(
        self,
        budget: int,
        seed: int,
        strategy: str = "random",
    ) -> CounterfactualBenchmarkResult:
        """Run one strategy, evaluating on the deconfounded causal metric.

        The optimizer trains on **confounded** data (``y0`` includes grid
        stress), so surrogate_only learns the biased landscape.  The final
        evaluation swaps ``y0`` to the deconfounded baseline so that
        ``y0 - y1 = effect`` — the true causal benefit.  This ensures
        regret is non-negative and measures distance to the causal oracle.
        """
        import time

        from causal_optimizer.engine.loop import ExperimentEngine

        valid_strategies = {"random", "surrogate_only", "causal"}
        if strategy not in valid_strategies:
            msg = f"Unknown strategy {strategy!r}, expected one of {sorted(valid_strategies)}"
            raise ValueError(msg)

        t_start = time.monotonic()

        data = self.generate()
        n = len(data)
        opt_end = int(n * 0.8)
        val_data = data.iloc[:opt_end].reset_index(drop=True)
        test_data = data.iloc[opt_end:].reset_index(drop=True)

        space = self.search_space()
        # Runner trains on CONFOUNDED data (optimizer sees biased landscape)
        runner = _PolicyRunner(val_data, self.treatment_cost)

        if strategy == "random":
            rng = np.random.default_rng(seed)
            best_obj = float("inf")
            best_params: dict[str, Any] | None = None
            for _ in range(budget):
                params = sample_random_params(space, rng)
                metrics = runner.run(params)
                obj = metrics["objective"]
                if obj < best_obj:
                    best_obj = obj
                    best_params = params
        else:
            graph = self.causal_graph() if strategy == "causal" else None
            engine = ExperimentEngine(
                search_space=space,
                runner=runner,
                causal_graph=graph,
                objective_name="objective",
                minimize=True,
                seed=seed,
                max_skips=0,
            )
            engine.run_loop(budget)
            best_result = engine.log.best_result("objective", minimize=True)
            best_params = best_result.parameters if best_result is not None else None

        # Evaluate on DECONFOUNDED test set.  Swap y0 to remove grid stress
        # so evaluate_policy computes y0-y1 = effect (the true causal benefit).
        # oracle_policy_value is already overridden to use true_treatment_effect
        # directly and doesn't need this swap, but eval_data is passed to it
        # for interface consistency.
        eval_data = test_data.copy()
        eval_data["y0"] = eval_data["_deconfounded_y0"]

        if best_params is not None:
            policy_value, decision_error = evaluate_policy(
                eval_data, best_params, self.treatment_cost
            )
        else:
            policy_value = 0.0
            oracle_treat = eval_data["true_treatment_effect"].values > self.treatment_cost
            decision_error = float(np.mean(oracle_treat))

        oracle_value = self.oracle_policy_value(eval_data)
        regret = oracle_value - policy_value

        return CounterfactualBenchmarkResult(
            strategy=strategy,
            budget=budget,
            seed=seed,
            policy_value=policy_value,
            oracle_value=oracle_value,
            regret=regret,
            decision_error_rate=decision_error,
            runtime_seconds=time.monotonic() - t_start,
        )
