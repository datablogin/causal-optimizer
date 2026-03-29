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
statistics are directly comparable.

Public API
----------
- :class:`HighNoiseDemandResponse`
- :class:`ConfoundedDemandResponse`
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from causal_optimizer.benchmarks.counterfactual_energy import (
    CounterfactualBenchmarkResult,
    DemandResponseScenario,
    _net_benefit,
    _propensity,
    _treatment_effect,
    evaluate_policy,
)
from causal_optimizer.benchmarks.runner import sample_random_params
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    CausalGraph,
    SearchSpace,
    Variable,
    VariableType,
)

# Number of nuisance dimensions added by the high-noise variant.
_NUM_NUISANCE_VARS: int = 10


# ── High-noise variant ─────────────────────────────────────────────


def _evaluate_high_noise_policy(
    data: pd.DataFrame,
    params: dict[str, Any],
    treatment_cost: float,
) -> tuple[float, float]:
    """Evaluate a threshold-based policy on high-noise counterfactual data.

    Identical to :func:`evaluate_policy` from the base benchmark -- the
    nuisance variables are ignored during policy evaluation because they
    have zero causal effect on the outcome.

    Args:
        data: DataFrame with counterfactual columns.
        params: Policy parameters (may include noise_var_* keys, which
            are silently ignored).
        treatment_cost: Cost per treatment event.

    Returns:
        Tuple of (policy_value, decision_error_rate).
    """
    # Delegate to the base evaluate_policy, which only reads the 3 real
    # causal parents + 2 original noise dims.  The noise_var_* keys in
    # params are harmlessly ignored.
    return evaluate_policy(data, params, treatment_cost)


class _HighNoisePolicyRunner:
    """ExperimentRunner for the high-noise variant."""

    def __init__(self, val_data: pd.DataFrame, treatment_cost: float) -> None:
        self._val_data = val_data
        self._treatment_cost = treatment_cost

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Evaluate one policy configuration and return metrics."""
        policy_value, decision_error = _evaluate_high_noise_policy(
            self._val_data, parameters, self._treatment_cost
        )
        return {
            "objective": -policy_value,
            "policy_value": policy_value,
            "decision_error_rate": decision_error,
        }


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

    Args:
        covariates: DataFrame with real covariate columns.
        seed: Random seed controlling treatment assignment randomness.
        treatment_cost: Fixed cost per demand-response event.
    """

    def generate(self) -> pd.DataFrame:
        """Generate data with nuisance columns appended.

        Calls the base generate() to produce counterfactual outcomes,
        then adds ``noise_var_0`` through ``noise_var_9`` as independent
        uniform [0, 1] columns.  These columns have no relationship to
        any outcome or treatment variable.
        """
        # Use parent's generate for the core counterfactual data
        df = super().generate()

        # Add independent nuisance columns
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
            # Original noise dims connect to base_load (not objective)
            ("treat_humidity_threshold", "base_load"),
            ("treat_day_filter", "base_load"),
        ]

        # Nuisance vars connect to a separate sink node
        nuisance_edges = [
            (f"noise_var_{i}", "nuisance_sink") for i in range(_NUM_NUISANCE_VARS)
        ]

        return CausalGraph(edges=base_edges + nuisance_edges)

    def run_benchmark(
        self,
        budget: int,
        seed: int,
        strategy: str = "random",
    ) -> CounterfactualBenchmarkResult:
        """Run one strategy on the high-noise variant.

        Same logic as the base run_benchmark, but uses the expanded
        search space and high-noise causal graph.

        Args:
            budget: Number of experiments (policy evaluations).
            seed: Random seed for the optimizer.
            strategy: One of ``"random"``, ``"surrogate_only"``, ``"causal"``.

        Returns:
            :class:`CounterfactualBenchmarkResult` with results.
        """
        valid_strategies = {"random", "surrogate_only", "causal"}
        if strategy not in valid_strategies:
            msg = (
                f"Unknown strategy {strategy!r}, "
                f"expected one of {sorted(valid_strategies)}"
            )
            raise ValueError(msg)

        t_start = time.monotonic()

        data = self.generate()
        n = len(data)
        opt_end = int(n * 0.8)
        val_data = data.iloc[:opt_end].reset_index(drop=True)
        test_data = data.iloc[opt_end:].reset_index(drop=True)

        space = self.search_space()
        runner = _HighNoisePolicyRunner(val_data, self.treatment_cost)

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

        if best_params is not None:
            policy_value, decision_error = _evaluate_high_noise_policy(
                test_data, best_params, self.treatment_cost
            )
        else:
            policy_value = 0.0
            oracle_treat = test_data["true_treatment_effect"].values > self.treatment_cost
            decision_error = float(np.mean(oracle_treat))

        oracle_value = self.oracle_policy_value(test_data)
        regret = oracle_value - policy_value
        runtime = time.monotonic() - t_start

        return CounterfactualBenchmarkResult(
            strategy=strategy,
            budget=budget,
            seed=seed,
            policy_value=policy_value,
            oracle_value=oracle_value,
            regret=regret,
            decision_error_rate=decision_error,
            runtime_seconds=runtime,
        )


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
    # Base propensity from temperature and hour
    base_prop = _propensity(temperature, hour_of_day)

    # Grid stress adds confounding: high stress -> more likely to treat
    # logit(p) += 1.5 * (grid_stress - 0.5)
    logit_base = np.log(base_prop / (1.0 - base_prop))
    logit_confounded = logit_base + 1.5 * (grid_stress - 0.5)
    confounded_prop = 1.0 / (1.0 + np.exp(-logit_confounded))

    return np.clip(confounded_prop, 0.05, 0.90)


def _evaluate_confounded_policy(
    data: pd.DataFrame,
    params: dict[str, Any],
    treatment_cost: float,
) -> tuple[float, float]:
    """Evaluate a threshold-based policy on confounded data.

    Same policy structure as the base benchmark.  The confounded
    outcomes are used for evaluation but the policy decision is
    based only on observable thresholds.

    Args:
        data: DataFrame with confounded counterfactual columns.
        params: Policy parameters from the search space.
        treatment_cost: Cost per treatment event.

    Returns:
        Tuple of (policy_value, decision_error_rate).
    """
    return evaluate_policy(data, params, treatment_cost)


class _ConfoundedPolicyRunner:
    """ExperimentRunner for the confounded variant."""

    def __init__(self, val_data: pd.DataFrame, treatment_cost: float) -> None:
        self._val_data = val_data
        self._treatment_cost = treatment_cost

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Evaluate one policy configuration and return metrics."""
        policy_value, decision_error = _evaluate_confounded_policy(
            self._val_data, parameters, self._treatment_cost
        )
        return {
            "objective": -policy_value,
            "policy_value": policy_value,
            "decision_error_rate": decision_error,
        }


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

    The causal graph includes a bidirected edge between the treatment
    variable and the outcome, marking the confounding and enabling
    POMIS-aware search to avoid the trap.

    **Confounding mechanism:**

    Grid stress U ~ Beta(2, 5) is unobserved.  It affects:
    - Y(0): base_load += 500 * U (higher stress -> higher load)
    - Propensity: logit(p) += 1.5 * (U - 0.5) (higher stress -> more treatment)

    The treatment effect itself is NOT affected by grid stress -- it is
    still a deterministic function of temperature and hour only.  But
    because treated units tend to have higher U (and thus higher Y(0)),
    the observed Y(0) - Y(1) difference is biased upward for treated
    units.

    Args:
        covariates: DataFrame with real covariate columns.
        seed: Random seed controlling treatment assignment randomness.
        treatment_cost: Fixed cost per demand-response event.
    """

    def generate(self) -> pd.DataFrame:
        """Generate confounded counterfactual data.

        Overrides the base generate() to introduce grid stress as a
        hidden confounder that shifts both base load and treatment
        propensity.
        """
        import pandas as pd_mod

        rng = np.random.default_rng(self._seed)
        df = self._covariates.copy()

        temp = df["temperature"].values
        hour = df["hour_of_day"].values

        # Hidden confounder: grid stress ~ Beta(2, 5), mean ~0.286
        grid_stress = rng.beta(2.0, 5.0, size=len(df))

        # Y(0) = base load + confounding shift from grid stress
        # High stress -> higher base load (by up to 500 MW)
        y0_base = df["target_load"].values.astype(np.float64)
        y0 = y0_base + 500.0 * grid_stress

        # Treatment effect: deterministic given covariates only (not grid stress)
        effect = _treatment_effect(temp, hour)

        # Y(1) = Y(0) - treatment effect
        y1 = y0 - effect

        # Treatment assignment: confounded propensity
        propensity = _confounded_propensity(temp, hour, grid_stress)
        treatment = (rng.random(len(df)) < propensity).astype(int)

        # Observed outcome: factual
        observed = np.where(treatment == 1, y1, y0)

        df["demand_response_event"] = treatment
        df["y0"] = y0
        df["y1"] = y1
        df["observed_outcome"] = observed
        df["true_treatment_effect"] = effect
        df["propensity"] = propensity
        df["grid_stress"] = grid_stress  # stored for analysis, but "hidden" from optimizer

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

        The graph structure is the same as the base benchmark, plus a
        bidirected edge between ``treat_temp_threshold`` and ``objective``
        signaling that an unobserved confounder (grid stress) affects
        both the treatment decision context and the outcome.

        In practice, the confounding acts through the treatment propensity
        and the base load.  The bidirected edge alerts POMIS computation
        to be conservative about intervention sets.
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
                # Grid stress confounds the treatment-outcome relationship:
                # it affects both the treatment decision context (through
                # propensity) and the outcome (through base load).
                ("treat_temp_threshold", "objective"),
            ],
        )

    def naive_policy(self, data: pd.DataFrame, cost: float) -> np.ndarray:
        """Compute the naive "best predictor" policy from biased observed data.

        The naive estimator computes treatment effect as the difference
        in mean outcome between treated and untreated groups, stratified
        by temperature bin.  Because grid stress confounds the treatment
        assignment, this naive estimate is biased upward -- treated units
        have systematically higher base load.

        The naive policy treats when the naive estimated effect exceeds
        the cost.  This should differ from the oracle policy (which uses
        the true treatment effect).

        Args:
            data: DataFrame with counterfactual columns including
                ``demand_response_event`` and ``observed_outcome``.
            cost: Treatment cost threshold.

        Returns:
            Boolean array of naive treatment decisions.
        """
        import pandas as pd_mod

        # Bin temperature into 5 bins for stratified estimation
        temp_bins = pd_mod.cut(data["temperature"], bins=5, labels=False)

        # Estimate treatment effect per bin from observed data
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
            # Naive effect = E[Y|T=0] - E[Y|T=1] (load reduction)
            naive_eff = float(
                untreated["observed_outcome"].mean() - treated["observed_outcome"].mean()
            )
            naive_effects[mask] = naive_eff

        return naive_effects > cost

    def run_benchmark(
        self,
        budget: int,
        seed: int,
        strategy: str = "random",
    ) -> CounterfactualBenchmarkResult:
        """Run one strategy on the confounded variant.

        Args:
            budget: Number of experiments (policy evaluations).
            seed: Random seed for the optimizer.
            strategy: One of ``"random"``, ``"surrogate_only"``, ``"causal"``.

        Returns:
            :class:`CounterfactualBenchmarkResult` with results.
        """
        valid_strategies = {"random", "surrogate_only", "causal"}
        if strategy not in valid_strategies:
            msg = (
                f"Unknown strategy {strategy!r}, "
                f"expected one of {sorted(valid_strategies)}"
            )
            raise ValueError(msg)

        t_start = time.monotonic()

        data = self.generate()
        n = len(data)
        opt_end = int(n * 0.8)
        val_data = data.iloc[:opt_end].reset_index(drop=True)
        test_data = data.iloc[opt_end:].reset_index(drop=True)

        space = self.search_space()
        runner = _ConfoundedPolicyRunner(val_data, self.treatment_cost)

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

        if best_params is not None:
            policy_value, decision_error = _evaluate_confounded_policy(
                test_data, best_params, self.treatment_cost
            )
        else:
            policy_value = 0.0
            oracle_treat = test_data["true_treatment_effect"].values > self.treatment_cost
            decision_error = float(np.mean(oracle_treat))

        oracle_value = self.oracle_policy_value(test_data)
        regret = oracle_value - policy_value
        runtime = time.monotonic() - t_start

        return CounterfactualBenchmarkResult(
            strategy=strategy,
            budget=budget,
            seed=seed,
            policy_value=policy_value,
            oracle_value=oracle_value,
            regret=regret,
            decision_error_rate=decision_error,
            runtime_seconds=runtime,
        )
