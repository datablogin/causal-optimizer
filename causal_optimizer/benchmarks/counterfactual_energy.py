"""Semi-synthetic counterfactual energy benchmark -- demand response scenario.

Uses real ERCOT covariates with known treatment effects to enable
counterfactual evaluation of optimization strategies.  The treatment
effect function is deterministic given covariates, so counterfactual
ground truth is exact.

Public API
----------
- :class:`DemandResponseScenario` -- generates semi-synthetic data.
- :class:`CounterfactualBenchmarkResult` -- result container.
- :func:`evaluate_policy` -- evaluate a threshold-based policy on data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from causal_optimizer.benchmarks.runner import sample_random_params
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    CausalGraph,
    SearchSpace,
    Variable,
    VariableType,
)


@dataclass
class CounterfactualBenchmarkResult:
    """Result of running one strategy on the counterfactual benchmark.

    Attributes:
        strategy: Optimization strategy name.
        budget: Number of experiments in the optimization run.
        seed: Random seed used.
        policy_value: Average net benefit under the learned policy
            (higher is better; measured as load reduction minus cost).
        oracle_value: Average net benefit under the oracle policy.
        regret: oracle_value - policy_value (non-negative by construction).
        treatment_effect_mae: MAE of estimated vs true effects.
        runtime_seconds: Wall-clock time for the full run.
    """

    strategy: str
    budget: int
    seed: int
    policy_value: float
    oracle_value: float
    regret: float
    treatment_effect_mae: float
    runtime_seconds: float


# ── Treatment effect function ────────────────────────────────────────


def _treatment_effect(temperature: np.ndarray, hour_of_day: np.ndarray) -> np.ndarray:
    """Compute the deterministic treatment effect (load reduction in MW).

    The effect is large on hot afternoons, near-zero on mild nights,
    and moderate otherwise.  No stochastic noise -- counterfactual
    truth is exact.

    Args:
        temperature: Array of temperatures (Fahrenheit).
        hour_of_day: Array of hour values (0-23).

    Returns:
        Non-negative array of treatment effects (load reduction).
    """
    temp = np.asarray(temperature, dtype=np.float64)
    hour = np.asarray(hour_of_day, dtype=np.float64)

    # Base effect: moderate everywhere
    effect = np.full_like(temp, 30.0)

    # Hot afternoon bonus: temp > 90F and hour in [14, 18]
    hot_afternoon = (temp > 90.0) & (hour >= 14.0) & (hour <= 18.0)
    effect = np.where(hot_afternoon, 150.0 + 2.0 * (temp - 90.0), effect)

    # Warm afternoon: temp > 80F and hour in [14, 18] but not hot
    warm_afternoon = (temp > 80.0) & (temp <= 90.0) & (hour >= 14.0) & (hour <= 18.0)
    effect = np.where(warm_afternoon, 60.0 + 1.0 * (temp - 80.0), effect)

    # Mild night suppression: temp < 70F or hour in [0, 6]
    mild_night = (temp < 70.0) | (hour <= 6.0)
    # Scale down: colder and later at night means less effect
    night_factor = np.clip((temp - 50.0) / 40.0, 0.0, 1.0)
    effect = np.where(mild_night, 5.0 * night_factor, effect)

    return np.maximum(effect, 0.0)


def _propensity(temperature: np.ndarray, hour_of_day: np.ndarray) -> np.ndarray:
    """Compute treatment propensity (probability of demand response event).

    Treatment is more likely on hot afternoons (mild confounding: treatment
    is more likely when it would be most effective).

    Args:
        temperature: Array of temperatures (Fahrenheit).
        hour_of_day: Array of hour values (0-23).

    Returns:
        Array of probabilities in [0.05, 0.90].
    """
    temp = np.asarray(temperature, dtype=np.float64)
    hour = np.asarray(hour_of_day, dtype=np.float64)

    # Strong propensity driven by temperature and afternoon hours.
    # Temperature component: centered at 80F, steep slope
    temp_z = 0.08 * (temp - 80.0)

    # Hour component: peaks at 16, decays away from afternoon
    hour_z = np.where(
        (hour >= 10.0) & (hour <= 22.0),
        0.3 * (1.0 - np.abs(hour - 16.0) / 8.0),
        -0.5,  # strong penalty for night hours
    )

    z = temp_z + hour_z
    propensity = 1.0 / (1.0 + np.exp(-z))

    return np.clip(propensity, 0.05, 0.90)


# ── Policy evaluation ────────────────────────────────────────────────


def _net_benefit(
    y0: np.ndarray,
    y1: np.ndarray,
    treat_mask: np.ndarray,
    treatment_cost: float,
) -> float:
    """Compute average net benefit of a treatment policy.

    Net benefit = sum over units of:
    - If treated: (y0 - y1) - cost  = load_reduction - cost
    - If untreated: 0

    This is the savings relative to never-treat baseline.
    Higher is better.

    Args:
        y0: Potential outcome under no treatment.
        y1: Potential outcome under treatment (y1 = y0 - effect).
        treat_mask: Boolean array of treatment decisions.
        treatment_cost: Cost per treatment event.

    Returns:
        Average net benefit (float).
    """
    # load_reduction = y0 - y1 = effect (non-negative)
    reduction = y0 - y1
    unit_benefit = np.where(treat_mask, reduction - treatment_cost, 0.0)
    return float(unit_benefit.mean())


def evaluate_policy(
    data: pd.DataFrame,
    params: dict[str, Any],
    treatment_cost: float,
) -> tuple[float, float]:
    """Evaluate a threshold-based treatment policy on counterfactual data.

    The policy treats when:
    - temperature >= treat_temp_threshold
    - hour_of_day in [treat_hour_start, treat_hour_end]
    - humidity >= treat_humidity_threshold (noise dimension)
    - day filter matches (noise dimension)

    Args:
        data: DataFrame with counterfactual columns (y0, y1, true_treatment_effect).
        params: Policy parameters from the search space.
        treatment_cost: Cost per treatment event.

    Returns:
        Tuple of (policy_value, treatment_effect_mae).
        policy_value is the average net benefit (higher is better).
    """
    temp_thresh = float(params.get("treat_temp_threshold", 80.0))
    hour_start = int(params.get("treat_hour_start", 14))
    hour_end = int(params.get("treat_hour_end", 18))
    humidity_thresh = float(params.get("treat_humidity_threshold", 0.0))
    day_filter = params.get("treat_day_filter", "all")

    # Decide who to treat
    # Handle hour_start > hour_end (e.g., 22 to 4 wraps around midnight)
    if hour_start <= hour_end:
        hour_mask = (data["hour_of_day"] >= hour_start) & (data["hour_of_day"] <= hour_end)
    else:
        hour_mask = (data["hour_of_day"] >= hour_start) | (data["hour_of_day"] <= hour_end)

    treat_mask = (data["temperature"] >= temp_thresh) & hour_mask

    # Noise dimensions (these should NOT matter for optimal policy)
    if humidity_thresh > 0:
        treat_mask = treat_mask & (data["humidity"] >= humidity_thresh)
    if day_filter == "weekday":
        treat_mask = treat_mask & (data["day_of_week"] < 5)
    elif day_filter == "weekend":
        treat_mask = treat_mask & (data["day_of_week"] >= 5)

    treat_arr = treat_mask.values.astype(bool)
    y0 = data["y0"].values
    y1 = data["y1"].values

    # Policy value = average net benefit relative to never-treat
    policy_value = _net_benefit(y0, y1, treat_arr, treatment_cost)

    # Treatment effect MAE: how well does the policy's decision boundary
    # approximate the true effect?  For untreated units, the policy
    # implicitly estimates effect < cost (we assign 0); for treated,
    # it estimates effect >= cost (we assign true effect).
    true_effect = data["true_treatment_effect"].values
    estimated_effect = np.where(treat_arr, true_effect, 0.0)
    mae = float(np.mean(np.abs(estimated_effect - true_effect)))

    return policy_value, mae


# ── Benchmark runner for policy evaluation ───────────────────────────


class _PolicyRunner:
    """ExperimentRunner that evaluates threshold policies on counterfactual data.

    Used internally by :meth:`DemandResponseScenario.run_benchmark` to
    connect the optimizer to the counterfactual evaluation.
    """

    def __init__(self, val_data: pd.DataFrame, treatment_cost: float) -> None:
        self._val_data = val_data
        self._treatment_cost = treatment_cost

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Evaluate one policy configuration and return metrics."""
        policy_value, effect_mae = evaluate_policy(
            self._val_data, parameters, self._treatment_cost
        )
        # The optimizer minimizes "objective", so negate the policy value
        # (higher policy value = better, so lower negative = better for minimizer)
        return {
            "objective": -policy_value,
            "policy_value": policy_value,
            "treatment_effect_mae": effect_mae,
        }


# ── Main scenario class ─────────────────────────────────────────────


class DemandResponseScenario:
    """Semi-synthetic demand-response scenario using real covariates.

    Generates counterfactual data by overlaying known treatment effects
    and propensity-driven treatment assignment onto real ERCOT covariate
    data.

    Args:
        covariates: DataFrame with real covariate columns (temperature,
            humidity, hour_of_day, day_of_week, is_holiday, target_load,
            and optionally load_lag_* columns).
        seed: Random seed controlling treatment assignment randomness.
        treatment_cost: Fixed cost per demand-response event.
    """

    def __init__(
        self,
        covariates: pd.DataFrame,
        seed: int = 0,
        treatment_cost: float = 50.0,
    ) -> None:
        self._covariates = covariates.copy()
        self._seed = seed
        self.treatment_cost = treatment_cost

    def generate(self) -> pd.DataFrame:
        """Generate semi-synthetic dataset with counterfactual outcomes.

        Returns:
            DataFrame with original covariates plus:
            - ``demand_response_event``: binary treatment (0/1)
            - ``y0``: potential outcome under no treatment
            - ``y1``: potential outcome under treatment
            - ``observed_outcome``: realized outcome
            - ``true_treatment_effect``: load reduction from treatment
            - ``propensity``: known treatment propensity
        """
        rng = np.random.default_rng(self._seed)
        df = self._covariates.copy()

        temp = df["temperature"].values
        hour = df["hour_of_day"].values

        # Potential outcomes
        # Y(0) = base load (the real target_load from ERCOT data)
        y0 = df["target_load"].values.astype(np.float64)

        # Treatment effect: deterministic given covariates
        effect = _treatment_effect(temp, hour)

        # Y(1) = base load - treatment effect (treatment reduces load)
        y1 = y0 - effect

        # Treatment assignment via propensity
        propensity = _propensity(temp, hour)
        treatment = (rng.random(len(df)) < propensity).astype(int)

        # Observed outcome: factual
        observed = np.where(treatment == 1, y1, y0)

        df["demand_response_event"] = treatment
        df["y0"] = y0
        df["y1"] = y1
        df["observed_outcome"] = observed
        df["true_treatment_effect"] = effect
        df["propensity"] = propensity

        return df

    @staticmethod
    def causal_graph() -> CausalGraph:
        """Return the known causal graph for the demand-response scenario.

        The graph encodes that temperature and hour_of_day are parents
        of both the treatment and the outcome, while humidity and
        day_of_week only affect base_load (not load_reduction).

        This gives the optimizer genuine non-parents to deprioritize.
        """
        return CausalGraph(
            edges=[
                ("temperature", "demand_response_event"),
                ("hour_of_day", "demand_response_event"),
                ("demand_response_event", "load_reduction"),
                ("temperature", "load_reduction"),
                ("hour_of_day", "load_reduction"),
                ("humidity", "base_load"),
                ("day_of_week", "base_load"),
            ],
        )

    @staticmethod
    def search_space() -> SearchSpace:
        """Return the policy search space.

        The optimizer searches over policy parameters (treatment
        decision thresholds), not model hyperparameters.  Variables
        ``treat_humidity_threshold`` and ``treat_day_filter`` are noise
        dimensions -- they are NOT parents of load_reduction in the
        causal graph.
        """
        return SearchSpace(
            variables=[
                Variable(
                    name="treat_temp_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=60.0,
                    upper=100.0,
                ),
                Variable(
                    name="treat_hour_start",
                    variable_type=VariableType.INTEGER,
                    lower=0,
                    upper=23,
                ),
                Variable(
                    name="treat_hour_end",
                    variable_type=VariableType.INTEGER,
                    lower=0,
                    upper=23,
                ),
                Variable(
                    name="treat_humidity_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=100.0,
                ),
                Variable(
                    name="treat_day_filter",
                    variable_type=VariableType.CATEGORICAL,
                    choices=["all", "weekday", "weekend"],
                ),
            ]
        )

    def oracle_policy_value(self, data: pd.DataFrame) -> float:
        """Compute the value of the oracle policy on the given data.

        The oracle treats exactly when the treatment effect exceeds
        the cost.  This is optimal because counterfactual truth is known.

        Returns:
            Average net benefit under the oracle policy (higher is better).
        """
        y0 = data["y0"].values
        y1 = data["y1"].values
        effect = data["true_treatment_effect"].values

        # Treat when effect > cost (load reduction exceeds cost)
        oracle_treat = effect > self.treatment_cost
        return _net_benefit(y0, y1, oracle_treat, self.treatment_cost)

    def run_benchmark(
        self,
        budget: int,
        seed: int,
        strategy: str = "random",
    ) -> CounterfactualBenchmarkResult:
        """Run one strategy on this scenario and return results.

        Generates data, splits into train/val/test by position, runs
        the optimizer on val, and evaluates the learned policy on test.

        Args:
            budget: Number of experiments (policy evaluations).
            seed: Random seed for the optimizer.
            strategy: One of ``"random"``, ``"surrogate_only"``, ``"causal"``.

        Returns:
            :class:`CounterfactualBenchmarkResult` with policy value,
            oracle value, regret, and treatment effect MAE.
        """
        t_start = time.monotonic()

        # Generate counterfactual data
        data = self.generate()
        n = len(data)

        # Split: 60/20/20 by position (data is already time-ordered)
        train_end = int(n * 0.6)
        val_end = train_end + int(n * 0.2)
        val_data = data.iloc[train_end:val_end].reset_index(drop=True)
        test_data = data.iloc[val_end:].reset_index(drop=True)

        space = self.search_space()
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
            )
            engine.run_loop(budget)
            best_result = engine.log.best_result("objective", minimize=True)
            best_params = best_result.parameters if best_result is not None else None

        # Evaluate on test set
        if best_params is not None:
            policy_value, effect_mae = evaluate_policy(
                test_data, best_params, self.treatment_cost
            )
        else:
            # No valid policy found; use never-treat as fallback
            policy_value = 0.0  # never-treat has zero net benefit
            effect_mae = float(test_data["true_treatment_effect"].mean())

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
            treatment_effect_mae=effect_mae,
            runtime_seconds=runtime,
        )
