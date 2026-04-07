"""Multi-threshold interaction policy benchmark -- positive control.

Uses real ERCOT covariates with a treatment effect function driven by
**interactions** between temperature, humidity, and hour-of-day.  Unlike
the base DemandResponseScenario (where the challenge is a single categorical
trap), this benchmark's difficulty comes from a nonlinear interaction surface:
the treatment effect is super-additive when multiple conditions co-occur
(hot AND humid AND afternoon).

**Intended causal advantage mechanism:**

The causal graph connects 4 real policy variables (temp threshold, humidity
threshold, hour start, hour end) to the objective and disconnects 3 noise
variables (wind speed, pressure, cloud cover).  A causal-aware optimizer
can focus its search budget on the 4-D real subspace.  A surrogate-only
optimizer must search a 7-D space with a complex interaction surface,
wasting budget on noise dimensions.  The interaction structure means that
marginal effects of individual variables are weak -- you need to get
MULTIPLE thresholds right simultaneously to capture the benefit.

Public API
----------
- :class:`InteractionPolicyScenario` -- generates semi-synthetic data.
- :func:`interaction_treatment_effect` -- deterministic interaction-driven effect.
- :func:`evaluate_interaction_policy` -- evaluate a multi-threshold policy.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from causal_optimizer.benchmarks.counterfactual_energy import (
    CounterfactualBenchmarkResult,
    net_benefit,
)
from causal_optimizer.benchmarks.runner import sample_random_params
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    CausalGraph,
    SearchSpace,
    Variable,
    VariableType,
)

# ── Treatment effect function (interaction-driven) ──────────────────


def interaction_treatment_effect(
    temperature: np.ndarray,
    humidity: np.ndarray,
    hour_of_day: np.ndarray,
) -> np.ndarray:
    """Compute the interaction-driven treatment effect (load reduction in MW).

    The effect has three components that **multiply** together, creating
    super-additive interactions.  Getting just one condition right yields
    a small effect; getting all three right yields a large effect.

    Components:
    - Temperature: sigmoid centered at 28C, slope 0.18.  Responds to heat.
    - Humidity: sigmoid centered at 55%, slope 0.06.  High humidity
      amplifies cooling load and therefore treatment benefit.
    - Hour: Gaussian peak at 15:00, sigma 3.0h.  Afternoon peak demand.

    The three-way product is scaled to a peak of ~400 MW when all three
    conditions are extreme (temp >> 35C, humidity >> 80%, hour ~ 15).
    Each marginal component contributes only ~30-80 MW alone, so the
    super-additive interaction is the dominant feature.

    Temperature is expected in Celsius, humidity in percent (0-100).

    Args:
        temperature: Array of temperatures (Celsius).
        humidity: Array of relative humidity values (0-100).
        hour_of_day: Array of hour values (0-23).

    Returns:
        Non-negative array of treatment effects (load reduction).
    """
    temp = np.asarray(temperature, dtype=np.float64)
    humid = np.asarray(humidity, dtype=np.float64)
    hour = np.asarray(hour_of_day, dtype=np.float64)

    # Temperature response: sigmoid centered at 28C
    temp_response = 1.0 / (1.0 + np.exp(-0.18 * (temp - 28.0)))

    # Humidity response: sigmoid centered at 55%
    humid_response = 1.0 / (1.0 + np.exp(-0.06 * (humid - 55.0)))

    # Hour response: Gaussian peak at 15:00, sigma=3.0h
    hour_response = np.exp(-0.5 * ((hour - 15.0) / 3.0) ** 2)

    # Main effects (small individually)
    main_temp = 80.0 * temp_response * 0.4  # ~32 MW at saturation
    main_humid = 60.0 * humid_response * 0.4  # ~24 MW at saturation
    main_hour = 50.0 * hour_response * 0.4  # ~20 MW at saturation

    # Two-way interactions (moderate)
    interact_temp_humid = 120.0 * temp_response * humid_response
    interact_temp_hour = 100.0 * temp_response * hour_response
    interact_humid_hour = 80.0 * humid_response * hour_response

    # Three-way interaction (dominant)
    interact_all = 250.0 * temp_response * humid_response * hour_response

    total = (
        main_temp
        + main_humid
        + main_hour
        + interact_temp_humid
        + interact_temp_hour
        + interact_humid_hour
        + interact_all
    )

    return np.maximum(total, 0.0)


def interaction_propensity(
    temperature: np.ndarray,
    humidity: np.ndarray,
    hour_of_day: np.ndarray,
) -> np.ndarray:
    """Compute treatment propensity for the interaction scenario.

    Treatment is more likely when conditions favor high load: hot, humid,
    and afternoon.  This creates mild confounding (treatment is more likely
    when it would be most effective).

    Args:
        temperature: Array of temperatures (Celsius).
        humidity: Array of relative humidity values (0-100).
        hour_of_day: Array of hour values (0-23).

    Returns:
        Array of probabilities in [0.05, 0.85].
    """
    temp = np.asarray(temperature, dtype=np.float64)
    humid = np.asarray(humidity, dtype=np.float64)
    hour = np.asarray(hour_of_day, dtype=np.float64)

    # Logit components
    temp_z = 0.10 * (temp - 26.0)
    humid_z = 0.02 * (humid - 50.0)
    hour_z = np.where(
        (hour >= 8.0) & (hour <= 22.0),
        0.25 * (1.0 - np.abs(hour - 15.0) / 8.0),
        -0.4,
    )

    z = temp_z + humid_z + hour_z
    prop = 1.0 / (1.0 + np.exp(-z))

    return np.clip(prop, 0.05, 0.85)


# ── Policy evaluation ────────────────────────────────────────────────


def evaluate_interaction_policy(
    data: pd.DataFrame,
    params: dict[str, Any],
    treatment_cost: float,
) -> tuple[float, float]:
    """Evaluate a multi-threshold treatment policy on counterfactual data.

    The policy treats when ALL of:
    - temperature >= policy_temp_threshold
    - humidity >= policy_humidity_threshold
    - hour_of_day in [policy_hour_start, policy_hour_end]

    Noise dimensions (noise_wind_speed, noise_pressure, noise_cloud_cover)
    are ignored -- they have no effect on the outcome.

    Args:
        data: DataFrame with counterfactual columns (y0, y1, true_treatment_effect).
        params: Policy parameters from the search space.
        treatment_cost: Cost per treatment event.

    Returns:
        Tuple of (policy_value, decision_error_rate).
    """
    temp_thresh = float(params.get("policy_temp_threshold", 28.0))
    humid_thresh = float(params.get("policy_humidity_threshold", 55.0))
    hour_start = int(params.get("policy_hour_start", 12))
    hour_end = int(params.get("policy_hour_end", 20))

    # Hour range: handle wrap-around
    if hour_start <= hour_end:
        hour_mask = (data["hour_of_day"] >= hour_start) & (data["hour_of_day"] <= hour_end)
    else:
        hour_mask = (data["hour_of_day"] >= hour_start) | (data["hour_of_day"] <= hour_end)

    treat_mask = (
        (data["temperature"] >= temp_thresh) & (data["humidity"] >= humid_thresh) & hour_mask
    )

    treat_arr = treat_mask.values.astype(bool)
    y0 = data["y0"].values
    y1 = data["y1"].values

    policy_value = net_benefit(y0, y1, treat_arr, treatment_cost)

    # Decision error rate: fraction that disagree with oracle
    true_effect = data["true_treatment_effect"].values
    oracle_treat = true_effect > treatment_cost
    decision_error = float(np.mean(treat_arr != oracle_treat))

    return policy_value, decision_error


# ── Policy runner ────────────────────────────────────────────────────


class InteractionPolicyRunner:
    """ExperimentRunner that evaluates multi-threshold policies on counterfactual data."""

    def __init__(self, val_data: pd.DataFrame, treatment_cost: float) -> None:
        self._val_data = val_data
        self._treatment_cost = treatment_cost

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Evaluate one policy configuration and return metrics."""
        policy_value, decision_error = evaluate_interaction_policy(
            self._val_data, parameters, self._treatment_cost
        )
        return {
            "objective": -policy_value,
            "policy_value": policy_value,
            "decision_error_rate": decision_error,
        }


# ── Main scenario class ─────────────────────────────────────────────


class InteractionPolicyScenario:
    """Multi-threshold interaction policy benchmark using real covariates.

    Generates counterfactual data where the treatment effect is driven
    by interactions between temperature, humidity, and hour-of-day.
    Unlike the base DemandResponseScenario, this benchmark has:

    1. No categorical trap -- all variables are continuous/integer.
    2. Super-additive interactions -- marginal effects are small,
       joint effects are large.
    3. Three noise dimensions that plausibly look like they could
       matter (wind speed, pressure, cloud cover) but have zero
       effect on the outcome.

    The causal advantage comes from knowing which variables are
    real parents of the objective (and therefore which dimensions
    to search intensively) in a landscape where interactions make
    the optimization surface complex.

    Args:
        covariates: DataFrame with real covariate columns.
        seed: Random seed controlling treatment assignment.
        treatment_cost: Fixed cost per treatment event.  The default
            (120.0) produces an oracle treat rate of ~25-35% with
            the interaction-based effect function on typical ERCOT
            covariate distributions.
    """

    def __init__(
        self,
        covariates: pd.DataFrame,
        seed: int = 0,
        treatment_cost: float = 120.0,
    ) -> None:
        self._covariates = covariates.copy()
        self._seed = seed
        self.treatment_cost = treatment_cost

    def generate(self) -> pd.DataFrame:
        """Generate semi-synthetic dataset with interaction-driven counterfactual outcomes.

        Returns:
            DataFrame with original covariates plus:
            - ``treatment_event``: binary treatment (0/1)
            - ``y0``: potential outcome under no treatment
            - ``y1``: potential outcome under treatment
            - ``observed_outcome``: realized outcome
            - ``true_treatment_effect``: load reduction from treatment
            - ``propensity``: known treatment propensity
        """
        rng = np.random.default_rng(self._seed)
        df = self._covariates.copy()

        temp = df["temperature"].values.astype(np.float64)
        humid = df["humidity"].values.astype(np.float64)
        hour = df["hour_of_day"].values.astype(np.float64)

        # Fill NaN covariates with neutral values to prevent NaN propagation.
        # Real ERCOT data has rare NaN humidity (~3 rows out of 26k).
        temp = np.where(np.isnan(temp), 20.0, temp)
        humid = np.where(np.isnan(humid), 50.0, humid)
        hour = np.where(np.isnan(hour), 12.0, hour)

        # Y(0) = base load
        y0 = df["target_load"].values.astype(np.float64)

        # Treatment effect: deterministic, interaction-driven
        effect = interaction_treatment_effect(temp, humid, hour)

        # Y(1) = base load - treatment effect
        y1 = y0 - effect

        # Treatment assignment via propensity
        prop_scores = interaction_propensity(temp, humid, hour)
        treatment = (rng.random(len(df)) < prop_scores).astype(int)

        # Observed outcome: factual
        observed = np.where(treatment == 1, y1, y0)

        df["treatment_event"] = treatment
        df["y0"] = y0
        df["y1"] = y1
        df["observed_outcome"] = observed
        df["true_treatment_effect"] = effect
        df["propensity"] = prop_scores

        return df

    @staticmethod
    def causal_graph() -> CausalGraph:
        """Return the known causal graph for the interaction policy scenario.

        Real policy variables (temp threshold, humidity threshold, hour
        start/end) are direct parents of ``objective``.  Noise dimensions
        (wind speed, pressure, cloud cover) connect to an isolated
        ``weather_noise`` node and are NOT ancestors of ``objective``.
        """
        return CausalGraph(
            edges=[
                ("policy_temp_threshold", "objective"),
                ("policy_humidity_threshold", "objective"),
                ("policy_hour_start", "objective"),
                ("policy_hour_end", "objective"),
                ("noise_wind_speed", "weather_noise"),
                ("noise_pressure", "weather_noise"),
                ("noise_cloud_cover", "weather_noise"),
            ],
        )

    @staticmethod
    def search_space() -> SearchSpace:
        """Return the policy search space.

        4 real policy parameters that interact to determine the treatment
        effect, plus 3 continuous noise dimensions with zero effect.
        No categorical variables -- the challenge is the interaction
        surface, not a categorical trap.
        """
        return SearchSpace(
            variables=[
                Variable(
                    name="policy_temp_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=10.0,
                    upper=42.0,
                ),
                Variable(
                    name="policy_humidity_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=10.0,
                    upper=95.0,
                ),
                Variable(
                    name="policy_hour_start",
                    variable_type=VariableType.INTEGER,
                    lower=0,
                    upper=23,
                ),
                Variable(
                    name="policy_hour_end",
                    variable_type=VariableType.INTEGER,
                    lower=0,
                    upper=23,
                ),
                Variable(
                    name="noise_wind_speed",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=30.0,
                ),
                Variable(
                    name="noise_pressure",
                    variable_type=VariableType.CONTINUOUS,
                    lower=980.0,
                    upper=1050.0,
                ),
                Variable(
                    name="noise_cloud_cover",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=100.0,
                ),
            ]
        )

    def oracle_policy_value(self, data: pd.DataFrame) -> float:
        """Compute the value of the oracle policy.

        The oracle treats exactly when the treatment effect exceeds cost.
        """
        y0 = data["y0"].values
        y1 = data["y1"].values
        effect = data["true_treatment_effect"].values
        oracle_treat = effect > self.treatment_cost
        return net_benefit(y0, y1, oracle_treat, self.treatment_cost)

    def run_benchmark(
        self,
        budget: int,
        seed: int,
        strategy: str = "random",
    ) -> CounterfactualBenchmarkResult:
        """Run one strategy on this scenario and return results.

        Generates data, splits into opt/test, runs the optimizer
        on opt, evaluates the learned policy on test.

        Args:
            budget: Number of experiments (policy evaluations).
            seed: Random seed for the optimizer.
            strategy: One of ``"random"``, ``"surrogate_only"``, ``"causal"``.

        Returns:
            :class:`CounterfactualBenchmarkResult` with policy value,
            oracle value, regret, and decision error rate.
        """
        valid_strategies = {"random", "surrogate_only", "causal"}
        if strategy not in valid_strategies:
            msg = f"Unknown strategy {strategy!r}, expected one of {sorted(valid_strategies)}"
            raise ValueError(msg)

        t_start = time.monotonic()

        data = self.generate()
        n = len(data)

        # Split: 80/20 opt/test by position
        opt_end = int(n * 0.8)
        val_data = data.iloc[:opt_end].reset_index(drop=True)
        test_data = data.iloc[opt_end:].reset_index(drop=True)

        space = self.search_space()
        runner = InteractionPolicyRunner(val_data, self.treatment_cost)

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

        # Evaluate on test set
        if best_params is not None:
            policy_value, decision_error = evaluate_interaction_policy(
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
