"""Semi-synthetic dose-response clinical benchmark.

Simulates a clinical-trial-style scenario where an optimizer searches
for the best treatment protocol (dose level, patient selection
thresholds) to maximize net patient benefit.

**Causal story:**

A treatment has a dose-response curve that depends on patient biomarker
level and disease severity.  The true effect follows an Emax (sigmoid)
model: high-biomarker, high-severity patients benefit most from
moderate-to-high doses, while low-biomarker patients get little benefit
regardless of dose.  The causal graph encodes this: ``dose_level``,
``biomarker_threshold``, and ``severity_threshold`` are direct parents
of the objective.  Three noise dimensions (``bmi_threshold``,
``age_threshold``, ``comorbidity_threshold``) have NO causal effect
on treatment benefit -- they are connected to a separate ``patient_risk``
node, not to ``objective``.

Causal knowledge helps by:
1. Pruning 3 noise dimensions from the search, focusing on 3 real ones.
2. Correctly identifying that biomarker and severity mediate efficacy.

**Data semantics:**

- Y(0): Baseline symptom score (higher = worse symptoms).
- Y(1): Symptom score after treatment at a fixed reference dose.
- true_treatment_effect: Y(0) - Y(1) = symptom reduction (non-negative).
- The optimizer searches over protocol parameters that decide:
  (a) What dose to use,
  (b) Which patients to treat (biomarker >= threshold, severity >= threshold).

**Evaluation:**

Net benefit = sum of (effect - cost) for treated patients.  The oracle
treats exactly the patients whose treatment effect exceeds the cost at
the protocol's dose level.

Public API
----------
- :class:`DoseResponseScenario` -- generates data, runs benchmarks.
- :class:`DoseResponseBenchmarkResult` -- result container.
- :func:`dose_response_effect` -- deterministic effect function.
- :func:`evaluate_protocol` -- evaluate a treatment protocol on data.
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
class DoseResponseBenchmarkResult:
    """Result of running one strategy on the dose-response benchmark.

    Attributes:
        strategy: Optimization strategy name.
        budget: Number of experiments in the optimization run.
        seed: Random seed used.
        policy_value: Average net benefit under the learned protocol
            (higher is better; symptom reduction minus treatment cost).
        oracle_value: Average net benefit under the oracle protocol.
        regret: oracle_value - policy_value (non-negative by construction).
        decision_error_rate: Fraction of treat/no-treat decisions that
            disagree with the oracle's optimal decision.
        runtime_seconds: Wall-clock time for the full run.
    """

    strategy: str
    budget: int
    seed: int
    policy_value: float
    oracle_value: float
    regret: float
    decision_error_rate: float
    runtime_seconds: float


# ── Dose-response effect function ───────────────────────────────────


def dose_response_effect(
    dose: np.ndarray,
    biomarker: np.ndarray,
    severity: np.ndarray,
) -> np.ndarray:
    """Compute the deterministic treatment effect (symptom reduction).

    Uses an Emax (sigmoid) dose-response model with biomarker-mediated
    efficacy.  The effect is:

        effect = Emax(dose, biomarker) * severity_modifier

    where:
    - Emax component: ``max_effect * dose^hill / (ED50^hill + dose^hill)``
    - max_effect scales with biomarker: ``20 + 60 * biomarker``
      (range 20-80 symptom points for bio in [0,1])
    - ED50 = 0.3 (dose at half-max effect)
    - hill = 2.5 (steepness of dose-response curve)
    - severity_modifier: ``0.3 + 0.7 * severity`` (range 0.3-1.0)

    The combination means:
    - High-biomarker, high-severity patients get up to 80 points reduction
    - Low-biomarker, low-severity patients get at most ~10 points
    - Zero dose always gives zero effect

    All inputs should be in [0, 1].  No stochastic noise -- the
    counterfactual truth is exact.

    Args:
        dose: Array of dose levels in [0, 1].
        biomarker: Array of biomarker values in [0, 1].
        severity: Array of severity values in [0, 1].

    Returns:
        Non-negative array of treatment effects (symptom reduction).
    """
    d = np.asarray(dose, dtype=np.float64)
    b = np.asarray(biomarker, dtype=np.float64)
    s = np.asarray(severity, dtype=np.float64)

    # Emax (Hill equation) parameters
    ed50 = 0.3
    hill = 2.5
    max_effect = 20.0 + 60.0 * b  # biomarker-mediated max effect

    # Avoid division by zero when dose=0: numerator is 0 so result is 0
    d_hill = np.power(np.maximum(d, 0.0), hill)
    ed50_hill = ed50**hill
    emax_response = max_effect * d_hill / (ed50_hill + d_hill)

    # Severity modifier: higher severity -> larger effect
    severity_mod = 0.3 + 0.7 * s

    result: np.ndarray = np.maximum(emax_response * severity_mod, 0.0)
    return result


# ── Protocol evaluation ─────────────────────────────────────────────


def _net_benefit(
    y0: np.ndarray,
    y1: np.ndarray,
    treat_mask: np.ndarray,
    treatment_cost: float,
) -> float:
    """Compute average net benefit of a treatment protocol.

    Average over patients of:
    - If treated: (y0 - y1) - cost = symptom_reduction - cost
    - If untreated: 0

    Higher is better.

    Args:
        y0: Potential outcome under no treatment (symptom score).
        y1: Potential outcome under treatment.
        treat_mask: Boolean array of treatment decisions.
        treatment_cost: Cost per treatment (in symptom-score units).

    Returns:
        Average net benefit (float).
    """
    reduction = y0 - y1
    unit_benefit = np.where(treat_mask, reduction - treatment_cost, 0.0)
    return float(unit_benefit.mean())


def evaluate_protocol(
    data: pd.DataFrame,
    params: dict[str, Any],
    treatment_cost: float,
) -> tuple[float, float]:
    """Evaluate a treatment protocol on counterfactual patient data.

    The protocol treats patients whose biomarker and severity exceed
    thresholds, at the specified dose level.  The dose level scales the
    treatment effect via the Emax curve.

    Args:
        data: DataFrame with counterfactual columns (y0, y1,
            true_treatment_effect, biomarker, severity, etc.).
        params: Protocol parameters from the search space.
        treatment_cost: Cost per treatment event.

    Returns:
        Tuple of (policy_value, decision_error_rate).
    """
    dose = float(params.get("dose_level", 0.5))
    bio_thresh = float(params.get("biomarker_threshold", 0.5))
    sev_thresh = float(params.get("severity_threshold", 0.5))
    bmi_thresh = float(params.get("bmi_threshold", 0.0))
    age_thresh = float(params.get("age_threshold", 0.0))
    comorbidity_thresh = float(params.get("comorbidity_threshold", 0.0))

    # Core selection: patients with biomarker and severity above thresholds
    treat_mask = (data["biomarker"] >= bio_thresh) & (data["severity"] >= sev_thresh)

    # Noise dimensions: these have no causal effect on treatment benefit.
    # When threshold > 0 they exclude patients (harmful); the optimal protocol
    # sets all three to 0.  The discontinuity at 0 is intentional: values in
    # (0, 1] restrict the patient pool while 0 disables filtering entirely.
    if bmi_thresh > 0:
        treat_mask = treat_mask & (data["bmi"] >= bmi_thresh)
    if age_thresh > 0:
        treat_mask = treat_mask & (data["age"] >= age_thresh)
    if comorbidity_thresh > 0:
        treat_mask = treat_mask & (data["comorbidity_score"] >= comorbidity_thresh)

    treat_arr = treat_mask.values.astype(bool)

    # Compute dose-adjusted effect for treated patients
    bio = data["biomarker"].values
    sev = data["severity"].values
    dose_arr = np.full(len(data), dose)
    effect_at_dose = dose_response_effect(dose_arr, bio, sev)

    # Potential outcomes at the protocol's dose level
    y0 = data["y0"].values
    y1_at_dose = y0 - effect_at_dose

    # Policy value: benefit from treating selected patients at this dose
    policy_value = _net_benefit(y0, y1_at_dose, treat_arr, treatment_cost)

    # Decision error: compare against the oracle at THIS dose level.
    # Note: this is dose-specific -- a protocol at dose=0.3 is compared
    # against the optimal selection *at dose=0.3*, not the global oracle
    # (which uses the reference dose).  Regret captures the full gap.
    oracle_treat = effect_at_dose > treatment_cost
    decision_error = float(np.mean(treat_arr != oracle_treat))

    return policy_value, decision_error


# ── Protocol runner for the optimizer ────────────────────────────────


class ProtocolRunner:
    """ExperimentRunner that evaluates treatment protocols on patient data.

    Used internally by :meth:`DoseResponseScenario.run_benchmark` to
    connect the optimizer to the counterfactual evaluation.
    """

    def __init__(self, val_data: pd.DataFrame, treatment_cost: float) -> None:
        self._val_data = val_data
        self._treatment_cost = treatment_cost

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Evaluate one protocol configuration and return metrics."""
        policy_value, decision_error = evaluate_protocol(
            self._val_data, parameters, self._treatment_cost
        )
        # Optimizer minimizes "objective", so negate the policy value
        return {
            "objective": -policy_value,
            "policy_value": policy_value,
            "decision_error_rate": decision_error,
        }


# ── Main scenario class ─────────────────────────────────────────────


class DoseResponseScenario:
    """Semi-synthetic dose-response clinical scenario.

    Generates a synthetic patient population with known covariate
    distributions and overlays a deterministic dose-response treatment
    effect.  No external data dependency -- covariates are generated
    internally, making this benchmark cheap and fully reproducible.

    Args:
        n_patients: Number of patients to generate.
        seed: Random seed controlling all data generation.
        treatment_cost: Fixed cost per treatment event (in symptom-score
            units).  The default (15.0) produces an oracle treat rate
            of roughly 25-40% depending on the dose level.
        reference_dose: The dose level used to generate the reference
            Y(1) column in the data.  Default 0.7 (moderate-high dose).
    """

    def __init__(
        self,
        n_patients: int = 1000,
        seed: int = 0,
        treatment_cost: float = 15.0,
        reference_dose: float = 0.7,
    ) -> None:
        self._n_patients = n_patients
        self._seed = seed
        self.treatment_cost = treatment_cost
        self._reference_dose = reference_dose

    def generate(self) -> pd.DataFrame:
        """Generate semi-synthetic patient data with counterfactual outcomes.

        Returns:
            DataFrame with patient covariates and counterfactual columns:
            - ``patient_id``: Unique integer ID.
            - ``age``: Patient age in [0, 1] (normalized).
            - ``biomarker``: Predictive biomarker in [0, 1].
            - ``severity``: Disease severity in [0, 1].
            - ``bmi``: Body mass index in [0, 1] (normalized, noise dim).
            - ``sex``: Binary 0/1 (noise dimension).
            - ``comorbidity_score``: Comorbidity burden in [0, 1] (noise dim).
            - ``y0``: Symptom score under no treatment.
            - ``y1``: Symptom score under treatment at reference dose.
            - ``true_treatment_effect``: y0 - y1 (symptom reduction).
            - ``observed_outcome``: Realized outcome.
            - ``treatment_assigned``: Binary 0/1 treatment assignment.
        """
        rng = np.random.default_rng(self._seed)
        n = self._n_patients

        # Patient covariates
        age = rng.beta(2.0, 5.0, size=n)  # skewed young
        biomarker = rng.beta(2.0, 2.0, size=n)  # symmetric around 0.5
        severity = rng.beta(1.5, 3.0, size=n)  # skewed mild
        bmi = rng.beta(2.0, 2.0, size=n)  # noise dimension
        sex = rng.integers(0, 2, size=n).astype(float)  # noise dimension
        comorbidity = rng.beta(1.0, 3.0, size=n)  # noise dimension

        # Baseline symptom score Y(0): depends on severity and age
        # Range roughly 30-100 points
        y0 = 30.0 + 50.0 * severity + 20.0 * age + 5.0 * rng.standard_normal(n)
        y0 = np.maximum(y0, 1.0)  # floor at 1

        # Treatment effect at reference dose
        effect = dose_response_effect(np.full(n, self._reference_dose), biomarker, severity)

        # Y(1) = Y(0) - effect
        y1 = y0 - effect

        # Treatment assignment: propensity depends on severity
        # (sicker patients more likely to be treated -- mild confounding)
        propensity_logit = -0.5 + 1.5 * severity + 0.5 * biomarker
        propensity_score = 1.0 / (1.0 + np.exp(-propensity_logit))
        treatment = (rng.random(n) < propensity_score).astype(int)

        # Observed outcome: factual
        observed = np.where(treatment == 1, y1, y0)

        df = pd.DataFrame(
            {
                "patient_id": np.arange(n),
                "age": age,
                "biomarker": biomarker,
                "severity": severity,
                "bmi": bmi,
                "sex": sex,
                "comorbidity_score": comorbidity,
                "y0": y0,
                "y1": y1,
                "true_treatment_effect": effect,
                "observed_outcome": observed,
                "treatment_assigned": treatment,
            }
        )
        return df

    @staticmethod
    def causal_graph() -> CausalGraph:
        """Return the known causal graph for the dose-response scenario.

        ``dose_level``, ``biomarker_threshold``, and ``severity_threshold``
        are direct parents of ``objective``.  The noise dimensions
        (``bmi_threshold``, ``age_threshold``, ``comorbidity_threshold``)
        connect to a separate ``patient_risk`` node.
        """
        return CausalGraph(
            edges=[
                ("dose_level", "objective"),
                ("biomarker_threshold", "objective"),
                ("severity_threshold", "objective"),
                ("bmi_threshold", "patient_risk"),
                ("age_threshold", "patient_risk"),
                ("comorbidity_threshold", "patient_risk"),
            ],
        )

    @staticmethod
    def search_space() -> SearchSpace:
        """Return the protocol search space.

        The optimizer searches over protocol parameters: dose level and
        patient selection thresholds.  ``bmi_threshold``, ``age_threshold``,
        and ``comorbidity_threshold`` are noise dimensions -- they do NOT
        affect treatment benefit in the causal graph.
        """
        return SearchSpace(
            variables=[
                Variable(
                    name="dose_level",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="biomarker_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="severity_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="bmi_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="age_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="comorbidity_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
            ]
        )

    def oracle_policy_value(self, data: pd.DataFrame) -> float:
        """Compute the value of the oracle protocol at the reference dose.

        The oracle treats exactly the patients whose effect at the
        *reference dose* (``self._reference_dose``, default 0.7) exceeds
        the treatment cost.  This is the global oracle -- it answers
        "what is the best achievable benefit at the reference dose?"

        Note: ``evaluate_protocol`` computes decision error against the
        oracle *at the protocol's chosen dose*, which is a different
        (dose-specific) quantity.  Regret = oracle_value - policy_value
        captures the full gap including suboptimal dose selection.

        Returns:
            Average net benefit under the oracle protocol.
        """
        y0 = data["y0"].values
        y1 = data["y1"].values
        effect = data["true_treatment_effect"].values
        oracle_treat = effect > self.treatment_cost
        return _net_benefit(y0, y1, oracle_treat, self.treatment_cost)

    def run_benchmark(
        self,
        budget: int,
        seed: int,
        strategy: str = "random",
    ) -> DoseResponseBenchmarkResult:
        """Run one strategy on this scenario and return results.

        Generates data, splits 80/20 into opt/test, runs the optimizer
        on opt, and evaluates the learned protocol on test.

        Args:
            budget: Number of experiments (protocol evaluations).
            seed: Random seed for the optimizer.
            strategy: One of ``"random"``, ``"surrogate_only"``, ``"causal"``.

        Returns:
            :class:`DoseResponseBenchmarkResult` with policy value,
            oracle value, regret, and decision error rate.
        """
        valid_strategies = {"random", "surrogate_only", "causal"}
        if strategy not in valid_strategies:
            msg = f"Unknown strategy {strategy!r}, expected one of {sorted(valid_strategies)}"
            raise ValueError(msg)

        t_start = time.monotonic()

        # Generate data
        data = self.generate()
        n = len(data)

        # Split: 80/20 opt/test
        opt_end = int(n * 0.8)
        val_data = data.iloc[:opt_end].reset_index(drop=True)
        test_data = data.iloc[opt_end:].reset_index(drop=True)

        space = self.search_space()
        runner = ProtocolRunner(val_data, self.treatment_cost)

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
                max_skips=0,  # disable skip -- protocol eval is cheap
            )
            engine.run_loop(budget)
            best_result = engine.log.best_result("objective", minimize=True)
            best_params = best_result.parameters if best_result is not None else None

        # Evaluate on test set
        if best_params is not None:
            policy_value, decision_error = evaluate_protocol(
                test_data, best_params, self.treatment_cost
            )
        else:
            policy_value = 0.0
            effect = test_data["true_treatment_effect"].values
            oracle_treat = effect > self.treatment_cost
            decision_error = float(np.mean(oracle_treat))

        oracle_value = self.oracle_policy_value(test_data)
        regret = oracle_value - policy_value

        runtime = time.monotonic() - t_start

        return DoseResponseBenchmarkResult(
            strategy=strategy,
            budget=budget,
            seed=seed,
            policy_value=policy_value,
            oracle_value=oracle_value,
            regret=regret,
            decision_error_rate=decision_error,
            runtime_seconds=runtime,
        )
