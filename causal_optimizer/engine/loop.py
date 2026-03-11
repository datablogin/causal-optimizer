"""Core experiment loop orchestrator.

This is the main entry point — it coordinates discovery, design, estimation,
optimization, evolution, prediction, and validation into a unified loop.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol

import numpy as np

from causal_optimizer.designer.screening import ScreeningDesigner, ScreeningResult
from causal_optimizer.estimator.effects import EffectEstimator
from causal_optimizer.evolution.map_elites import MAPElites
from causal_optimizer.predictor.off_policy import OffPolicyPredictor
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
)

logger = logging.getLogger(__name__)

# Minimum requirements before calling estimate_improvement() for keep/discard.
# _MIN_KEPT=5 ensures the estimator uses its full statistical method (bootstrap
# or difference).  With fewer than 5 kept experiments the estimator falls back
# to a greedy comparison (method="greedy"), so calling it from the engine would
# add no statistical value over the engine's own greedy fallback.
_MIN_KEPT = 5
_MIN_DISCARDED = 2


class ExperimentRunner(Protocol):
    """Protocol for domain-specific experiment execution."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run an experiment with given parameters, return metrics."""
        ...


class ExperimentEngine:
    """Orchestrates the causal optimization loop.

    The loop:
        1. Discover — learn/update causal graph from experiment history
        2. Screen — use DoE to identify important variables and interactions
        3. Estimate — robustly estimate treatment effects of past experiments
        4. Prioritize — use CBO to select next experiment (observation vs intervention)
        5. Evolve — maintain diverse solution population via MAP-Elites
        6. Predict — off-policy evaluation of candidate experiments
        7. Validate — sensitivity analysis on findings
    """

    def __init__(
        self,
        search_space: SearchSpace,
        runner: ExperimentRunner,
        objective_name: str = "objective",
        minimize: bool = True,
        causal_graph: CausalGraph | None = None,
        max_skips: int = 3,
        max_screening_attempts: int = 3,
        descriptor_names: list[str] | None = None,
        epsilon_mode: bool = False,
        n_max: int = 100,
        seed: int | None = None,
        effect_method: str = "bootstrap",
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> None:
        """Initialize the experiment engine.

        Args:
            seed: Seed for the epsilon controller's RNG only. Controls
                reproducibility of observe/intervene decisions in
                ``OffPolicyPredictor``. Does **not** seed other random
                sources in the engine (MAP-Elites sampling, bootstrap CI).
            effect_method: Method used by :class:`EffectEstimator` to assess
                statistical significance in keep/discard decisions.  Valid
                values are ``"difference"`` and ``"bootstrap"`` (default).
                ``"aipw"`` is accepted but automatically falls back to
                ``"bootstrap"`` with a logged warning (AIPW requires
                treatment/control assignment not available in improvement
                tests).
            confidence_level: Confidence level for statistical tests (default
                0.95 → alpha = 0.05).  Passed directly to
                :class:`~causal_optimizer.estimator.effects.EffectEstimator`.
            n_bootstrap: Number of bootstrap samples used when
                ``effect_method="bootstrap"`` (default 1000).  Passed to
                :class:`~causal_optimizer.estimator.effects.EffectEstimator`.
        """
        _valid_effect_methods = {"difference", "bootstrap", "aipw"}
        if effect_method not in _valid_effect_methods:
            raise ValueError(
                f"Invalid effect_method {effect_method!r}. "
                f"Must be one of {sorted(_valid_effect_methods)}."
            )

        self.search_space = search_space
        self.runner = runner
        self.objective_name = objective_name
        self.minimize = minimize
        self.causal_graph = causal_graph
        self.log = ExperimentLog()
        self._phase: str = "exploration"
        self._effect_estimator = EffectEstimator(
            method=effect_method,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        )
        self._predictor = OffPolicyPredictor(epsilon_mode=epsilon_mode, n_max=n_max, seed=seed)
        self._max_skips = max_skips
        self._screening_result: ScreeningResult | None = None
        self._screened_focus_variables: list[str] | None = None
        self._pomis_sets: list[frozenset[str]] | None = None
        self._screening_attempts: int = 0
        self._max_screening_attempts: int = max_screening_attempts
        self._descriptor_names = descriptor_names
        self._archive: MAPElites | None = (
            MAPElites(descriptor_names, minimize=minimize) if descriptor_names else None
        )

    def run_experiment(self, parameters: dict[str, Any]) -> ExperimentResult:
        """Execute a single experiment and log the result."""
        experiment_id = str(uuid.uuid4())[:8]
        logger.info(f"Running experiment {experiment_id} with {parameters}")

        try:
            metrics = self.runner.run(parameters)
            status = self._evaluate_status(metrics)
        except Exception as e:
            logger.error(f"Experiment {experiment_id} crashed: {e}")
            metrics = {self.objective_name: float("inf") if self.minimize else float("-inf")}
            status = ExperimentStatus.CRASH

        result = ExperimentResult(
            experiment_id=experiment_id,
            parameters=parameters,
            metrics=metrics,
            status=status,
            metadata={"phase": self._phase},
        )
        self.log.results.append(result)

        # Fit the off-policy predictor with updated history
        self._predictor.fit(self.log, self.search_space, self.objective_name)

        # Add to MAP-Elites archive if configured (skip crashed experiments)
        if (
            self._archive is not None
            and self._descriptor_names
            and status != ExperimentStatus.CRASH
        ):
            fitness = metrics.get(
                self.objective_name, float("inf") if self.minimize else float("-inf")
            )
            descriptors = self._extract_descriptors(metrics)
            if descriptors:
                self._archive.add(result, fitness, descriptors)

        return result

    def suggest_next(self) -> dict[str, Any]:
        """Suggest the next experiment parameters.

        Uses the current phase to determine strategy:
        - exploration: DoE-based screening
        - optimization: CBO with causal graph
        - exploitation: best-neighborhood search (with MAP-Elites diversity)
        """
        from causal_optimizer.optimizer.suggest import suggest_parameters

        # In exploitation phase, 50% of the time sample from MAP-Elites archive
        if self._phase == "exploitation" and self._archive is not None and self._archive.archive:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                elite = self._archive.sample_elite()
                if elite is not None:
                    logger.info("Sampling from MAP-Elites archive for diversity")
                    return suggest_parameters(
                        search_space=self.search_space,
                        experiment_log=self.log,
                        causal_graph=self.causal_graph,
                        phase=self._phase,
                        minimize=self.minimize,
                        objective_name=self.objective_name,
                        screened_variables=self._screened_focus_variables,
                        base_parameters=elite.parameters,
                    )

        # Only pass pomis_sets during optimization phase (not used in exploitation)
        pomis_sets = self._pomis_sets if self._phase == "optimization" else None
        return suggest_parameters(
            search_space=self.search_space,
            experiment_log=self.log,
            causal_graph=self.causal_graph,
            phase=self._phase,
            minimize=self.minimize,
            objective_name=self.objective_name,
            screened_variables=self._screened_focus_variables,
            pomis_sets=pomis_sets,
        )

    def step(self) -> ExperimentResult:
        """Run one iteration: suggest → check predictor → run → log → update.

        Uses the off-policy predictor to skip experiments that are predicted
        to have poor outcomes (with high confidence). If an experiment is
        skipped, the prediction is noted (but NOT added to the experiment log)
        and a new suggestion is generated, up to max_skips times.
        """
        skips = 0
        while True:
            parameters = self.suggest_next()

            # Check if the predictor recommends observing (not running) this experiment
            if skips < self._max_skips and not self._predictor.should_run_experiment(parameters):
                skips += 1
                # Get predicted outcome for logging purposes (not logged to experiment log)
                prediction = self._predictor.predict(parameters)
                if prediction is not None:
                    logger.info(
                        "Observation (predicted): objective=%.4f "
                        "(skip %d/%d), suggesting new parameters",
                        prediction.expected_value,
                        skips,
                        self._max_skips,
                    )
                else:
                    logger.info(
                        "Off-policy predictor recommends skipping experiment "
                        "(skip %d/%d), suggesting new parameters",
                        skips,
                        self._max_skips,
                    )
                continue

            if skips >= self._max_skips:
                logger.info(
                    f"Reached max skips ({self._max_skips}), "
                    f"running experiment regardless of prediction"
                )

            break

        result = self.run_experiment(parameters)
        self._update_phase()
        return result

    def run_loop(self, n_experiments: int) -> ExperimentLog:
        """Run the full optimization loop for n experiments."""
        if self._predictor.epsilon_mode and n_experiments != self._predictor.n_max:
            logger.warning(
                "epsilon_mode is enabled with n_max=%d but run_loop was called with "
                "n_experiments=%d. For best results, set n_max equal to n_experiments.",
                self._predictor.n_max,
                n_experiments,
            )
        for i in range(n_experiments):
            result = self.step()
            best = self.log.best_result(self.objective_name, self.minimize)
            best_val = best.metrics.get(self.objective_name) if best else None
            logger.info(
                f"[{i + 1}/{n_experiments}] "
                f"status={result.status.value} "
                f"objective={result.metrics.get(self.objective_name):.6f} "
                f"best={best_val}"
            )
        return self.log

    def _is_improvement_significant(self, current_objective: float) -> bool | None:
        """Check if the current objective is a statistically significant improvement.

        Delegates to :attr:`_effect_estimator` via
        :meth:`~causal_optimizer.estimator.effects.EffectEstimator.estimate_improvement`,
        which compares *current_objective* against the distribution of kept
        experiments in the log.

        Returns:
            True if improvement is statistically significant.
            False if the change is within noise.
            None if insufficient data for statistical evaluation (fall back to greedy).
        """
        kept = [
            r.metrics.get(self.objective_name)
            for r in self.log.results
            if r.status == ExperimentStatus.KEEP and r.metrics.get(self.objective_name) is not None
        ]
        discarded = [
            r.metrics.get(self.objective_name)
            for r in self.log.results
            if r.status == ExperimentStatus.DISCARD
            and r.metrics.get(self.objective_name) is not None
        ]

        # Not enough history for statistical evaluation — fall back to greedy.
        # Require at least 5 kept experiments (aligns with estimate_improvement's
        # own greedy-fallback threshold) and 2 discarded for contrast.
        if len(kept) < _MIN_KEPT or len(discarded) < _MIN_DISCARDED:
            return None

        estimate = self._effect_estimator.estimate_improvement(
            experiment_log=self.log,
            current_value=current_objective,
            objective_name=self.objective_name,
            minimize=self.minimize,
        )

        logger.debug(
            "Statistical evaluation via %s: %s",
            estimate.method,
            estimate.summary,
        )

        # The guard above ensures len(kept) >= _MIN_KEPT (2), which matches the
        # estimator's own "insufficient_data" threshold (< 2 kept).  In practice
        # estimate_improvement will never return "insufficient_data" here, but the
        # check is kept as an explicit safety net for future callers who may bypass
        # the guard or change _MIN_KEPT without updating the estimator threshold.
        if estimate.method == "insufficient_data":
            return None

        return estimate.is_significant

    def _evaluate_status(self, metrics: dict[str, float]) -> ExperimentStatus:
        """Determine if this result should be kept.

        Uses bootstrap-based statistical testing when enough history is available.
        Falls back to simple greedy comparison otherwise.
        """
        current_objective = metrics.get(self.objective_name)
        if current_objective is None:
            return ExperimentStatus.CRASH

        best = self.log.best_result(self.objective_name, self.minimize)
        if best is None:
            return ExperimentStatus.KEEP

        # Try statistical evaluation first
        sig_result = self._is_improvement_significant(current_objective)
        if sig_result is not None:
            if sig_result:
                logger.info("Statistical evaluation: significant improvement detected — KEEP")
                return ExperimentStatus.KEEP
            else:
                # Not statistically significant, but still check greedy
                # (the statistical test may be conservative)
                fallback = float("inf") if self.minimize else float("-inf")
                best_objective = best.metrics.get(self.objective_name, fallback)
                is_better = (
                    current_objective < best_objective
                    if self.minimize
                    else current_objective > best_objective
                )
                if is_better:
                    logger.info(
                        "Statistical evaluation: not significant, "
                        "but greedy improvement detected — KEEP"
                    )
                    return ExperimentStatus.KEEP
                else:
                    logger.info("Statistical evaluation: no improvement — DISCARD")
                    return ExperimentStatus.DISCARD

        # Fall back to greedy comparison when insufficient data
        fallback = float("inf") if self.minimize else float("-inf")
        best_objective = best.metrics.get(self.objective_name, fallback)
        if self.minimize:
            is_better = current_objective < best_objective
        else:
            is_better = current_objective > best_objective
        return ExperimentStatus.KEEP if is_better else ExperimentStatus.DISCARD

    def _update_phase(self) -> None:
        """Transition between optimization phases based on experiment count."""
        n = len(self.log.results)
        old_phase = self._phase

        if n < 10:
            self._phase = "exploration"
        elif n < 50:
            self._phase = "optimization"
        else:
            self._phase = "exploitation"

        # Run screening and POMIS when transitioning from exploration to optimization
        # (also covers direct exploration → exploitation if max_screening_attempts is large)
        if old_phase == "exploration" and self._phase in ("optimization", "exploitation"):
            if self._screening_attempts < self._max_screening_attempts:
                self._run_screening()
            else:
                # Max screening retries exceeded — proceed with all variables
                logger.warning(
                    "Max screening attempts (%d) reached with no important variables; "
                    "proceeding with all variables",
                    self._max_screening_attempts,
                )
                self._screened_focus_variables = self.search_space.variable_names
            if self._phase == "optimization":  # screening may have reverted to exploration
                self._compute_pomis()

    def _run_screening(self) -> None:
        """Run screening analysis to identify important variables."""
        self._screening_attempts += 1
        designer = ScreeningDesigner(self.search_space)
        result = designer.screen(self.log, self.objective_name)
        self._screening_result = result

        if result.important_variables:
            self._screened_focus_variables = result.important_variables
            # Reset counter on success so future transitions can screen again
            self._screening_attempts = 0
            logger.info(
                "Screening identified important variables: %s",
                result.important_variables,
            )
        else:
            # No important variables found — extend exploration
            self._phase = "exploration"
            self._screened_focus_variables = None
            logger.info(
                "Screening found no important variables above threshold; "
                "extending exploration phase"
            )

        if result.interactions:
            logger.info(
                "Screening identified interactions: %s",
                list(result.interactions.keys()),
            )

        logger.info("Screening summary:\n%s", result.summary)

    def _compute_pomis(self) -> None:
        """Compute POMIS sets if the causal graph has confounders."""
        if self.causal_graph is None or not self.causal_graph.has_confounders:
            return

        try:
            from causal_optimizer.optimizer.pomis import compute_pomis

            self._pomis_sets = compute_pomis(self.causal_graph, self.objective_name)
            logger.info("POMIS sets: %s", self._pomis_sets)
        except Exception:
            logger.warning(
                "POMIS computation failed or unavailable, continuing without", exc_info=True
            )
            self._pomis_sets = None

    def _extract_descriptors(self, metrics: dict[str, float]) -> dict[str, float]:
        """Extract descriptor values from metrics for MAP-Elites."""
        if not self._descriptor_names:
            return {}
        return {name: metrics[name] for name in self._descriptor_names if name in metrics}
