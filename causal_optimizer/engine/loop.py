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

# Minimum requirements for statistical evaluation
_MIN_EXPERIMENTS = 5
_MIN_KEPT = 2
_MIN_DISCARDED = 2
_N_BOOTSTRAP = 1000
_ALPHA_EARLY = 0.1  # permissive threshold for < 20 experiments
_ALPHA_LATE = 0.05  # stricter threshold for >= 20 experiments


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
        descriptor_names: list[str] | None = None,
        epsilon_mode: bool = False,
        n_max: int = 100,
        seed: int | None = None,
    ) -> None:
        self.search_space = search_space
        self.runner = runner
        self.objective_name = objective_name
        self.minimize = minimize
        self.causal_graph = causal_graph
        self.log = ExperimentLog()
        self._phase: str = "exploration"
        self._predictor = OffPolicyPredictor(epsilon_mode=epsilon_mode, n_max=n_max, seed=seed)
        self._max_skips = max_skips
        self._screening_result: ScreeningResult | None = None
        self._screened_focus_variables: list[str] | None = None
        self._pomis_sets: list[frozenset[str]] | None = None
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

        # Add to MAP-Elites archive if configured
        if self._archive is not None and self._descriptor_names:
            fitness = metrics.get(self.objective_name, float("inf"))
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
        skipped, a new suggestion is generated, up to max_skips times.
        """
        skips = 0
        while True:
            parameters = self.suggest_next()

            # Check if the predictor recommends skipping this experiment
            if skips < self._max_skips and not self._predictor.should_run_experiment(parameters):
                skips += 1
                logger.info(
                    f"Off-policy predictor recommends skipping experiment "
                    f"(skip {skips}/{self._max_skips}), suggesting new parameters"
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
            best = self.log.best_result
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

        Uses bootstrap confidence intervals on the distribution of kept objectives
        to determine if the current value represents a real improvement over noise.

        Returns:
            True if improvement is statistically significant.
            False if the change is within noise.
            None if insufficient data for statistical evaluation (fall back to greedy).
        """
        n_total = len(self.log.results)
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

        # Not enough history for statistical evaluation
        if n_total < _MIN_EXPERIMENTS or len(kept) < _MIN_KEPT or len(discarded) < _MIN_DISCARDED:
            return None

        kept_arr = np.array(kept)
        best_objective = float(np.min(kept_arr) if self.minimize else np.max(kept_arr))

        # Bootstrap the difference: current_objective - best_objective
        # For minimization, a negative difference means improvement
        observed_diff = current_objective - best_objective

        rng = np.random.default_rng(seed=None)
        bootstrap_diffs = np.empty(_N_BOOTSTRAP)
        for i in range(_N_BOOTSTRAP):
            boot_sample = rng.choice(kept_arr, size=len(kept_arr), replace=True)
            boot_best = float(np.min(boot_sample) if self.minimize else np.max(boot_sample))
            bootstrap_diffs[i] = current_objective - boot_best

        # Adaptive alpha: permissive early, stricter later
        alpha = _ALPHA_EARLY if n_total < 20 else _ALPHA_LATE

        if self.minimize:
            # For minimization: improvement means current < best (negative diff)
            # Check if the observed diff is significantly negative
            # (below the lower bound of the bootstrap CI)
            threshold = float(np.percentile(bootstrap_diffs, 100 * alpha))
            is_significant = observed_diff < threshold
        else:
            # For maximization: improvement means current > best (positive diff)
            # Check if the observed diff is significantly positive
            # (above the upper bound of the bootstrap CI)
            threshold = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha)))
            is_significant = observed_diff > threshold

        logger.debug(
            f"Statistical evaluation: diff={observed_diff:.6f}, "
            f"threshold={threshold:.6f}, alpha={alpha}, significant={is_significant}"
        )

        return is_significant

    def _evaluate_status(self, metrics: dict[str, float]) -> ExperimentStatus:
        """Determine if this result should be kept.

        Uses bootstrap-based statistical testing when enough history is available.
        Falls back to simple greedy comparison otherwise.
        """
        current_objective = metrics.get(self.objective_name)
        if current_objective is None:
            return ExperimentStatus.CRASH

        best = self.log.best_result
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
                best_objective = best.metrics.get(self.objective_name, float("inf"))
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
        best_objective = best.metrics.get(self.objective_name, float("inf"))
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
        if old_phase == "exploration" and self._phase == "optimization":
            self._run_screening()
            if self._phase == "optimization":  # screening may have reverted to exploration
                self._compute_pomis()

    def _run_screening(self) -> None:
        """Run screening analysis to identify important variables."""
        designer = ScreeningDesigner(self.search_space)
        result = designer.screen(self.log, self.objective_name)
        self._screening_result = result

        if result.important_variables:
            self._screened_focus_variables = result.important_variables
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
