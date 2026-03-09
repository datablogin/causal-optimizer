"""Core experiment loop orchestrator.

This is the main entry point — it coordinates discovery, design, estimation,
optimization, evolution, prediction, and validation into a unified loop.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Protocol

import numpy as np

from causal_optimizer.designer.screening import ScreeningDesigner, ScreeningResult
from causal_optimizer.evolution.map_elites import MAPElites
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
)

logger = logging.getLogger(__name__)


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
        descriptor_names: list[str] | None = None,
    ) -> None:
        self.search_space = search_space
        self.runner = runner
        self.objective_name = objective_name
        self.minimize = minimize
        self.causal_graph = causal_graph
        self.log = ExperimentLog()
        self._phase: str = "exploration"
        self._screening_result: ScreeningResult | None = None
        self._screened_focus_variables: list[str] | None = None
        self._descriptor_names = descriptor_names
        self._archive: MAPElites | None = (
            MAPElites(descriptor_names, minimize=minimize)
            if descriptor_names
            else None
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
        if (
            self._phase == "exploitation"
            and self._archive is not None
            and self._archive.archive
        ):
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

        return suggest_parameters(
            search_space=self.search_space,
            experiment_log=self.log,
            causal_graph=self.causal_graph,
            phase=self._phase,
            minimize=self.minimize,
            objective_name=self.objective_name,
            screened_variables=self._screened_focus_variables,
        )

    def step(self) -> ExperimentResult:
        """Run one iteration: suggest → run → log → update."""
        parameters = self.suggest_next()
        result = self.run_experiment(parameters)
        self._update_phase()
        return result

    def run_loop(self, n_experiments: int) -> ExperimentLog:
        """Run the full optimization loop for n experiments."""
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

    def _evaluate_status(self, metrics: dict[str, float]) -> ExperimentStatus:
        """Determine if this result should be kept."""
        current_objective = metrics.get(self.objective_name)
        if current_objective is None:
            return ExperimentStatus.CRASH

        best = self.log.best_result
        if best is None:
            return ExperimentStatus.KEEP

        best_objective = best.metrics.get(self.objective_name, float("inf"))
        if self.minimize:
            return ExperimentStatus.KEEP if current_objective < best_objective else ExperimentStatus.DISCARD
        else:
            return ExperimentStatus.KEEP if current_objective > best_objective else ExperimentStatus.DISCARD

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

        # Run screening when transitioning from exploration to optimization
        if old_phase == "exploration" and self._phase == "optimization":
            self._run_screening()

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

    def _extract_descriptors(
        self, metrics: dict[str, float]
    ) -> dict[str, float]:
        """Extract descriptor values from metrics for MAP-Elites."""
        if not self._descriptor_names:
            return {}
        return {
            name: metrics[name]
            for name in self._descriptor_names
            if name in metrics
        }
