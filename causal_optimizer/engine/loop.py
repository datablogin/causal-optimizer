"""Core experiment loop orchestrator.

This is the main entry point — it coordinates discovery, design, estimation,
optimization, evolution, prediction, and validation into a unified loop.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Protocol

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
    ) -> None:
        self.search_space = search_space
        self.runner = runner
        self.objective_name = objective_name
        self.minimize = minimize
        self.causal_graph = causal_graph
        self.log = ExperimentLog()
        self._phase: str = "exploration"

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
        return result

    def suggest_next(self) -> dict[str, Any]:
        """Suggest the next experiment parameters.

        Uses the current phase to determine strategy:
        - exploration: DoE-based screening
        - optimization: CBO with causal graph
        - exploitation: best-neighborhood search
        """
        from causal_optimizer.optimizer.suggest import suggest_parameters

        return suggest_parameters(
            search_space=self.search_space,
            experiment_log=self.log,
            causal_graph=self.causal_graph,
            phase=self._phase,
            minimize=self.minimize,
            objective_name=self.objective_name,
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
        if n < 10:
            self._phase = "exploration"
        elif n < 50:
            self._phase = "optimization"
        else:
            self._phase = "exploitation"
