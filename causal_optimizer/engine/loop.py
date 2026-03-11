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

    #: Valid values for the ``discovery_method`` parameter.
    _VALID_DISCOVERY_METHODS: frozenset[str] = frozenset({"correlation", "pc", "notears"})

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
        discovery_method: str | None = None,
        discovery_threshold: float = 0.3,
        discovery_bidir_threshold: float = 0.7,
    ) -> None:
        """Initialize the experiment engine.

        Args:
            seed: Seed for the epsilon controller's RNG only. Controls
                reproducibility of observe/intervene decisions in
                ``OffPolicyPredictor``. Does **not** seed other random
                sources in the engine (MAP-Elites sampling, bootstrap CI).
            discovery_method: Algorithm used to learn a causal graph from
                experiment data at the exploration→optimization phase
                transition.  Valid values are ``"correlation"``, ``"pc"``,
                and ``"notears"``.  Set to ``None`` (default) to disable
                auto-discovery (backward-compatible).  When a *causal_graph*
                is also provided the discovered graph is logged but the prior
                graph is **not** replaced (hybrid mode).
            discovery_threshold: Correlation threshold forwarded to
                :class:`~causal_optimizer.discovery.graph_learner.GraphLearner`.
                Variable pairs with |r| ≤ this value are not connected.
                Defaults to ``0.3`` (the ``GraphLearner`` default).
            discovery_bidir_threshold: For the correlation method, pairs of
                non-outcome variables with |r| above *this* value get a
                bidirected edge (X ↔ Y) rather than a directed edge.  Forwarded
                to ``GraphLearner(bidir_threshold=...)``.  Defaults to ``0.7``.
        """
        if discovery_method is not None and discovery_method not in self._VALID_DISCOVERY_METHODS:
            raise ValueError(
                f"discovery_method={discovery_method!r} is not valid; "
                f"choose one of {sorted(self._VALID_DISCOVERY_METHODS)} or None"
            )

        self.search_space = search_space
        self.runner = runner
        self.objective_name = objective_name
        self.minimize = minimize
        # _causal_graph is the single source of truth for the active graph.
        # self.causal_graph is kept as a public alias that stays in sync.
        self._causal_graph: CausalGraph | None = causal_graph
        # Separate reference to the user-supplied prior so _run_auto_discovery can
        # distinguish "no prior was given" from "auto-discovery already ran once".
        # This is important when _run_screening reverts the phase to exploration and
        # _run_auto_discovery is called again: without this distinction, the
        # auto-discovered graph from the first call would be treated as a user prior
        # and block re-discovery from the richer dataset.
        self._prior_causal_graph: CausalGraph | None = causal_graph
        self._discovery_method: str | None = discovery_method
        self._discovery_threshold: float = discovery_threshold
        self._discovery_bidir_threshold: float = discovery_bidir_threshold
        self._discovered_graph: CausalGraph | None = None
        self.log = ExperimentLog()
        self._phase: str = "exploration"
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

    @property
    def causal_graph(self) -> CausalGraph | None:
        """The active causal graph (user-supplied prior or auto-discovered)."""
        return self._causal_graph

    @causal_graph.setter
    def causal_graph(self, graph: CausalGraph | None) -> None:
        self._causal_graph = graph
        # Keep _prior_causal_graph in sync so that post-construction assignments
        # (e.g. engine.causal_graph = domain_graph) are treated as user priors
        # and protect against being overwritten by _run_auto_discovery.
        self._prior_causal_graph = graph

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
                        causal_graph=self._causal_graph,
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
            causal_graph=self._causal_graph,
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
            # Auto-discover causal graph from data before screening
            self._run_auto_discovery()

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

    def _run_auto_discovery(self) -> None:
        """Discover a causal graph from experiment data if ``discovery_method`` is set.

        Two operating modes:

        - **No user prior** (``self._prior_causal_graph is None``): the discovered
          graph becomes the active graph used for optimization.  If discovery runs
          again after a screening-induced phase revert, the previously auto-discovered
          graph is overwritten with the richer dataset.
        - **Hybrid mode** (user supplied a prior graph): the discovered graph is
          computed and logged for informational purposes, but the prior graph is
          *not* replaced.
        """
        if self._discovery_method is None:
            return

        from causal_optimizer.discovery.graph_learner import GraphLearner

        learner = GraphLearner(
            method=self._discovery_method,
            threshold=self._discovery_threshold,
            bidir_threshold=self._discovery_bidir_threshold,
        )
        try:
            discovered = learner.learn(self.log, objective_name=self.objective_name)
        except Exception as exc:
            logger.error(
                "Auto-discovery failed (%s: %s), continuing without causal graph",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return

        self._discovered_graph = discovered

        n_nodes = len(discovered.nodes)
        n_edges = len(discovered.edges)
        n_bidir = len(discovered.bidirected_edges)
        logger.info(
            "Discovered causal graph with %d nodes, %d edges (%d bidirected)",
            n_nodes,
            n_edges,
            n_bidir,
        )

        if self._prior_causal_graph is None:
            # No user-supplied prior — use the discovered graph going forward.
            # This also handles the re-discovery case after a screening revert:
            # the new graph (with more samples) replaces the old auto-discovered one.
            # Assign directly to _causal_graph (bypassing the property setter) so
            # that _prior_causal_graph stays None — the auto-discovered graph must
            # not be promoted to a user prior, otherwise re-discovery after a
            # screening revert would enter hybrid mode and stop updating the graph.
            self._causal_graph = discovered
        else:
            # Hybrid mode: user-supplied prior is preserved; discovered graph is informational only
            logger.info(
                "Hybrid mode: prior causal graph retained; discovered graph logged but not applied"
            )

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
        if self._causal_graph is None or not self._causal_graph.has_confounders:
            return

        try:
            from causal_optimizer.optimizer.pomis import compute_pomis

            self._pomis_sets = compute_pomis(self._causal_graph, self.objective_name)
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
