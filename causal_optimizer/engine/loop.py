"""Core experiment loop orchestrator.

This is the main entry point — it coordinates discovery, design, estimation,
optimization, evolution, prediction, and validation into a unified loop.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from causal_optimizer.designer.screening import ScreeningDesigner, ScreeningResult
from causal_optimizer.estimator.effects import EffectEstimator
from causal_optimizer.evolution.map_elites import MAPElites
from causal_optimizer.predictor.off_policy import OffPolicyPredictor
from causal_optimizer.types import (
    CausalGraph,
    Constraint,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    ObjectiveSpec,
    ParetoResult,
    SearchSpace,
)
from causal_optimizer.validator.sensitivity import RobustnessReport, SensitivityValidator

if TYPE_CHECKING:
    from causal_optimizer.storage.sqlite import ExperimentStore

logger = logging.getLogger(__name__)

# Minimum requirements before calling estimate_improvement() for keep/discard.
# _MIN_KEPT=5 ensures the estimator uses its full statistical method (bootstrap
# or difference).  With fewer than 5 kept experiments the estimator falls back
# to a greedy comparison (method="greedy"), so calling it from the engine would
# add no statistical value over the engine's own greedy fallback.
_MIN_KEPT = 5
_MIN_DISCARDED = 2

# Phase transition thresholds (by result count)
_EXPLORATION_LIMIT = 10
_OPTIMIZATION_LIMIT = 50


@dataclass(frozen=True)
class ValidationRecord:
    """A single validation result paired with its phase-transition context."""

    report: RobustnessReport
    old_phase: str
    new_phase: str


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

    #: Valid values for the ``strategy`` parameter.
    _VALID_STRATEGIES: frozenset[str] = frozenset({"bayesian", "causal_gp"})

    #: Valid values for the ``effect_method`` parameter.
    _VALID_EFFECT_METHODS: frozenset[str] = frozenset(
        {"difference", "bootstrap", "aipw", "observational"}
    )

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
        effect_method: str = "bootstrap",
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        objectives: list[ObjectiveSpec] | None = None,
        constraints: list[Constraint] | None = None,
        store: ExperimentStore | None = None,
        experiment_id: str | None = None,
        strategy: str = "bayesian",
    ) -> None:
        """Initialize the experiment engine.

        Args:
            seed: Seed for reproducibility of random operations in
                ``OffPolicyPredictor`` and ``EffectEstimator`` bootstrap
                sampling.  Does **not** seed MAP-Elites archive sampling.
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
            effect_method: Method used by :class:`EffectEstimator` to assess
                statistical significance in keep/discard decisions.  Valid
                values are ``"difference"``, ``"bootstrap"`` (default),
                ``"aipw"``, and ``"observational"``.
            confidence_level: Confidence level for statistical tests (default
                0.95 → alpha = 0.05).  Passed directly to
                :class:`~causal_optimizer.estimator.effects.EffectEstimator`.
            n_bootstrap: Number of bootstrap samples used when
                ``effect_method="bootstrap"`` (default 1000).  Passed to
                :class:`~causal_optimizer.estimator.effects.EffectEstimator`.
            strategy: Optimization strategy for the optimization phase.
                ``"bayesian"`` (default) uses Ax/BoTorch; ``"causal_gp"``
                uses the experimental CBO surrogate with separate GPs per
                causal mechanism.  Requires a causal graph; falls back to
                ``"bayesian"`` if no graph is provided.
        """
        if discovery_method is not None and discovery_method not in self._VALID_DISCOVERY_METHODS:
            raise ValueError(
                f"discovery_method={discovery_method!r} is not valid; "
                f"choose one of {sorted(self._VALID_DISCOVERY_METHODS)} or None"
            )
        if discovery_bidir_threshold < discovery_threshold:
            raise ValueError(
                f"discovery_bidir_threshold ({discovery_bidir_threshold!r}) must be >= "
                f"discovery_threshold ({discovery_threshold!r}); "
                "when bidir_threshold < threshold, bidirected edges are unreachable"
            )
        if effect_method not in self._VALID_EFFECT_METHODS:
            raise ValueError(
                f"effect_method={effect_method!r} is not valid; "
                f"choose one of {sorted(self._VALID_EFFECT_METHODS)}"
            )
        if objectives is not None and not any(o.name == objective_name for o in objectives):
            raise ValueError(
                f"objective_name={objective_name!r} not found in objectives "
                f"{[o.name for o in objectives]}; the primary objective must "
                f"appear in the objectives list"
            )
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"strategy={strategy!r} is not valid; "
                f"choose one of {sorted(self._VALID_STRATEGIES)}"
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
        self._strategy: str = strategy
        self._seed: int | None = seed
        self.log = ExperimentLog()
        self._phase: str = "exploration"
        self._effect_method = effect_method
        self._effect_estimator = EffectEstimator(
            method=effect_method,
            causal_graph=causal_graph,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed,
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
        self._validator = SensitivityValidator()
        #: Validation records from phase transitions.  Each record bundles a
        #: :class:`RobustnessReport` with the phase-transition context.
        self.validation_records: list[ValidationRecord] = []

        #: Multi-objective specifications.  If ``None``, single-objective mode.
        self._objectives: list[ObjectiveSpec] | None = objectives
        #: Optimization constraints.  If ``None``, no constraints applied.
        self._constraints: list[Constraint] | None = constraints

        # Persistence support — validate that store and experiment_id are
        # either both provided or both omitted.
        if (store is None) != (experiment_id is None):
            raise ValueError("store and experiment_id must be provided together or both omitted")
        self._store: ExperimentStore | None = store
        self._experiment_id: str | None = experiment_id
        if store is not None and experiment_id is not None:
            store.create_experiment(experiment_id, search_space)

    @classmethod
    def resume(
        cls,
        store: ExperimentStore,
        experiment_id: str,
        runner: ExperimentRunner,
        **engine_kwargs: Any,
    ) -> ExperimentEngine:
        """Resume an interrupted experiment from a persistent store.

        Loads the ``ExperimentLog`` from *store*, reconstructs phase from
        result count, and returns a ready-to-continue engine.
        """
        log = store.load_log(experiment_id)

        # Ensure search_space is passed (required for engine construction)
        if "search_space" not in engine_kwargs:
            raise TypeError("resume() requires 'search_space' keyword argument")

        engine = cls(
            runner=runner,
            store=store,
            experiment_id=experiment_id,
            **engine_kwargs,
        )

        # Restore log and infer phase from result count
        engine.log = log
        n = len(log.results)
        if n < _EXPLORATION_LIMIT:
            engine._phase = "exploration"
        elif n < _OPTIMIZATION_LIMIT:
            engine._phase = "optimization"
        else:
            engine._phase = "exploitation"

        # Refit the off-policy predictor with restored history
        if n > 0:
            engine._predictor.fit(engine.log, engine.search_space, engine.objective_name)

            # Repopulate MAP-Elites archive so exploitation diversity is preserved
            if engine._archive is not None and engine._descriptor_names:
                for past_result in engine.log.results:
                    if past_result.status != ExperimentStatus.CRASH:
                        fitness = past_result.metrics.get(
                            engine.objective_name,
                            float("inf") if engine.minimize else float("-inf"),
                        )
                        descriptors = engine._extract_descriptors(past_result.metrics)
                        if descriptors:
                            engine._archive.add(past_result, fitness, descriptors)

        return engine

    @property
    def validation_results(self) -> list[RobustnessReport]:
        """Convenience accessor: just the reports from all validation records.

        Returns a fresh list on every call.  For phase-transition context, use
        :attr:`validation_records` directly.
        """
        return [r.report for r in self.validation_records]

    @property
    def pomis_sets(self) -> list[frozenset[str]] | None:
        """POMIS intervention sets computed during the optimization phase.

        Read-only: computed internally by :meth:`_compute_pomis` at the
        exploration-to-optimization phase transition. Not user-settable.
        """
        return self._pomis_sets

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

    @property
    def pareto_front(self) -> list[ExperimentResult]:
        """Return the Pareto front of KEEP results.

        If multi-objective mode is active, returns non-dominated results.
        In single-objective mode, falls back to the single best result.
        """
        if self._objectives is not None and len(self._objectives) > 1:
            return self.log.pareto_front(self._objectives)
        # Single-objective: wrap best result as a 1-element list
        objectives = self._objectives or [
            ObjectiveSpec(name=self.objective_name, minimize=self.minimize)
        ]
        return self.log.pareto_front(objectives)

    def run_experiment(self, parameters: dict[str, Any]) -> ExperimentResult:
        """Execute a single experiment and log the result."""
        experiment_id = str(uuid.uuid4())[:8]
        logger.info(f"Running experiment {experiment_id} with {parameters}")

        constraint_violated = False
        try:
            metrics = self.runner.run(parameters)
            status, constraint_violated = self._evaluate_status(metrics)
        except Exception as e:
            logger.error(f"Experiment {experiment_id} crashed: {e}")
            metrics = {self.objective_name: float("inf") if self.minimize else float("-inf")}
            status = ExperimentStatus.CRASH

        metadata: dict[str, Any] = {"phase": self._phase}
        if constraint_violated:
            metadata["constraint_violated"] = True

        result = ExperimentResult(
            experiment_id=experiment_id,
            parameters=parameters,
            metrics=metrics,
            status=status,
            metadata=metadata,
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
            step_count = len(self.log.results)
            flip_seed = (self._seed + step_count) if self._seed is not None else None
            rng = np.random.default_rng(flip_seed)
            if rng.random() < 0.5:
                elite_seed = (self._seed + step_count + 1) if self._seed is not None else None
                elite = self._archive.sample_elite(seed=elite_seed)
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
                        objectives=self._objectives,
                        strategy=self._strategy,
                        seed=self._seed,
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
            objectives=self._objectives,
            strategy=self._strategy,
            seed=self._seed,
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

        # Persist to store if configured
        if self._store is not None and self._experiment_id is not None:
            step_index = len(self.log.results) - 1
            self._store.append_result(self._experiment_id, result, step_index)

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

        # The guard above ensures len(kept) >= _MIN_KEPT (5), which is stricter
        # than the estimator's "insufficient_data" threshold (< 2 kept).  With
        # _MIN_KEPT=5 the estimator always uses a statistical test here, but the
        # check below is kept as a defensive safety net for future callers who
        # may bypass the guard or lower _MIN_KEPT without updating this code.
        if estimate.method == "insufficient_data":
            return None

        return estimate.is_significant

    def _check_constraints(self, metrics: dict[str, float]) -> bool:
        """Return True if all constraints are satisfied."""
        if self._constraints is None:
            return True
        return all(c.is_satisfied(metrics) for c in self._constraints)

    def _evaluate_status(self, metrics: dict[str, float]) -> tuple[ExperimentStatus, bool]:
        """Determine if this result should be kept.

        Uses bootstrap-based statistical testing when enough history is available.
        Falls back to simple greedy comparison otherwise.  Constraints are checked
        first — any violation causes an immediate DISCARD.

        In multi-objective mode, Pareto dominance replaces scalar comparison:
        a result is KEEP if no existing KEEP result dominates it on all objectives.

        Returns:
            A ``(status, constraint_violated)`` tuple.  ``constraint_violated`` is
            ``True`` only when the result was discarded due to a constraint violation.
        """
        # Multi-objective: check that at least one objective metric is present
        if self._objectives is not None and len(self._objectives) > 1:
            if not any(obj.name in metrics for obj in self._objectives):
                return ExperimentStatus.CRASH, False

            # Check constraints first — violated results are always discarded.
            if not self._check_constraints(metrics):
                return ExperimentStatus.DISCARD, True

            return self._evaluate_multi_objective(metrics), False

        # Single-objective: require the primary objective metric
        current_objective = metrics.get(self.objective_name)
        if current_objective is None:
            return ExperimentStatus.CRASH, False

        # Check constraints first — violated results are always discarded.
        if not self._check_constraints(metrics):
            return ExperimentStatus.DISCARD, True

        # Single-objective path (unchanged from before)
        best = self.log.best_result(self.objective_name, self.minimize)
        if best is None:
            return ExperimentStatus.KEEP, False

        # Try statistical evaluation first
        sig_result = self._is_improvement_significant(current_objective)
        if sig_result is not None:
            if sig_result:
                logger.info("Statistical evaluation: significant improvement detected — KEEP")
                return ExperimentStatus.KEEP, False
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
                    return ExperimentStatus.KEEP, False
                else:
                    logger.info("Statistical evaluation: no improvement — DISCARD")
                    return ExperimentStatus.DISCARD, False

        # Fall back to greedy comparison when insufficient data
        fallback = float("inf") if self.minimize else float("-inf")
        best_objective = best.metrics.get(self.objective_name, fallback)
        if self.minimize:
            is_better = current_objective < best_objective
        else:
            is_better = current_objective > best_objective
        status = ExperimentStatus.KEEP if is_better else ExperimentStatus.DISCARD
        return status, False

    def _evaluate_multi_objective(self, metrics: dict[str, float]) -> ExperimentStatus:
        """Evaluate a result using Pareto dominance for multi-objective mode.

        A new result is KEEP if no existing KEEP result dominates it on all
        objectives.  When a new result is accepted, any existing KEEP results
        that the new result dominates are downgraded to DISCARD so that
        ``status == KEEP`` remains consistent with Pareto non-dominance.
        """
        assert self._objectives is not None  # noqa: S101
        # Create a temporary result to test dominance against
        temp = ExperimentResult(
            experiment_id="__temp__",
            parameters={},
            metrics=metrics,
            status=ExperimentStatus.KEEP,
        )
        kept = [r for r in self.log.results if r.status == ExperimentStatus.KEEP]
        for existing in kept:
            if ParetoResult.dominated_by(temp, existing, self._objectives):
                return ExperimentStatus.DISCARD

        # Downgrade existing KEEP results that the new result dominates
        for existing in kept:
            if ParetoResult.dominated_by(existing, temp, self._objectives):
                existing.status = ExperimentStatus.DISCARD
        return ExperimentStatus.KEEP

    def _update_phase(self) -> None:
        """Transition between optimization phases based on experiment count."""
        n = len(self.log.results)
        old_phase = self._phase

        if n < _EXPLORATION_LIMIT:
            self._phase = "exploration"
        elif n < _OPTIMIZATION_LIMIT:
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

        # Validate after screening has confirmed (or skipped) the transition.
        # This ensures reports only describe transitions that actually stuck —
        # screening can revert exploration→optimization back to exploration.
        if old_phase != self._phase:
            self._run_validation(old_phase, self._phase)

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

        _discovery_caveats = {
            "correlation": (
                "heuristic only; bidirected edges are proxies, not identified confounders — "
                "treat results as approximate guidance"
            ),
            "pc": (
                "assumes causal sufficiency and faithfulness; undirected CPDAG edges are "
                "represented as bidirected (conservative placeholder, not identified confounders)"
            ),
            "notears": (
                "continuous optimisation method; may not converge on small or noisy samples"
            ),
        }
        caveat = _discovery_caveats.get(self._discovery_method, "")
        logger.info(
            "Running %r causal discovery (%s)",
            self._discovery_method,
            caveat,
        )

        from causal_optimizer.discovery.graph_learner import GraphLearner

        learner = GraphLearner(
            method=self._discovery_method,
            threshold=self._discovery_threshold,
            bidir_threshold=self._discovery_bidir_threshold,
        )
        try:
            discovered = learner.learn(self.log, objective_name=self.objective_name)
        except (ValueError, ImportError, RuntimeError) as exc:
            # Gracefully degrade on expected failures (bad data, missing dep,
            # algorithm convergence issues).  Other exceptions propagate so
            # programming errors are not silently swallowed.
            logger.error(
                "Auto-discovery failed (%s: %s), continuing without causal graph",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return

        # Validate that the directed-edge subgraph is a DAG (no cycles).
        # Cycles can occur with heuristic discovery methods; a cyclic graph
        # breaks topological ordering used by POMIS computation.
        if self._has_directed_cycle(discovered):
            logger.error(
                "Discovered graph contains directed cycles; discarding to avoid breaking "
                "topological ordering in downstream POMIS computation"
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

            # Rebuild the effect estimator with the active causal graph whenever
            # a graph with edges becomes available from auto-discovery.  Skip empty
            # graphs (0 edges) since they provide no structural information and
            # would silently disable DoWhy's causal identification for observational.
            # Preserve all non-graph settings from the existing estimator so that
            # any future parameters added to EffectEstimator are not silently dropped.
            if self._causal_graph is not None and len(self._causal_graph.edges) > 0:
                self._effect_estimator = EffectEstimator(
                    method=self._effect_method,
                    causal_graph=self._causal_graph,
                    confidence_level=self._effect_estimator.confidence_level,
                    n_bootstrap=self._effect_estimator.n_bootstrap,
                    obs_method=self._effect_estimator.obs_method,
                )
        else:
            # Hybrid mode: user-supplied prior is preserved; discovered graph is informational only
            logger.info(
                "Hybrid mode: prior causal graph retained; discovered graph logged but not applied"
            )

    @staticmethod
    def _has_directed_cycle(graph: CausalGraph) -> bool:
        """Return True if the directed-edge subgraph contains a cycle.

        Uses DFS with a grey/black colouring scheme.  Bidirected edges are
        intentionally ignored because they do not impose causal ordering;
        only ``graph.edges`` (directed edges) are checked.
        """
        adjacency: dict[str, list[str]] = {n: [] for n in graph.nodes}
        for u, v in graph.edges:
            # Unconditionally append; if u is not in nodes, a KeyError is raised
            # so callers learn about malformed graphs rather than silently skipping.
            adjacency[u].append(v)

        # 0 = unvisited, 1 = in-stack (grey), 2 = done (black)
        color: dict[str, int] = {n: 0 for n in graph.nodes}

        def dfs(node: str) -> bool:
            color[node] = 1
            for nbr in adjacency.get(node, []):
                if color[nbr] == 1:
                    return True  # back-edge → cycle
                if color[nbr] == 0 and dfs(nbr):
                    return True
            color[node] = 2
            return False

        return any(dfs(n) for n in graph.nodes if color[n] == 0)

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

    def _run_validation(self, old_phase: str, new_phase: str) -> None:
        """Run sensitivity validation at a confirmed phase transition.

        Compares the first half of all experiments so far (baseline) against
        the second half (improved) to assess whether the optimization trajectory
        is producing robust improvements within the outgoing phase.  This is a
        within-phase trend check, not a cross-phase comparison — no experiments
        from the new phase exist yet at the moment of transition.

        If the result is not robust, a warning is logged but the phase transition
        is not blocked — the engine always makes forward progress.
        """
        results = self.log.results
        if len(results) < 4:
            # Need at least 2 baseline + 2 improved for validation.
            # SensitivityValidator also guards at < 2 per group; this engine-side
            # check avoids the overhead of building ID lists and calling the
            # validator when the result is guaranteed to be "insufficient data."
            return

        midpoint = len(results) // 2
        earlier_ids = [r.experiment_id for r in results[:midpoint]]
        later_ids = [r.experiment_id for r in results[midpoint:]]

        try:
            report = self._validator.validate_improvement(
                experiment_log=self.log,
                baseline_experiments=earlier_ids,
                improved_experiments=later_ids,
                objective_name=self.objective_name,
            )
        except (ValueError, RuntimeError, ArithmeticError):
            logger.warning(
                "Sensitivity validation failed at %s→%s transition, continuing without",
                old_phase,
                new_phase,
                exc_info=True,
            )
            return

        # Verify the effect direction matches the optimization goal.
        # SensitivityValidator uses abs(effect) for SNR/E-value, so is_robust
        # can be True even when the effect goes the wrong way (regression).
        # Note: effect_size == 0.0 is treated as "not improving" (no change).
        improving = (report.effect_size < 0) if self.minimize else (report.effect_size > 0)
        if report.is_robust and not improving:
            logger.warning(
                "Validation at %s→%s: effect direction is wrong "
                "(effect=%.4f, minimize=%s); treating as non-robust",
                old_phase,
                new_phase,
                report.effect_size,
                self.minimize,
            )
            # Override: a "robust" regression is not a real improvement.
            report = replace(
                report,
                is_robust=False,
                summary=(
                    f"Effect direction is wrong for optimization goal "
                    f"(effect={report.effect_size:.6f}, minimize={self.minimize}); "
                    f"originally: {report.summary}"
                ),
            )

        self.validation_records.append(
            ValidationRecord(report=report, old_phase=old_phase, new_phase=new_phase)
        )

        if report.is_robust:
            logger.info(
                "Validation at %s→%s transition: optimization trend is robust (%s)",
                old_phase,
                new_phase,
                report.summary,
            )
        else:
            logger.warning(
                "Validation at %s→%s transition: optimization trend is NOT robust (%s); "
                "continuing with phase transition",
                old_phase,
                new_phase,
                report.summary,
            )

    def _extract_descriptors(self, metrics: dict[str, float]) -> dict[str, float]:
        """Extract descriptor values from metrics for MAP-Elites."""
        if not self._descriptor_names:
            return {}
        return {name: metrics[name] for name in self._descriptor_names if name in metrics}
