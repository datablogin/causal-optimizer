"""Research advisor — orchestrates analyses and synthesizes recommendations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from causal_optimizer.diagnostics.convergence import analyze_convergence
from causal_optimizer.diagnostics.coverage import analyze_coverage
from causal_optimizer.diagnostics.models import (
    ConfidenceLevel,
    ConvergenceAnalysis,
    CoverageAnalysis,
    DiagnosticReport,
    Recommendation,
    RecommendationType,
    RobustnessAnalysis,
    VariableSignalAnalysis,
    VariableSignalClass,
)
from causal_optimizer.diagnostics.robustness import analyze_robustness
from causal_optimizer.diagnostics.variable_signal import analyze_variable_signal

if TYPE_CHECKING:
    from causal_optimizer.engine.loop import ExperimentEngine
    from causal_optimizer.evolution.map_elites import MAPElites
    from causal_optimizer.types import CausalGraph, ExperimentLog, SearchSpace

logger = logging.getLogger(__name__)


class ResearchAdvisor:
    """Analyze experiment runs and produce actionable research recommendations.

    Usage::

        advisor = ResearchAdvisor(objective_name="yield", minimize=False)
        report = advisor.analyze(engine)
        print(report.summary())

    Or from a raw log (e.g., loaded from SQLite)::

        report = advisor.analyze_from_log(log, search_space)
    """

    def __init__(
        self,
        objective_name: str = "objective",
        minimize: bool = True,
        total_budget: int | None = None,
    ) -> None:
        self.objective_name = objective_name
        self.minimize = minimize
        self.total_budget = total_budget

    def analyze(self, engine: ExperimentEngine) -> DiagnosticReport:
        """Produce a full diagnostic report from a completed or in-progress engine."""
        return self.analyze_from_log(
            experiment_log=engine.log,
            search_space=engine.search_space,
            causal_graph=engine.causal_graph,
            pomis_sets=engine.pomis_sets,
            archive=engine.archive,
            phase=engine.phase,
            experiment_id=engine.experiment_id,
        )

    def analyze_from_log(
        self,
        experiment_log: ExperimentLog,
        search_space: SearchSpace,
        causal_graph: CausalGraph | None = None,
        pomis_sets: list[frozenset[str]] | None = None,
        archive: MAPElites | None = None,
        phase: str = "unknown",
        experiment_id: str | None = None,
    ) -> DiagnosticReport:
        """Produce a diagnostic report from raw components."""
        signal = analyze_variable_signal(
            experiment_log, search_space, self.objective_name, self.minimize, causal_graph
        )
        convergence = analyze_convergence(
            experiment_log, self.objective_name, self.minimize, self.total_budget
        )
        coverage = analyze_coverage(
            experiment_log, search_space, self.objective_name, causal_graph, pomis_sets, archive
        )
        robustness = analyze_robustness(
            experiment_log, self.objective_name, self.minimize, search_space
        )
        recommendations = _synthesize_recommendations(
            signal, convergence, coverage, robustness, search_space
        )

        return DiagnosticReport(
            experiment_id=experiment_id,
            n_experiments=len(experiment_log.results),
            current_phase=phase,
            variable_signal=signal,
            convergence=convergence,
            coverage=coverage,
            robustness=robustness,
            recommendations=recommendations,
        )


def _synthesize_recommendations(
    signal: VariableSignalAnalysis,
    convergence: ConvergenceAnalysis,
    coverage: CoverageAnalysis,
    robustness: RobustnessAnalysis,
    search_space: SearchSpace,
) -> list[Recommendation]:
    """Generate ranked recommendations from all analyses."""
    recs: list[Recommendation] = []

    # --- EXPLORE: untested causal ancestors ---
    if coverage.ancestors_never_intervened:
        for var in coverage.ancestors_never_intervened:
            recs.append(
                Recommendation(
                    rank=0,
                    rec_type=RecommendationType.EXPLORE,
                    confidence=ConfidenceLevel.HIGH,
                    title=f"Explore {var}: causal ancestor never tested",
                    description=(
                        f"Variable {var} is a causal ancestor of the objective but was never "
                        f"varied across experiments. It may have a significant effect."
                    ),
                    evidence=[
                        f"{var} is in the causal graph as an ancestor of the objective",
                        f"{var} was held constant across all experiments",
                    ],
                    next_step=(
                        f"Run 5-10 experiments varying {var} while holding others at best values"
                    ),
                    expected_info_gain=0.9,
                    variables_involved=[var],
                )
            )

    # --- EXPLORE: unexplored POMIS sets ---
    if coverage.pomis_sets_unexplored:
        for pset in coverage.pomis_sets_unexplored:
            vars_str = ", ".join(pset)
            recs.append(
                Recommendation(
                    rank=0,
                    rec_type=RecommendationType.EXPLORE,
                    confidence=ConfidenceLevel.HIGH,
                    title=f"Explore POMIS set {{{vars_str}}}",
                    description=(
                        f"The intervention set {{{vars_str}}} is a POMIS-optimal set but was "
                        f"never fully explored. These variables together may unlock better results."
                    ),
                    evidence=[
                        f"POMIS analysis identified {{{vars_str}}} as a minimal intervention set",
                        "Not all variables in this set were varied together",
                    ],
                    next_step=f"Run experiments varying {vars_str} jointly",
                    expected_info_gain=0.85,
                    variables_involved=list(pset),
                )
            )

    # --- EXPLOIT: abandoned climb ---
    if convergence.abandoned_climb:
        ratio = abs(convergence.improvement_rate_late) / max(
            abs(convergence.improvement_rate_early), 1e-10
        )
        gain = min(0.8, 0.8 * ratio)
        recs.append(
            Recommendation(
                rank=0,
                rec_type=RecommendationType.EXPLOIT,
                confidence=ConfidenceLevel.MEDIUM,
                title="Continue optimization: improvement still active",
                description=(
                    f"The objective was still improving when the run ended "
                    f"(late improvement rate: {convergence.improvement_rate_late:.4f}). "
                    f"Additional budget may yield better results."
                ),
                evidence=[
                    f"Late improvement rate: {convergence.improvement_rate_late:.4f}",
                    f"Best found at step {convergence.best_at_step}",
                    f"{convergence.steps_since_improvement} steps since last improvement",
                ],
                next_step="Resume the experiment with additional budget (10-20 more steps)",
                expected_info_gain=gain,
            )
        )

    # --- EXPLORE: low MAP-Elites coverage ---
    if coverage.map_elites_coverage is not None and coverage.map_elites_coverage < 0.5:
        gain = 0.7 * (1.0 - coverage.map_elites_coverage)
        recs.append(
            Recommendation(
                rank=0,
                rec_type=RecommendationType.EXPLORE,
                confidence=ConfidenceLevel.LOW,
                title=f"Expand diversity: MAP-Elites {coverage.map_elites_coverage:.0%} filled",
                description=(
                    f"The MAP-Elites diversity archive is only {coverage.map_elites_coverage:.0%} "
                    f"filled. Unexplored behavioral regions may contain better solutions."
                ),
                evidence=[
                    *(
                        [
                            f"Archive: {coverage.map_elites_filled_cells}"
                            f"/{coverage.map_elites_total_cells} cells"
                        ]
                        if coverage.map_elites_total_cells is not None
                        else []
                    ),
                    f"Coverage: {coverage.map_elites_coverage:.1%}",
                ],
                next_step="Run more exploitation steps to fill the diversity archive",
                expected_info_gain=gain,
            )
        )

    # --- PIVOT: plateau detected ---
    if convergence.plateaued:
        recs.append(
            Recommendation(
                rank=0,
                rec_type=RecommendationType.PIVOT,
                confidence=ConfidenceLevel.MEDIUM,
                title="Objective has plateaued",
                description=(
                    f"No meaningful improvement in the late phase of optimization "
                    f"(late improvement rate: {convergence.improvement_rate_late:.6f}). "
                    f"Consider changing strategy or exploring a different objective."
                ),
                evidence=[
                    f"Best at step {convergence.best_at_step} of "
                    f"{convergence.best_at_step + convergence.steps_since_improvement}",
                    f"{convergence.steps_since_improvement} steps without improvement",
                    f"Late improvement rate near zero: {convergence.improvement_rate_late:.6f}",
                ],
                next_step=(
                    "Consider multi-objective optimization, a different causal graph, "
                    "or relaxing constraints"
                ),
                expected_info_gain=0.65,
            )
        )

    # --- PIVOT: best result not robust ---
    if not robustness.best_result_robust and robustness.signal_to_noise > 0:
        gain = min(0.6, 0.6 * (1.0 - robustness.signal_to_noise / 3.0))
        gain = max(0.1, gain)
        recs.append(
            Recommendation(
                rank=0,
                rec_type=RecommendationType.PIVOT,
                confidence=ConfidenceLevel.MEDIUM,
                title="Best result may not be robust",
                description=(
                    f"Sensitivity analysis indicates the best result may be noise "
                    f"(SNR: {robustness.signal_to_noise:.2f}, e-value: {robustness.e_value:.2f})."
                ),
                evidence=[
                    f"Signal-to-noise ratio: {robustness.signal_to_noise:.2f}",
                    f"E-value: {robustness.e_value:.2f}",
                    f"Top-K consistency: {robustness.top_k_consistency:.2f}",
                ],
                next_step="Increase sample size or use more robust evaluation (bootstrap CI)",
                expected_info_gain=gain,
            )
        )

    # --- PIVOT: top-K inconsistency ---
    # Skip only when robustness had insufficient data (effect_size==0 and snr==0)
    has_robustness_data = robustness.effect_size != 0.0 or robustness.signal_to_noise != 0.0
    if robustness.top_k_consistency < 0.3 and has_robustness_data:
        gain = 0.55 * (1.0 - robustness.top_k_consistency)
        recs.append(
            Recommendation(
                rank=0,
                rec_type=RecommendationType.PIVOT,
                confidence=ConfidenceLevel.LOW,
                title="Top results disagree on parameter values",
                description=(
                    f"The top 3 results have low parameter similarity "
                    f"({robustness.top_k_consistency:.0%}), suggesting the landscape "
                    f"is noisy or multimodal."
                ),
                evidence=[
                    f"Top-K parameter consistency: {robustness.top_k_consistency:.0%}",
                ],
                next_step="Increase exploration budget or add a causal graph to focus the search",
                expected_info_gain=gain,
            )
        )

    # --- DROP: low-signal variables ---
    for var_report in signal.variables:
        if var_report.signal_class != VariableSignalClass.LOW_SIGNAL:
            continue
        importance = var_report.importance_score or 0.0
        gain = 0.5 * (1.0 - importance)
        recs.append(
            Recommendation(
                rank=0,
                rec_type=RecommendationType.DROP,
                confidence=(
                    ConfidenceLevel.MEDIUM
                    if var_report.effect_significant is False
                    else ConfidenceLevel.LOW
                ),
                title=f"Drop {var_report.variable_name}: no signal detected",
                description=(
                    f"Variable {var_report.variable_name} shows low importance "
                    f"(score: {importance:.4f}) and no statistically significant effect "
                    f"across {var_report.n_experiments_varied} experiments."
                ),
                evidence=[
                    f"Screening importance: {importance:.4f}",
                    f"Effect significant: {var_report.effect_significant}",
                    f"Experiments varied: {var_report.n_experiments_varied}",
                ],
                next_step=(
                    f"Fix {var_report.variable_name} at its best-known value to reduce search space"
                ),
                expected_info_gain=gain,
                variables_involved=[var_report.variable_name],
            )
        )

    # Sort by expected_info_gain descending and assign ranks
    recs.sort(key=lambda r: r.expected_info_gain, reverse=True)
    for i, rec in enumerate(recs):
        rec.rank = i + 1

    return recs
