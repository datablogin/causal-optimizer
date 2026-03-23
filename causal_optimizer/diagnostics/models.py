"""Pydantic output models for diagnostic reports and recommendations."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RecommendationType(str, Enum):
    """Type of research recommendation."""

    EXPLOIT = "exploit"  # Deepen search in a promising region
    EXPLORE = "explore"  # Investigate untested variables/regions
    DROP = "drop"  # Stop varying low-signal variables
    PIVOT = "pivot"  # Change strategy (multi-obj, new graph, etc.)


class ConfidenceLevel(str, Enum):
    """Confidence level for a recommendation."""

    HIGH = "high"  # Backed by causal structure + statistical evidence
    MEDIUM = "medium"  # Backed by one source
    LOW = "low"  # Heuristic only


class VariableSignalClass(str, Enum):
    """Classification of a variable's signal strength."""

    HIGH_SIGNAL = "high_signal"
    LOW_SIGNAL = "low_signal"
    UNTESTED = "untested"


class VariableSignalReport(BaseModel):
    """Signal analysis for a single variable."""

    variable_name: str
    signal_class: VariableSignalClass
    importance_score: float | None = None
    effect_estimate: float | None = None
    effect_significant: bool | None = None
    n_experiments_varied: int = 0
    value_range_explored: tuple[float, float] | None = None


class VariableSignalAnalysis(BaseModel):
    """Aggregate variable signal analysis."""

    variables: list[VariableSignalReport]
    high_signal_count: int = 0
    low_signal_count: int = 0
    untested_count: int = 0


class ConvergenceAnalysis(BaseModel):
    """Analysis of optimization convergence behavior."""

    plateaued: bool
    improvement_rate: float
    improvement_rate_early: float
    improvement_rate_late: float
    best_objective: float
    best_at_step: int
    budget_remaining_fraction: float = 0.0
    steps_since_improvement: int
    abandoned_climb: bool


class CoverageAnalysis(BaseModel):
    """Analysis of search space and causal structure coverage.

    Ancestor and POMIS coverage use all non-crash experiments (KEEP + DISCARD).
    ``kept_varied_vars`` reports the narrower KEEP-only subset for consumers
    that need to distinguish retained-frontier coverage from exploration coverage.
    """

    pomis_sets_total: int | None = None
    pomis_sets_explored: int | None = None
    pomis_sets_unexplored: list[list[str]] | None = None
    ancestor_variables: list[str] | None = None
    ancestors_intervened: list[str] | None = None
    ancestors_never_intervened: list[str] | None = None
    map_elites_coverage: float | None = None
    map_elites_filled_cells: int | None = None
    map_elites_total_cells: int | None = None
    search_space_coverage: float | None = None
    kept_varied_vars: list[str] | None = Field(
        default=None,
        description=(
            "Variables with variation across KEEP-only experiments. "
            "None when no KEEP experiments exist; empty list when KEEP "
            "experiments exist but all variables are constant."
        ),
    )


class RobustnessAnalysis(BaseModel):
    """Analysis of result robustness and statistical reliability."""

    best_result_robust: bool
    signal_to_noise: float
    e_value: float
    effect_size: float
    summary: str
    top_k_consistency: float


class ObservationalVariableReport(BaseModel):
    """Observational analysis for a single variable."""

    variable_name: str
    identifiable: bool
    identification_method: str | None = None
    obs_estimate: float | None = None
    obs_ci: tuple[float, float] | None = None
    exp_estimate: float | None = None
    agreement: float | None = None


class ObservationalAnalysis(BaseModel):
    """Aggregate observational signal analysis."""

    n_identifiable: int
    n_variables: int
    variables: list[ObservationalVariableReport]
    obs_experimental_agreement: float | None = None
    recommendation: str


class Recommendation(BaseModel):
    """A ranked research recommendation."""

    rank: int
    rec_type: RecommendationType
    confidence: ConfidenceLevel
    title: str
    description: str
    evidence: list[str]
    next_step: str
    expected_info_gain: float
    variables_involved: list[str] = Field(default_factory=list)


class DiagnosticReport(BaseModel):
    """Complete diagnostic report with analyses and recommendations."""

    experiment_id: str | None = None
    n_experiments: int
    current_phase: str
    variable_signal: VariableSignalAnalysis
    convergence: ConvergenceAnalysis
    coverage: CoverageAnalysis
    robustness: RobustnessAnalysis
    observational: ObservationalAnalysis | None = None
    recommendations: list[Recommendation]

    def summary(self) -> str:
        """Human-readable text summary for CLI output."""
        lines: list[str] = []
        lines.append(
            f"Diagnostic Report ({self.n_experiments} experiments, phase: {self.current_phase})"
        )
        lines.append("")

        # Variable signal
        vs = self.variable_signal
        lines.append(
            f"Variables: {vs.high_signal_count} high-signal, "
            f"{vs.low_signal_count} low-signal, {vs.untested_count} untested"
        )

        # Convergence
        c = self.convergence
        if c.plateaued:
            lines.append(
                f"Convergence: PLATEAUED (best {c.best_objective:.4f} at step {c.best_at_step})"
            )
        elif c.abandoned_climb:
            lines.append(
                f"Convergence: ABANDONED CLIMB (still improving, "
                f"best {c.best_objective:.4f} at step {c.best_at_step})"
            )
        else:
            lines.append(f"Convergence: best {c.best_objective:.4f} at step {c.best_at_step}")

        # Coverage
        cov = self.coverage
        if cov.pomis_sets_total is not None:
            lines.append(
                f"POMIS coverage: {cov.pomis_sets_explored}/{cov.pomis_sets_total} sets explored"
            )
        if cov.ancestors_never_intervened:
            lines.append(f"Untested ancestors: {', '.join(cov.ancestors_never_intervened)}")
        if cov.map_elites_coverage is not None:
            lines.append(f"MAP-Elites coverage: {cov.map_elites_coverage:.0%}")
        if cov.search_space_coverage is not None:
            lines.append(f"Search space coverage: {cov.search_space_coverage:.0%}")

        # Observational
        if self.observational is not None:
            obs = self.observational
            lines.append(f"Observational: {obs.n_identifiable}/{obs.n_variables} identifiable")
            if obs.obs_experimental_agreement is not None:
                lines.append(f"Obs-experimental agreement: {obs.obs_experimental_agreement:.0%}")
            lines.append(f"Observational recommendation: {obs.recommendation}")

        # Robustness
        r = self.robustness
        robust_label = "ROBUST" if r.best_result_robust else "NOT ROBUST"
        lines.append(f"Robustness: {robust_label} ({r.summary})")

        # Recommendations
        if self.recommendations:
            lines.append("")
            lines.append("Research Directions:")
            for rec in self.recommendations:
                lines.append(
                    f"  #{rec.rank} [{rec.rec_type.value.upper()}] {rec.title} "
                    f"(confidence: {rec.confidence.value})"
                )
                for ev in rec.evidence:
                    lines.append(f"       - {ev}")
                lines.append(f"       Next: {rec.next_step}")
        else:
            lines.append("")
            lines.append("No actionable recommendations.")

        return "\n".join(lines)
