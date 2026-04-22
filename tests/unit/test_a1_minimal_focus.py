"""Tests for the Sprint 37 A1 minimal-focus heuristic and engine flag.

The A1 contract (Sprint 36 recommendation, Sprint 37 issue #197):

1. Add an explicit engine flag ``pomis_minimal_focus`` defaulting to ``False``.
2. When the flag is **off**, behavior is preserved exactly.
3. When the flag is **on** and graph ancestors cover the whole search space,
   allow focus restriction to ``screened_variables ∩ ancestors`` -- but
   **only if** that intersection is a non-empty proper subset of the
   search space.
4. Otherwise, fall back to the current ancestor/full-space behavior.
5. The same rule applies in both optimization and exploitation.

These tests pin the contract before the implementation lands.  They use the
shared helper :func:`_apply_minimal_focus_a1` and the threaded engine flag.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.optimizer.suggest import (
    _apply_minimal_focus_a1,
    suggest_parameters,
)
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="z", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
    )


def _full_ancestor_graph() -> CausalGraph:
    """Graph where every search variable is a direct parent of the objective."""
    return CausalGraph(
        edges=[("x", "objective"), ("y", "objective"), ("z", "objective")],
    )


def _partial_ancestor_graph() -> CausalGraph:
    """Graph where only some search variables are ancestors of the objective."""
    return CausalGraph(edges=[("x", "objective"), ("y", "objective")])


def _make_log(n: int = 5, phase: str | None = None) -> ExperimentLog:
    import numpy as np

    rng = np.random.default_rng(42)
    results = []
    for i in range(n):
        params = {v: float(rng.uniform(0, 10)) for v in ("x", "y", "z")}
        metadata: dict[str, Any] = {}
        if phase is not None:
            metadata["phase"] = phase
        results.append(
            ExperimentResult(
                experiment_id=str(i),
                parameters=params,
                metrics={"objective": params["x"] + params["y"]},
                status=ExperimentStatus.KEEP,
                metadata=metadata,
            )
        )
    return ExperimentLog(results=results)


# ── Helper-level tests ───────────────────────────────────────────────


class TestApplyMinimalFocusA1:
    def test_disabled_returns_base_focus_unchanged(self) -> None:
        ss = _make_search_space()
        graph = _full_ancestor_graph()
        base = ["x", "y", "z"]
        result = _apply_minimal_focus_a1(
            base_focus=base,
            search_space=ss,
            causal_graph=graph,
            objective_name="objective",
            screened_variables=["x"],
            enable=False,
        )
        assert result == base

    def test_no_graph_returns_base_focus_unchanged(self) -> None:
        ss = _make_search_space()
        base = ["x", "y", "z"]
        result = _apply_minimal_focus_a1(
            base_focus=base,
            search_space=ss,
            causal_graph=None,
            objective_name="objective",
            screened_variables=["x"],
            enable=True,
        )
        assert result == base

    def test_no_screening_returns_base_focus_unchanged(self) -> None:
        ss = _make_search_space()
        graph = _full_ancestor_graph()
        base = ["x", "y", "z"]
        result = _apply_minimal_focus_a1(
            base_focus=base,
            search_space=ss,
            causal_graph=graph,
            objective_name="objective",
            screened_variables=None,
            enable=True,
        )
        assert result == base

    def test_empty_screening_returns_base_focus_unchanged(self) -> None:
        ss = _make_search_space()
        graph = _full_ancestor_graph()
        base = ["x", "y", "z"]
        result = _apply_minimal_focus_a1(
            base_focus=base,
            search_space=ss,
            causal_graph=graph,
            objective_name="objective",
            screened_variables=[],
            enable=True,
        )
        assert result == base

    def test_partial_ancestors_returns_base_focus_unchanged(self) -> None:
        """When ancestors do not cover the full search space, A1 must not bind."""
        ss = _make_search_space()
        graph = _partial_ancestor_graph()  # x, y are ancestors; z is not
        base = ["x", "y"]
        result = _apply_minimal_focus_a1(
            base_focus=base,
            search_space=ss,
            causal_graph=graph,
            objective_name="objective",
            screened_variables=["x"],
            enable=True,
        )
        assert result == base

    def test_full_ancestors_proper_subset_screening_restricts(self) -> None:
        """The binding case: ancestors cover space, screening is a proper subset."""
        ss = _make_search_space()
        graph = _full_ancestor_graph()
        base = ["x", "y", "z"]
        result = _apply_minimal_focus_a1(
            base_focus=base,
            search_space=ss,
            causal_graph=graph,
            objective_name="objective",
            screened_variables=["x", "y"],
            enable=True,
        )
        assert set(result) == {"x", "y"}

    def test_full_ancestors_screening_equals_space_returns_base(self) -> None:
        """When screening already keeps every variable, no proper subset exists."""
        ss = _make_search_space()
        graph = _full_ancestor_graph()
        base = ["x", "y", "z"]
        result = _apply_minimal_focus_a1(
            base_focus=base,
            search_space=ss,
            causal_graph=graph,
            objective_name="objective",
            screened_variables=["x", "y", "z"],
            enable=True,
        )
        assert set(result) == set(base)

    def test_full_ancestors_screening_disjoint_returns_base(self) -> None:
        """An empty intersection is not a valid restriction; fall back."""
        ss = _make_search_space()
        graph = _full_ancestor_graph()
        base = ["x", "y", "z"]
        result = _apply_minimal_focus_a1(
            base_focus=base,
            search_space=ss,
            causal_graph=graph,
            objective_name="objective",
            screened_variables=["nonsense"],
            enable=True,
        )
        assert set(result) == set(base)

    def test_full_ancestors_single_screened_variable_restricts(self) -> None:
        ss = _make_search_space()
        graph = _full_ancestor_graph()
        result = _apply_minimal_focus_a1(
            base_focus=["x", "y", "z"],
            search_space=ss,
            causal_graph=graph,
            objective_name="objective",
            screened_variables=["y"],
            enable=True,
        )
        assert result == ["y"]


# ── suggest_parameters integration ───────────────────────────────────


class TestSuggestParametersThreadsA1:
    def test_optimization_passes_minimal_focus_to_bayesian(self) -> None:
        ss = _make_search_space()
        log = _make_log(n=20, phase="optimization")
        graph = _full_ancestor_graph()

        with patch("causal_optimizer.optimizer.suggest._suggest_bayesian") as mock_bayesian:
            mock_bayesian.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
            suggest_parameters(
                ss,
                log,
                causal_graph=graph,
                phase="optimization",
                minimize=True,
                objective_name="objective",
                screened_variables=["x", "y"],
                pomis_minimal_focus=True,
            )
            call_kwargs = mock_bayesian.call_args.kwargs
            focus = call_kwargs.get("focus_variables", [])
            assert set(focus) == {"x", "y"}

    def test_optimization_minimal_focus_off_passes_existing_intersection(self) -> None:
        """When the flag is off, optimization preserves its current intersection logic."""
        ss = _make_search_space()
        log = _make_log(n=20, phase="optimization")
        graph = _full_ancestor_graph()

        with patch("causal_optimizer.optimizer.suggest._suggest_bayesian") as mock_bayesian:
            mock_bayesian.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
            suggest_parameters(
                ss,
                log,
                causal_graph=graph,
                phase="optimization",
                minimize=True,
                objective_name="objective",
                screened_variables=["x", "y"],
                pomis_minimal_focus=False,
            )
            call_kwargs = mock_bayesian.call_args.kwargs
            focus = call_kwargs.get("focus_variables", [])
            # Optimization already intersects graph_focus with screening when both
            # are present; that pre-existing behavior is preserved.
            assert set(focus) == {"x", "y"}

    def test_exploitation_minimal_focus_on_restricts_perturbation(self) -> None:
        ss = _make_search_space()
        log = _make_log(n=60)
        graph = _full_ancestor_graph()

        with patch("causal_optimizer.optimizer.suggest._suggest_exploitation") as mock_exploit:
            mock_exploit.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
            suggest_parameters(
                ss,
                log,
                causal_graph=graph,
                phase="exploitation",
                minimize=True,
                objective_name="objective",
                screened_variables=["x"],
                pomis_minimal_focus=True,
            )
            focus = mock_exploit.call_args.kwargs.get("focus_variables", [])
            assert focus == ["x"]

    def test_exploitation_minimal_focus_off_uses_full_space(self) -> None:
        """With flag off, exploitation keeps the full ancestor space."""
        ss = _make_search_space()
        log = _make_log(n=60)
        graph = _full_ancestor_graph()

        with patch("causal_optimizer.optimizer.suggest._suggest_exploitation") as mock_exploit:
            mock_exploit.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
            suggest_parameters(
                ss,
                log,
                causal_graph=graph,
                phase="exploitation",
                minimize=True,
                objective_name="objective",
                screened_variables=["x"],
                pomis_minimal_focus=False,
            )
            focus = mock_exploit.call_args.kwargs.get("focus_variables", [])
            assert set(focus) == {"x", "y", "z"}


# ── ExperimentEngine plumbing ────────────────────────────────────────


class _DummyRunner:
    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        return {"objective": float(parameters.get("x", 0.0))}


class TestExperimentEnginePlumbing:
    def test_default_pomis_minimal_focus_is_false(self) -> None:
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_DummyRunner(),
        )
        assert engine.pomis_minimal_focus is False

    def test_pomis_minimal_focus_can_be_enabled(self) -> None:
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_DummyRunner(),
            pomis_minimal_focus=True,
        )
        assert engine.pomis_minimal_focus is True

    def test_engine_threads_flag_into_optimization(self) -> None:
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_DummyRunner(),
            objective_name="objective",
            minimize=False,
            causal_graph=_full_ancestor_graph(),
            pomis_minimal_focus=True,
        )
        engine._phase = "optimization"
        engine._screened_focus_variables = ["x"]
        with patch("causal_optimizer.optimizer.suggest.suggest_parameters") as mock_sp:
            mock_sp.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
            engine.suggest_next()
            assert mock_sp.call_args.kwargs["pomis_minimal_focus"] is True

    def test_engine_threads_flag_into_exploitation_b80(self) -> None:
        """The flag must be threaded through exploitation too -- B80 crosses
        the optimization→exploitation phase boundary at experiment 50, so
        exploitation must not silently revert to full-space behavior."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_DummyRunner(),
            objective_name="objective",
            minimize=False,
            causal_graph=_full_ancestor_graph(),
            pomis_minimal_focus=True,
        )
        # Force exploitation phase (B80 crosses the >= 50 boundary)
        engine._phase = "exploitation"
        engine._screened_focus_variables = ["x"]
        with patch("causal_optimizer.optimizer.suggest.suggest_parameters") as mock_sp:
            mock_sp.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
            engine.suggest_next()
            assert mock_sp.call_args.kwargs["pomis_minimal_focus"] is True
