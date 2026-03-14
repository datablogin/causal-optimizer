"""Tests for multi-objective Pareto front support."""

from __future__ import annotations

from causal_optimizer.types import (
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    ObjectiveSpec,
    ParetoResult,
    SearchSpace,
    Variable,
    VariableType,
)


def _make_result(
    eid: str,
    metrics: dict[str, float],
    status: ExperimentStatus = ExperimentStatus.KEEP,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=eid,
        parameters={"x": 0.0},
        metrics=metrics,
        status=status,
    )


class TestParetoResult:
    """Tests for ParetoResult.dominated_by."""

    def test_dominated_when_worse_on_all(self) -> None:
        objectives = [
            ObjectiveSpec(name="a", minimize=True),
            ObjectiveSpec(name="b", minimize=True),
        ]
        dominated = _make_result("d", {"a": 3.0, "b": 3.0})
        dominator = _make_result("r", {"a": 1.0, "b": 1.0})
        pr = ParetoResult(front=[dominated])
        assert pr.dominated_by(dominated, dominator, objectives)

    def test_not_dominated_when_better_on_one(self) -> None:
        objectives = [
            ObjectiveSpec(name="a", minimize=True),
            ObjectiveSpec(name="b", minimize=True),
        ]
        a = _make_result("a", {"a": 1.0, "b": 3.0})
        b = _make_result("b", {"a": 3.0, "b": 1.0})
        pr = ParetoResult(front=[a])
        assert not pr.dominated_by(a, b, objectives)

    def test_not_dominated_when_equal(self) -> None:
        objectives = [
            ObjectiveSpec(name="a", minimize=True),
            ObjectiveSpec(name="b", minimize=True),
        ]
        a = _make_result("a", {"a": 2.0, "b": 2.0})
        b = _make_result("b", {"a": 2.0, "b": 2.0})
        pr = ParetoResult(front=[a])
        assert not pr.dominated_by(a, b, objectives)


class TestExperimentLogParetoFront:
    """Tests for ExperimentLog.pareto_front."""

    def test_pareto_front_single_objective_same_as_best(self) -> None:
        """With one objective, pareto_front returns the single best KEEP result."""
        objectives = [ObjectiveSpec(name="obj", minimize=True)]
        log = ExperimentLog(
            results=[
                _make_result("a", {"obj": 3.0}),
                _make_result("b", {"obj": 1.0}),
                _make_result("c", {"obj": 5.0}),
                _make_result("d", {"obj": 2.0}, ExperimentStatus.DISCARD),
            ]
        )
        front = log.pareto_front(objectives)
        assert len(front) == 1
        assert front[0].experiment_id == "b"

    def test_pareto_front_two_objectives_nondominated(self) -> None:
        """With 3 results [(1,3),(2,2),(3,1)], all 3 are non-dominated."""
        objectives = [
            ObjectiveSpec(name="a", minimize=True),
            ObjectiveSpec(name="b", minimize=True),
        ]
        log = ExperimentLog(
            results=[
                _make_result("r1", {"a": 1.0, "b": 3.0}),
                _make_result("r2", {"a": 2.0, "b": 2.0}),
                _make_result("r3", {"a": 3.0, "b": 1.0}),
            ]
        )
        front = log.pareto_front(objectives)
        assert len(front) == 3

    def test_pareto_front_dominated_excluded(self) -> None:
        """Result (2,3) is dominated by (1,2); assert excluded."""
        objectives = [
            ObjectiveSpec(name="a", minimize=True),
            ObjectiveSpec(name="b", minimize=True),
        ]
        log = ExperimentLog(
            results=[
                _make_result("winner", {"a": 1.0, "b": 2.0}),
                _make_result("loser", {"a": 2.0, "b": 3.0}),
            ]
        )
        front = log.pareto_front(objectives)
        assert len(front) == 1
        assert front[0].experiment_id == "winner"

    def test_pareto_front_maximize_objective(self) -> None:
        """With maximize objectives, higher is better."""
        objectives = [
            ObjectiveSpec(name="a", minimize=False),
            ObjectiveSpec(name="b", minimize=False),
        ]
        log = ExperimentLog(
            results=[
                _make_result("high", {"a": 5.0, "b": 5.0}),
                _make_result("low", {"a": 1.0, "b": 1.0}),
            ]
        )
        front = log.pareto_front(objectives)
        assert len(front) == 1
        assert front[0].experiment_id == "high"

    def test_pareto_front_ignores_non_keep(self) -> None:
        """Only KEEP results are considered for the Pareto front."""
        objectives = [ObjectiveSpec(name="a", minimize=True)]
        log = ExperimentLog(
            results=[
                _make_result("good", {"a": 1.0}, ExperimentStatus.KEEP),
                _make_result("crashed", {"a": 0.5}, ExperimentStatus.CRASH),
                _make_result("disc", {"a": 0.3}, ExperimentStatus.DISCARD),
            ]
        )
        front = log.pareto_front(objectives)
        assert len(front) == 1
        assert front[0].experiment_id == "good"


class TestEngineMultiObjective:
    """Tests for ExperimentEngine multi-objective integration."""

    def test_engine_multi_objective_keeps_nondominated(self) -> None:
        """Run 20 steps on ToyGraphBiObjective; pareto_front is non-empty and non-dominated."""
        import numpy as np

        from causal_optimizer.benchmarks.toy_graph import ToyGraphBiObjective
        from causal_optimizer.engine.loop import ExperimentEngine

        bench = ToyGraphBiObjective(rng=np.random.default_rng(42))
        objectives = [
            ObjectiveSpec(name="objective", minimize=True),
            ObjectiveSpec(name="cost", minimize=True),
        ]
        engine = ExperimentEngine(
            search_space=bench.search_space(),
            runner=bench,
            objectives=objectives,
            seed=42,
        )
        engine.run_loop(20)
        front = engine.pareto_front
        assert len(front) > 0

        # Verify all front members are non-dominated
        for i, a in enumerate(front):
            for j, b in enumerate(front):
                if i == j:
                    continue
                assert not ParetoResult.dominated_by(a, b, objectives), (
                    f"Result {b.experiment_id} dominates {a.experiment_id} "
                    f"but both are in the Pareto front"
                )

    def test_engine_single_objective_backward_compat(self) -> None:
        """Engine with no objectives param behaves identically to current behavior."""
        import numpy as np

        from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark
        from causal_optimizer.engine.loop import ExperimentEngine

        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        engine = ExperimentEngine(
            search_space=bench.search_space(),
            runner=bench,
            objective_name="objective",
            minimize=True,
            seed=42,
        )
        engine.run_loop(10)
        best = engine.log.best_result("objective", minimize=True)
        assert best is not None
        # pareto_front should still work (falls back to single objective)
        front = engine.pareto_front
        assert len(front) >= 1

    def test_engine_multi_objective_evaluate_status_nondominated_keep(self) -> None:
        """Multi-objective: non-dominated results should be KEEP."""
        from causal_optimizer.engine.loop import ExperimentEngine

        space = SearchSpace(
            variables=[
                Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ]
        )

        class DummyRunner:
            def __init__(self) -> None:
                self.call_count = 0

            def run(self, parameters: dict[str, object]) -> dict[str, float]:
                self.call_count += 1
                # Return alternating trade-offs
                if self.call_count % 2 == 0:
                    return {"objective": 1.0, "cost": 3.0}
                return {"objective": 3.0, "cost": 1.0}

        objectives = [
            ObjectiveSpec(name="objective", minimize=True),
            ObjectiveSpec(name="cost", minimize=True),
        ]
        engine = ExperimentEngine(
            search_space=space,
            runner=DummyRunner(),
            objectives=objectives,
            seed=42,
        )
        engine.run_loop(4)

        kept = [r for r in engine.log.results if r.status == ExperimentStatus.KEEP]
        # Both trade-off patterns should be kept since neither dominates the other
        assert len(kept) >= 2


class TestParetoFrontMixedObjectives:
    """Tests for mixed minimize/maximize multi-objective."""

    def test_pareto_front_mixed_min_max(self) -> None:
        """Mixed minimize/maximize: non-dominated results are correct."""
        objectives = [
            ObjectiveSpec(name="loss", minimize=True),
            ObjectiveSpec(name="throughput", minimize=False),
        ]
        log = ExperimentLog(
            results=[
                # Low loss, high throughput — clearly best
                _make_result("best", {"loss": 1.0, "throughput": 10.0}),
                # High loss, low throughput — dominated
                _make_result("worst", {"loss": 5.0, "throughput": 2.0}),
                # Trade-off: low loss but low throughput
                _make_result("trade", {"loss": 0.5, "throughput": 1.0}),
            ]
        )
        front = log.pareto_front(objectives)
        front_ids = {r.experiment_id for r in front}
        # "best" dominates "worst" (lower loss AND higher throughput)
        assert "worst" not in front_ids
        # "best" and "trade" are non-dominated (trade has lower loss)
        assert "best" in front_ids
        assert "trade" in front_ids


class TestParetoFrontEdgeCases:
    """Edge case tests for pareto_front."""

    def test_pareto_front_empty_log(self) -> None:
        """pareto_front with no results returns empty list."""
        objectives = [ObjectiveSpec(name="a", minimize=True)]
        log = ExperimentLog(results=[])
        assert log.pareto_front(objectives) == []

    def test_engine_pareto_front_empty(self) -> None:
        """engine.pareto_front before any experiments returns empty list."""
        from causal_optimizer.engine.loop import ExperimentEngine

        space = SearchSpace(
            variables=[
                Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ]
        )

        class NullRunner:
            def run(self, parameters: dict[str, object]) -> dict[str, float]:
                return {"objective": 0.0}

        engine = ExperimentEngine(
            search_space=space,
            runner=NullRunner(),
            seed=42,
        )
        assert engine.pareto_front == []

    def test_scalarized_key_not_in_metrics_after_suggest(self) -> None:
        """Internal scalarized key should not leak into result metrics."""
        from causal_optimizer.engine.loop import ExperimentEngine

        space = SearchSpace(
            variables=[
                Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ]
        )

        class SimpleRunner:
            def run(self, parameters: dict[str, object]) -> dict[str, float]:
                return {"objective": 1.0, "cost": 2.0}

        objectives = [
            ObjectiveSpec(name="objective", minimize=True),
            ObjectiveSpec(name="cost", minimize=True),
        ]
        engine = ExperimentEngine(
            search_space=space,
            runner=SimpleRunner(),
            objectives=objectives,
            seed=42,
        )
        engine.run_loop(5)

        # Verify no internal scalarized key leaked into metrics
        for result in engine.log.results:
            assert "__scalarized_objective__" not in result.metrics


class TestObjectiveNameValidation:
    """Tests for objective_name validation against objectives."""

    def test_objective_name_not_in_objectives_raises(self) -> None:
        """Engine raises ValueError if objective_name not in objectives."""
        import pytest

        from causal_optimizer.engine.loop import ExperimentEngine

        space = SearchSpace(
            variables=[
                Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ]
        )

        class NullRunner:
            def run(self, parameters: dict[str, object]) -> dict[str, float]:
                return {"loss": 0.0}

        objectives = [
            ObjectiveSpec(name="loss", minimize=True),
            ObjectiveSpec(name="cost", minimize=True),
        ]
        with pytest.raises(ValueError, match="objective_name.*not found"):
            ExperimentEngine(
                search_space=space,
                runner=NullRunner(),
                objectives=objectives,
                objective_name="objective",  # not in objectives
                seed=42,
            )
