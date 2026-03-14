"""Tests for constrained optimization support."""

from __future__ import annotations

from causal_optimizer.types import (
    Constraint,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)


def _make_space() -> SearchSpace:
    return SearchSpace(
        variables=[Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0)]
    )


class SequenceRunner:
    """Runner that returns metrics from a predefined sequence."""

    def __init__(self, sequence: list[dict[str, float]]) -> None:
        self._sequence = sequence
        self._idx = 0

    def run(self, parameters: dict[str, object]) -> dict[str, float]:
        metrics = self._sequence[self._idx % len(self._sequence)]
        self._idx += 1
        return dict(metrics)


class TestConstraintUpperBound:
    def test_constraint_upper_bound_discards(self) -> None:
        """Result with metric > upper_bound gets DISCARD."""
        from causal_optimizer.engine.loop import ExperimentEngine

        constraints = [Constraint(metric_name="latency", upper_bound=100.0)]
        # First result is good (KEEP), second violates constraint
        runner = SequenceRunner(
            [
                {"objective": 1.0, "latency": 50.0},  # good
                {"objective": 0.5, "latency": 150.0},  # violates
            ]
        )
        engine = ExperimentEngine(
            search_space=_make_space(),
            runner=runner,
            constraints=constraints,
            seed=42,
        )
        engine.step()  # first: latency=50 < 100 -> KEEP
        result2 = engine.step()  # second: latency=150 > 100 -> DISCARD
        assert result2.status == ExperimentStatus.DISCARD


class TestConstraintLowerBound:
    def test_constraint_lower_bound_discards(self) -> None:
        """Result with metric < lower_bound gets DISCARD."""
        from causal_optimizer.engine.loop import ExperimentEngine

        constraints = [Constraint(metric_name="throughput", lower_bound=10.0)]
        runner = SequenceRunner(
            [
                {"objective": 1.0, "throughput": 20.0},  # good
                {"objective": 0.5, "throughput": 5.0},  # violates
            ]
        )
        engine = ExperimentEngine(
            search_space=_make_space(),
            runner=runner,
            constraints=constraints,
            seed=42,
        )
        engine.step()
        result2 = engine.step()
        assert result2.status == ExperimentStatus.DISCARD


class TestConstraintSatisfied:
    def test_constraint_satisfied_keeps(self) -> None:
        """Result within bounds gets KEEP (assuming improvement)."""
        from causal_optimizer.engine.loop import ExperimentEngine

        constraints = [Constraint(metric_name="latency", upper_bound=100.0, lower_bound=0.0)]
        runner = SequenceRunner(
            [
                {"objective": 5.0, "latency": 50.0},  # first
                {"objective": 3.0, "latency": 50.0},  # improvement, within bounds
            ]
        )
        engine = ExperimentEngine(
            search_space=_make_space(),
            runner=runner,
            constraints=constraints,
            seed=42,
        )
        engine.step()
        result2 = engine.step()
        assert result2.status == ExperimentStatus.KEEP


class TestConstraintMetadata:
    def test_constraint_metadata_tag(self) -> None:
        """Discarded result has constraint_violated=True in metadata."""
        from causal_optimizer.engine.loop import ExperimentEngine

        constraints = [Constraint(metric_name="latency", upper_bound=100.0)]
        runner = SequenceRunner(
            [
                {"objective": 1.0, "latency": 50.0},
                {"objective": 0.5, "latency": 200.0},
            ]
        )
        engine = ExperimentEngine(
            search_space=_make_space(),
            runner=runner,
            constraints=constraints,
            seed=42,
        )
        engine.step()
        result2 = engine.step()
        assert result2.metadata.get("constraint_violated") is True


class TestConstraintBackwardCompat:
    def test_engine_constraint_backward_compat(self) -> None:
        """Engine with no constraints param works as before."""
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
        kept = [r for r in engine.log.results if r.status == ExperimentStatus.KEEP]
        assert len(kept) >= 1
