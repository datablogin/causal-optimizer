"""Tests for SQLite experiment store and engine persistence integration."""

from __future__ import annotations

from typing import Any

import pytest

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.storage.sqlite import ExperimentStore
from causal_optimizer.types import (
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)


def _make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        ]
    )


def _make_result(
    exp_id: str = "exp-001",
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    status: ExperimentStatus = ExperimentStatus.KEEP,
    metadata: dict[str, Any] | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=exp_id,
        parameters=params or {"x": 1.0, "y": 2.0},
        metrics=metrics or {"objective": 5.0},
        status=status,
        metadata=metadata or {"phase": "exploration"},
    )


class QuadraticRunner:
    """Simple test runner: f(x, y) = x^2 + y^2."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": x**2 + y**2}


# --- ExperimentStore tests ---


def test_store_create_and_load_empty() -> None:
    """Create an experiment, load_log returns an empty ExperimentLog."""
    store = ExperimentStore(":memory:")
    store.create_experiment("exp-001", _make_search_space())
    log = store.load_log("exp-001")
    assert isinstance(log, ExperimentLog)
    assert len(log.results) == 0


def test_store_append_and_load() -> None:
    """Append 3 results, load_log returns all 3 in order."""
    store = ExperimentStore(":memory:")
    ss = _make_search_space()
    store.create_experiment("exp-001", ss)

    for i in range(3):
        result = _make_result(
            exp_id=f"r-{i}",
            params={"x": float(i), "y": float(i + 1)},
            metrics={"objective": float(i * i)},
        )
        store.append_result("exp-001", result, step=i)

    log = store.load_log("exp-001")
    assert len(log.results) == 3
    # Check ordering
    assert log.results[0].parameters["x"] == 0.0
    assert log.results[1].parameters["x"] == 1.0
    assert log.results[2].parameters["x"] == 2.0


def test_store_result_roundtrip() -> None:
    """ExperimentResult survives serialize/deserialize with all fields."""
    store = ExperimentStore(":memory:")
    ss = _make_search_space()
    store.create_experiment("exp-001", ss)

    original = ExperimentResult(
        experiment_id="r-roundtrip",
        parameters={"x": 3.14, "y": -2.71},
        metrics={"objective": 17.19, "secondary": 42.0},
        status=ExperimentStatus.DISCARD,
        metadata={"phase": "optimization", "extra": [1, 2, 3]},
    )
    store.append_result("exp-001", original, step=0)

    log = store.load_log("exp-001")
    loaded = log.results[0]

    assert loaded.experiment_id == original.experiment_id
    assert loaded.parameters == original.parameters
    assert loaded.metrics == original.metrics
    assert loaded.status == original.status
    assert loaded.metadata == original.metadata


def test_store_list_experiments() -> None:
    """list_experiments returns the experiments created."""
    store = ExperimentStore(":memory:")
    ss = _make_search_space()
    store.create_experiment("exp-001", ss)
    store.create_experiment("exp-002", ss)

    experiments = store.list_experiments()
    assert len(experiments) == 2
    ids = {e["id"] for e in experiments}
    assert ids == {"exp-001", "exp-002"}


def test_store_delete() -> None:
    """After delete, load_log raises KeyError."""
    store = ExperimentStore(":memory:")
    ss = _make_search_space()
    store.create_experiment("exp-001", ss)
    store.append_result("exp-001", _make_result(), step=0)

    store.delete_experiment("exp-001")

    import pytest

    with pytest.raises(KeyError):
        store.load_log("exp-001")


def test_store_in_memory() -> None:
    """:memory: path works for testing."""
    store = ExperimentStore(":memory:")
    ss = _make_search_space()
    store.create_experiment("mem-test", ss)
    store.append_result("mem-test", _make_result(), step=0)
    log = store.load_log("mem-test")
    assert len(log.results) == 1


# --- Engine persistence integration tests ---


def test_engine_with_store_persists_each_step() -> None:
    """Run engine 5 steps with store; assert 5 rows in DB."""
    store = ExperimentStore(":memory:")
    ss = _make_search_space()

    engine = ExperimentEngine(
        search_space=ss,
        runner=QuadraticRunner(),
        store=store,
        experiment_id="eng-001",
    )

    for _ in range(5):
        engine.step()

    log = store.load_log("eng-001")
    assert len(log.results) == 5


def test_engine_resume_continues_from_step() -> None:
    """Run 5 steps, create new engine via resume(), run 5 more; assert 10 total."""
    store = ExperimentStore(":memory:")
    ss = _make_search_space()

    engine = ExperimentEngine(
        search_space=ss,
        runner=QuadraticRunner(),
        store=store,
        experiment_id="eng-resume",
    )
    for _ in range(5):
        engine.step()

    # Resume with a new engine instance
    engine2 = ExperimentEngine.resume(
        store=store,
        experiment_id="eng-resume",
        runner=QuadraticRunner(),
        search_space=ss,
    )

    for _ in range(5):
        engine2.step()

    log = store.load_log("eng-resume")
    assert len(log.results) == 10
    # Engine2 should also have 10 results in its in-memory log
    assert len(engine2.log.results) == 10


def test_engine_rejects_store_without_experiment_id() -> None:
    """Passing store without experiment_id raises ValueError."""
    store = ExperimentStore(":memory:")
    with pytest.raises(ValueError, match="store and experiment_id must be provided together"):
        ExperimentEngine(
            search_space=_make_search_space(),
            runner=QuadraticRunner(),
            store=store,
        )


def test_engine_rejects_experiment_id_without_store() -> None:
    """Passing experiment_id without store raises ValueError."""
    with pytest.raises(ValueError, match="store and experiment_id must be provided together"):
        ExperimentEngine(
            search_space=_make_search_space(),
            runner=QuadraticRunner(),
            experiment_id="test-id",
        )


def test_engine_resume_infers_optimization_phase() -> None:
    """Resume with 15 results infers optimization phase."""
    store = ExperimentStore(":memory:")
    ss = _make_search_space()

    # Create experiment and manually insert 15 results to simulate past run
    store.create_experiment("opt-phase", ss)
    for i in range(15):
        result = _make_result(
            exp_id=f"r-{i}",
            params={"x": float(i), "y": float(i)},
            metrics={"objective": float(i * i)},
            metadata={"phase": "exploration" if i < 10 else "optimization"},
        )
        store.append_result("opt-phase", result, step=i)

    engine = ExperimentEngine.resume(
        store=store,
        experiment_id="opt-phase",
        runner=QuadraticRunner(),
        search_space=ss,
    )

    assert engine._phase == "optimization"
    assert len(engine.log.results) == 15
