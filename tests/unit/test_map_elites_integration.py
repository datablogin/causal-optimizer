"""Tests for MAP-Elites integration in the experiment engine."""

from typing import Any

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.evolution.map_elites import MAPElites
from causal_optimizer.types import (
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)


def make_search_space() -> SearchSpace:
    return SearchSpace(variables=[
        Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
    ])


class MetricRunner:
    """Runner that returns objective plus descriptor metrics."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {
            "objective": x**2 + y**2,
            "total_spend": abs(x) + abs(y),
            "channel_diversity": abs(x - y),
        }


def test_archive_created_when_descriptor_names_provided():
    """Archive should be created when descriptor_names are given."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=MetricRunner(),
        descriptor_names=["total_spend", "channel_diversity"],
    )
    assert engine._archive is not None
    assert isinstance(engine._archive, MAPElites)
    assert engine._archive.descriptor_names == ["total_spend", "channel_diversity"]


def test_archive_is_none_when_no_descriptor_names():
    """Archive should be None when no descriptor_names are given."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=MetricRunner(),
    )
    assert engine._archive is None


def test_results_added_to_archive_after_experiment():
    """Results should be added to the archive after each experiment."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=MetricRunner(),
        descriptor_names=["total_spend", "channel_diversity"],
    )

    assert len(engine._archive.archive) == 0

    engine.run_experiment({"x": 1.0, "y": 2.0})

    assert len(engine._archive.archive) > 0


def test_multiple_experiments_populate_archive():
    """Running multiple experiments should add diverse entries to archive."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=MetricRunner(),
        descriptor_names=["total_spend", "channel_diversity"],
    )

    engine.run_experiment({"x": 1.0, "y": 1.0})
    engine.run_experiment({"x": 4.0, "y": 0.0})
    engine.run_experiment({"x": 0.0, "y": 4.0})

    # Should have some entries (possibly different cells)
    assert len(engine._archive.archive) >= 1


def test_exploitation_can_sample_from_archive():
    """In exploitation phase, the engine should be able to sample from archive."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=MetricRunner(),
        descriptor_names=["total_spend", "channel_diversity"],
    )

    # Populate archive with some results
    for i in range(5):
        engine.run_experiment({"x": float(i), "y": float(i)})

    assert len(engine._archive.archive) > 0

    # Verify archive can sample
    elite = engine._archive.sample_elite()
    assert elite is not None
    assert isinstance(elite, ExperimentResult)


def test_extract_descriptors():
    """_extract_descriptors should extract the right keys from metrics."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=MetricRunner(),
        descriptor_names=["total_spend", "channel_diversity"],
    )

    metrics = {
        "objective": 5.0,
        "total_spend": 3.0,
        "channel_diversity": 1.0,
        "extra_metric": 99.0,
    }

    descriptors = engine._extract_descriptors(metrics)
    assert descriptors == {"total_spend": 3.0, "channel_diversity": 1.0}


def test_extract_descriptors_missing_keys():
    """_extract_descriptors should skip missing descriptor keys."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=MetricRunner(),
        descriptor_names=["total_spend", "nonexistent"],
    )

    metrics = {"objective": 5.0, "total_spend": 3.0}
    descriptors = engine._extract_descriptors(metrics)
    assert descriptors == {"total_spend": 3.0}


def test_map_elites_add_and_sample():
    """Basic MAPElites add/sample operations."""
    archive = MAPElites(
        descriptor_names=["d1", "d2"],
        n_bins=5,
        minimize=True,
    )

    result = ExperimentResult(
        experiment_id="test_1",
        parameters={"x": 1.0},
        metrics={"objective": 5.0},
        status=ExperimentStatus.KEEP,
    )

    added = archive.add(result, fitness=5.0, descriptors={"d1": 0.5, "d2": 0.5})
    assert added is True
    assert len(archive.archive) == 1

    sampled = archive.sample_elite()
    assert sampled is not None
    assert sampled.experiment_id == "test_1"


def test_map_elites_keeps_better_fitness():
    """MAPElites should keep the better result in a cell."""
    archive = MAPElites(
        descriptor_names=["d1"],
        n_bins=5,
        minimize=True,
    )

    r1 = ExperimentResult(
        experiment_id="worse",
        parameters={"x": 1.0},
        metrics={"objective": 10.0},
        status=ExperimentStatus.KEEP,
    )
    r2 = ExperimentResult(
        experiment_id="better",
        parameters={"x": 0.5},
        metrics={"objective": 2.0},
        status=ExperimentStatus.KEEP,
    )

    archive.add(r1, fitness=10.0, descriptors={"d1": 0.5})
    archive.add(r2, fitness=2.0, descriptors={"d1": 0.5})

    sampled = archive.sample_elite()
    assert sampled is not None
    assert sampled.experiment_id == "better"


def test_archive_not_populated_without_descriptor_names():
    """Without descriptor_names, no archive operations should happen."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=MetricRunner(),
    )

    engine.run_experiment({"x": 1.0, "y": 2.0})
    assert engine._archive is None
