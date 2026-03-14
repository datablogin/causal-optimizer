"""Tests for seed reproducibility across all random paths.

Verifies that threading a seed through the engine, suggest_parameters,
_random_sample, _suggest_exploitation, and MAPElites produces
deterministic results.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.evolution.map_elites import MAPElites
from causal_optimizer.optimizer.suggest import (
    _random_sample,
    _suggest_exploitation,
    suggest_parameters,
)
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


class _QuadraticRunner:
    """f(x, y) = x**2 + y**2, minimum at origin."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": x**2 + y**2}


# ---------------------------------------------------------------------------
# Task 1 tests: _random_sample seeded
# ---------------------------------------------------------------------------


def test_random_sample_seeded() -> None:
    """_random_sample with the same seed returns identical results."""
    ss = _make_search_space()
    a = _random_sample(ss, seed=42)
    b = _random_sample(ss, seed=42)
    assert a == b, f"Expected identical samples with seed=42, got {a} vs {b}"


def test_random_sample_different_seeds_differ() -> None:
    """_random_sample with different seeds returns different results."""
    ss = _make_search_space()
    a = _random_sample(ss, seed=42)
    b = _random_sample(ss, seed=99)
    assert a != b, "Different seeds should produce different samples"


# ---------------------------------------------------------------------------
# Task 1 tests: _suggest_exploitation seeded
# ---------------------------------------------------------------------------


def test_exploitation_seeded() -> None:
    """_suggest_exploitation with the same seed is deterministic."""
    ss = _make_search_space()
    log = ExperimentLog()
    log.results.append(
        ExperimentResult(
            experiment_id="e1",
            parameters={"x": 1.0, "y": 2.0},
            metrics={"objective": 5.0},
            status=ExperimentStatus.KEEP,
        )
    )
    a = _suggest_exploitation(ss, log, minimize=True, objective_name="objective", seed=42)
    b = _suggest_exploitation(ss, log, minimize=True, objective_name="objective", seed=42)
    assert a == b, f"Expected identical exploitation with seed=42, got {a} vs {b}"


def test_exploitation_different_seeds_differ() -> None:
    """_suggest_exploitation with different seeds returns different results."""
    ss = _make_search_space()
    log = ExperimentLog()
    log.results.append(
        ExperimentResult(
            experiment_id="e1",
            parameters={"x": 1.0, "y": 2.0},
            metrics={"objective": 5.0},
            status=ExperimentStatus.KEEP,
        )
    )
    a = _suggest_exploitation(ss, log, minimize=True, objective_name="objective", seed=42)
    b = _suggest_exploitation(ss, log, minimize=True, objective_name="objective", seed=99)
    assert a != b, "Different seeds should produce different exploitation results"


# ---------------------------------------------------------------------------
# Task 1 tests: MAPElites seeded sampling
# ---------------------------------------------------------------------------


def test_map_elites_sample_seeded() -> None:
    """sample_elite with the same seed returns the same elite."""
    archive = MAPElites(descriptor_names=["d1"], n_bins=5, minimize=True)
    # Add several elites so there's something to randomly choose from
    for i in range(10):
        result = ExperimentResult(
            experiment_id=f"e{i}",
            parameters={"x": float(i)},
            metrics={"objective": float(i)},
            status=ExperimentStatus.KEEP,
        )
        archive.add(result, fitness=float(i), descriptors={"d1": float(i)})

    a = archive.sample_elite(seed=42)
    b = archive.sample_elite(seed=42)
    assert a is not None and b is not None
    assert a.experiment_id == b.experiment_id


def test_map_elites_sample_diverse_seeded() -> None:
    """sample_diverse with the same seed returns identical selections."""
    archive = MAPElites(descriptor_names=["d1"], n_bins=10, minimize=True)
    for i in range(10):
        result = ExperimentResult(
            experiment_id=f"e{i}",
            parameters={"x": float(i)},
            metrics={"objective": float(i)},
            status=ExperimentStatus.KEEP,
        )
        archive.add(result, fitness=float(i), descriptors={"d1": float(i)})

    a = archive.sample_diverse(n=3, seed=42)
    b = archive.sample_diverse(n=3, seed=42)
    assert [r.experiment_id for r in a] == [r.experiment_id for r in b]


# ---------------------------------------------------------------------------
# Task 2 tests: Engine-level seed determinism
# ---------------------------------------------------------------------------


def test_engine_seeded_deterministic() -> None:
    """Two engine runs with the same seed produce identical results."""
    ss = _make_search_space()

    results_a = []
    engine_a = ExperimentEngine(search_space=ss, runner=_QuadraticRunner(), seed=42)
    for _ in range(5):
        result = engine_a.step()
        results_a.append(result.parameters)

    results_b = []
    engine_b = ExperimentEngine(search_space=ss, runner=_QuadraticRunner(), seed=42)
    for _ in range(5):
        result = engine_b.step()
        results_b.append(result.parameters)

    assert results_a == results_b, (
        "Two engines with seed=42 should produce identical parameter sequences"
    )


def test_engine_different_seeds_differ() -> None:
    """Two engine runs with different seeds produce different results."""
    ss = _make_search_space()

    results_a = []
    engine_a = ExperimentEngine(search_space=ss, runner=_QuadraticRunner(), seed=42)
    for _ in range(5):
        result = engine_a.step()
        results_a.append(result.parameters)

    results_b = []
    engine_b = ExperimentEngine(search_space=ss, runner=_QuadraticRunner(), seed=99)
    for _ in range(5):
        result = engine_b.step()
        results_b.append(result.parameters)

    assert results_a != results_b, (
        "Engines with different seeds should produce different parameter sequences"
    )


def test_engine_unseeded_varies() -> None:
    """Unseeded engine runs may produce different results.

    We run 5 independent unseeded sequences and check that not all are identical.
    This is a statistical test -- very unlikely to fail by chance with continuous params.
    """
    ss = _make_search_space()
    all_sequences: list[list[dict[str, Any]]] = []

    for _ in range(5):
        engine = ExperimentEngine(search_space=ss, runner=_QuadraticRunner(), seed=None)
        seq = []
        for _ in range(3):
            result = engine.step()
            seq.append(result.parameters)
        all_sequences.append(seq)

    # At least two sequences should differ
    unique = {tuple(tuple(sorted(p.items())) for p in seq) for seq in all_sequences}
    assert len(unique) > 1, "Unseeded runs should not all produce identical results"


def test_suggest_parameters_passes_seed_to_exploitation() -> None:
    """suggest_parameters in exploitation phase passes seed through."""
    ss = _make_search_space()
    log = ExperimentLog()
    # Need enough results so exploitation doesn't fall back
    for i in range(5):
        log.results.append(
            ExperimentResult(
                experiment_id=f"e{i}",
                parameters={"x": float(i), "y": float(i)},
                metrics={"objective": float(i**2)},
                status=ExperimentStatus.KEEP,
            )
        )

    a = suggest_parameters(ss, log, phase="exploitation", seed=42)
    b = suggest_parameters(ss, log, phase="exploitation", seed=42)
    assert a == b, "suggest_parameters with seed=42 in exploitation should be deterministic"
