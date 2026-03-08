"""Tests for core types."""

import numpy as np

from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)


def test_variable_validation():
    var = Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
    assert var.validate_value(0.5)
    assert not var.validate_value(-0.1)
    assert not var.validate_value(1.1)


def test_causal_graph_adjacency():
    graph = CausalGraph(edges=[("a", "b"), ("b", "c")])
    assert graph.nodes == ["a", "b", "c"]
    adj = graph.adjacency_matrix
    assert adj.shape == (3, 3)
    assert adj[0, 1] == 1  # a -> b
    assert adj[1, 2] == 1  # b -> c
    assert adj[0, 2] == 0  # no direct a -> c


def test_experiment_log_best():
    log = ExperimentLog(results=[
        ExperimentResult(experiment_id="1", parameters={"x": 1}, metrics={"objective": 5.0}, status=ExperimentStatus.KEEP),
        ExperimentResult(experiment_id="2", parameters={"x": 2}, metrics={"objective": 3.0}, status=ExperimentStatus.KEEP),
        ExperimentResult(experiment_id="3", parameters={"x": 3}, metrics={"objective": 7.0}, status=ExperimentStatus.DISCARD),
    ])
    best = log.best_result
    assert best is not None
    assert best.experiment_id == "2"
    assert best.metrics["objective"] == 3.0


def test_experiment_log_to_dataframe():
    log = ExperimentLog(results=[
        ExperimentResult(experiment_id="1", parameters={"x": 1}, metrics={"objective": 5.0}, status=ExperimentStatus.KEEP),
    ])
    df = log.to_dataframe()
    assert len(df) == 1
    assert "x" in df.columns
    assert "objective" in df.columns
