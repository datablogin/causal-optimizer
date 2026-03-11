"""Tests for core types."""

from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
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


def test_causal_graph_bidirected_edges():
    graph = CausalGraph(
        edges=[("x", "z"), ("z", "y")],
        bidirected_edges=[("x", "y")],
    )
    assert set(graph.nodes) == {"x", "y", "z"}
    assert graph.has_confounders

    # c-components: x and y are connected via bidirected edge
    components = graph.c_components()
    xy_component = next(c for c in components if "x" in c)
    assert "y" in xy_component
    assert "z" not in xy_component


def test_causal_graph_no_confounders():
    graph = CausalGraph(edges=[("a", "b"), ("b", "c")])
    assert not graph.has_confounders
    components = graph.c_components()
    assert all(len(c) == 1 for c in components)


def test_causal_graph_ancestors_descendants():
    graph = CausalGraph(edges=[("a", "b"), ("b", "c"), ("b", "d"), ("d", "e")])
    assert graph.ancestors("e") == {"a", "b", "d"}
    assert graph.ancestors("a") == set()
    assert graph.descendants("b") == {"c", "d", "e"}
    assert graph.descendants("e") == set()


def test_causal_graph_parents_children():
    graph = CausalGraph(edges=[("a", "c"), ("b", "c"), ("c", "d")])
    assert graph.parents("c") == {"a", "b"}
    assert graph.children("c") == {"d"}


def test_causal_graph_do_operator():
    graph = CausalGraph(
        edges=[("a", "b"), ("b", "c"), ("a", "c")],
        bidirected_edges=[("a", "c")],
    )
    # do(b) removes incoming edges to b and bidirected edges involving b
    g2 = graph.do({"b"})
    assert ("a", "b") not in g2.edges
    assert ("b", "c") in g2.edges  # outgoing edges preserved
    assert ("a", "c") in g2.edges  # unrelated edges preserved
    assert ("a", "c") in g2.bidirected_edges  # b not in this bidirected edge


def test_causal_graph_subgraph():
    graph = CausalGraph(
        edges=[("a", "b"), ("b", "c"), ("c", "d")],
        bidirected_edges=[("a", "d")],
    )
    sg = graph.subgraph({"b", "c"})
    assert sg.edges == [("b", "c")]
    assert sg.bidirected_edges == []
    assert set(sg.nodes) == {"b", "c"}


def test_experiment_log_best():
    log = ExperimentLog(
        results=[
            ExperimentResult(
                experiment_id="1",
                parameters={"x": 1},
                metrics={"objective": 5.0},
                status=ExperimentStatus.KEEP,
            ),
            ExperimentResult(
                experiment_id="2",
                parameters={"x": 2},
                metrics={"objective": 3.0},
                status=ExperimentStatus.KEEP,
            ),
            ExperimentResult(
                experiment_id="3",
                parameters={"x": 3},
                metrics={"objective": 7.0},
                status=ExperimentStatus.DISCARD,
            ),
        ]
    )
    # Default args: minimize=True, objective_name="objective"
    best = log.best_result()
    assert best is not None
    assert best.experiment_id == "2"
    assert best.metrics["objective"] == 3.0


def test_experiment_log_best_result_minimize():
    """best_result with minimize=True returns the smallest objective."""
    log = ExperimentLog(
        results=[
            ExperimentResult(
                experiment_id="1",
                parameters={},
                metrics={"objective": 10.0},
                status=ExperimentStatus.KEEP,
            ),
            ExperimentResult(
                experiment_id="2",
                parameters={},
                metrics={"objective": 2.0},
                status=ExperimentStatus.KEEP,
            ),
            ExperimentResult(
                experiment_id="3",
                parameters={},
                metrics={"objective": 5.0},
                status=ExperimentStatus.KEEP,
            ),
        ]
    )
    best = log.best_result(minimize=True)
    assert best is not None
    assert best.experiment_id == "2"


def test_experiment_log_best_result_maximize():
    """best_result with minimize=False returns the largest objective."""
    log = ExperimentLog(
        results=[
            ExperimentResult(
                experiment_id="1",
                parameters={},
                metrics={"objective": 10.0},
                status=ExperimentStatus.KEEP,
            ),
            ExperimentResult(
                experiment_id="2",
                parameters={},
                metrics={"objective": 2.0},
                status=ExperimentStatus.KEEP,
            ),
            ExperimentResult(
                experiment_id="3",
                parameters={},
                metrics={"objective": 5.0},
                status=ExperimentStatus.KEEP,
            ),
        ]
    )
    best = log.best_result(minimize=False)
    assert best is not None
    assert best.experiment_id == "1"


def test_experiment_log_best_result_custom_objective():
    """best_result with a custom objective_name uses the correct metric."""
    log = ExperimentLog(
        results=[
            ExperimentResult(
                experiment_id="1",
                parameters={},
                metrics={"loss": 0.5, "accuracy": 0.8},
                status=ExperimentStatus.KEEP,
            ),
            ExperimentResult(
                experiment_id="2",
                parameters={},
                metrics={"loss": 0.2, "accuracy": 0.95},
                status=ExperimentStatus.KEEP,
            ),
        ]
    )
    best_loss = log.best_result(objective_name="loss", minimize=True)
    assert best_loss is not None
    assert best_loss.experiment_id == "2"

    best_acc = log.best_result(objective_name="accuracy", minimize=False)
    assert best_acc is not None
    assert best_acc.experiment_id == "2"


def test_experiment_log_best_result_no_kept():
    """best_result returns None when no results have KEEP status."""
    log = ExperimentLog(
        results=[
            ExperimentResult(
                experiment_id="1",
                parameters={},
                metrics={"objective": 1.0},
                status=ExperimentStatus.DISCARD,
            ),
            ExperimentResult(
                experiment_id="2",
                parameters={},
                metrics={"objective": 2.0},
                status=ExperimentStatus.CRASH,
            ),
        ]
    )
    assert log.best_result() is None


def test_experiment_log_best_result_backward_compat():
    """Default args match old property behavior: objective_name='objective', minimize=True."""
    log = ExperimentLog(
        results=[
            ExperimentResult(
                experiment_id="a",
                parameters={},
                metrics={"objective": 100.0},
                status=ExperimentStatus.KEEP,
            ),
            ExperimentResult(
                experiment_id="b",
                parameters={},
                metrics={"objective": 1.0},
                status=ExperimentStatus.KEEP,
            ),
        ]
    )
    best = log.best_result()
    assert best is not None
    assert best.experiment_id == "b"


def test_experiment_log_to_dataframe():
    log = ExperimentLog(
        results=[
            ExperimentResult(
                experiment_id="1",
                parameters={"x": 1},
                metrics={"objective": 5.0},
                status=ExperimentStatus.KEEP,
            ),
        ]
    )
    df = log.to_dataframe()
    assert len(df) == 1
    assert "x" in df.columns
    assert "objective" in df.columns
