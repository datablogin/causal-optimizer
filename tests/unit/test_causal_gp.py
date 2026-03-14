"""Tests for the CausalGPSurrogate class.

Tests are organized into:
1. Graceful degradation when botorch is unavailable
2. Fit runs without crash on ToyGraph data
3. Predict interventional returns correct shape
4. Suggest returns valid params within search space
5. Graph topology is respected (monotone chain)
6. Engine integration with strategy="causal_gp"
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("botorch")

from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toygraph_log(n: int = 10, seed: int = 42) -> ExperimentLog:
    """Generate n experiment results from ToyGraph."""
    bench = ToyGraphBenchmark(noise_scale=0.1, rng=np.random.default_rng(seed))
    ss = ToyGraphBenchmark.search_space()
    rng = np.random.default_rng(seed)
    log = ExperimentLog()
    for i in range(n):
        params: dict[str, Any] = {}
        for var in ss.variables:
            assert var.lower is not None and var.upper is not None
            params[var.name] = float(rng.uniform(var.lower, var.upper))
        metrics = bench.run(params)
        log.results.append(
            ExperimentResult(
                experiment_id=f"exp_{i}",
                parameters=params,
                metrics=metrics,
                status=ExperimentStatus.KEEP,
            )
        )
    return log


def _make_chain_graph() -> CausalGraph:
    """A -> B -> Y monotone increasing chain."""
    return CausalGraph(
        edges=[("A", "B"), ("B", "Y")],
        bidirected_edges=[],
    )


def _make_chain_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="A", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="B", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
    )


def _make_chain_log(n: int = 20, seed: int = 0) -> ExperimentLog:
    """Generate log from a monotone chain: B = 2*A + noise, Y = 3*B + noise."""
    rng = np.random.default_rng(seed)
    log = ExperimentLog()
    for i in range(n):
        a = float(rng.uniform(0, 10))
        b = 2.0 * a + float(rng.normal(0, 0.1))
        y = 3.0 * b + float(rng.normal(0, 0.1))
        log.results.append(
            ExperimentResult(
                experiment_id=f"chain_{i}",
                parameters={"A": a, "B": b},
                metrics={"Y": y},
                status=ExperimentStatus.KEEP,
            )
        )
    return log


# ---------------------------------------------------------------------------
# Test 1: graceful degradation — ImportError when botorch unavailable
# ---------------------------------------------------------------------------


def test_causal_gp_requires_botorch() -> None:
    """When botorch is not importable, CausalGPSurrogate raises ImportError."""
    with patch.dict(
        sys.modules,
        {"botorch": None, "botorch.models": None, "gpytorch": None, "gpytorch.mlls": None},
    ):
        import causal_optimizer.optimizer.causal_gp as cgp_module

        importlib.reload(cgp_module)

        with pytest.raises(ImportError, match="botorch"):
            cgp_module.CausalGPSurrogate(
                search_space=ToyGraphBenchmark.search_space(),
                causal_graph=ToyGraphBenchmark.causal_graph(),
                objective_name="objective",
            )

    # Restore module state
    import causal_optimizer.optimizer.causal_gp as cgp_restore

    importlib.reload(cgp_restore)


# ---------------------------------------------------------------------------
# Test 2: fit runs without crash on ToyGraph observations
# ---------------------------------------------------------------------------


def test_causal_gp_fit_runs_without_crash() -> None:
    """Fit on 10 ToyGraph observations completes without exception."""
    from causal_optimizer.optimizer.causal_gp import CausalGPSurrogate

    surrogate = CausalGPSurrogate(
        search_space=ToyGraphBenchmark.search_space(),
        causal_graph=ToyGraphBenchmark.causal_graph(),
        objective_name="objective",
        minimize=True,
        seed=42,
    )
    log = _make_toygraph_log(n=10, seed=42)
    surrogate.fit(log)  # should not raise


# ---------------------------------------------------------------------------
# Test 3: predict_interventional returns (float, float) tuple
# ---------------------------------------------------------------------------


def test_causal_gp_predict_interventional_shape() -> None:
    """predict_interventional returns a (mean, std) tuple of floats."""
    from causal_optimizer.optimizer.causal_gp import CausalGPSurrogate

    surrogate = CausalGPSurrogate(
        search_space=ToyGraphBenchmark.search_space(),
        causal_graph=ToyGraphBenchmark.causal_graph(),
        objective_name="objective",
        minimize=True,
        seed=42,
    )
    log = _make_toygraph_log(n=10, seed=42)
    surrogate.fit(log)

    mean, std = surrogate.predict_interventional({"x": 1.0, "z": 2.0})

    assert isinstance(mean, float), f"Expected float mean, got {type(mean)}"
    assert isinstance(std, float), f"Expected float std, got {type(std)}"
    assert std >= 0.0, f"Expected non-negative std, got {std}"


# ---------------------------------------------------------------------------
# Test 4: suggest returns valid params within search space
# ---------------------------------------------------------------------------


def test_causal_gp_suggest_returns_valid_params() -> None:
    """suggest() returns params within search space bounds."""
    from causal_optimizer.optimizer.causal_gp import CausalGPSurrogate

    ss = ToyGraphBenchmark.search_space()
    surrogate = CausalGPSurrogate(
        search_space=ss,
        causal_graph=ToyGraphBenchmark.causal_graph(),
        objective_name="objective",
        minimize=True,
        seed=42,
    )
    log = _make_toygraph_log(n=10, seed=42)
    surrogate.fit(log)

    params = surrogate.suggest(n_candidates=50)

    assert isinstance(params, dict)
    for var in ss.variables:
        assert var.name in params, f"Missing variable {var.name}"
        assert var.lower is not None and var.upper is not None
        val = params[var.name]
        assert var.lower <= val <= var.upper, f"{var.name}={val} out of [{var.lower}, {var.upper}]"


# ---------------------------------------------------------------------------
# Test 5: respects graph topology — monotone chain ordering
# ---------------------------------------------------------------------------


def test_causal_gp_respects_graph_topology() -> None:
    """With chain A->B->Y (monotone increasing SCM), do(A=hi) > do(A=lo)."""
    from causal_optimizer.optimizer.causal_gp import CausalGPSurrogate

    graph = _make_chain_graph()
    ss = _make_chain_search_space()
    log = _make_chain_log(n=30, seed=0)

    surrogate = CausalGPSurrogate(
        search_space=ss,
        causal_graph=graph,
        objective_name="Y",
        minimize=True,
        seed=0,
    )
    surrogate.fit(log)

    mean_lo, _ = surrogate.predict_interventional({"A": 1.0})
    mean_hi, _ = surrogate.predict_interventional({"A": 9.0})

    # The SCM is Y = 3*(2*A) = 6*A, so do(A=9) should give much higher Y than do(A=1)
    assert mean_hi > mean_lo, (
        f"Expected do(A=9) > do(A=1) for monotone chain, "
        f"got mean_hi={mean_hi:.4f} <= mean_lo={mean_lo:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: engine integration with strategy="causal_gp"
# ---------------------------------------------------------------------------


def test_causal_gp_in_engine_with_strategy() -> None:
    """Run ExperimentEngine with strategy='causal_gp' for 15 steps; no crashes."""
    from causal_optimizer.engine.loop import ExperimentEngine

    bench = ToyGraphBenchmark(noise_scale=0.1, rng=np.random.default_rng(0))
    engine = ExperimentEngine(
        search_space=ToyGraphBenchmark.search_space(),
        runner=bench,
        causal_graph=ToyGraphBenchmark.causal_graph(),
        objective_name="objective",
        minimize=True,
        seed=0,
        max_skips=0,
        strategy="causal_gp",
    )

    for _ in range(15):
        engine.step()

    assert len(engine.log.results) == 15
    # Should have passed into optimization phase (10+ experiments)
    assert engine._phase == "optimization"
