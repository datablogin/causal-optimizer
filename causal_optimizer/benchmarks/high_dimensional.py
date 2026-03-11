"""High-dimensional sparse benchmark: 20 variables, only 3 causal ancestors."""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class HighDimensionalSparseBenchmark:
    """20 observed variables, only X1→X2→X3→objective are causal.

    Structural equations:
        X1 = parameter (intervened or sampled uniformly)
        X2 = sin(X1) + epsilon
        X3 = X2^2 / 10 + epsilon
        Y  = cos(X3) - X3 / 5 + epsilon
        X4..X20 = independent noise (no causal effect on Y)

    The causal chain X1→X2→X3→objective means the optimal strategy
    is to intervene on a subset of {X1, X2, X3}. Variables X4-X20
    are irrelevant distractors that waste budget if explored.

    This is a minimization problem: objective = -Y (negate so lower is better).
    """

    N_VARIABLES = 20
    N_CAUSAL = 3

    def __init__(self, noise_scale: float = 0.1, rng: np.random.Generator | None = None) -> None:
        self.noise_scale = noise_scale
        self.rng = rng or np.random.default_rng()

    @staticmethod
    def search_space() -> SearchSpace:
        variables = [
            Variable(
                name=f"x{i}",
                variable_type=VariableType.CONTINUOUS,
                lower=-5.0,
                upper=5.0,
            )
            for i in range(1, HighDimensionalSparseBenchmark.N_VARIABLES + 1)
        ]
        return SearchSpace(variables=variables)

    @staticmethod
    def causal_graph() -> CausalGraph:
        """Chain graph: x1→x2→x3→objective. x4-x20 are disconnected."""
        return CausalGraph(
            edges=[("x1", "x2"), ("x2", "x3"), ("x3", "objective")],
            bidirected_edges=[],
        )

    @staticmethod
    def known_pomis() -> list[frozenset[str]]:
        """Minimal intervention sets on the causal chain.

        For a chain x1->x2->x3->objective with no bidirected edges (no
        confounders), the POMIS algorithm yields only {x3} — the direct
        parent of the outcome. Intervening on x3 alone controls the
        entire causal path, so supersets are pruned as non-minimal.
        """
        return [frozenset({"x3"})]

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run the SCM with partial intervention semantics.

        Causal variables not provided follow their structural equations.
        Non-causal variables (x4-x20) are ignored — they have no effect on Y.

        Structural noise draws are guarded: RNG draws only occur for
        variables that follow their structural equations (not intervened on).
        Note that the RNG stream position still depends on which variables
        are intervened on — different intervention sets consume different
        numbers of draws before the y-noise call. This matches the standard
        partial intervention semantics used by the other benchmarks.
        """
        x1 = parameters["x1"] if "x1" in parameters else self.rng.uniform(-5.0, 5.0)
        x2 = (
            parameters["x2"]
            if "x2" in parameters
            else np.sin(x1) + self.rng.normal(0, self.noise_scale)
        )
        x3 = (
            parameters["x3"]
            if "x3" in parameters
            else x2**2 / 10 + self.rng.normal(0, self.noise_scale)
        )

        y = np.cos(x3) - x3 / 5 + self.rng.normal(0, self.noise_scale)
        return {"objective": -y}
