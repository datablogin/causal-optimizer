"""Interaction SCM benchmark: X1 and X2 interact — individually harmful, jointly helpful.

Structural equation:
    Y = -X1 - X2 + 3*X1*X2 + noise

Optimal point: X1=1, X2=1 → Y=1 (minimization: objective = -Y = -1).
Greedy hill-climbing discards both X1=1 and X2=1 because individually they
increase cost; only together do they produce the global optimum.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class InteractionSCM:
    """5-variable SCM where X1 and X2 have a synergistic interaction.

    The structural equation is::

        Y = -x1 - x2 + 3*x1*x2 + noise

    Variables X3, X4, X5 are irrelevant (independent Uniform[0,1]).

    This is natively a maximisation problem (higher Y is better). The runner
    negates Y so the engine can minimise, i.e. ``objective = -Y``.

    Args:
        noise_scale: Standard deviation of the Gaussian noise term.
        rng: NumPy random generator. A default one is created if not provided.
    """

    def __init__(
        self,
        noise_scale: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.noise_scale = noise_scale
        self.rng = rng or np.random.default_rng()

    @staticmethod
    def search_space() -> SearchSpace:
        """Return the 5-variable search space [0, 1]^5."""
        return SearchSpace(
            variables=[
                Variable(
                    name="x1",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="x2",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="x3",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="x4",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="x5",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
            ]
        )

    @staticmethod
    def causal_graph() -> CausalGraph:
        """Return the true causal graph: only X1 and X2 affect objective.

        X3, X4, and X5 are independent dummy variables with no causal path
        to the outcome, so they are omitted from the graph. This matches the
        structural equation Y = -x1 - x2 + 3*x1*x2 + noise, which does not
        reference x3, x4, or x5.
        """
        return CausalGraph(
            edges=[
                ("x1", "objective"),
                ("x2", "objective"),
            ],
            bidirected_edges=[],
        )

    @staticmethod
    def known_pomis() -> list[frozenset[str]]:
        """Known POMIS: intervening on {x1, x2} is sufficient."""
        return [frozenset({"x1", "x2"})]

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Evaluate the SCM at the given parameters.

        Variables not provided default to 0.0.

        Args:
            parameters: Dict mapping variable names to values.

        Returns:
            Dict with key ``"objective"`` = -Y (negated so engine minimises).
        """
        x1 = float(parameters.get("x1", 0.0))
        x2 = float(parameters.get("x2", 0.0))
        noise = float(self.rng.normal(0.0, self.noise_scale))
        y = -x1 - x2 + 3.0 * x1 * x2 + noise
        return {"objective": -y}  # negate: minimising -Y = maximising Y
