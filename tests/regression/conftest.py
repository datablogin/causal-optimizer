"""Regression test configuration.

All tests in this directory should be marked ``@pytest.mark.slow``.
Deselect with: ``pytest -m "not slow"``

Note: ``pytestmark`` in conftest.py does NOT propagate to sibling test
modules — each test class must be decorated with ``@pytest.mark.slow``
explicitly. This conftest exists for shared fixtures and configuration.

Test files
----------
test_convergence.py
    Convergence regression on ToyGraphBenchmark (causal vs. random vs. surrogate).
test_high_dim_convergence.py
    Convergence regression on HighDimensionalSparseBenchmark (20 vars, 3 causal).
test_pomis_pruning.py
    POMIS pruning regression: verifies CompleteGraphBenchmark.known_pomis() matches
    Aglietti et al. (5 sets) and that the POMIS-guided engine explores ≤ 1/5 of the
    naive 64-subset search space.
test_interaction_detection.py
    Interaction detection regression: verifies ScreeningDesigner detects the x1*x2
    interaction in the InteractionSCM, that naive main-effects RF has meaningful
    lower R² than one with the product term, and that the causal engine with
    discovery_method='correlation' finds the interaction optimum.
"""

from __future__ import annotations
