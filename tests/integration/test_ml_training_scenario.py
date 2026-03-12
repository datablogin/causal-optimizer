"""Integration test: MLTrainingAdapter end-to-end through ExperimentEngine.

Validates that the full optimization loop works with the ML training domain
adapter's 9-variable mixed-type search space (continuous, integer, categorical)
and prior causal graph.

This is especially important for testing edge cases in:
  - Categorical variable handling in screening (pd.to_numeric coercion)
  - Boolean-free but categorical-heavy search spaces
  - The surrogate model with mixed types (RF feature encoding)
  - POMIS computation with a larger, more complex graph

Uses epsilon_mode=False to avoid convex hull issues with categorical variables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causal_optimizer.domain_adapters.ml_training import MLTrainingAdapter
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.optimizer.pomis import compute_pomis


class MLTrainingSimRunner:
    """Simulated ML training runner with known causal structure.

    The objective (val_loss) depends on learning_rate, batch_size, n_layers,
    n_heads, hidden_dim, dropout, weight_decay, optimizer choice, and activation.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        lr = float(parameters.get("learning_rate", 1e-3))
        batch_size = int(parameters.get("batch_size", 32))
        n_layers = int(parameters.get("n_layers", 6))
        n_heads = int(parameters.get("n_heads", 8))
        hidden_dim = int(parameters.get("hidden_dim", 512))
        dropout = float(parameters.get("dropout", 0.1))
        weight_decay = float(parameters.get("weight_decay", 0.01))
        optimizer_name = parameters.get("optimizer", "adamw")
        activation = parameters.get("activation", "gelu")

        # Simulate causal mechanisms
        # Learning rate -> gradient scale -> stability
        gradient_scale = lr * 1000
        if gradient_scale > 10:
            gradient_scale = 10 + (gradient_scale - 10) * 0.1  # diminishing
        training_stability = max(0, 1.0 - abs(gradient_scale - 1.0) * 0.3)

        # Batch size -> gradient noise, throughput, memory
        gradient_noise = 1.0 / np.sqrt(batch_size)
        training_stability *= max(0.1, 1.0 - gradient_noise * 0.5)
        throughput = batch_size * 0.8
        memory_usage = hidden_dim * n_layers * 4 / 1e6  # rough MB

        # Architecture -> model capacity
        model_capacity = np.log(n_layers * n_heads * hidden_dim + 1)

        # Optimizer effect
        optimizer_map = {"adamw": 1.0, "muon": 0.95, "sgd": 1.2, "lion": 0.98}
        opt_factor = optimizer_map.get(optimizer_name, 1.0)
        convergence_speed = 1.0 / opt_factor

        # Activation effect
        activation_map = {"gelu": 1.0, "swiglu": 0.95, "relu": 1.1}
        act_factor = activation_map.get(activation, 1.0)
        gradient_flow = 1.0 / act_factor

        training_stability *= gradient_flow

        # Regularization
        regularization = dropout * 0.5 + weight_decay * 10

        # tokens_seen depends on throughput
        tokens_seen = throughput * 100

        # Val loss: lower is better
        # Good capacity + good stability + some regularization -> low loss
        val_loss = (
            2.0
            - model_capacity * 0.15
            - training_stability * 0.3
            - convergence_speed * 0.1
            - min(regularization, 0.5) * 0.2
            - np.log(tokens_seen + 1) * 0.05
            + self._rng.normal(0, 0.05)
        )
        val_loss = max(0.01, val_loss)

        return {
            "val_loss": val_loss,
            "memory_usage": memory_usage + self._rng.normal(0, 0.1),
            "model_capacity": model_capacity,
        }


@pytest.mark.slow
class TestMLTrainingScenario:
    """End-to-end test of ExperimentEngine with MLTrainingAdapter."""

    def test_engine_runs_30_experiments_without_crash(self) -> None:
        """The engine should complete 30 experiments with the 9-var mixed-type space."""
        adapter = MLTrainingAdapter()
        runner = MLTrainingSimRunner(seed=42)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="val_loss",
            minimize=True,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=42,
        )

        log = engine.run_loop(30)

        assert len(log.results) == 30
        keep_count = sum(1 for r in log.results if r.status.value == "keep")
        assert keep_count >= 1, "Expected at least one KEEP result"

    def test_no_crash_results(self) -> None:
        """No experiments should crash with valid simulated data."""
        adapter = MLTrainingAdapter()
        runner = MLTrainingSimRunner(seed=123)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="val_loss",
            minimize=True,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=123,
        )

        log = engine.run_loop(30)

        crash_count = sum(1 for r in log.results if r.status.value == "crash")
        assert crash_count == 0, f"Expected 0 crashes, got {crash_count}"

    def test_categorical_variables_in_screening(self) -> None:
        """Screening should handle categorical variables (optimizer, activation) gracefully.

        The ML training space has 2 categorical variables. The screening designer
        uses pd.to_numeric with errors='coerce', which converts categorical strings
        to NaN then fills with 0. This should not crash.
        """
        adapter = MLTrainingAdapter()
        runner = MLTrainingSimRunner(seed=42)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="val_loss",
            minimize=True,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=42,
        )

        # Run past exploration to trigger screening
        log = engine.run_loop(15)

        # Should not crash and screening should have run
        assert len(log.results) == 15
        # Screening result should exist (might have retried)
        # The key assertion: no crash during screening with categorical vars

    def test_pomis_prunes_9var_space(self) -> None:
        """POMIS should produce intervention sets for the 9-variable ML space."""
        adapter = MLTrainingAdapter()
        graph = adapter.get_prior_graph()
        space = adapter.get_search_space()

        pomis_sets = compute_pomis(graph, "val_loss")

        assert len(pomis_sets) >= 1, "Expected at least one POMIS set"
        # At least one POMIS set should have fewer variables than the full space
        all_search_vars = set(space.variable_names)
        has_pruning = any(len(pset & all_search_vars) < len(all_search_vars) for pset in pomis_sets)
        assert has_pruning, f"Expected POMIS to prune the 9-variable space, got sets: {pomis_sets}"

    def test_phase_transitions_with_mixed_types(self) -> None:
        """Engine should transition phases correctly with mixed variable types."""
        adapter = MLTrainingAdapter()
        runner = MLTrainingSimRunner(seed=42)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="val_loss",
            minimize=True,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=42,
        )

        phases_seen: set[str] = set()
        for _ in range(30):
            result = engine.step()
            phase = result.metadata.get("phase", "unknown")
            phases_seen.add(phase)

        assert "exploration" in phases_seen
        assert "optimization" in phases_seen

    def test_surrogate_handles_categorical_encoding(self) -> None:
        """The RF surrogate should handle categorical string values without error.

        When Ax is not available, the engine falls back to the RF surrogate
        (via _suggest_surrogate). The surrogate uses .astype(float, errors='ignore')
        which leaves strings as-is, then fillna(0). This test verifies the full
        path works for a space with categoricals.
        """
        adapter = MLTrainingAdapter()
        runner = MLTrainingSimRunner(seed=42)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="val_loss",
            minimize=True,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=42,
        )

        # Run enough experiments to get into optimization phase
        log = engine.run_loop(20)

        # Should complete without errors
        assert len(log.results) == 20

    def test_pomis_computed_during_optimization(self) -> None:
        """POMIS sets should be computed when entering optimization phase."""
        adapter = MLTrainingAdapter()
        runner = MLTrainingSimRunner(seed=42)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="val_loss",
            minimize=True,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=42,
        )

        engine.run_loop(15)

        # Graph has confounders, so POMIS should be computed
        assert engine._pomis_sets is not None, "POMIS sets should be computed"
        assert len(engine._pomis_sets) >= 1

    def test_multiple_seeds_produce_valid_results(self) -> None:
        """Different seeds should all produce valid (non-crash) results.

        Full determinism is not guaranteed because LatinHypercube and other
        internal RNG sources are not all seeded by the engine seed. Instead
        we verify that multiple seeds all complete successfully.
        """
        adapter = MLTrainingAdapter()

        for seed in [42, 77, 123]:
            runner = MLTrainingSimRunner(seed=seed)
            engine = ExperimentEngine(
                search_space=adapter.get_search_space(),
                runner=runner,
                objective_name="val_loss",
                minimize=True,
                causal_graph=adapter.get_prior_graph(),
                epsilon_mode=False,
                seed=seed,
            )
            log = engine.run_loop(15)
            assert len(log.results) == 15
            crashes = sum(1 for r in log.results if r.status.value == "crash")
            assert crashes == 0, f"Seed {seed} produced {crashes} crashes"

    def test_all_variable_types_represented_in_parameters(self) -> None:
        """Each experiment result should contain all 9 variables with correct types."""
        adapter = MLTrainingAdapter()
        runner = MLTrainingSimRunner(seed=42)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="val_loss",
            minimize=True,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=42,
        )

        log = engine.run_loop(5)

        expected_vars = {v.name for v in adapter.get_search_space().variables}
        for result in log.results:
            param_names = set(result.parameters.keys())
            assert expected_vars == param_names, f"Expected vars {expected_vars}, got {param_names}"
            # Check categorical vars are valid choices
            assert result.parameters["optimizer"] in ["adamw", "muon", "sgd", "lion"], (
                f"Invalid optimizer: {result.parameters['optimizer']}"
            )
            assert result.parameters["activation"] in ["gelu", "swiglu", "relu"], (
                f"Invalid activation: {result.parameters['activation']}"
            )
