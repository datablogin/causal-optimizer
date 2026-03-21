"""Unit tests for MLTrainingAdapter simulator.

Tests the structural equations, failure modes, categorical handling,
noise, seed support, and metric outputs of the ML training simulator.
"""

from __future__ import annotations

import pytest

from causal_optimizer.domain_adapters.ml_training import MLTrainingAdapter


class TestMLTrainingAdapterBasics:
    """Basic adapter contract tests."""

    def test_get_search_space_has_9_variables(self) -> None:
        adapter = MLTrainingAdapter()
        space = adapter.get_search_space()
        assert len(space.variables) == 9

    def test_get_prior_graph_returns_graph(self) -> None:
        adapter = MLTrainingAdapter()
        graph = adapter.get_prior_graph()
        assert graph is not None
        assert len(graph.edges) > 0
        assert len(graph.bidirected_edges) == 2

    def test_get_descriptor_names(self) -> None:
        adapter = MLTrainingAdapter()
        assert adapter.get_descriptor_names() == ["memory_usage", "model_capacity"]

    def test_get_objective_name(self) -> None:
        adapter = MLTrainingAdapter()
        assert adapter.get_objective_name() == "val_loss"

    def test_get_minimize_is_true(self) -> None:
        adapter = MLTrainingAdapter()
        assert adapter.get_minimize() is True


class TestMLTrainingSimulator:
    """Tests for the run_experiment simulator."""

    def test_run_experiment_returns_three_metrics(self) -> None:
        adapter = MLTrainingAdapter(seed=42)
        params = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 6,
            "n_heads": 8,
            "hidden_dim": 512,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        metrics = adapter.run_experiment(params)
        assert "val_loss" in metrics
        assert "memory_usage" in metrics
        assert "model_capacity" in metrics

    def test_val_loss_is_positive(self) -> None:
        adapter = MLTrainingAdapter(seed=42)
        params = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 6,
            "n_heads": 8,
            "hidden_dim": 512,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        metrics = adapter.run_experiment(params)
        assert metrics["val_loss"] > 0

    def test_memory_usage_is_positive(self) -> None:
        adapter = MLTrainingAdapter(seed=42)
        params = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 12,
            "n_heads": 8,
            "hidden_dim": 1024,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        metrics = adapter.run_experiment(params)
        assert metrics["memory_usage"] > 0


class TestMLTrainingFailureModes:
    """Tests for realistic ML training failure modes."""

    def test_high_lr_causes_divergence(self) -> None:
        """Very high learning rate should cause high val_loss (divergence)."""
        base = {
            "batch_size": 64,
            "n_layers": 6,
            "n_heads": 8,
            "hidden_dim": 512,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        good_lr = MLTrainingAdapter(seed=42).run_experiment({**base, "learning_rate": 1e-3})
        bad_lr = MLTrainingAdapter(seed=42).run_experiment({**base, "learning_rate": 1e-1})
        assert bad_lr["val_loss"] > good_lr["val_loss"]

    def test_large_model_no_reg_overfits(self) -> None:
        """Large model with no regularization should overfit (higher val_loss)."""
        big_no_reg = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 24,
            "n_heads": 16,
            "hidden_dim": 2048,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        big_with_reg = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 24,
            "n_heads": 16,
            "hidden_dim": 2048,
            "dropout": 0.3,
            "weight_decay": 0.1,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        no_reg = MLTrainingAdapter(seed=42).run_experiment(big_no_reg)
        with_reg = MLTrainingAdapter(seed=42).run_experiment(big_with_reg)
        assert no_reg["val_loss"] > with_reg["val_loss"]

    def test_tiny_model_underfits(self) -> None:
        """Very small model should underfit (higher val_loss than a well-sized one)."""
        tiny = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 2,
            "n_heads": 1,
            "hidden_dim": 128,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        medium = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 12,
            "n_heads": 8,
            "hidden_dim": 1024,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        tiny_loss = MLTrainingAdapter(seed=42).run_experiment(tiny)
        medium_loss = MLTrainingAdapter(seed=42).run_experiment(medium)
        assert tiny_loss["val_loss"] > medium_loss["val_loss"]


class TestMLTrainingCategoricals:
    """Tests for categorical variable handling."""

    def test_optimizer_choices_affect_loss(self) -> None:
        """Different optimizers should produce different val_loss values."""
        base = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 6,
            "n_heads": 8,
            "hidden_dim": 512,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "activation": "gelu",
        }
        losses = {}
        for opt in ["adamw", "sgd", "muon", "lion"]:
            adapter = MLTrainingAdapter(seed=42)
            losses[opt] = adapter.run_experiment({**base, "optimizer": opt})["val_loss"]

        # SGD should be worse than AdamW (lower gradient scale factor)
        assert losses["sgd"] > losses["adamw"]

    def test_activation_choices_affect_loss(self) -> None:
        """Different activations should produce different val_loss values."""
        base = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 6,
            "n_heads": 8,
            "hidden_dim": 512,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
        }
        losses = {}
        for act in ["gelu", "swiglu", "relu"]:
            adapter = MLTrainingAdapter(seed=42)
            losses[act] = adapter.run_experiment({**base, "activation": act})["val_loss"]

        # swiglu should be best (factor 1.1)
        assert losses["swiglu"] < losses["relu"]

    def test_categorical_encoding_values(self) -> None:
        """Categorical variables should map to specific numeric factors."""
        adapter = MLTrainingAdapter(seed=42)
        # adamw=1.0, sgd=0.8, muon=1.1, lion=0.95
        # Just verify the method handles them without error
        for opt in ["adamw", "sgd", "muon", "lion"]:
            for act in ["gelu", "swiglu", "relu"]:
                params = {
                    "learning_rate": 1e-3,
                    "batch_size": 64,
                    "n_layers": 6,
                    "n_heads": 8,
                    "hidden_dim": 512,
                    "dropout": 0.1,
                    "weight_decay": 0.01,
                    "optimizer": opt,
                    "activation": act,
                }
                metrics = adapter.run_experiment(params)
                assert metrics["val_loss"] > 0


class TestMLTrainingSeedReproducibility:
    """Tests for seed-based reproducibility."""

    def test_same_seed_same_result(self) -> None:
        params = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 6,
            "n_heads": 8,
            "hidden_dim": 512,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        m1 = MLTrainingAdapter(seed=42).run_experiment(params)
        m2 = MLTrainingAdapter(seed=42).run_experiment(params)
        assert m1 == m2

    def test_different_seed_different_result(self) -> None:
        # Use moderate params that won't clamp to val_loss floor
        params = {
            "learning_rate": 1e-2,
            "batch_size": 32,
            "n_layers": 4,
            "n_heads": 2,
            "hidden_dim": 256,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "sgd",
            "activation": "relu",
        }
        m1 = MLTrainingAdapter(seed=42).run_experiment(params)
        m2 = MLTrainingAdapter(seed=99).run_experiment(params)
        assert m1["val_loss"] != pytest.approx(m2["val_loss"], abs=1e-10)


class TestMLTrainingMemory:
    """Tests for memory usage behavior."""

    def test_larger_model_uses_more_memory(self) -> None:
        """Larger hidden_dim and n_layers should use more memory."""
        small = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 2,
            "n_heads": 1,
            "hidden_dim": 128,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        large = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "n_layers": 24,
            "n_heads": 16,
            "hidden_dim": 2048,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "activation": "gelu",
        }
        small_mem = MLTrainingAdapter(seed=42).run_experiment(small)["memory_usage"]
        large_mem = MLTrainingAdapter(seed=42).run_experiment(large)["memory_usage"]
        assert large_mem > small_mem
