"""Tests for public API re-exports and packaging metadata."""

from __future__ import annotations


class TestTopLevelImports:
    """Test that all key public API types are importable from the top-level package."""

    def test_experiment_engine(self) -> None:
        from causal_optimizer import ExperimentEngine

        assert ExperimentEngine is not None

    def test_causal_graph(self) -> None:
        from causal_optimizer import CausalGraph

        assert CausalGraph is not None

    def test_constraint(self) -> None:
        from causal_optimizer import Constraint

        assert Constraint is not None

    def test_experiment_log(self) -> None:
        from causal_optimizer import ExperimentLog

        assert ExperimentLog is not None

    def test_experiment_result(self) -> None:
        from causal_optimizer import ExperimentResult

        assert ExperimentResult is not None

    def test_experiment_status(self) -> None:
        from causal_optimizer import ExperimentStatus

        assert ExperimentStatus is not None

    def test_objective_spec(self) -> None:
        from causal_optimizer import ObjectiveSpec

        assert ObjectiveSpec is not None

    def test_search_space(self) -> None:
        from causal_optimizer import SearchSpace

        assert SearchSpace is not None

    def test_variable(self) -> None:
        from causal_optimizer import Variable

        assert Variable is not None

    def test_variable_type(self) -> None:
        from causal_optimizer import VariableType

        assert VariableType is not None

    def test_all_imports_at_once(self) -> None:
        """Verify all public API names can be imported in a single statement."""
        from causal_optimizer import (
            CausalGraph,
            Constraint,
            ExperimentEngine,
            ExperimentLog,
            ExperimentResult,
            ExperimentStatus,
            ObjectiveSpec,
            SearchSpace,
            Variable,
            VariableType,
        )

        # All should be the real classes, not None
        for cls in [
            CausalGraph,
            Constraint,
            ExperimentEngine,
            ExperimentLog,
            ExperimentResult,
            ExperimentStatus,
            ObjectiveSpec,
            SearchSpace,
            Variable,
            VariableType,
        ]:
            assert cls is not None

    def test_imports_are_correct_types(self) -> None:
        """Verify re-exports point to the actual implementation classes."""
        from causal_optimizer import ExperimentEngine
        from causal_optimizer.engine.loop import ExperimentEngine as DirectEngine

        assert ExperimentEngine is DirectEngine

        from causal_optimizer import SearchSpace
        from causal_optimizer.types import SearchSpace as DirectSearchSpace

        assert SearchSpace is DirectSearchSpace


class TestStorageImport:
    """Test that ExperimentStore is importable from causal_optimizer.storage."""

    def test_storage_import(self) -> None:
        from causal_optimizer.storage import ExperimentStore

        assert ExperimentStore is not None

    def test_storage_import_is_correct_type(self) -> None:
        from causal_optimizer.storage import ExperimentStore
        from causal_optimizer.storage.sqlite import ExperimentStore as DirectStore

        assert ExperimentStore is DirectStore


class TestEngineImport:
    """Test that ExperimentEngine is importable from causal_optimizer.engine."""

    def test_engine_import(self) -> None:
        from causal_optimizer.engine import ExperimentEngine

        assert ExperimentEngine is not None

    def test_engine_import_is_correct_type(self) -> None:
        from causal_optimizer.engine import ExperimentEngine
        from causal_optimizer.engine.loop import ExperimentEngine as DirectEngine

        assert ExperimentEngine is DirectEngine


class TestDunderAll:
    """Test that __all__ is defined and complete."""

    def test_top_level_all(self) -> None:
        import causal_optimizer

        assert hasattr(causal_optimizer, "__all__")
        expected = {
            "CausalGraph",
            "Constraint",
            "ExperimentEngine",
            "ExperimentLog",
            "ExperimentResult",
            "ExperimentStatus",
            "ObjectiveSpec",
            "SearchSpace",
            "Variable",
            "VariableType",
        }
        assert expected <= set(causal_optimizer.__all__)
