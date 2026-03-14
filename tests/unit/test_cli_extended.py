"""Tests for extended CLI flags and DomainAdapter configuration hooks.

Covers:
- DomainAdapter configuration hook defaults
- CLI flags for --objective-name, --minimize/--maximize, --strategy, --discovery-method
- CLI flag override priority (CLI > adapter > engine default)
- Report using correct objective name
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from causal_optimizer.cli import (
    _adapter_engine_kwargs,
    _apply_cli_overrides,
    _cmd_report,
    _cmd_resume,
    _cmd_run,
)
from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import (
    Constraint,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    ObjectiveSpec,
    SearchSpace,
    Variable,
    VariableType,
)


class _StubAdapter(DomainAdapter):
    """Minimal adapter that returns a single variable and fixed metrics."""

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ]
        )

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        return {"objective": 0.5}


class _CustomAdapter(DomainAdapter):
    """Adapter that overrides all configuration hooks."""

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ]
        )

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        return {"loss": 0.42}

    def get_objective_name(self) -> str:
        return "loss"

    def get_minimize(self) -> bool:
        return False

    def get_strategy(self) -> str:
        return "causal_gp"

    def get_objectives(self) -> list[ObjectiveSpec] | None:
        return [ObjectiveSpec(name="loss", minimize=False)]

    def get_constraints(self) -> list[Constraint] | None:
        return [Constraint(metric_name="cost", upper_bound=100.0)]

    def get_discovery_method(self) -> str | None:
        return "correlation"


class TestAdapterConfigHooksDefaults:
    """Test that DomainAdapter configuration hooks have correct defaults."""

    def test_adapter_config_hooks_defaults(self) -> None:
        adapter = _StubAdapter()
        assert adapter.get_objective_name() == "objective"
        assert adapter.get_minimize() is True
        assert adapter.get_strategy() == "bayesian"
        assert adapter.get_objectives() is None
        assert adapter.get_constraints() is None
        assert adapter.get_discovery_method() is None


class TestAdapterEngineKwargs:
    """Test that _adapter_engine_kwargs pulls config from adapter hooks."""

    def test_custom_adapter_kwargs(self) -> None:
        adapter = _CustomAdapter()
        kwargs = _adapter_engine_kwargs(adapter)
        assert kwargs["objective_name"] == "loss"
        assert kwargs["minimize"] is False
        assert kwargs["strategy"] == "causal_gp"
        assert kwargs["objectives"] == [ObjectiveSpec(name="loss", minimize=False)]
        assert kwargs["constraints"] == [Constraint(metric_name="cost", upper_bound=100.0)]
        assert kwargs["discovery_method"] == "correlation"

    def test_default_adapter_kwargs_omits_none(self) -> None:
        adapter = _StubAdapter()
        kwargs = _adapter_engine_kwargs(adapter)
        assert kwargs["objective_name"] == "objective"
        assert kwargs["minimize"] is True
        assert kwargs["strategy"] == "bayesian"
        # None values should not be present
        assert "objectives" not in kwargs
        assert "constraints" not in kwargs
        assert "discovery_method" not in kwargs


class TestCLIMinimizeFlag:
    """Test --minimize flag."""

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_cli_minimize_flag(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = False
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = None
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id=None,
                seed=None,
                objective_name=None,
                minimize=True,
                maximize=False,
                strategy=None,
                discovery_method=None,
            )
            _cmd_run(args)

        _, call_kwargs = mock_engine_cls.call_args
        assert call_kwargs["minimize"] is True


class TestCLIMaximizeFlag:
    """Test --maximize flag."""

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_cli_maximize_flag(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = False
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = None
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id=None,
                seed=None,
                objective_name=None,
                minimize=False,
                maximize=True,
                strategy=None,
                discovery_method=None,
            )
            _cmd_run(args)

        _, call_kwargs = mock_engine_cls.call_args
        assert call_kwargs["minimize"] is False


class TestCLIStrategyFlag:
    """Test --strategy flag."""

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_cli_strategy_flag(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = False
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = None
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id=None,
                seed=None,
                objective_name=None,
                minimize=False,
                maximize=False,
                strategy="causal_gp",
                discovery_method=None,
            )
            _cmd_run(args)

        _, call_kwargs = mock_engine_cls.call_args
        assert call_kwargs["strategy"] == "causal_gp"


class TestCLIObjectiveNameFlag:
    """Test --objective-name flag."""

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_cli_objective_name_flag(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = False
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = None
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id=None,
                seed=None,
                objective_name="custom_metric",
                minimize=False,
                maximize=False,
                strategy=None,
                discovery_method=None,
            )
            _cmd_run(args)

        _, call_kwargs = mock_engine_cls.call_args
        assert call_kwargs["objective_name"] == "custom_metric"


class TestCLIDiscoveryMethodFlag:
    """Test --discovery-method flag."""

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_cli_discovery_method_flag(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = False
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = None
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id=None,
                seed=None,
                objective_name=None,
                minimize=False,
                maximize=False,
                strategy=None,
                discovery_method="pc",
            )
            _cmd_run(args)

        _, call_kwargs = mock_engine_cls.call_args
        assert call_kwargs["discovery_method"] == "pc"


class TestCLIFlagOverridesAdapter:
    """Test that CLI flags override adapter configuration hooks."""

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_cli_flag_overrides_adapter(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        # _CustomAdapter returns objective_name="loss", minimize=False, strategy="causal_gp"
        adapter = _CustomAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = False
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = None
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id=None,
                seed=None,
                # CLI flags override adapter hooks
                objective_name="my_metric",
                minimize=True,
                maximize=False,
                strategy="bayesian",
                discovery_method="notears",
            )
            _cmd_run(args)

        _, call_kwargs = mock_engine_cls.call_args
        assert call_kwargs["objective_name"] == "my_metric"
        assert call_kwargs["minimize"] is True
        assert call_kwargs["strategy"] == "bayesian"
        assert call_kwargs["discovery_method"] == "notears"

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_cli_flag_overrides_adapter_resume(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Verify CLI flags override adapter hooks in the resume path too."""
        adapter = _CustomAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = None
        mock_engine_cls.resume.return_value = mock_engine

        args = argparse.Namespace(
            adapter="mod:Cls",
            budget=5,
            db="test.db",
            id="existing-exp",
            # CLI flags override adapter hooks
            objective_name="override_obj",
            minimize=True,
            maximize=False,
            strategy="bayesian",
            discovery_method="notears",
        )
        _cmd_resume(args)

        _, call_kwargs = mock_engine_cls.resume.call_args
        assert call_kwargs["objective_name"] == "override_obj"
        assert call_kwargs["minimize"] is True
        assert call_kwargs["strategy"] == "bayesian"
        assert call_kwargs["discovery_method"] == "notears"


class TestReportUsesCorrectObjective:
    """Test that report command uses actual objective name, not hardcoded 'objective'."""

    @patch("causal_optimizer.cli.ExperimentStore")
    def test_report_uses_correct_objective(
        self,
        mock_store_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        best = ExperimentResult(
            experiment_id="r1",
            parameters={"x": 0.5},
            metrics={"custom_obj": 1.23, "other": 9.9},
            status=ExperimentStatus.KEEP,
            metadata={"phase": "exploration"},
        )
        log = ExperimentLog(results=[best])

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.load_log.return_value = log
        mock_store_cls.return_value = mock_store

        args = argparse.Namespace(
            id="test-exp",
            db="test.db",
            format="table",
            objective_name="custom_obj",
            minimize=False,
            maximize=False,
        )
        _cmd_report(args)

        captured = capsys.readouterr()
        # The report should show the actual objective value, not 'N/A'
        assert "1.23" in captured.out

    @patch("causal_optimizer.cli.ExperimentStore")
    def test_report_maximize_selects_largest(
        self,
        mock_store_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Report with --maximize should select the result with the largest value."""
        low = ExperimentResult(
            experiment_id="r-low",
            parameters={"x": 0.1},
            metrics={"score": 1.0},
            status=ExperimentStatus.KEEP,
            metadata={"phase": "exploration"},
        )
        high = ExperimentResult(
            experiment_id="r-high",
            parameters={"x": 0.9},
            metrics={"score": 9.0},
            status=ExperimentStatus.KEEP,
            metadata={"phase": "exploration"},
        )
        log = ExperimentLog(results=[low, high])

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.load_log.return_value = log
        mock_store_cls.return_value = mock_store

        args = argparse.Namespace(
            id="test-exp",
            db="test.db",
            format="table",
            objective_name="score",
            minimize=False,
            maximize=True,
        )
        _cmd_report(args)

        captured = capsys.readouterr()
        # With --maximize, the best result should be 9.0, not 1.0
        assert "9.0" in captured.out

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_run_prints_correct_objective(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = False
        mock_store_cls.return_value = mock_store

        best = ExperimentResult(
            experiment_id="r2",
            parameters={"x": 0.5},
            metrics={"my_obj": 3.14},
            status=ExperimentStatus.KEEP,
            metadata={},
        )
        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = best
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id=None,
                seed=None,
                objective_name="my_obj",
                minimize=False,
                maximize=False,
                strategy=None,
                discovery_method=None,
            )
            _cmd_run(args)

        captured = capsys.readouterr()
        assert "3.14" in captured.out


class TestApplyCliOverrides:
    """Direct unit tests for _apply_cli_overrides."""

    def test_no_flags_set_preserves_kwargs(self) -> None:
        kwargs: dict[str, Any] = {
            "objective_name": "loss",
            "minimize": False,
            "strategy": "causal_gp",
            "discovery_method": "correlation",
        }
        args = argparse.Namespace(
            objective_name=None,
            minimize=False,
            maximize=False,
            strategy=None,
            discovery_method=None,
        )
        _apply_cli_overrides(kwargs, args)
        # All values should be preserved from the adapter
        assert kwargs["objective_name"] == "loss"
        assert kwargs["minimize"] is False
        assert kwargs["strategy"] == "causal_gp"
        assert kwargs["discovery_method"] == "correlation"

    def test_all_flags_override(self) -> None:
        kwargs: dict[str, Any] = {
            "objective_name": "loss",
            "minimize": False,
            "strategy": "causal_gp",
        }
        args = argparse.Namespace(
            objective_name="revenue",
            minimize=True,
            maximize=False,
            strategy="bayesian",
            discovery_method="pc",
        )
        _apply_cli_overrides(kwargs, args)
        assert kwargs["objective_name"] == "revenue"
        assert kwargs["minimize"] is True
        assert kwargs["strategy"] == "bayesian"
        assert kwargs["discovery_method"] == "pc"

    def test_neither_minimize_nor_maximize_preserves_adapter(self) -> None:
        """When neither --minimize nor --maximize is set, adapter value is kept."""
        kwargs: dict[str, Any] = {"minimize": False}
        args = argparse.Namespace(
            objective_name=None,
            minimize=False,
            maximize=False,
            strategy=None,
            discovery_method=None,
        )
        _apply_cli_overrides(kwargs, args)
        assert kwargs["minimize"] is False


class TestBestResultUsesObjective:
    """Test that best_result() is called with configured objective and minimize."""

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_run_calls_best_result_with_objective(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = False
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = None
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id=None,
                seed=None,
                objective_name="custom",
                minimize=False,
                maximize=True,
                strategy=None,
                discovery_method=None,
            )
            _cmd_run(args)

        # Verify best_result was called with the correct objective_name and minimize
        mock_engine.log.best_result.assert_called_once_with(objective_name="custom", minimize=False)


class TestResumePrintsObjective:
    """Test that _cmd_resume prints the correct objective value."""

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_resume_prints_correct_objective(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store_cls.return_value = mock_store

        best = ExperimentResult(
            experiment_id="r3",
            parameters={"x": 0.5},
            metrics={"my_metric": 7.77},
            status=ExperimentStatus.KEEP,
            metadata={},
        )
        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = best
        mock_engine_cls.resume.return_value = mock_engine

        args = argparse.Namespace(
            adapter="mod:Cls",
            budget=5,
            db="test.db",
            id="existing-exp",
            objective_name="my_metric",
            minimize=False,
            maximize=True,
            strategy=None,
            discovery_method=None,
        )
        _cmd_resume(args)

        captured = capsys.readouterr()
        assert "7.77" in captured.out
        assert "my_metric" in captured.out

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_resume_not_found_exits(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """_cmd_resume exits with error when experiment ID is not found."""
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store_cls.return_value = mock_store

        mock_engine_cls.resume.side_effect = KeyError("not found")

        args = argparse.Namespace(
            adapter="mod:Cls",
            budget=5,
            db="test.db",
            id="nonexistent",
            objective_name=None,
            minimize=False,
            maximize=False,
            strategy=None,
            discovery_method=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_resume(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err


class TestReportJsonFormat:
    """Test _cmd_report with JSON output format."""

    @patch("causal_optimizer.cli.ExperimentStore")
    def test_report_json_includes_experiment_data(
        self,
        mock_store_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        result = ExperimentResult(
            experiment_id="r1",
            parameters={"x": 0.5},
            metrics={"loss": 0.42},
            status=ExperimentStatus.KEEP,
            metadata={"phase": "exploration"},
        )
        log = ExperimentLog(results=[result])

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.load_log.return_value = log
        mock_store_cls.return_value = mock_store

        args = argparse.Namespace(
            id="test-exp",
            db="test.db",
            format="json",
            objective_name="loss",
            minimize=False,
            maximize=False,
        )
        _cmd_report(args)

        captured = capsys.readouterr()
        import json

        data = json.loads(captured.out)
        assert data["experiment_id"] == "test-exp"
        assert data["total_steps"] == 1
        assert data["n_kept"] == 1
        assert data["best_result"] is not None

    @patch("causal_optimizer.cli.ExperimentStore")
    def test_report_no_kept_results(
        self,
        mock_store_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Report shows 'No kept results' when all experiments are discarded."""
        result = ExperimentResult(
            experiment_id="r1",
            parameters={"x": 0.5},
            metrics={"obj": 0.9},
            status=ExperimentStatus.DISCARD,
            metadata={"phase": "exploration"},
        )
        log = ExperimentLog(results=[result])

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.load_log.return_value = log
        mock_store_cls.return_value = mock_store

        args = argparse.Namespace(
            id="test-exp",
            db="test.db",
            format="table",
            objective_name=None,
            minimize=False,
            maximize=False,
        )
        _cmd_report(args)

        captured = capsys.readouterr()
        assert "No kept results." in captured.out

    @patch("causal_optimizer.cli.ExperimentStore")
    def test_report_not_found_exits(
        self,
        mock_store_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """_cmd_report exits with error when experiment ID is not found."""
        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.load_log.side_effect = KeyError("not found")
        mock_store_cls.return_value = mock_store

        args = argparse.Namespace(
            id="nonexistent",
            db="test.db",
            format="table",
            objective_name=None,
            minimize=False,
            maximize=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_report(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err


class TestRunWithSeed:
    """Test _cmd_run with --seed flag."""

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_run_passes_seed_to_engine(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = False
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.log.best_result.return_value = None
        mock_engine_cls.return_value = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id=None,
                seed=42,
                objective_name=None,
                minimize=False,
                maximize=False,
                strategy=None,
                discovery_method=None,
            )
            _cmd_run(args)

        _, call_kwargs = mock_engine_cls.call_args
        assert call_kwargs["seed"] == 42

    @patch("causal_optimizer.cli._load_adapter")
    @patch("causal_optimizer.cli.ExperimentEngine")
    @patch("causal_optimizer.cli.ExperimentStore")
    def test_run_existing_id_exits(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_load: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """_cmd_run exits with error when experiment ID already exists."""
        adapter = _StubAdapter()
        mock_load.return_value = adapter

        mock_store = MagicMock()
        mock_store.__enter__ = MagicMock(return_value=mock_store)
        mock_store.__exit__ = MagicMock(return_value=False)
        mock_store.experiment_exists.return_value = True
        mock_store_cls.return_value = mock_store

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            args = argparse.Namespace(
                adapter="mod:Cls",
                budget=1,
                db=f.name,
                id="existing-id",
                seed=None,
                objective_name=None,
                minimize=False,
                maximize=False,
                strategy=None,
                discovery_method=None,
            )
            with pytest.raises(SystemExit) as exc_info:
                _cmd_run(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "already exists" in captured.err
