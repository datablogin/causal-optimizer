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
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from causal_optimizer.cli import (
    _adapter_engine_kwargs,
    _cmd_report,
    _cmd_run,
)

if TYPE_CHECKING:
    import pytest
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
        )
        _cmd_report(args)

        captured = capsys.readouterr()
        # The report should show the actual objective value, not 'N/A'
        assert "1.23" in captured.out

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
