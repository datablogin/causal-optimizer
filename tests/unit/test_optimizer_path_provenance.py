"""Tests for optimizer-path provenance (Sprint 28).

Verifies that benchmark provenance captures which optimizer path ran
(Ax/BoTorch vs RF fallback) and the reason for fallback when applicable.
"""

from __future__ import annotations


class TestDetectOptimizerPath:
    """Tests for the optimizer path detection function."""

    def test_detect_returns_dict(self):
        from causal_optimizer.benchmarks.provenance import detect_optimizer_path

        result = detect_optimizer_path()
        assert isinstance(result, dict)

    def test_detect_has_required_keys(self):
        from causal_optimizer.benchmarks.provenance import detect_optimizer_path

        result = detect_optimizer_path()
        assert "optimizer_path" in result
        assert "ax_available" in result
        assert "botorch_available" in result
        assert "fallback_reason" in result

    def test_optimizer_path_is_valid_value(self):
        from causal_optimizer.benchmarks.provenance import detect_optimizer_path

        result = detect_optimizer_path()
        assert result["optimizer_path"] in ("ax_botorch", "rf_fallback")

    def test_ax_available_is_bool(self):
        from causal_optimizer.benchmarks.provenance import detect_optimizer_path

        result = detect_optimizer_path()
        assert isinstance(result["ax_available"], bool)

    def test_botorch_available_is_bool(self):
        from causal_optimizer.benchmarks.provenance import detect_optimizer_path

        result = detect_optimizer_path()
        assert isinstance(result["botorch_available"], bool)

    def test_fallback_reason_is_none_or_string(self):
        from causal_optimizer.benchmarks.provenance import detect_optimizer_path

        result = detect_optimizer_path()
        assert result["fallback_reason"] is None or isinstance(result["fallback_reason"], str)

    def test_path_consistent_with_availability(self):
        """If both ax and botorch are available, path should be ax_botorch.
        If either is missing, path should be rf_fallback."""
        from causal_optimizer.benchmarks.provenance import detect_optimizer_path

        result = detect_optimizer_path()
        if result["ax_available"] and result["botorch_available"]:
            assert result["optimizer_path"] == "ax_botorch"
            assert result["fallback_reason"] is None
        else:
            assert result["optimizer_path"] == "rf_fallback"
            assert result["fallback_reason"] is not None


class TestDetectOptimizerPathFallback:
    """Tests for the fallback detection path using mocks."""

    def test_rf_fallback_when_ax_missing(self):
        """When ax import fails, should detect rf_fallback."""
        from unittest.mock import patch

        from causal_optimizer.benchmarks.provenance import detect_optimizer_path

        with patch.dict("sys.modules", {"ax": None}):
            # Force reimport detection by calling the function
            # Note: detect_optimizer_path tries import ax directly
            pass

        # Use a more direct approach: mock the import itself
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "ax":
                raise ImportError("mocked: ax not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = detect_optimizer_path()

        assert result["optimizer_path"] == "rf_fallback"
        assert result["ax_available"] is False
        assert "ax-platform not installed" in result["fallback_reason"]

    def test_rf_fallback_when_both_missing(self):
        """When both ax and botorch imports fail, should detect rf_fallback."""
        import builtins
        from unittest.mock import patch

        from causal_optimizer.benchmarks.provenance import detect_optimizer_path

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("ax", "botorch"):
                raise ImportError(f"mocked: {name} not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = detect_optimizer_path()

        assert result["optimizer_path"] == "rf_fallback"
        assert result["ax_available"] is False
        assert result["botorch_available"] is False
        assert "ax-platform not installed" in result["fallback_reason"]
        assert "botorch not installed" in result["fallback_reason"]


class TestCollectProvenanceIncludesOptimizerPath:
    """Tests that collect_provenance includes optimizer path fields."""

    def test_provenance_has_optimizer_path_key(self):
        from causal_optimizer.benchmarks.provenance import collect_provenance

        prov = collect_provenance(seeds=[0], budgets=[20], strategies=["random"])
        assert "optimizer_path" in prov

    def test_provenance_optimizer_path_is_dict(self):
        from causal_optimizer.benchmarks.provenance import collect_provenance

        prov = collect_provenance(seeds=[0], budgets=[20], strategies=["random"])
        assert isinstance(prov["optimizer_path"], dict)

    def test_provenance_optimizer_path_has_required_keys(self):
        from causal_optimizer.benchmarks.provenance import collect_provenance

        prov = collect_provenance(seeds=[0], budgets=[20], strategies=["random"])
        op = prov["optimizer_path"]
        assert "optimizer_path" in op
        assert "ax_available" in op
        assert "botorch_available" in op
        assert "fallback_reason" in op
