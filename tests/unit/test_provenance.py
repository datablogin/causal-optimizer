"""Tests for benchmark provenance metadata capture.

Verifies that collect_provenance() captures all required metadata fields,
handles missing optional packages gracefully, and produces stable dataset
hashes.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

from causal_optimizer.benchmarks.provenance import (
    collect_provenance,
    dataset_hash,
)

# ── Required keys ────────────────────────────────────────────────────


_REQUIRED_TOP_KEYS = {
    "git_sha",
    "python_version",
    "package_versions",
    "command_line",
    "timestamp",
    "seeds",
    "budgets",
    "strategies",
}


class TestCollectProvenanceKeys:
    """collect_provenance returns a dict with all required top-level keys."""

    def test_all_required_keys_present(self) -> None:
        prov = collect_provenance(
            seeds=[0, 1],
            budgets=[20, 40],
            strategies=["random", "causal"],
        )
        assert isinstance(prov, dict)
        missing = _REQUIRED_TOP_KEYS - set(prov.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_git_sha_is_string(self) -> None:
        prov = collect_provenance(seeds=[0], budgets=[10], strategies=["random"])
        assert isinstance(prov["git_sha"], str)
        # Should be a hex SHA or "unknown" if not in a repo
        assert len(prov["git_sha"]) >= 1

    def test_python_version_matches_runtime(self) -> None:
        prov = collect_provenance(seeds=[0], budgets=[10], strategies=["random"])
        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert prov["python_version"] == expected

    def test_timestamp_is_iso8601(self) -> None:
        prov = collect_provenance(seeds=[0], budgets=[10], strategies=["random"])
        ts = prov["timestamp"]
        assert isinstance(ts, str)
        # ISO 8601 contains 'T' separator and should be parseable
        assert "T" in ts
        from datetime import datetime

        datetime.fromisoformat(ts)  # raises ValueError if invalid

    def test_command_line_is_list_of_strings(self) -> None:
        prov = collect_provenance(seeds=[0], budgets=[10], strategies=["random"])
        assert isinstance(prov["command_line"], list)
        for item in prov["command_line"]:
            assert isinstance(item, str)

    def test_seeds_budgets_strategies_round_trip(self) -> None:
        seeds = [0, 1, 2]
        budgets = [10, 20, 40]
        strategies = ["random", "surrogate_only", "causal"]
        prov = collect_provenance(seeds=seeds, budgets=budgets, strategies=strategies)
        assert prov["seeds"] == seeds
        assert prov["budgets"] == budgets
        assert prov["strategies"] == strategies


# ── Package versions ─────────────────────────────────────────────────


class TestPackageVersions:
    """Package version capture handles both installed and missing packages."""

    def test_package_versions_is_dict(self) -> None:
        prov = collect_provenance(seeds=[0], budgets=[10], strategies=["random"])
        assert isinstance(prov["package_versions"], dict)

    def test_core_packages_present(self) -> None:
        """numpy, scipy, scikit-learn should always be installed."""
        prov = collect_provenance(seeds=[0], budgets=[10], strategies=["random"])
        versions = prov["package_versions"]
        for pkg in ("numpy", "scipy", "scikit-learn"):
            assert pkg in versions, f"Missing core package: {pkg}"
            assert versions[pkg] != "not installed", f"{pkg} should be installed"

    def test_missing_optional_package_records_not_installed(self) -> None:
        """Simulate a missing optional package -- should not crash."""
        from importlib.metadata import PackageNotFoundError

        original_version = __import__("importlib.metadata", fromlist=["version"]).version

        def fake_version(pkg: str) -> str:
            if pkg == "ax-platform":
                raise PackageNotFoundError(pkg)
            return original_version(pkg)

        with patch(
            "causal_optimizer.benchmarks.provenance.version", side_effect=fake_version
        ):
            prov = collect_provenance(seeds=[0], budgets=[10], strategies=["random"])
            versions = prov["package_versions"]
            assert isinstance(versions, dict)
            assert versions["ax-platform"] == "not installed"
            # Core packages should still resolve
            assert versions["numpy"] != "not installed"

    def test_optional_packages_listed(self) -> None:
        """All tracked optional packages appear in the dict."""
        prov = collect_provenance(seeds=[0], budgets=[10], strategies=["random"])
        versions = prov["package_versions"]
        expected_optional = {"ax-platform", "botorch", "torch", "gpytorch"}
        for pkg in expected_optional:
            assert pkg in versions, f"Optional package {pkg} not in versions dict"


# ── Dataset hash ─────────────────────────────────────────────────────


class TestDatasetHash:
    """dataset_hash produces stable, content-based hashes."""

    def test_stable_hash_same_content(self, tmp_path: Path) -> None:
        """Same file content produces the same hash."""
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n")
        h1 = dataset_hash(str(f))
        h2 = dataset_hash(str(f))
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different file content produces a different hash."""
        f1 = tmp_path / "data1.csv"
        f2 = tmp_path / "data2.csv"
        f1.write_text("a,b,c\n1,2,3\n")
        f2.write_text("a,b,c\n7,8,9\n")
        assert dataset_hash(str(f1)) != dataset_hash(str(f2))

    def test_hash_is_hex_string(self, tmp_path: Path) -> None:
        """Hash should be a hex-encoded SHA256 string."""
        f = tmp_path / "data.csv"
        f.write_text("hello\n")
        h = dataset_hash(str(f))
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex digest length
        int(h, 16)  # raises ValueError if not valid hex

    def test_missing_file_returns_none(self) -> None:
        """Non-existent file path should return None, not crash."""
        result = dataset_hash("/nonexistent/path/to/data.csv")
        assert result is None

    def test_hash_with_dataset_path_in_provenance(self, tmp_path: Path) -> None:
        """When dataset_path is provided, provenance includes dataset info."""
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\n1,2\n")
        prov = collect_provenance(
            seeds=[0],
            budgets=[10],
            strategies=["random"],
            dataset_path=str(f),
        )
        assert "dataset_path" in prov
        assert "dataset_hash" in prov
        assert prov["dataset_path"] == str(f)
        assert prov["dataset_hash"] is not None

    def test_provenance_missing_dataset_path(self) -> None:
        """When dataset_path is omitted, dataset fields are absent or None."""
        prov = collect_provenance(seeds=[0], budgets=[10], strategies=["random"])
        # Should not crash; dataset_path/hash are either absent or None
        assert prov.get("dataset_path") is None
        assert prov.get("dataset_hash") is None


# ── JSON serialization ───────────────────────────────────────────────


class TestProvenanceSerialization:
    """Provenance block must be JSON-serializable for artifact embedding."""

    def test_json_serializable(self) -> None:
        prov = collect_provenance(seeds=[0, 1], budgets=[20], strategies=["random"])
        # Should not raise
        dumped = json.dumps(prov)
        loaded = json.loads(dumped)
        assert loaded == prov

    def test_json_serializable_with_dataset(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("x,y\n1,2\n")
        prov = collect_provenance(
            seeds=[0],
            budgets=[10],
            strategies=["random"],
            dataset_path=str(f),
        )
        dumped = json.dumps(prov)
        loaded = json.loads(dumped)
        assert loaded == prov
