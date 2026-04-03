"""Benchmark provenance metadata capture for reproducible artifacts.

Provides utilities to record the runtime environment, package versions,
dataset identity, and run parameters alongside benchmark JSON output.
This metadata allows diagnosing environment drift when benchmark results
change across reruns.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Packages whose versions are tracked in provenance metadata.
# Core packages are always installed; optional packages may be absent.
_CORE_PACKAGES: list[str] = ["numpy", "scipy", "scikit-learn"]
_OPTIONAL_PACKAGES: list[str] = ["ax-platform", "botorch", "torch", "gpytorch"]
_ALL_TRACKED_PACKAGES: list[str] = _CORE_PACKAGES + _OPTIONAL_PACKAGES


def _get_git_sha() -> str:
    """Return the current git commit SHA, or ``"unknown"`` if unavailable."""
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def _get_package_versions() -> dict[str, str]:
    """Collect installed versions for all tracked packages.

    Missing optional packages are recorded as ``"not installed"`` rather
    than raising an error.
    """
    versions: dict[str, str] = {}
    for pkg in _ALL_TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def dataset_hash(path: str) -> str | None:
    """Compute a SHA-256 hex digest of a file's contents.

    Reads the file in 64 KiB chunks to avoid loading large files entirely
    into memory.

    Args:
        path: Filesystem path to the file.

    Returns:
        64-character hex string, or ``None`` if the file does not exist
        or cannot be read.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def collect_provenance(
    *,
    seeds: list[int],
    budgets: list[int],
    strategies: list[str],
    dataset_path: str | None = None,
) -> dict[str, object]:
    """Capture a provenance metadata block for a benchmark run.

    The returned dict is JSON-serializable and designed to be added as a
    ``"provenance"`` key in benchmark JSON artifacts.  Existing readers
    that do not expect this key will ignore it.

    Args:
        seeds: RNG seeds used in the run.
        budgets: Experiment budgets used.
        strategies: Strategy names used.
        dataset_path: Optional path to the dataset file.  When provided,
            the file is hashed to enable content-based identity checks.

    Returns:
        A JSON-serializable dict with provenance metadata including
        ``git_sha``, ``python_version``, ``package_versions``,
        ``command_line``, ``timestamp``, ``seeds``, ``budgets``,
        ``strategies``, and optionally ``dataset_path`` / ``dataset_hash``.
    """
    vi = sys.version_info
    python_version = f"{vi.major}.{vi.minor}.{vi.micro}"

    prov: dict[str, object] = {
        "git_sha": _get_git_sha(),
        "python_version": python_version,
        "package_versions": _get_package_versions(),
        "command_line": sys.argv[:],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "budgets": budgets,
        "strategies": strategies,
        "dataset_path": None,
        "dataset_hash": None,
    }

    if dataset_path is not None:
        prov["dataset_path"] = dataset_path
        prov["dataset_hash"] = dataset_hash(dataset_path)

    return prov
