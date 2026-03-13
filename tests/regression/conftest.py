"""Regression test configuration.

All tests in this directory should be marked ``@pytest.mark.slow``.
Deselect with: ``pytest -m "not slow"``

Note: ``pytestmark`` in conftest.py does NOT propagate to sibling test
modules — each test class must be decorated with ``@pytest.mark.slow``
explicitly. This conftest exists for shared fixtures and configuration.
"""

from __future__ import annotations
