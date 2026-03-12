"""Regression test configuration.

All tests in this directory are marked slow by default. They run multi-seed
benchmark comparisons and take significant time. Deselect with:

    pytest -m "not slow"
"""

from __future__ import annotations

import pytest

# Apply @pytest.mark.slow to every test in this directory automatically.
pytestmark = pytest.mark.slow
