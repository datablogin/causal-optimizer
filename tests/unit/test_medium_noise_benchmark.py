"""Tests for the medium-noise demand-response benchmark variant (Sprint 27).

MediumNoiseDemandResponse adds 4 nuisance dimensions to the base
demand-response benchmark (9 total: 3 real + 2 original noise + 4 nuisance),
sitting between the base (5D) and high-noise (15D) variants to map the
causal/surrogate crossover boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.benchmarks.counterfactual_energy import DemandResponseScenario
from causal_optimizer.benchmarks.counterfactual_variants import HighNoiseDemandResponse


class TestMediumNoiseDemandResponse:
    """Tests for the medium-noise variant."""

    def test_class_exists(self):
        """MediumNoiseDemandResponse should be importable."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        assert MediumNoiseDemandResponse is not None

    def test_search_space_has_9_variables(self):
        """Medium-noise should have 9 variables: 5 base + 4 nuisance."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        space = MediumNoiseDemandResponse.search_space()
        assert len(space.variables) == 9

    def test_search_space_includes_base_variables(self):
        """Medium-noise should include all base search space variables."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        base_names = {v.name for v in DemandResponseScenario.search_space().variables}
        medium_names = {v.name for v in MediumNoiseDemandResponse.search_space().variables}
        assert base_names.issubset(medium_names)

    def test_search_space_includes_4_nuisance_vars(self):
        """Medium-noise should add exactly 4 nuisance dimensions."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        space = MediumNoiseDemandResponse.search_space()
        nuisance = [v for v in space.variables if v.name.startswith("noise_var_")]
        assert len(nuisance) == 4

    def test_search_space_between_base_and_high_noise(self):
        """Medium-noise dimensionality should be between base and high-noise."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        base_count = len(DemandResponseScenario.search_space().variables)
        medium_count = len(MediumNoiseDemandResponse.search_space().variables)
        high_count = len(HighNoiseDemandResponse.search_space().variables)
        assert base_count < medium_count < high_count

    def test_causal_graph_excludes_nuisance_from_objective(self):
        """Nuisance variables should not be ancestors of objective."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        graph = MediumNoiseDemandResponse.causal_graph()
        ancestors = graph.ancestors("objective")
        for i in range(4):
            assert f"noise_var_{i}" not in ancestors

    def test_causal_graph_preserves_real_parents(self):
        """The 3 real parents should still be ancestors of objective."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        graph = MediumNoiseDemandResponse.causal_graph()
        ancestors = graph.ancestors("objective")
        for parent in ["treat_temp_threshold", "treat_hour_start", "treat_hour_end"]:
            assert parent in ancestors

    def test_generate_includes_nuisance_columns(self):
        """Generated data should include noise_var_0 through noise_var_3."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        scenario = MediumNoiseDemandResponse(covariates=_make_minimal_covariates(), seed=42)
        df = scenario.generate()
        for i in range(4):
            assert f"noise_var_{i}" in df.columns

    def test_generate_does_not_include_high_noise_vars(self):
        """Medium-noise should NOT have noise_var_4 through noise_var_9."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        scenario = MediumNoiseDemandResponse(covariates=_make_minimal_covariates(), seed=42)
        df = scenario.generate()
        for i in range(4, 10):
            assert f"noise_var_{i}" not in df.columns

    def test_oracle_matches_base(self):
        """Oracle should match the base variant (nuisance vars don't affect outcomes)."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        covariates = _make_minimal_covariates()
        base = DemandResponseScenario(covariates=covariates, seed=42)
        medium = MediumNoiseDemandResponse(covariates=covariates, seed=42)

        base_data = base.generate()
        medium_data = medium.generate()

        base_oracle = base.oracle_policy_value(base_data)
        medium_oracle = medium.oracle_policy_value(medium_data)

        assert base_oracle == pytest.approx(medium_oracle, rel=1e-6)

    def test_nuisance_vars_are_independent_uniform(self):
        """Nuisance vars should be uniform [0, 1] and independent of outcomes."""
        from causal_optimizer.benchmarks.counterfactual_variants import (
            MediumNoiseDemandResponse,
        )

        scenario = MediumNoiseDemandResponse(covariates=_make_minimal_covariates(), seed=42)
        df = scenario.generate()

        for i in range(4):
            col = df[f"noise_var_{i}"]
            assert col.min() >= 0.0
            assert col.max() <= 1.0
            # Should be roughly uniform
            assert 0.3 < col.mean() < 0.7


def _make_minimal_covariates():
    """Create minimal covariates for testing."""
    import pandas as pd

    n = 200
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
            "target_load": rng.normal(3000, 500, n),
            "temperature": rng.normal(25, 10, n),
            "humidity": rng.normal(50, 20, n),
            "hour_of_day": np.tile(np.arange(24), n // 24 + 1)[:n],
            "day_of_week": np.tile(np.arange(7), n // 7 + 1)[:n],
            "is_holiday": np.zeros(n, dtype=int),
            "load_lag_1h": rng.normal(3000, 500, n),
            "load_lag_24h": rng.normal(3000, 500, n),
        }
    )
