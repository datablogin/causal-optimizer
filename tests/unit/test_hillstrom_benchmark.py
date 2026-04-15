"""Unit tests for the Sprint 31 Hillstrom benchmark harness.

Covers the invariants pinned by the Sprint 31 Hillstrom benchmark contract:

1. slice mapping: primary ``Womens E-Mail vs No E-Mail`` and pooled
   ``Any E-Mail vs No E-Mail``.
2. per-slice propensity: exactly ``0.5`` on the primary slice,
   exactly ``2.0 / 3.0`` on the pooled slice.
3. projected prior graph: 7 edges over the active-only subgraph with
   the 3 frozen Hillstrom dimensions removed.
4. active search space: exactly the 3 tuned dimensions
   (``eligibility_threshold``, ``regularization``,
   ``treatment_budget_pct``).
5. wrapped runner pre-bakes the 3 frozen dimensions into every
   forwarded parameter dict.
6. null-control permutation preserves the marginal of ``spend`` and
   is deterministic under a fixed seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.benchmarks.hillstrom import (
    HILLSTROM_FROZEN_PARAMS,
    HILLSTROM_POOLED_PROPENSITY,
    HILLSTROM_PRIMARY_PROPENSITY,
    HillstromPolicyRunner,
    HillstromSliceType,
    hillstrom_active_search_space,
    hillstrom_null_baseline,
    hillstrom_projected_prior_graph,
    load_hillstrom_slice,
    permute_hillstrom_spend,
)
from causal_optimizer.domain_adapters.marketing_logs import MarketingLogAdapter
from causal_optimizer.types import VariableType

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "hillstrom_fixture.csv"


@pytest.fixture
def raw_hillstrom() -> pd.DataFrame:
    """Load the committed Hillstrom-shaped fixture CSV."""
    return pd.read_csv(FIXTURE_PATH)


# ── Slice mapping ────────────────────────────────────────────────────


class TestSliceMapping:
    """Primary and pooled slice column mapping."""

    def test_primary_slice_drops_mens(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        mens_count = int((raw_hillstrom["segment"] == "Mens E-Mail").sum())
        assert mens_count > 0, "fixture should contain Mens E-Mail rows"
        expected_n = len(raw_hillstrom) - mens_count
        assert len(df) == expected_n

    def test_primary_slice_has_both_arms(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        assert set(df["treatment"].unique()) == {0, 1}

    def test_primary_slice_treatment_maps_correctly(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        # Count Womens rows in the fixture — they should all map to treatment=1
        womens_in_raw = int((raw_hillstrom["segment"] == "Womens E-Mail").sum())
        assert int((df["treatment"] == 1).sum()) == womens_in_raw
        # Every control row should have come from "No E-Mail" in the raw frame
        control_in_raw = int((raw_hillstrom["segment"] == "No E-Mail").sum())
        assert int((df["treatment"] == 0).sum()) == control_in_raw

    def test_pooled_slice_keeps_all_rows(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.POOLED)
        assert len(df) == len(raw_hillstrom)

    def test_pooled_slice_treatment_is_any_email(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.POOLED)
        email_count = int(
            (
                (raw_hillstrom["segment"] == "Mens E-Mail")
                | (raw_hillstrom["segment"] == "Womens E-Mail")
            ).sum()
        )
        assert int((df["treatment"] == 1).sum()) == email_count

    def test_outcome_is_spend(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        assert "outcome" in df.columns
        # outcome should be a pass-through of the original spend values for the
        # non-dropped rows
        non_mens = raw_hillstrom[raw_hillstrom["segment"] != "Mens E-Mail"]
        np.testing.assert_allclose(np.sort(df["outcome"].values), np.sort(non_mens["spend"].values))

    def test_channel_constant_email(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        assert (df["channel"] == "email").all()

    def test_segment_bucketed_from_history_segment(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        assert set(df["segment"].unique()) <= {"low", "medium", "high_value"}
        # Every high_value row must come from one of the top three history_segment
        # buckets in the raw frame
        high_raw_segments = raw_hillstrom[
            raw_hillstrom["history_segment"].isin(
                ["5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +"]
            )
        ]
        # After dropping Mens, the count of high_value rows in the reshaped frame
        # must equal the count of high-segment non-Mens rows in the raw frame
        high_non_mens = int((high_raw_segments["segment"] != "Mens E-Mail").sum())
        assert int((df["segment"] == "high_value").sum()) == high_non_mens

    def test_cost_constant(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        # Treated rows share a single fixed cost; control rows are zero
        treated_costs = df.loc[df["treatment"] == 1, "cost"].unique()
        control_costs = df.loc[df["treatment"] == 0, "cost"].unique()
        assert len(treated_costs) == 1
        assert len(control_costs) == 1
        assert float(treated_costs[0]) > 0.0
        assert float(control_costs[0]) == 0.0


# ── Propensity invariants ────────────────────────────────────────────


class TestPropensityInvariants:
    """The two contract-level propensity pin-downs."""

    def test_primary_propensity_exactly_half(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        # Must be exactly 0.5 on every row — no rounded 0.5000001 drift
        assert (df["propensity"] == 0.5).all()
        assert HILLSTROM_PRIMARY_PROPENSITY == 0.5

    def test_pooled_propensity_exactly_two_thirds(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.POOLED)
        expected = 2.0 / 3.0
        # Every row must equal 2/3 exactly (bitwise equality after 2.0/3.0
        # computation); this is the invariant the Sprint 31 smoke test asserts
        assert (df["propensity"] == expected).all()
        assert expected == HILLSTROM_POOLED_PROPENSITY

    def test_pooled_propensity_is_not_one_half(self) -> None:
        # Regression guard against the "most likely implementation bug" flagged
        # in the Sprint 31 contract: swapping 0.5 onto the pooled slice.
        assert HILLSTROM_POOLED_PROPENSITY != 0.5


# ── Projected prior graph ────────────────────────────────────────────


class TestProjectedPriorGraph:
    """The 7-edge sub-DAG enumerated in Sprint 31 contract Section 4a.i."""

    def test_edge_count_is_seven(self) -> None:
        graph = hillstrom_projected_prior_graph()
        assert len(graph.edges) == 7

    def test_edges_match_contract_exactly(self) -> None:
        graph = hillstrom_projected_prior_graph()
        expected = {
            ("eligibility_threshold", "treated_fraction"),
            ("treatment_budget_pct", "treated_fraction"),
            ("regularization", "treated_fraction"),
            ("regularization", "policy_value"),
            ("treated_fraction", "total_cost"),
            ("treated_fraction", "policy_value"),
            ("treated_fraction", "effective_sample_size"),
        }
        assert set(graph.edges) == expected

    def test_no_frozen_variable_is_a_tail(self) -> None:
        graph = hillstrom_projected_prior_graph()
        frozen_nodes = {"email_share", "social_share_of_remainder", "min_propensity_clip"}
        tails = {u for u, _ in graph.edges}
        assert not (tails & frozen_nodes), "no frozen variable may appear as an edge tail"

    def test_no_frozen_variable_appears_as_node(self) -> None:
        graph = hillstrom_projected_prior_graph()
        frozen_nodes = {"email_share", "social_share_of_remainder", "min_propensity_clip"}
        assert not (set(graph.nodes) & frozen_nodes)

    def test_full_adapter_graph_has_all_seven_projected_edges(
        self, raw_hillstrom: pd.DataFrame
    ) -> None:
        # Sanity check: the 7 projected edges must all be present in the
        # full adapter graph — i.e., the projection is a strict sub-DAG.
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        adapter = MarketingLogAdapter(data=df, seed=0)
        full = set(adapter.get_prior_graph().edges)
        projected = set(hillstrom_projected_prior_graph().edges)
        assert projected.issubset(full)
        # And the projection must drop exactly 7 edges (14 - 7 = 7)
        assert len(full - projected) == 7


# ── Active search space ──────────────────────────────────────────────


class TestActiveSearchSpace:
    """The 3-variable tuned search space."""

    def test_active_space_has_three_variables(self) -> None:
        space = hillstrom_active_search_space()
        assert len(space.variables) == 3

    def test_active_space_names(self) -> None:
        space = hillstrom_active_search_space()
        assert set(space.variable_names) == {
            "eligibility_threshold",
            "regularization",
            "treatment_budget_pct",
        }

    def test_active_space_does_not_contain_frozen_vars(self) -> None:
        space = hillstrom_active_search_space()
        names = set(space.variable_names)
        assert "email_share" not in names
        assert "social_share_of_remainder" not in names
        assert "min_propensity_clip" not in names

    def test_active_space_bounds_match_adapter(self) -> None:
        # Bounds must be inherited from MarketingLogAdapter to keep the
        # search space numerically identical to the adapter's native ranges
        # for the 3 tuned dimensions.
        space = hillstrom_active_search_space()
        adapter_space = MarketingLogAdapter(
            data=pd.DataFrame(
                {
                    "treatment": [0, 1],
                    "outcome": [0.0, 1.0],
                    "cost": [0.0, 0.0],
                }
            )
        ).get_search_space()
        adapter_vars = {v.name: v for v in adapter_space.variables}
        for var in space.variables:
            assert var.variable_type == VariableType.CONTINUOUS
            assert var.lower == adapter_vars[var.name].lower
            assert var.upper == adapter_vars[var.name].upper


class TestFrozenParameterConstants:
    """The frozen-dimension constants must match the Sprint 31 contract."""

    def test_frozen_params_dict(self) -> None:
        assert HILLSTROM_FROZEN_PARAMS == {
            "email_share": 1.0,
            "social_share_of_remainder": 0.0,
            "min_propensity_clip": 0.01,
        }


# ── Wrapped runner ───────────────────────────────────────────────────


class TestHillstromPolicyRunner:
    """The runner that pre-bakes frozen dims into every parameter dict."""

    def test_runner_injects_frozen_params(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        adapter = MarketingLogAdapter(data=df, seed=0)
        runner = HillstromPolicyRunner(adapter=adapter)
        active_params = {
            "eligibility_threshold": 0.3,
            "regularization": 1.0,
            "treatment_budget_pct": 0.5,
        }
        # The runner must accept an active-only dict and still produce
        # finite policy_value / total_cost
        metrics = runner.run(active_params)
        assert "policy_value" in metrics
        assert np.isfinite(metrics["policy_value"])
        # ... and it must not mutate the caller's dict
        assert "email_share" not in active_params
        assert "min_propensity_clip" not in active_params

    def test_runner_forwarded_params_match_frozen_constants(
        self, raw_hillstrom: pd.DataFrame
    ) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        adapter = MarketingLogAdapter(data=df, seed=0)
        runner = HillstromPolicyRunner(adapter=adapter)
        active_params = {
            "eligibility_threshold": 0.0,
            "regularization": 0.001,
            "treatment_budget_pct": 1.0,
        }
        forwarded = runner._forward_params(active_params)
        for key, val in HILLSTROM_FROZEN_PARAMS.items():
            assert forwarded[key] == val, f"{key}: frozen at {val}, got {forwarded[key]}"
        for key in ("eligibility_threshold", "regularization", "treatment_budget_pct"):
            assert forwarded[key] == active_params[key]


# ── Null-control path ────────────────────────────────────────────────


class TestNullControl:
    """Permuted-outcome null control setup."""

    def test_permutation_preserves_multiset(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        shuffled = permute_hillstrom_spend(df, seed=0)
        # Multiset equality: every original value appears exactly once in the shuffled
        np.testing.assert_allclose(
            np.sort(shuffled["outcome"].values), np.sort(df["outcome"].values)
        )

    def test_permutation_preserves_treatment_column(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        shuffled = permute_hillstrom_spend(df, seed=0)
        assert (shuffled["treatment"].values == df["treatment"].values).all()

    def test_permutation_preserves_propensity_column(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        shuffled = permute_hillstrom_spend(df, seed=0)
        np.testing.assert_array_equal(shuffled["propensity"].values, df["propensity"].values)

    def test_permutation_is_deterministic_under_same_seed(
        self, raw_hillstrom: pd.DataFrame
    ) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        a = permute_hillstrom_spend(df, seed=42)
        b = permute_hillstrom_spend(df, seed=42)
        np.testing.assert_array_equal(a["outcome"].values, b["outcome"].values)

    def test_null_baseline_equals_raw_mean_spend(self, raw_hillstrom: pd.DataFrame) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        mu = hillstrom_null_baseline(df)
        # Contract Section 5g: μ = mean(spend) on the reshaped frame, and
        # shuffling preserves the column so permuted frames have the same μ
        expected = float(df["outcome"].mean())
        assert mu == expected
        shuffled = permute_hillstrom_spend(df, seed=123)
        assert hillstrom_null_baseline(shuffled) == mu

    def test_null_control_runs_on_fixture(self, raw_hillstrom: pd.DataFrame) -> None:
        """Smoke test: the adapter + runner pipeline runs end-to-end on a
        shuffled frame without raising, returning a finite policy_value.
        """
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        shuffled = permute_hillstrom_spend(df, seed=7)
        adapter = MarketingLogAdapter(data=shuffled, seed=0)
        runner = HillstromPolicyRunner(adapter=adapter)
        metrics = runner.run(
            {
                "eligibility_threshold": 0.2,
                "regularization": 0.5,
                "treatment_budget_pct": 0.5,
            }
        )
        assert np.isfinite(metrics["policy_value"])
        assert np.isfinite(metrics["effective_sample_size"])


# ── Defensive guards + HillstromScenario property coverage ─────────


class TestDefensiveGuards:
    """Validation errors and property accessors."""

    def test_load_hillstrom_slice_rejects_missing_columns(self) -> None:
        bad = pd.DataFrame({"segment": ["Womens E-Mail"], "spend": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            load_hillstrom_slice(bad, slice_type=HillstromSliceType.PRIMARY)

    def test_permute_hillstrom_spend_rejects_missing_outcome(
        self, raw_hillstrom: pd.DataFrame
    ) -> None:
        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        with pytest.raises(ValueError, match="column 'not_an_outcome' not in reshaped frame"):
            permute_hillstrom_spend(df, seed=0, outcome_col="not_an_outcome")

    def test_scenario_slice_type_property(self, raw_hillstrom: pd.DataFrame) -> None:
        from causal_optimizer.benchmarks.hillstrom import HillstromScenario

        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.POOLED)
        assert scenario.slice_type is HillstromSliceType.POOLED

    def test_secondary_outcomes_empty_when_visit_or_conversion_missing(
        self, raw_hillstrom: pd.DataFrame
    ) -> None:
        from causal_optimizer.benchmarks.hillstrom import _secondary_outcomes_under_policy

        df = load_hillstrom_slice(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        # Drop the retained secondary outcome columns; the guard must
        # return an empty dict rather than raising or silently producing
        # garbage aggregates.
        without_secondary = df.drop(columns=["visit", "conversion"])
        result = _secondary_outcomes_under_policy(without_secondary, {})
        assert result == {}
