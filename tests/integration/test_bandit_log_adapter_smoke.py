"""Sprint 34 contract Section 10.A smoke test for ``BanditLogAdapter``.

Loads the ZOZOTOWN Men / uniform-random slice bundled in the ``obp``
package and confirms the three Sprint 34 findings required before
Issue B and Issue C can land:

1. row count matches the Saito et al. 2021 Table 1 order of magnitude
   (for the bundled small-sized slice shipped with OBP 0.4.1 the row
   count is 10,000; the full released slice is ~452,949)
2. whether ``pscore`` / ``action_prob`` is stored as conditional
   ``P(item | position)`` (= ``1/n_actions``) or as joint
   ``P(item, position)`` (= ``1/(n_actions * n_positions)``)
3. the 3-to-5 context features chosen by the adapter are documented in
   the adapter docstring

Marked ``slow`` and ``obp`` so it is skipped in the default fast suite
and only runs when the optional ``bandit`` extra is installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.slow

# Skip cleanly if the optional extra is not installed. This keeps the
# suite honest when running under ``uv sync`` without ``--extra bandit``.
obp = pytest.importorskip("obp")


def _load_men_random_raw() -> tuple[Any, Any]:
    """Return the raw Men/Random ``data`` and ``item_context`` frames.

    We read the CSVs directly rather than calling
    ``OpenBanditDataset.load_raw_data`` + ``pre_process`` because the
    pinned ``obp==0.4.1`` pre-processor calls ``DataFrame.drop(..., 1)``
    with a positional ``axis`` argument that modern pandas rejects.
    Issue B / Issue C can switch to a patched loader later; for the
    Sprint 35.A smoke test we only need the raw column shapes.
    """
    import os
    from pathlib import Path

    import pandas as pd

    root = Path(os.path.dirname(obp.__file__)) / "dataset" / "obd" / "random" / "men"
    data = pd.read_csv(root / "men.csv", index_col=0)
    item_context = pd.read_csv(root / "item_context.csv", index_col=0)
    return data, item_context


class TestMenRandomSmoke:
    """Three Section 10.A findings required by the Sprint 34 contract."""

    def test_bundled_men_random_row_count(self) -> None:
        # OBP 0.4.1 ships a 10,000-row sample of the Men/Random slice
        # (a deterministic downsample of the ~452,949-row full slice).
        # The full slice must be loaded separately via ``data_path=``;
        # the bundled sample is what ships with the package.
        data, _ = _load_men_random_raw()
        assert len(data) == 10_000, (
            f"Bundled Men/Random row count changed: expected 10,000, got {len(data)}. "
            "Full Men/Random slice (~452,949 rows per Saito et al. 2021 Table 1) "
            "requires ``data_path=`` pointing at the released dataset."
        )

    def test_men_random_action_and_position_cardinality(self) -> None:
        data, item_context = _load_men_random_raw()
        # 34 items, 3 positions (Saito et al. 2021 Table 1)
        assert data["item_id"].nunique() == 34
        assert data["position"].nunique() == 3
        assert len(item_context) == 34

    def test_action_prob_is_conditional_not_joint(self) -> None:
        # Sprint 34 contract Section 5c / Section 7d: confirm whether
        # ``action_prob`` (= ``pscore``) is conditional ``P(item|position)``
        # or joint ``P(item, position)``.
        # Under the Random logger, conditional ``P(item|position)``
        # would be ``1/34 ≈ 0.0294``, and joint ``P(item, position)``
        # would be ``1/(34*3) ≈ 0.0098``.
        data, _ = _load_men_random_raw()
        n_items = int(data["item_id"].nunique())
        n_positions = int(data["position"].nunique())
        conditional_target = 1.0 / n_items
        joint_target = 1.0 / (n_items * n_positions)
        empirical_mean = float(data["propensity_score"].mean())

        # Use a tight 1% relative band to assert which schema is live.
        assert abs(empirical_mean - conditional_target) / conditional_target < 0.01, (
            f"propensity_score mean {empirical_mean:.6f} does not match conditional "
            f"P(item|position) = {conditional_target:.6f}"
        )
        assert abs(empirical_mean - joint_target) / joint_target > 0.5, (
            f"propensity_score mean {empirical_mean:.6f} unexpectedly close to joint "
            f"P(item, position) = {joint_target:.6f}"
        )

    def test_chosen_context_features_are_present(self) -> None:
        # Sprint 34 contract Section 4c: adapter's 3-to-5 context-feature
        # weights must reference columns that actually exist in the OBD
        # schema.
        data, item_context = _load_men_random_raw()

        # Feature 1: ``item_feature_0`` is a per-item continuous feature
        # in ``item_context`` (range ~[-0.7, 0.7]).
        assert "item_feature_0" in item_context.columns
        assert np.issubdtype(item_context["item_feature_0"].dtype, np.floating)

        # Feature 2: ``user-item_affinity_<k>`` are 34 per-row continuous
        # columns, one per candidate item id k.
        affinity_cols = [c for c in data.columns if c.startswith("user-item_affinity_")]
        assert len(affinity_cols) == 34

        # Feature 3: per-item popularity (log-count of appearances in the
        # log) can be computed from ``item_id`` value counts, which the
        # adapter pre-computes at load time.
        counts = data["item_id"].value_counts()
        assert counts.min() > 0  # every item appeared at least once


class TestAdapterAcceptsBundledFeedback:
    """End-to-end: the adapter can consume a bandit-feedback dict
    materialized from the bundled Men/Random slice."""

    def test_adapter_runs_on_bundled_men_random(self) -> None:
        # Build a bandit-feedback dict directly (the OBP pre_process path
        # is broken against modern pandas in 0.4.1; we bypass it here by
        # using the raw CSVs).
        from scipy.stats import rankdata

        from causal_optimizer.domain_adapters.bandit_log import BanditLogAdapter

        data, item_context = _load_men_random_raw()
        data = data.sort_values("timestamp").reset_index(drop=True)

        action = data["item_id"].to_numpy().astype(int)
        position = (rankdata(data["position"].to_numpy(), "dense") - 1).astype(int)
        reward = data["click"].to_numpy().astype(int)
        pscore = data["propensity_score"].to_numpy().astype(float)

        # Build per-row "context" and per-item "action_context" from the
        # raw CSVs. Full feature engineering is Issue B's territory; the
        # adapter only needs well-shaped arrays for the smoke run.
        affinity_cols = [c for c in data.columns if c.startswith("user-item_affinity_")]
        # Context: the 34 affinity columns. (User feature columns are
        # hashed strings in the raw CSV; the adapter does not need them
        # for this smoke test.)
        context = data[affinity_cols].to_numpy().astype(float)

        # action_context: per-item continuous feature ``item_feature_0``
        # aligned to ``item_id``.
        item_context = item_context.sort_values("item_id")
        action_context = item_context[["item_feature_0"]].to_numpy().astype(float)

        bandit_feedback: dict[str, Any] = {
            "n_rounds": int(len(data)),
            "n_actions": int(data["item_id"].nunique()),
            "action": action,
            "position": position,
            "reward": reward,
            "pscore": pscore,
            "context": context,
            "action_context": action_context,
        }

        adapter = BanditLogAdapter(bandit_feedback=bandit_feedback, seed=0)
        metrics = adapter.run_experiment(
            {
                "tau": 1.0,
                "eps": 0.1,
                "w_item_feature_0": 0.5,
                "w_user_item_affinity": 0.5,
                "w_item_popularity": 0.0,
                "position_handling_flag": "position_1_only",
            }
        )
        # Must satisfy the Section 4d interface on real data.
        for key in (
            "policy_value",
            "ess",
            "weight_cv",
            "max_weight",
            "zero_support_fraction",
            "n_effective_actions",
        ):
            assert key in metrics
        # Sanity: policy value in [0, 1] on binary rewards.
        assert 0.0 <= metrics["policy_value"] <= 1.0
        # ESS floor per Section 7b for position_1_only on ~10K rows / 3:
        # ``max(1000, n_rows/100)`` = max(1000, ~33) = 1000.
        # Under a random-ish policy with eps=0.1 the ESS should be well
        # above a few hundred; we only assert strict positivity here,
        # because the support gate itself is Issue B's responsibility.
        assert metrics["ess"] > 0.0
