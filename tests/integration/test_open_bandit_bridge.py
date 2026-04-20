"""Sprint 35 bridge integration test: wire ``BanditLogAdapter`` into the
Open Bandit OPE path.

Covers the three known seams in a single end-to-end flow so Issue #187
(the Sprint 35.C benchmark report) can consume a ready-to-run path:

1. **Position normalization.** Track B's ``_validate_positions`` fails
   loudly on anything other than 0-indexed contiguous integers. The
   bridge must remap OBD-style positions before the OPE wrapper (and
   ``run_section_7_gates``) consume them.
2. **Conditional propensity schema propagation.** Men/Random's
   ``action_prob`` is conditional ``P(item | position)``; the
   :func:`propensity_sanity_gate` default is joint. The bridge must
   thread ``PROPENSITY_SCHEMA_CONDITIONAL`` into the gate when the
   adapter was built for Men/Random so the gate targets ``1/n_items``
   instead of ``1/(n_items * n_positions)``.
3. **Non-placeholder OBP version provenance.** The merged Track B
   placeholder (``OBP_VERSION_PLACEHOLDER``) must be swapped for the
   real ``obp.__version__`` string in the ``GateReport.provenance``
   dict. The bridge writes the version via a public helper so a missing
   optional extra reports an explicit sentinel rather than crashing.

The test also exercises the narrow ``BanditLogAdapter.to_bandit_feedback``
helper surface, which converts adapter state into the OBP-shaped dict
the Track B evaluator consumes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causal_optimizer.benchmarks.open_bandit import (
    OBP_VERSION_PLACEHOLDER,
    PROPENSITY_SCHEMA_CONDITIONAL,
    PROPENSITY_SCHEMA_JOINT,
    GateReport,
    evaluate_open_bandit_policy,
    get_obp_version,
    normalize_positions,
    propensity_sanity_gate,
    run_section_7_gates,
    uniform_policy,
)
from causal_optimizer.domain_adapters.bandit_log import BanditLogAdapter


def _build_men_random_like_feedback(
    *, n_rounds: int, n_actions: int, n_positions: int, seed: int
) -> dict[str, Any]:
    """Return a Men/Random-shaped feedback dict for the bridge test.

    Uses conditional ``P(item | position) = 1 / n_actions`` to mirror the
    empirical OBD finding from Sprint 35.A. Positions are deliberately
    emitted 1-indexed so the test can prove the bridge remaps them.
    """
    rng = np.random.default_rng(seed)
    action = rng.integers(low=0, high=n_actions, size=n_rounds).astype(np.int64)
    # Emit positions as 1-indexed + non-contiguous (1, 2, 3 with gaps
    # later simulated by picking {1, 3}), to exercise the remap path.
    raw_positions = rng.choice([1, 2, 3], size=n_rounds).astype(np.int64)
    reward = rng.binomial(n=1, p=0.03, size=n_rounds).astype(float)
    pscore = np.full(n_rounds, 1.0 / n_actions, dtype=float)
    context = rng.standard_normal((n_rounds, n_actions)).astype(float)
    action_context = rng.standard_normal((n_actions, 1)).astype(float)
    return {
        "n_rounds": int(n_rounds),
        "n_actions": int(n_actions),
        "action": action,
        "position": raw_positions,
        "reward": reward,
        "pscore": pscore,
        "context": context,
        "action_context": action_context,
    }


# ── Seam 1: position normalization ────────────────────────────────────


class TestPositionNormalization:
    """``normalize_positions`` remaps any integer labels to 0-indexed
    contiguous integers while preserving row-level ordering."""

    def test_normalize_positions_remaps_one_indexed_to_zero_indexed(self) -> None:
        raw = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
        out = normalize_positions(raw)
        assert out.dtype == np.int64
        # Minimum drops to 0; unique values become {0, 1, 2}.
        assert int(out.min()) == 0
        assert sorted(np.unique(out).tolist()) == [0, 1, 2]
        # Row ordering of the *ranks* must be preserved.
        # Position 1 rows map to rank 0, position 2 to rank 1, position 3 to 2.
        assert out.tolist() == [0, 1, 2, 0, 1, 2]

    def test_normalize_positions_handles_gaps(self) -> None:
        # OBD loaders occasionally emit non-contiguous integer labels
        # (e.g. after filtering a campaign). The remapped output must
        # collapse the gap so Track B's ``_validate_positions`` passes.
        raw = np.array([5, 5, 10, 10, 20], dtype=np.int64)
        out = normalize_positions(raw)
        assert out.tolist() == [0, 0, 1, 1, 2]

    def test_normalize_positions_idempotent_on_already_normalized(self) -> None:
        raw = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        out = normalize_positions(raw)
        assert out.tolist() == raw.tolist()

    def test_normalize_positions_empty_array(self) -> None:
        raw = np.array([], dtype=np.int64)
        out = normalize_positions(raw)
        assert out.shape == (0,)


# ── Seam 2: OBP version provenance ────────────────────────────────────


class TestObpVersionProvenance:
    """``get_obp_version`` returns a real version string when the
    optional extra is installed and a sentinel otherwise."""

    def test_get_obp_version_returns_real_version_when_installed(self) -> None:
        obp = pytest.importorskip("obp", reason="requires optional 'bandit' extra")
        version = get_obp_version()
        assert isinstance(version, str)
        assert version == obp.__version__
        # Most importantly: it is not the placeholder that Track B shipped.
        assert version != OBP_VERSION_PLACEHOLDER
        # Pinned to the 0.4.x series (see pyproject bandit extra).
        assert version.startswith("0.4.")

    def test_get_obp_version_returns_sentinel_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate a caller without the ``bandit`` extra. The bridge
        # helper must not raise; it must return a clearly-marked
        # sentinel string so downstream provenance dicts keep the key
        # under graceful-degradation conditions.
        import builtins

        real_import = builtins.__import__

        def _reject_obp(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "obp" or name.startswith("obp."):
                raise ImportError(f"simulated: {name} not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _reject_obp)
        version = get_obp_version()
        assert isinstance(version, str)
        assert version != OBP_VERSION_PLACEHOLDER
        assert "unavailable" in version.lower() or "not installed" in version.lower()

    def test_get_obp_version_returns_sentinel_when_version_attr_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Defense-in-depth: a forked OBP without ``__version__`` must
        # still produce a usable provenance string rather than raising.
        # This test needs ``obp`` importable so we can strip the
        # attribute off a real module object; skip cleanly if the
        # optional ``bandit`` extra is not installed.
        obp = pytest.importorskip("obp", reason="requires optional 'bandit' extra")

        monkeypatch.delattr(obp, "__version__", raising=False)
        version = get_obp_version()
        assert isinstance(version, str)
        assert version != OBP_VERSION_PLACEHOLDER
        assert "unavailable" in version.lower() or "not installed" in version.lower()


# ── Seam 3: conditional schema propagation ────────────────────────────


class TestConditionalSchemaPropagation:
    """A Men/Random-shaped pscore must pass the propensity sanity gate
    under the conditional schema (and fail under the default joint one)."""

    def test_conditional_schema_passes_on_one_over_n_actions(self) -> None:
        # Men/Random target: 1 / 34.
        n_actions = 34
        n_positions = 3
        pscore = np.full(5_000, 1.0 / n_actions, dtype=float)
        result = propensity_sanity_gate(
            pscore=pscore,
            schema=PROPENSITY_SCHEMA_CONDITIONAL,
            n_actions=n_actions,
            n_positions=n_positions,
        )
        assert result.passed is True
        assert result.schema == PROPENSITY_SCHEMA_CONDITIONAL
        assert result.target == pytest.approx(1.0 / n_actions)
        assert result.relative_deviation < 0.10

    def test_joint_schema_fails_on_conditional_shaped_pscore(self) -> None:
        # Cross-check: the same pscore fails under the default joint
        # schema, proving the schema parameter is actually consumed by
        # the gate and not dropped silently. This is the regression that
        # the bridge test guards against.
        n_actions = 34
        n_positions = 3
        pscore = np.full(5_000, 1.0 / n_actions, dtype=float)
        result = propensity_sanity_gate(
            pscore=pscore,
            schema=PROPENSITY_SCHEMA_JOINT,
            n_actions=n_actions,
            n_positions=n_positions,
        )
        assert result.passed is False


# ── Adapter → bandit_feedback helper ──────────────────────────────────


class TestToBanditFeedbackHelper:
    """``BanditLogAdapter.to_bandit_feedback`` returns an OBP-shaped dict
    with normalized positions so the Track B evaluator can consume it
    without further re-shaping."""

    def test_returns_dict_with_track_b_expected_keys(self) -> None:
        bf = _build_men_random_like_feedback(n_rounds=200, n_actions=5, n_positions=3, seed=42)
        adapter = BanditLogAdapter(bandit_feedback=bf, seed=42)
        out = adapter.to_bandit_feedback()
        # Track B's ``evaluate_open_bandit_policy`` reads these keys; the
        # helper must emit them all.
        for key in ("n_rounds", "n_actions", "action", "reward", "pscore", "position"):
            assert key in out, f"missing key: {key}"
        # Shape preservation.
        assert out["n_rounds"] == 200
        assert out["n_actions"] == 5
        assert np.asarray(out["action"]).shape == (200,)
        assert np.asarray(out["reward"]).shape == (200,)
        assert np.asarray(out["pscore"]).shape == (200,)
        assert np.asarray(out["position"]).shape == (200,)

    def test_positions_are_normalized_to_zero_indexed_contiguous(self) -> None:
        bf = _build_men_random_like_feedback(n_rounds=200, n_actions=5, n_positions=3, seed=42)
        adapter = BanditLogAdapter(bandit_feedback=bf, seed=42)
        out = adapter.to_bandit_feedback()
        positions = np.asarray(out["position"], dtype=int)
        # Post-condition: 0-indexed, contiguous. ``_validate_positions``
        # in Track B would otherwise raise.
        assert int(positions.min()) == 0
        unique = np.unique(positions)
        assert unique.tolist() == list(range(len(unique)))

    def test_exposes_conditional_schema_for_men_random(self) -> None:
        # Bridge contract: adapter advertises which propensity schema
        # its feedback dict uses, so downstream callers threading the
        # bandit_feedback dict into ``run_section_7_gates`` can pass the
        # correct schema without out-of-band knowledge.
        bf = _build_men_random_like_feedback(n_rounds=200, n_actions=5, n_positions=3, seed=42)
        adapter = BanditLogAdapter(bandit_feedback=bf, seed=42)
        assert adapter.propensity_schema == PROPENSITY_SCHEMA_CONDITIONAL


# ── End-to-end bridge flow: all three seams together ──────────────────


class TestBridgeEndToEnd:
    """Exercises all three seams in a single integration pass.

    - Build Men/Random-like feedback with *raw* 1-indexed positions.
    - Hand the adapter state to ``to_bandit_feedback`` to normalize.
    - Pass ``PROPENSITY_SCHEMA_CONDITIONAL`` into ``run_section_7_gates``.
    - Assert the returned provenance dict carries a real OBP version.
    """

    def test_full_bridge_flow_produces_clean_gate_report(self) -> None:
        n_rounds = 2_000
        n_actions = 34
        n_positions = 3
        bf_raw = _build_men_random_like_feedback(
            n_rounds=n_rounds,
            n_actions=n_actions,
            n_positions=n_positions,
            seed=7,
        )
        adapter = BanditLogAdapter(bandit_feedback=bf_raw, seed=7)

        # Seam: adapter → bandit_feedback with normalized positions.
        bf = adapter.to_bandit_feedback()

        # Build a uniform evaluation policy on the normalized feedback
        # shape; this is the minimum honest stand-in for the optimized
        # policy that the Sprint 35.C benchmark will compute.
        n_rows_used = bf["n_rounds"]
        policies = {
            "random": uniform_policy(n_rounds=n_rows_used, n_actions=n_actions),
        }

        # Seeded per-seed diagnostics so all gates have well-defined
        # inputs. We evaluate the same uniform policy on the adapter's
        # feedback so SNIPW ≈ mean reward, DR ≈ mean reward, and their
        # relative divergence is ~0 — cross-check gate passes.
        clip = 1.0 / (2 * n_actions * n_positions)
        per_seed_ess: list[float] = []
        per_seed_zero_support: list[float] = []
        snipw_per_seed: list[float] = []
        dr_per_seed: list[float] = []
        for _ in range(3):
            out = evaluate_open_bandit_policy(
                bf,
                uniform_policy(n_rounds=n_rows_used, n_actions=n_actions),
                min_propensity_clip=clip,
            )
            per_seed_ess.append(out["ess"])
            per_seed_zero_support.append(out["zero_support_fraction"])
            snipw_per_seed.append(out["policy_value"])
            dr_per_seed.append(out["policy_value"])  # uniform ↔ uniform parity

        # Seam: conditional schema propagation into the gate bundle.
        report = run_section_7_gates(
            bandit_feedback=bf,
            strategy_policies=policies,
            per_seed_ess=per_seed_ess,
            per_seed_zero_support=per_seed_zero_support,
            snipw_per_seed=snipw_per_seed,
            dr_per_seed=dr_per_seed,
            n_actions=n_actions,
            n_positions=n_positions,
            schema=PROPENSITY_SCHEMA_CONDITIONAL,
            permutation_seed=1,
        )
        assert isinstance(report, GateReport)
        # The ESS floor on 2_000 rows is max(1000, 20) = 1000; with a
        # uniform-vs-uniform eval on conditional propensity the Kish
        # ESS lands at n_rounds, so the ESS gate passes.
        assert report.ess.passed
        # Conditional schema: empirical mean of pscore (= 1/n_actions)
        # matches the conditional target exactly.
        assert report.propensity_sanity.passed
        assert report.propensity_sanity.schema == PROPENSITY_SCHEMA_CONDITIONAL
        assert report.propensity_sanity.target == pytest.approx(1.0 / n_actions)

        # Seam: provenance carries a real OBP version, not the Track B
        # placeholder.
        provenance = report.provenance
        assert "obp_version" in provenance
        assert provenance["obp_version"] != OBP_VERSION_PLACEHOLDER
        # Installed via the ``bandit`` extra (skip if someone runs this
        # test without the extra, but the top-level pytest marker would
        # have already excluded it — we guard defensively).
        obp = pytest.importorskip("obp")
        assert provenance["obp_version"] == obp.__version__
        # Schema is recorded in provenance for reproducibility.
        assert provenance["schema"] == PROPENSITY_SCHEMA_CONDITIONAL

    def test_positions_pass_track_b_validator_after_bridge(self) -> None:
        # Direct regression for seam 1: feed raw 1-indexed positions
        # through the bridge and confirm the Track B null-control path
        # (which invokes ``_validate_positions``) does not raise.
        from causal_optimizer.benchmarks.open_bandit import permute_rewards_stratified

        bf_raw = _build_men_random_like_feedback(n_rounds=500, n_actions=4, n_positions=3, seed=0)
        adapter = BanditLogAdapter(bandit_feedback=bf_raw, seed=0)
        bf = adapter.to_bandit_feedback()
        # Would raise on raw 1-indexed positions.
        permuted = permute_rewards_stratified(bf, seed=0)
        assert permuted["reward"].shape == (500,)
