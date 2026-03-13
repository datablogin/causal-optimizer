"""Interaction detection regression tests.

Tests that factorial screening catches the X1×X2 interaction that naive
main-effects analysis misses. The SCM is:

    Y = -X1 - X2 + 3*X1*X2 + noise

So:
    X1=0, X2=0 → Y ≈ 0          (baseline)
    X1=1, X2=0 → Y ≈ -1         (individually harmful)
    X1=0, X2=1 → Y ≈ -1         (individually harmful)
    X1=1, X2=1 → Y ≈ 1          (jointly beneficial — the interaction)

Negated for minimisation: objective = -Y.

Three additional dummy variables (X3, X4, X5) are irrelevant.

Key claim: ScreeningDesigner.screen() detects the (x1, x2) interaction in
the residuals by fitting a model *with* the product term x1*x2 vs *without*
it and measuring the R² improvement. This is the H-statistic approach.

Note: ``pytestmark`` in conftest.py does NOT propagate to sibling test
modules — each test class must be decorated with ``@pytest.mark.slow``.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from causal_optimizer.benchmarks.interaction_scm import InteractionSCM
from causal_optimizer.designer.screening import ScreeningDesigner
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_NOISE_SCALE = 0.05
# 40 experiments for screening: well above ScreeningDesigner's minimum of 10,
# provides robust signal for the x1*x2 interaction with 5-fold cross-validation.
_LOG_SIZE = 40


def _make_interaction_log(n: int = _LOG_SIZE, seed: int = 42) -> ExperimentLog:
    """Build an ExperimentLog from random continuous samples of the interaction SCM.

    Samples are drawn uniformly from [0, 1]^5 to give the ScreeningDesigner
    the continuous variation needed to detect the x1*x2 interaction signal.
    """
    root_rng = np.random.default_rng(seed)
    param_rng = np.random.default_rng(int(root_rng.integers(0, 2**32)))
    noise_rng = np.random.default_rng(int(root_rng.integers(0, 2**32)))
    scm = InteractionSCM(noise_scale=_NOISE_SCALE, rng=noise_rng)

    results: list[ExperimentResult] = []
    for _ in range(n):
        params = {
            "x1": float(param_rng.uniform(0.0, 1.0)),
            "x2": float(param_rng.uniform(0.0, 1.0)),
            "x3": float(param_rng.uniform(0.0, 1.0)),
            "x4": float(param_rng.uniform(0.0, 1.0)),
            "x5": float(param_rng.uniform(0.0, 1.0)),
        }
        metrics = scm.run(params)
        results.append(
            ExperimentResult(
                experiment_id=str(uuid.uuid4()),
                parameters=params,
                metrics=metrics,
                status=ExperimentStatus.KEEP,
            )
        )
    return ExperimentLog(results=results)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestScreeningDetectsInteraction:
    """ScreeningDesigner must detect the (x1, x2) interaction."""

    def test_screening_detects_interaction(self) -> None:
        """ScreeningDesigner.screen() reports (x1, x2) in interactions dict.

        We build an ExperimentLog with 40 experiments sampled uniformly from
        [0,1]^5 and verify that the screening result contains the (x1, x2)
        pair. The pair key may appear in either order ((x1, x2) or (x2, x1)).
        """
        log = _make_interaction_log(n=_LOG_SIZE, seed=0)
        space = InteractionSCM.search_space()
        designer = ScreeningDesigner(search_space=space, importance_threshold=0.01)
        result = designer.screen(log)

        interaction_pairs = set(result.interactions.keys())
        detected = ("x1", "x2") in interaction_pairs or ("x2", "x1") in interaction_pairs

        assert detected, (
            f"Expected (x1, x2) interaction not found.\n"
            f"Detected interactions: {interaction_pairs}\n"
            f"Main effects: {result.main_effects}"
        )


@pytest.mark.slow
class TestGreedyMissesInteraction:
    """Naive main-effects analysis fails to detect the x1*x2 interaction."""

    def test_greedy_misses_interaction(self) -> None:
        """A naive RF trained without an interaction term has lower R² than one with it.

        This demonstrates that greedy/main-effects optimisation is blind to the
        x1*x2 synergy. Without the explicit interaction feature, the model
        cannot capture the non-linear joint effect — its R² is significantly
        lower than a model that includes the product term.

        We average across 3 seeds so the test is robust to dataset variance.
        The mean cross-validated R² improvement from adding the x1*x2 feature
        must exceed 0.02 (2 percentage points) — a conservative threshold well
        below the observed mean gain of ~5pp.

        Cross-validation (cv=5) is used instead of in-sample R² to avoid
        over-fitting artefacts that could inflate the apparent gap.
        """
        n_avg_seeds = 3
        gaps: list[float] = []

        for seed in range(n_avg_seeds):
            log = _make_interaction_log(n=_LOG_SIZE, seed=seed)
            df = log.to_dataframe()
            x1 = df["x1"].values
            x2 = df["x2"].values
            x_main = df[["x1", "x2"]].values
            x_with_inter = np.column_stack([x1, x2, x1 * x2])
            y = df["objective"].values

            rf_without = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=seed)
            score_without = float(
                np.mean(cross_val_score(rf_without, x_main, y, cv=5, scoring="r2"))
            )

            rf_with = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=seed)
            score_with = float(
                np.mean(cross_val_score(rf_with, x_with_inter, y, cv=5, scoring="r2"))
            )

            gaps.append(score_with - score_without)

        mean_gap = float(np.mean(gaps))
        assert mean_gap > 0.02, (
            f"Expected mean R² improvement from interaction term > 0.02, "
            f"got {mean_gap:.4f} (per-seed gaps: {[f'{g:.4f}' for g in gaps]}).\n"
            "The interaction signal may not be strong enough in this dataset."
        )


@pytest.mark.slow
class TestCausalFindsInteraction:
    """ExperimentEngine with causal discovery reliably finds the interaction optimum."""

    def test_causal_finds_interaction(self) -> None:
        """ExperimentEngine with discovery_method='correlation' finds X1≈1, X2≈1.

        Over 30 steps the engine should discover that X1 and X2 are important
        correlated inputs and converge toward X1=1, X2=1 (objective ≈ -1).
        At least one kept result must have X1 > 0.7 and X2 > 0.7.
        """
        rng = np.random.default_rng(0)
        scm = InteractionSCM(noise_scale=_NOISE_SCALE, rng=rng)
        space = InteractionSCM.search_space()

        engine = ExperimentEngine(
            search_space=space,
            runner=scm,
            causal_graph=None,
            discovery_method="correlation",
        )
        for _ in range(30):
            engine.step()

        kept = [r for r in engine.log.results if r.status == ExperimentStatus.KEEP]
        # Require x1, x2 > 0.7 rather than 1.0: the engine only runs 30 steps,
        # noise (scale=0.05) prevents exact convergence, and correlation-based
        # discovery is a heuristic that may not fully constrain the search.
        found = any(
            float(r.parameters.get("x1", 0.0)) > 0.7 and float(r.parameters.get("x2", 0.0)) > 0.7
            for r in kept
        )

        assert found, (
            "ExperimentEngine with discovery_method='correlation' never found a kept "
            "result with X1>0.7 and X2>0.7 after 30 steps.\n"
            "Best kept results (top 5 by objective):\n"
            + "\n".join(
                f"  x1={r.parameters.get('x1', '?'):.3f}, "
                f"x2={r.parameters.get('x2', '?'):.3f}, "
                f"obj={r.metrics.get('objective', '?'):.3f}"
                for r in sorted(kept, key=lambda x: x.metrics.get("objective", 0))[:5]
            )
        )
