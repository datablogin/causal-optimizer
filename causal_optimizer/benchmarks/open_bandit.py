"""Sprint 35 Open Bandit OPE stack and Section 7 support gates.

Stub module for the RED stage of TDD.  Implementations land in the GREEN
stage; tests import these symbols and must fail for the right reason
(``NotImplementedError``) rather than ``ImportError``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

PROPENSITY_SCHEMA_CONDITIONAL: Literal["conditional"] = "conditional"
PROPENSITY_SCHEMA_JOINT: Literal["joint"] = "joint"
DEFAULT_PROPENSITY_SCHEMA = PROPENSITY_SCHEMA_JOINT

PropensitySchema = Literal["conditional", "joint"]


def compute_min_propensity_clip(*, n_actions: int, n_positions: int) -> float:
    raise NotImplementedError


def generate_synthetic_bandit_feedback(
    *,
    n_rounds: int,
    n_actions: int,
    n_positions: int,
    seed: int,
    propensity_schema: PropensitySchema = DEFAULT_PROPENSITY_SCHEMA,
    reward_mean_by_action: list[float] | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


def uniform_policy(n_rounds: int, n_actions: int) -> np.ndarray:
    raise NotImplementedError


def peaked_policy(
    *, n_rounds: int, n_actions: int, best_action: int, peak_mass: float
) -> np.ndarray:
    raise NotImplementedError


def degenerate_policy(
    *, n_rounds: int, n_actions: int, support_action: int
) -> np.ndarray:
    raise NotImplementedError


def compute_snipw(
    bandit_feedback: dict[str, Any],
    action_dist: np.ndarray,
    *,
    min_propensity_clip: float,
) -> float:
    raise NotImplementedError


def compute_dm(action_dist: np.ndarray, reward_hat: np.ndarray) -> float:
    raise NotImplementedError


def compute_dr(
    bandit_feedback: dict[str, Any],
    action_dist: np.ndarray,
    reward_hat: np.ndarray,
    *,
    min_propensity_clip: float,
) -> float:
    raise NotImplementedError


def evaluate_policy(
    bandit_feedback: dict[str, Any],
    action_dist: np.ndarray,
    *,
    min_propensity_clip: float,
    reward_hat: np.ndarray | None = None,
    estimator: str = "snipw",
) -> dict[str, float]:
    raise NotImplementedError


def permute_rewards_stratified(
    bandit_feedback: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    raise NotImplementedError


class NullControlResult:
    passed: bool


class ESSResult:
    passed: bool


class ZeroSupportResult:
    passed: bool


class PropensitySanityResult:
    passed: bool


class DrSnipwCrossCheckResult:
    passed: bool


class GateReport:
    null_control: NullControlResult
    ess: ESSResult
    zero_support: ZeroSupportResult
    propensity_sanity: PropensitySanityResult
    dr_cross_check: DrSnipwCrossCheckResult

    def all_passed(self) -> bool:
        raise NotImplementedError


def null_control_gate(
    bandit_feedback: dict[str, Any],
    *,
    strategy_policies: dict[str, np.ndarray],
    permutation_seed: int,
    min_propensity_clip: float | None = None,
    band_multiplier: float = 1.05,
    strategy_value_overrides: dict[str, float] | None = None,
) -> NullControlResult:
    raise NotImplementedError


def ess_gate(*, per_seed_ess: list[float], n_rows: int) -> ESSResult:
    raise NotImplementedError


def zero_support_gate(*, per_seed_zero_support: list[float]) -> ZeroSupportResult:
    raise NotImplementedError


def propensity_sanity_gate(
    *,
    pscore: np.ndarray,
    schema: PropensitySchema,
    n_actions: int,
    n_positions: int,
    tolerance_relative: float = 0.10,
) -> PropensitySanityResult:
    raise NotImplementedError


def snipw_dr_cross_check_gate(
    *,
    snipw_per_seed: list[float],
    dr_per_seed: list[float],
    tolerance_relative: float = 0.25,
) -> DrSnipwCrossCheckResult:
    raise NotImplementedError


def run_section_7_gates(
    *,
    bf: dict[str, Any],
    strategy_policies: dict[str, np.ndarray],
    per_seed_ess: list[float],
    per_seed_zero_support: list[float],
    snipw_per_seed: list[float],
    dr_per_seed: list[float],
    n_actions: int,
    n_positions: int,
    schema: PropensitySchema,
    permutation_seed: int,
    min_propensity_clip: float | None = None,
) -> GateReport:
    raise NotImplementedError
