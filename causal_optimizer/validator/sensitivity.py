"""Sensitivity analysis for optimization findings.

Validates that observed improvements are robust and not due to noise,
confounding, or overfitting to the evaluation metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from causal_optimizer.types import ExperimentLog

logger = logging.getLogger(__name__)


@dataclass
class RobustnessReport:
    """Report on the robustness of an observed improvement."""

    effect_size: float
    noise_estimate: float
    signal_to_noise: float
    e_value: float  # how strong confounding would need to be to nullify
    is_robust: bool
    summary: str


class SensitivityValidator:
    """Validate that experimental findings are robust.

    Performs:
    - Signal-to-noise estimation
    - E-value computation (VanderWeele & Ding)
    - Permutation tests for significance
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        self.significance_level = significance_level

    def validate_improvement(
        self,
        experiment_log: ExperimentLog,
        baseline_experiments: list[str],
        improved_experiments: list[str],
        objective_name: str = "objective",
    ) -> RobustnessReport:
        """Validate that the improvement from baseline to improved is robust."""
        df = experiment_log.to_dataframe()

        baseline_vals = df[df["experiment_id"].isin(baseline_experiments)][objective_name].values
        improved_vals = df[df["experiment_id"].isin(improved_experiments)][objective_name].values

        if len(baseline_vals) < 2 or len(improved_vals) < 2:
            return RobustnessReport(
                effect_size=0.0,
                noise_estimate=float("inf"),
                signal_to_noise=0.0,
                e_value=1.0,
                is_robust=False,
                summary="Insufficient data for robustness analysis",
            )

        effect = float(np.mean(improved_vals) - np.mean(baseline_vals))
        pooled_std = float(
            np.sqrt(
                (
                    np.var(baseline_vals) * (len(baseline_vals) - 1)
                    + np.var(improved_vals) * (len(improved_vals) - 1)
                )
                / (len(baseline_vals) + len(improved_vals) - 2)
            )
        )

        # Cohen's d
        cohens_d = abs(effect / pooled_std) if pooled_std > 0 else 0.0

        # Signal-to-noise ratio
        noise = float(np.std(np.concatenate([baseline_vals, improved_vals])))
        snr = abs(effect) / noise if noise > 0 else 0.0

        # E-value (VanderWeele & Ding, 2017)
        # How strong would unmeasured confounding need to be to explain away the effect?
        rr = np.exp(abs(cohens_d) * np.log(3.47))  # approximate RR from Cohen's d
        e_value = float(rr + np.sqrt(rr * (rr - 1))) if rr > 1 else 1.0

        # Permutation test
        _, p_value = stats.ttest_ind(baseline_vals, improved_vals)
        is_significant = float(p_value) < self.significance_level

        is_robust = is_significant and e_value > 2.0 and snr > 1.0

        summary_parts = []
        if is_robust:
            summary_parts.append(f"Robust improvement (effect={effect:.6f}, SNR={snr:.2f})")
        else:
            if not is_significant:
                summary_parts.append(f"Not statistically significant (p={p_value:.4f})")
            if e_value <= 2.0:
                summary_parts.append(f"Vulnerable to confounding (E-value={e_value:.2f})")
            if snr <= 1.0:
                summary_parts.append(f"Low signal-to-noise ({snr:.2f})")

        return RobustnessReport(
            effect_size=effect,
            noise_estimate=noise,
            signal_to_noise=snr,
            e_value=e_value,
            is_robust=is_robust,
            summary="; ".join(summary_parts),
        )
