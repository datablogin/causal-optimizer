"""Robust treatment effect estimation for experiments.

Wraps doubly-robust estimators (AIPW, TMLE) to determine whether an
experimental change truly helped or was just noise. Provides confidence
intervals to support keep/discard decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from causal_optimizer.types import ExperimentLog

logger = logging.getLogger(__name__)


@dataclass
class EffectEstimate:
    """Estimated causal effect of a change."""

    point_estimate: float
    confidence_interval: tuple[float, float]
    p_value: float
    is_significant: bool
    method: str

    @property
    def summary(self) -> str:
        sig = "significant" if self.is_significant else "not significant"
        return (
            f"Effect: {self.point_estimate:.6f} "
            f"CI: [{self.confidence_interval[0]:.6f}, {self.confidence_interval[1]:.6f}] "
            f"p={self.p_value:.4f} ({sig}) [{self.method}]"
        )


class EffectEstimator:
    """Estimate whether experimental changes have true causal effects.

    Supports:
    - 'difference': simple difference in means (baseline)
    - 'bootstrap': bootstrap confidence intervals
    - 'aipw': augmented IPW via causal-inference library
    """

    def __init__(
        self,
        method: str = "bootstrap",
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> None:
        self.method = method
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap

    def estimate_effect(
        self,
        experiment_log: ExperimentLog,
        treatment_param: str,
        treatment_value: Any,
        control_value: Any,
        objective_name: str = "objective",
    ) -> EffectEstimate:
        """Estimate the causal effect of changing `treatment_param` from control to treatment."""
        df = experiment_log.to_dataframe()

        treated = df[df[treatment_param] == treatment_value][objective_name].values
        control = df[df[treatment_param] == control_value][objective_name].values

        if len(treated) < 2 or len(control) < 2:
            return EffectEstimate(
                point_estimate=0.0,
                confidence_interval=(float("-inf"), float("inf")),
                p_value=1.0,
                is_significant=False,
                method="insufficient_data",
            )

        if self.method == "difference":
            return self._difference_estimate(treated, control)
        elif self.method == "bootstrap":
            return self._bootstrap_estimate(treated, control)
        elif self.method == "aipw":
            return self._aipw_estimate(experiment_log, treatment_param, treatment_value, control_value, objective_name)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _difference_estimate(
        self, treated: np.ndarray, control: np.ndarray
    ) -> EffectEstimate:
        """Simple difference in means with t-test."""
        effect = float(np.mean(treated) - np.mean(control))
        t_stat, p_value = stats.ttest_ind(treated, control)
        se = float(np.sqrt(np.var(treated) / len(treated) + np.var(control) / len(control)))
        alpha = 1 - self.confidence_level
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (effect - z * se, effect + z * se)

        return EffectEstimate(
            point_estimate=effect,
            confidence_interval=ci,
            p_value=float(p_value),
            is_significant=p_value < (1 - self.confidence_level),
            method="difference",
        )

    def _bootstrap_estimate(
        self, treated: np.ndarray, control: np.ndarray
    ) -> EffectEstimate:
        """Bootstrap confidence interval for treatment effect."""
        rng = np.random.default_rng(42)
        effects = np.empty(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            t_boot = rng.choice(treated, size=len(treated), replace=True)
            c_boot = rng.choice(control, size=len(control), replace=True)
            effects[i] = np.mean(t_boot) - np.mean(c_boot)

        point_estimate = float(np.mean(treated) - np.mean(control))
        alpha = 1 - self.confidence_level
        ci = (
            float(np.percentile(effects, 100 * alpha / 2)),
            float(np.percentile(effects, 100 * (1 - alpha / 2))),
        )
        # Approximate p-value from bootstrap distribution
        p_value = float(2 * min(np.mean(effects <= 0), np.mean(effects >= 0)))

        return EffectEstimate(
            point_estimate=point_estimate,
            confidence_interval=ci,
            p_value=p_value,
            is_significant=p_value < (1 - self.confidence_level),
            method="bootstrap",
        )

    def _aipw_estimate(
        self,
        experiment_log: ExperimentLog,
        treatment_param: str,
        treatment_value: Any,
        control_value: Any,
        objective_name: str,
    ) -> EffectEstimate:
        """AIPW estimation via causal-inference library."""
        try:
            from causal_inference.estimators.aipw import AIPW
        except ImportError:
            logger.warning("causal-inference not installed, falling back to bootstrap")
            df = experiment_log.to_dataframe()
            treated = df[df[treatment_param] == treatment_value][objective_name].values
            control = df[df[treatment_param] == control_value][objective_name].values
            return self._bootstrap_estimate(treated, control)

        df = experiment_log.to_dataframe()
        treatment = (df[treatment_param] == treatment_value).astype(int).values
        outcome = df[objective_name].values
        covariates = df.drop(
            columns=["experiment_id", "status", treatment_param, objective_name],
            errors="ignore",
        ).select_dtypes(include=[np.number])

        aipw = AIPW()
        result = aipw.estimate(
            treatment=treatment,
            outcome=outcome,
            covariates=covariates.values,
        )

        return EffectEstimate(
            point_estimate=result.ate,
            confidence_interval=(result.ci_lower, result.ci_upper),
            p_value=result.p_value,
            is_significant=result.p_value < (1 - self.confidence_level),
            method="aipw",
        )
