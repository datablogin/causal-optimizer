"""Robust treatment effect estimation for experiments.

Wraps doubly-robust estimators (AIPW, TMLE) to determine whether an
experimental change truly helped or was just noise. Provides confidence
intervals to support keep/discard decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from causal_optimizer.types import CausalGraph, ExperimentLog

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
    - 'observational': DoWhy backdoor/frontdoor/IV adjustment (requires causal_graph)

    When ``method='observational'``, the ``obs_method`` parameter controls
    which DoWhy identification strategy is used (``"backdoor"``, ``"frontdoor"``,
    or ``"iv"``).
    """

    #: Valid values for the ``obs_method`` parameter.
    _VALID_OBS_METHODS: frozenset[str] = frozenset({"backdoor", "frontdoor", "iv"})

    def __init__(
        self,
        method: str = "bootstrap",
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        causal_graph: CausalGraph | None = None,
        obs_method: str = "backdoor",
    ) -> None:
        if obs_method not in self._VALID_OBS_METHODS:
            raise ValueError(
                f"obs_method={obs_method!r} is not valid; "
                f"choose one of {sorted(self._VALID_OBS_METHODS)}"
            )
        self.method = method
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.causal_graph = causal_graph
        self.obs_method = obs_method

    def estimate_effect(
        self,
        experiment_log: ExperimentLog,
        treatment_param: str,
        treatment_value: Any,
        control_value: Any,
        objective_name: str = "objective",
    ) -> EffectEstimate:
        """Estimate the causal effect of changing `treatment_param` from control to treatment."""
        if self.method == "observational":
            return self._observational_estimate(
                experiment_log,
                treatment_param,
                float(treatment_value),
                float(control_value),
                objective_name,
            )

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
            return self._aipw_estimate(
                experiment_log,
                treatment_param,
                treatment_value,
                control_value,
                objective_name,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _difference_estimate(self, treated: np.ndarray, control: np.ndarray) -> EffectEstimate:
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
            is_significant=p_value < alpha,  # alpha = 1 - confidence_level (e.g. 0.05 for 95% CI)
            method="difference",
        )

    def _bootstrap_estimate(
        self,
        treated: np.ndarray,
        control: np.ndarray,
        small_sample: bool = False,
    ) -> EffectEstimate:
        """Bootstrap confidence interval for treatment effect.

        Args:
            treated: Outcome values for the treatment group.
            control: Outcome values for the control group.
            small_sample: When True (or when total n < 10), uses 100 bootstrap
                samples instead of ``self.n_bootstrap`` for efficiency.
        """
        rng = np.random.default_rng(42)
        n_total = len(treated) + len(control)
        n_iter = 100 if (small_sample or n_total < 10) else self.n_bootstrap
        effects = np.empty(n_iter)

        for i in range(n_iter):
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

    def estimate_improvement(
        self,
        experiment_log: ExperimentLog,
        current_value: float,
        objective_name: str = "objective",
        minimize: bool = True,
    ) -> EffectEstimate:
        """Estimate whether ``current_value`` is a significant improvement over history.

        Compares ``current_value`` against the distribution of kept experiments
        in ``experiment_log``.  Returns an :class:`EffectEstimate` where
        ``is_significant=True`` means the current experiment is a statistically
        meaningful improvement over the best-so-far.

        Args:
            experiment_log: History of past experiments.
            current_value: The objective value of the candidate experiment.
            objective_name: Metric key to compare on.
            minimize: ``True`` if lower values are better.

        Returns:
            :class:`EffectEstimate` with ``is_significant`` indicating whether
            the improvement is statistically meaningful.

            - Fewer than 2 kept experiments: permissive fallback
              (``is_significant=True``) — no comparison is possible.
            - 2–4 kept experiments: greedy comparison — ``is_significant=True``
              only when the current value strictly beats the best-so-far.
            - 5+ kept experiments: full statistical test (bootstrap or t-test).
        """
        from causal_optimizer.types import ExperimentStatus

        kept_values = [
            r.metrics[objective_name]
            for r in experiment_log.results
            if r.status == ExperimentStatus.KEEP and objective_name in r.metrics
        ]

        # Permissive fallback: truly too few kept experiments (< 2) for any comparison
        if len(kept_values) < 2:
            return EffectEstimate(
                point_estimate=0.0,
                confidence_interval=(float("-inf"), float("inf")),
                p_value=1.0,
                is_significant=True,
                method="insufficient_data",
            )

        # Small sample: not enough kept history for reliable bootstrap testing
        # (< 5 kept). Fall back to a greedy comparison: a better result is
        # considered significant, a worse result is not.  This avoids both
        # false positives (keeping noise) and false negatives (discarding real
        # improvements) when data is scarce.
        if len(kept_values) < 5:
            kept_arr_small = np.array(kept_values, dtype=float)
            best_small = float(np.min(kept_arr_small) if minimize else np.max(kept_arr_small))
            point_est = float(current_value - best_small)
            is_better = current_value < best_small if minimize else current_value > best_small
            return EffectEstimate(
                point_estimate=point_est,
                confidence_interval=(float("-inf"), float("inf")),
                p_value=0.0 if is_better else 1.0,
                is_significant=is_better,  # greedy comparison for small samples
                method=self.method,
            )

        kept_arr = np.array(kept_values, dtype=float)

        best = float(np.min(kept_arr)) if minimize else float(np.max(kept_arr))

        if self.method == "difference":
            # Use t-test between current_value and kept distribution
            effect = float(current_value - best)
            _, p_value = stats.ttest_1samp(kept_arr, popmean=current_value)
            p_value_f = float(p_value)
            se = float(np.std(kept_arr) / np.sqrt(len(kept_arr)))
            alpha_ci = 1 - self.confidence_level
            z = stats.norm.ppf(1 - alpha_ci / 2)
            ci = (effect - z * se, effect + z * se)

            if minimize:
                is_significant = current_value < best and p_value_f < (1 - self.confidence_level)
            else:
                is_significant = current_value > best and p_value_f < (1 - self.confidence_level)

            return EffectEstimate(
                point_estimate=effect,
                confidence_interval=ci,
                p_value=p_value_f,
                is_significant=is_significant,
                method="difference",
            )

        elif self.method in ("bootstrap", "aipw", "observational"):
            # Bootstrap CI for the best-so-far in the kept distribution.
            # If current_value falls outside the CI on the improvement side,
            # it is significantly better than any historically kept result.
            rng = np.random.default_rng(42)
            n_iter = 100 if len(kept_arr) < 10 else self.n_bootstrap
            boot_bests = np.empty(n_iter)
            for i in range(n_iter):
                boot = rng.choice(kept_arr, size=len(kept_arr), replace=True)
                boot_bests[i] = float(np.min(boot) if minimize else np.max(boot))

            point_estimate = current_value - best
            alpha = 1 - self.confidence_level

            # CI for the best-so-far under bootstrap
            ci_best_lo = float(np.percentile(boot_bests, 100 * alpha / 2))
            ci_best_hi = float(np.percentile(boot_bests, 100 * (1 - alpha / 2)))

            if minimize:
                # Significant improvement: current_value < lower bound of best CI
                is_significant = current_value < ci_best_lo
                # p-value: fraction of bootstrap bests at or below current_value
                p_value = float(np.mean(boot_bests <= current_value))
            else:
                # Significant improvement: current_value > upper bound of best CI
                is_significant = current_value > ci_best_hi
                # p-value: fraction of bootstrap bests at or above current_value
                p_value = float(np.mean(boot_bests >= current_value))

            # CI of the point estimate (current - best) derived from best CI
            ci_lo = float(current_value - ci_best_hi)
            ci_hi = float(current_value - ci_best_lo)

            return EffectEstimate(
                point_estimate=point_estimate,
                confidence_interval=(ci_lo, ci_hi),
                p_value=p_value,
                is_significant=is_significant,
                method=self.method,
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

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

    def _observational_bootstrap_fallback(
        self,
        experiment_log: ExperimentLog,
        treatment_param: str,
        treatment_value: float,
        control_value: float,
        objective_name: str,
    ) -> EffectEstimate:
        """Bootstrap fallback when observational estimation fails or is unavailable."""
        import pandas as pd

        df = experiment_log.to_dataframe()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
        treated_arr = df[df[treatment_param] == treatment_value][objective_name].values
        control_arr = df[df[treatment_param] == control_value][objective_name].values
        if len(treated_arr) >= 2 and len(control_arr) >= 2:
            return self._bootstrap_estimate(treated_arr, control_arr)
        all_vals = df[objective_name].values
        half = max(1, len(all_vals) // 2)
        return self._bootstrap_estimate(all_vals[:half], all_vals[half:])

    def _observational_estimate(
        self,
        experiment_log: ExperimentLog,
        treatment_param: str,
        treatment_value: float,
        control_value: float,
        objective_name: str,
    ) -> EffectEstimate:
        """Estimate causal effect via DoWhy observational adjustment.

        Requires ``self.causal_graph`` to be set.  Falls back to bootstrap
        when DoWhy is not installed or the effect is not identifiable.
        """
        if self.causal_graph is None:
            raise ValueError(
                "method='observational' requires causal_graph to be provided "
                "to EffectEstimator.__init__"
            )

        from causal_optimizer.estimator.observational import ObservationalEstimator

        obs_estimator = ObservationalEstimator(
            causal_graph=self.causal_graph,
            method=self.obs_method,
            confidence_level=self.confidence_level,
        )

        try:
            treatment_est = obs_estimator.estimate_intervention(
                experiment_log=experiment_log,
                treatment_var=treatment_param,
                treatment_value=treatment_value,
                objective_name=objective_name,
            )
            control_est = obs_estimator.estimate_intervention(
                experiment_log=experiment_log,
                treatment_var=treatment_param,
                treatment_value=control_value,
                objective_name=objective_name,
            )
        except ImportError:
            logger.warning(
                "dowhy not installed, falling back to bootstrap for observational method"
            )
            return self._observational_bootstrap_fallback(
                experiment_log, treatment_param, treatment_value, control_value, objective_name
            )

        if not treatment_est.identified or not control_est.identified:
            logger.warning("Effect not identifiable via %s; falling back to bootstrap", self.method)
            return self._observational_bootstrap_fallback(
                experiment_log, treatment_param, treatment_value, control_value, objective_name
            )

        point_estimate = treatment_est.expected_outcome - control_est.expected_outcome
        ci_lo = treatment_est.confidence_interval[0] - control_est.confidence_interval[1]
        ci_hi = treatment_est.confidence_interval[1] - control_est.confidence_interval[0]

        # Approximate p-value from the CI width using the configured confidence level
        alpha = 1.0 - self.confidence_level
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_width = ci_hi - ci_lo
        se = ci_width / (2 * z_crit) if ci_width > 0 else 1.0
        z_stat = abs(point_estimate) / se if se > 0 else 0.0
        p_value = float(2 * stats.norm.sf(z_stat))

        return EffectEstimate(
            point_estimate=point_estimate,
            confidence_interval=(ci_lo, ci_hi),
            p_value=p_value,
            is_significant=p_value < (1 - self.confidence_level),
            method=treatment_est.method,
        )
