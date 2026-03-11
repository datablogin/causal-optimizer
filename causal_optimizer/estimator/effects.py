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

from causal_optimizer.types import ExperimentStatus

if TYPE_CHECKING:
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
            is_significant=p_value < (1 - self.confidence_level),
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
            the improvement is statistically meaningful.  When fewer than 2
            kept experiments are available, returns a permissive estimate
            (``is_significant=True``) to avoid premature pruning.  Between 2
            and 4 kept experiments, uses a greedy comparison (faster, less
            data required).  With 5+ kept experiments, uses the configured
            statistical method.
        """
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
        # (2–4 kept). Use greedy comparison: a better result is considered
        # significant, a worse result is not.  This avoids both false positives
        # (keeping noise) and false negatives (discarding real improvements).
        if len(kept_values) < 5:
            kept_arr_small = np.array(kept_values, dtype=float)
            best_small = float(np.min(kept_arr_small) if minimize else np.max(kept_arr_small))
            point_est = float(current_value - best_small)
            is_better = current_value < best_small if minimize else current_value > best_small
            return EffectEstimate(
                point_estimate=point_est,
                confidence_interval=(float("-inf"), float("inf")),
                p_value=0.0 if is_better else 1.0,
                is_significant=is_better,
                method="greedy",
            )

        kept_arr = np.array(kept_values, dtype=float)
        best = float(np.min(kept_arr) if minimize else np.max(kept_arr))

        if self.method == "difference":
            # One-sided t-test: test whether current_value is significantly
            # better than the kept distribution.
            # H0: mean(kept_arr) == current_value.
            # For minimize=True, "better" means current_value < mean(kept),
            # so we use alternative="greater" (kept mean > current_value).
            # For minimize=False, alternative="less" (kept mean < current_value).
            # Combined with the directional guard (current_value < best / > best),
            # this gives a coherent one-sided significance test.
            effect = float(current_value - best)
            if minimize:
                result = stats.ttest_1samp(kept_arr, popmean=current_value, alternative="greater")
            else:
                result = stats.ttest_1samp(kept_arr, popmean=current_value, alternative="less")
            p_value_f = float(result.pvalue)
            # Infinite CI bounds: the CI for min(kept_arr) requires bootstrap;
            # callers should rely on is_significant rather than CI bounds.
            if minimize:
                is_significant = current_value < best and p_value_f < (1 - self.confidence_level)
            else:
                is_significant = current_value > best and p_value_f < (1 - self.confidence_level)
            return EffectEstimate(
                point_estimate=effect,
                confidence_interval=(float("-inf"), float("inf")),
                p_value=p_value_f,
                is_significant=is_significant,
                method="difference",
            )

        elif self.method == "bootstrap":
            return self._bootstrap_improvement(kept_arr, current_value, best, minimize)

        elif self.method == "aipw":
            # AIPW (augmented IPW) requires a treatment/control split that is not
            # available in the improvement context — fall back to bootstrap and warn.
            logger.warning(
                "estimate_improvement does not support 'aipw'; falling back to bootstrap"
            )
            return self._bootstrap_improvement(kept_arr, current_value, best, minimize)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _bootstrap_improvement(
        self,
        kept_arr: np.ndarray,
        current_value: float,
        best: float,
        minimize: bool,
    ) -> EffectEstimate:
        """Bootstrap significance test for improvement over kept distribution.

        Uses a one-sided t-test with a bootstrap-estimated p-value to test
        whether *current_value* is significantly better than the mean of
        *kept_arr*, combined with a directional guard against *best*.

        A bootstrap-of-the-minimum approach is *not* used here because the
        bootstrapped minimum is heavily concentrated at ``min(kept_arr)``
        (probability ≈ 63% per resample), so ``ci_best_lo ≈ min(kept_arr)``
        and the test degenerates to the greedy check.  Instead we bootstrap
        the *mean* of the kept distribution so the CI is meaningful.
        """
        rng = np.random.default_rng()
        n_iter = 100 if len(kept_arr) < 10 else self.n_bootstrap
        boot_means = np.empty(n_iter)
        for i in range(n_iter):
            boot = rng.choice(kept_arr, size=len(kept_arr), replace=True)
            boot_means[i] = float(np.mean(boot))

        point_estimate = current_value - best
        alpha = 1 - self.confidence_level
        ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

        # p-value: fraction of bootstrap means worse than (on the non-improvement
        # side of) current_value.  For minimize=True, "worse" means greater than
        # current_value; a very good candidate has nearly all boot_means above it,
        # giving a small p-value (strong significance).
        if minimize:
            p_value = float(np.mean(boot_means > current_value))
            # Significant if current_value is below the CI lower bound of the
            # kept mean AND is a raw improvement over the best-so-far.
            is_significant = current_value < best and current_value < ci_lo
        else:
            p_value = float(np.mean(boot_means < current_value))
            is_significant = current_value > best and current_value > ci_hi

        return EffectEstimate(
            point_estimate=point_estimate,
            confidence_interval=(ci_lo, ci_hi),
            p_value=p_value,
            is_significant=is_significant,
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
