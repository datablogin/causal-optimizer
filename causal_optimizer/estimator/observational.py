"""Observational treatment effect estimation via DoWhy.

Estimates E[Y | do(X=x)] from observational data using causal identification
methods (backdoor adjustment, frontdoor adjustment, instrumental variables).

Requires the optional ``dowhy`` dependency:
    uv sync --extra causal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from causal_optimizer.types import CausalGraph, ExperimentLog

logger = logging.getLogger(__name__)

# Latent variable name used when converting bidirected edges
_LATENT_PREFIX = "U_"


def causal_graph_to_dowhy_str(graph: CausalGraph) -> str:
    """Convert a :class:`CausalGraph` to a DoWhy-compatible DOT digraph string.

    Directed edges (X → Y) are passed through as-is. Bidirected edges
    (X ↔ Y) are converted to a pair of directed edges from a synthetic latent
    node ``U_X_Y`` so that DoWhy understands the unobserved confounder.

    Args:
        graph: Our internal causal graph representation.

    Returns:
        A DOT-format digraph string accepted by ``dowhy.CausalModel``.
    """
    lines: list[str] = ["digraph {"]

    # Declare all observed nodes explicitly so isolated nodes (no edges) are
    # visible to DoWhy and don't silently disappear from graph analysis.
    for node in graph.nodes:
        lines.append(f'    "{node}";')

    # Directed edges
    for src, dst in graph.edges:
        lines.append(f'    "{src}" -> "{dst}";')

    # Bidirected edges → latent common cause nodes
    # DoWhy identifies unobserved nodes via the `observed="no"` attribute.
    for i, (u, v) in enumerate(graph.bidirected_edges):
        latent = f"{_LATENT_PREFIX}{i}_{u}_{v}"
        lines.append(f'    "{latent}" [label="Unobserved Confounders", observed="no"];')
        lines.append(f'    "{latent}" -> "{u}";')
        lines.append(f'    "{latent}" -> "{v}";')

    lines.append("}")
    return "\n".join(lines)


@dataclass
class ObservationalEstimate:
    """Result of an observational causal effect estimation.

    Attributes:
        expected_outcome: Predicted E[Y | do(X=x)].
        confidence_interval: (lower, upper) bounds at the configured confidence level.
        method: Which adjustment strategy was used (e.g., ``"backdoor.linear_regression"``).
        identified: ``True`` if the effect was identifiable from the causal graph;
            ``False`` if we fell back to a naive estimate.
    """

    expected_outcome: float
    confidence_interval: tuple[float, float]
    method: str
    identified: bool


class ObservationalEstimator:
    """Estimate E[Y | do(X=x)] from observational data using DoWhy.

    Uses causal identification methods to correct for confounding when the
    causal graph provides sufficient structure.  Gracefully degrades to a
    random-forest surrogate prediction when:

    * ``dowhy`` is not installed, or
    * the effect is not identifiable from the graph.

    Args:
        causal_graph: The causal graph describing the data-generating process.
        method: Identification strategy — ``"backdoor"`` (default),
            ``"frontdoor"``, or ``"iv"``.
        confidence_level: Confidence level for the interval (default 0.95).
    """

    _METHOD_MAP: dict[str, str] = {
        "backdoor": "backdoor.linear_regression",
        "frontdoor": "frontdoor.two_stage_regression",
        "iv": "iv.instrumental_variable",
    }

    def __init__(
        self,
        causal_graph: CausalGraph,
        method: str = "backdoor",
        confidence_level: float = 0.95,
    ) -> None:
        if method not in self._METHOD_MAP:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(self._METHOD_MAP)}")
        self.causal_graph = causal_graph
        self.method = method
        self.confidence_level = confidence_level

    def estimate_intervention(
        self,
        experiment_log: ExperimentLog,
        treatment_var: str,
        treatment_value: float,
        objective_name: str = "objective",
    ) -> ObservationalEstimate:
        """Estimate E[Y | do(treatment_var = treatment_value)].

        Args:
            experiment_log: Historical observations to use as data.
            treatment_var: Name of the treatment variable.
            treatment_value: The value of the intervention.
            objective_name: Name of the outcome column in the data.

        Returns:
            :class:`ObservationalEstimate` with ``identified=True`` when
            DoWhy could identify the effect, or ``identified=False`` with an
            RF-surrogate fallback otherwise.

        Raises:
            ImportError: If ``dowhy`` is not installed.
        """
        try:
            from dowhy import CausalModel
        except ImportError as exc:
            raise ImportError(
                "The 'dowhy' package is required for ObservationalEstimator. "
                "Install it with: uv sync --extra causal"
            ) from exc

        df = experiment_log.to_dataframe()

        # Exclude non-numeric / metadata columns
        drop_cols = {"experiment_id", "status"}
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        graph_str = causal_graph_to_dowhy_str(self.causal_graph)
        dowhy_method = self._METHOD_MAP[self.method]

        try:
            model = CausalModel(
                data=df,
                treatment=treatment_var,
                outcome=objective_name,
                graph=graph_str,
            )
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

            # Check if the requested identification strategy found a valid estimand
            estimand_key = self.method  # "backdoor", "frontdoor", or "iv"
            estimand_info = identified_estimand.estimands.get(estimand_key)
            if not estimand_info:
                logger.warning(
                    "Effect of %s on %s is not identifiable via %s. "
                    "Returning identified=False with RF surrogate.",
                    treatment_var,
                    objective_name,
                    self.method,
                )
                return self._rf_fallback(df, treatment_var, treatment_value, objective_name)

            estimate = model.estimate_effect(
                identified_estimand,
                method_name=dowhy_method,
            )

            # Guard against None or NaN estimate values (DoWhy can return NaN
            # for numerically degenerate cases, e.g. singular covariance matrix).
            if estimate.value is None or (
                isinstance(estimate.value, (float, np.floating)) and np.isnan(estimate.value)
            ):
                logger.warning(
                    "DoWhy returned None/NaN estimate for %s method. Falling back.",
                    self.method,
                )
                return self._rf_fallback(df, treatment_var, treatment_value, objective_name)

            # Convert estimate to E[Y | do(T=t)].
            # - backdoor (linear regression): estimate.value is the slope coef;
            #   E[Y|do(T=t)] = intercept + coef * t.
            # - frontdoor/iv: estimate.value is the ATE (scalar shift); intercept
            #   attribute is not set by DoWhy.  Approximate using mean(Y) as
            #   baseline: E[Y|do(T=t)] ≈ mean(Y) + ATE * (t - mean(T)).
            intercept_raw = getattr(estimate, "intercept", None)
            if self.method == "backdoor" and intercept_raw is not None:
                intercept_val = float(intercept_raw)
                expected_outcome = intercept_val + float(estimate.value) * treatment_value
            else:
                # frontdoor/iv: estimate.value is the ATE (scalar shift), not a slope.
                # Approximate E[Y|do(T=t)] ≈ mean(Y) + ATE * (t - mean(T)).
                if treatment_var not in df.columns:
                    raise ValueError(
                        f"treatment_var={treatment_var!r} not found in data columns. "
                        f"Available columns: {list(df.columns)}"
                    )
                baseline = float(np.mean(df[objective_name].values))
                t_mean = float(np.mean(df[treatment_var].values))
                expected_outcome = baseline + float(estimate.value) * (treatment_value - t_mean)

            # Confidence interval from standard error
            # CI on E[Y|do(T=t)] = intercept + coef * t.
            # We build the CI from the SE of the ATE (slope) and ensure a
            # minimum non-zero width so the interval is meaningful even when
            # treatment_value is near zero (where coef * t → 0).
            from scipy import stats as scipy_stats

            alpha_ci = 1.0 - self.confidence_level
            z_ci = float(scipy_stats.norm.ppf(1.0 - alpha_ci / 2.0))
            try:
                se_arr = estimate.get_standard_error()
                se_ate = float(np.ravel(se_arr)[0]) if se_arr is not None else 0.0
            except Exception:
                se_ate = 0.0

            # SE of predicted outcome at treatment_value:
            # σ_{ŷ} ≈ |t| * se_ate  +  floor (outcome residual SE / √n)
            outcome_col = objective_name
            try:
                floor_se = float(np.std(df[outcome_col].values)) / max(1.0, float(np.sqrt(len(df))))
            except Exception:
                floor_se = 0.0
            # Apply a minimum floor to prevent zero-width CIs even when
            # treatment_value=0 and floor_se is negligibly small.
            pred_se = max(abs(treatment_value) * se_ate + floor_se, 1e-6)
            ci_lo_v = expected_outcome - z_ci * pred_se
            ci_hi_v = expected_outcome + z_ci * pred_se
            ci: tuple[float, float] = (ci_lo_v, ci_hi_v)

            return ObservationalEstimate(
                expected_outcome=expected_outcome,
                confidence_interval=ci,
                method=f"observational/{dowhy_method}",
                identified=True,
            )

        except ImportError:
            raise
        except Exception as exc:
            logger.warning("DoWhy estimation failed (%s). Using RF fallback.", exc)
            return self._rf_fallback(df, treatment_var, treatment_value, objective_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rf_fallback(
        self,
        df: pd.DataFrame,
        treatment_var: str,
        treatment_value: float,
        objective_name: str,
    ) -> ObservationalEstimate:
        """Predict E[Y | X=x] using a random-forest surrogate (no causal adjustment)."""
        from sklearn.ensemble import RandomForestRegressor

        feature_cols = [c for c in df.columns if c != objective_name]
        features_df = df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
        y = df[objective_name].values

        if len(features_df) < 5 or features_df.shape[1] == 0:
            return ObservationalEstimate(
                expected_outcome=float(np.mean(y)) if len(y) > 0 else 0.0,
                confidence_interval=(float("-inf"), float("inf")),
                method="fallback/rf_insufficient_data",
                identified=False,
            )

        rng = np.random.default_rng(42)
        rf = RandomForestRegressor(n_estimators=50, random_state=int(rng.integers(0, 2**31)))
        rf.fit(features_df.values, y)

        # Build a representative point with treatment_var set to treatment_value
        point = features_df.mean().to_dict()
        if treatment_var in point:
            point[treatment_var] = treatment_value

        point_arr = np.array([[point.get(c, 0.0) for c in features_df.columns]])
        pred = float(rf.predict(point_arr)[0])

        # Rough CI: ±z std of individual tree predictions (respects confidence_level)
        tree_preds = np.array([t.predict(point_arr)[0] for t in rf.estimators_])
        std = float(np.std(tree_preds))
        from scipy import stats as scipy_stats

        alpha = 1.0 - self.confidence_level
        z_rf = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
        ci: tuple[float, float] = (pred - z_rf * std, pred + z_rf * std)

        return ObservationalEstimate(
            expected_outcome=pred,
            confidence_interval=ci,
            method="fallback/rf_surrogate",
            identified=False,
        )
