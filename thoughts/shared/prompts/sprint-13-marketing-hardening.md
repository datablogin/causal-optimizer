# Sprint 13 — Marketing Adapter Hardening

## Context

Post-merge audit of `MarketingLogAdapter` surfaced three issues (GitHub #45, #46, #47). These range from a silent correctness bug to missing input validation to a performance bottleneck in the off-policy predictor's observational estimation path. All three target the marketing adapter surface.

**Base branch:** `main`

Read `CLAUDE.md` for project conventions. Read the GitHub issues for full context:
- `gh issue view 45` — Performance: cache and delay observational estimates
- `gh issue view 46` — Bug: zero-support policies silently return population mean
- `gh issue view 47` — Validation: reject invalid treatment/propensity values

---

## Prompt

Stand up one agent per feature. Each agent must follow this exact workflow:

  /tdd → implement → /polish → gh pr create → /gauntlet → report PR URL

Features this sprint:

- **sprint-13/marketing-perf** (GitHub #45) — Fix the off-policy predictor hot path so marketing fixture runs stay within wall-clock budgets. (1) In `_should_run_heuristic()` in `predictor/off_policy.py`, move the cheap `model_quality < 0.3` and `model is None` guards *before* calling `self.predict()`. (2) Cache the `Prediction` returned during `should_run_experiment()` on the predictor instance (`_last_prediction`) so `engine/loop.py:step()` can reuse it for logging instead of calling `predict()` a second time. (3) Gate `_observational_predict()` behind a minimum history threshold — only attempt DoWhy estimation when `len(experiment_log) >= 20` (not the existing `min_history=5`). Add a class-level `obs_min_history: int = 20` parameter to `OffPolicyPredictor.__init__`. (4) Suppress repeated `statsmodels` and `pygraphviz` warnings in `_observational_predict()` using `warnings.catch_warnings()` with `filterwarnings("ignore")` scoped to the DoWhy call. (5) Add a timed regression test: `tests/regression/test_marketing_perf.py` that runs `MarketingLogAdapter` integration pipeline (15 experiments) and asserts wall-clock < 15s. Mark `@pytest.mark.slow`. (6) Verify `uv run python examples/marketing_logs.py` completes in < 30s. Reference: `causal_optimizer/predictor/off_policy.py:128-211`, `causal_optimizer/engine/loop.py:491-514`.

- **sprint-13/zero-support-guard** (GitHub #46) — Fix `MarketingLogAdapter.run_experiment()` so zero-support policies are not silently rewarded. In `marketing_logs.py`, when `weight_sum == 0`: (1) set `policy_value` to a *pessimistic fallback* — use `float(outcome.min())` for maximize objectives (the worst observed outcome), not the population mean; (2) set `effective_sample_size` to `0.0` (already done); (3) add a new boolean metric `zero_support: float = 1.0` (use `0.0` when support exists, `1.0` when it doesn't — keep all metrics numeric per contract); (4) keep the existing `logger.warning`. Add unit tests in `tests/unit/test_marketing_log_adapter.py`: construct a DataFrame where the proposed policy has zero support in the logged data (e.g., all rows are treatment=1 but policy says don't treat), assert `policy_value < population_mean`, `effective_sample_size == 0.0`, `zero_support == 1.0`. Also add a test for the normal case: `zero_support == 0.0`. Reference: `causal_optimizer/domain_adapters/marketing_logs.py:271-281`.

- **sprint-13/input-validation** (GitHub #47) — Add schema validation to `MarketingLogAdapter._validate_data()` for binary treatment and propensity bounds. In `marketing_logs.py:_validate_data()`: (1) After existing NaN checks, validate that treatment column contains only values in `{0, 1}` — raise `ValueError` with message `"Treatment column '{col}' must be binary (0/1), found values: {bad_values}"`. (2) If propensity column exists, validate all values are in `[0, 1]` — raise `ValueError` with message `"Propensity column '{col}' values must be in [0, 1], found range [{min}, {max}]"`. (3) Warn (via `logger.warning`) when one treatment arm is entirely absent (all 0s or all 1s). Add unit tests in `tests/unit/test_marketing_log_adapter.py`: test that non-binary treatment raises ValueError, test that out-of-range propensity raises ValueError, test that single-arm data logs a warning (use `caplog` fixture), test that valid binary data passes. Reference: `causal_optimizer/domain_adapters/marketing_logs.py:87-114`.

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Each agent works in an isolated worktree
- Do NOT merge — leave PRs open for human review
- Merge order after human approval: input-validation → zero-support-guard → marketing-perf
- After each merge, next PR rebases on main and re-runs `uv run pytest -m "not slow"`
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)
