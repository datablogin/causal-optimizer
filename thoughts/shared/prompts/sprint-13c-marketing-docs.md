# Sprint 13c — Marketing Adapter Doc Update

## Context

Issues #45, #46, and #47 changed MarketingLogAdapter validation, metrics, and zero-support behavior, but the adapter documentation was not updated (GitHub #55).

**Base branch:** `main`

Read `CLAUDE.md` for project conventions. Read the GitHub issue for full context:
- `gh issue view 55` — Update MarketingLogAdapter docs after issues #45-47

---

## Prompt

Stand up one agent. Follow this exact workflow:

  read current code → update docs → /polish → gh pr create → /gauntlet → report PR URL

Feature:

- **sprint-13c/marketing-docs** (GitHub #55) — Update `thoughts/shared/docs/marketing-log-adapter.md` to match the current implementation. Before writing anything, read `causal_optimizer/domain_adapters/marketing_logs.py` (the full file) and `causal_optimizer/predictor/off_policy.py` (the `__init__` signature and `_observational_predict` method) to verify current behavior. Then make these changes to the doc:
  1. **Validation section**: expand the bullet to list all current validations — missing columns, empty DataFrames, NaN values, binary treatment `{0, 1}` enforcement, propensity values in `[0, 1]`, and warnings for single-arm data and boundary propensities.
  2. **Metrics table**: add a row for `zero_support` — `float`, `1.0 when no logged observations match the proposed policy (zero IPS support), 0.0 otherwise`.
  3. **Known Assumptions and Limitations**: replace item 5 ("No positivity violation warnings") entirely. The adapter now validates propensity bounds at construction, clips propensities at evaluation time, returns `zero_support = 1.0` and uses a pessimistic fallback (`outcome.min()`) for `policy_value` when support is zero, and emits a `logger.warning`. Write this clearly so a user understands what happens when their policy has no support in the logged data and *why* the fallback is pessimistic (to prevent the optimizer from rewarding unsupported policies).
  4. **Nice-to-have**: add a short "Runtime notes" section after "Known Assumptions" mentioning that the off-policy predictor gates observational (DoWhy) estimation behind `obs_min_history` (default 20) to avoid expensive causal estimation on small experiment logs. This is relevant context for users who enable `epsilon_mode=True` and wonder why observational estimates only appear after 20 experiments.
  5. **Do not change** the search variables table, prior graph, descriptor names, or split strategy sections — verify they are still accurate and leave them alone.
  6. **Parquet loading**: the validation section currently says `data_path` accepts CSV. Update to mention that `.parquet` files are also supported (detected by file extension).

Rules:
- This is a docs-only PR — do not modify any Python source files
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Agent works in an isolated worktree
- Do NOT merge — leave PR open for human review
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)
