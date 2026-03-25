# PR #64 Verification Commands

## PR Details

- **URL:** https://github.com/datablogin/causal-optimizer/pull/64
- **Branch:** `sprint-14/benchmark-tests-docs`
- **Commit:** `9a0600c` (polish pass) / `5e72d6e` (initial)
- **Worktree:** `/Users/robertwelborn/Projects/causal-optimizer/.claude/worktrees/agent-a4b89242`

## Changes

### New files

1. **`tests/integration/test_predictive_energy_smoke.py`** — 5 smoke tests calling `run_strategy` from the benchmark script on the 200-row fixture dataset with budget=3. Tests each strategy ("random", "surrogate_only", "causal") and verifies `PredictiveBenchmarkResult` fields are populated (`test_mae > 0`, `best_validation_mae > 0`, `runtime_seconds > 0`, gap auto-computed).

2. **`tests/regression/test_predictive_energy_reproducibility.py`** — 3 reproducibility tests marked `@pytest.mark.slow`. Runs `run_strategy("random", budget=3, seed=42)` twice and asserts `best_validation_mae`, `test_mae`, and `selected_parameters` are exactly equal across runs.

3. **`thoughts/shared/docs/predictive-energy-benchmark.md`** — Benchmark documentation covering dataset contract, locked split, strategies, output artifact schema, example commands, key rule (test performance required for claims), and limitations.

4. **`thoughts/shared/plans/06-energy-predictive-benchmark-handoff.md`** — Updated handoff doc reflecting shipped API divergences from the original scaffold (auto-computed gap, `runtime_seconds`, `split_timestamp`-based splitting, direct random sampling, `None` return on all-crash, CLI enhancements).

5. **`thoughts/shared/prompts/sprint-14-pr64-verification.md`** — This file.

## Lint & Format

```bash
uv run ruff check tests/integration/test_predictive_energy_smoke.py tests/regression/test_predictive_energy_reproducibility.py
uv run ruff format --check tests/integration/test_predictive_energy_smoke.py tests/regression/test_predictive_energy_reproducibility.py
```

## Type Check

```bash
uv run mypy causal_optimizer/
```

## New Tests Only

```bash
# Smoke tests (5 tests, ~26s)
uv run pytest tests/integration/test_predictive_energy_smoke.py -v

# Reproducibility tests (3 tests, ~2s, marked slow)
uv run pytest tests/regression/test_predictive_energy_reproducibility.py -v
```

## Full Test Suite

```bash
uv run pytest -m "not slow"
```

## Manual Smoke Test

```bash
# Run all three strategies on fixture data with tiny budget
uv run python scripts/energy_predictive_benchmark.py \
  --data-path tests/fixtures/energy_load_fixture.csv \
  --budgets 3 \
  --seeds 0 \
  --strategies random,surrogate_only,causal \
  --output /tmp/predictive_energy_results_pr64.json

# Verify JSON artifact is valid and has all fields
python3 -c "
import json
data = json.load(open('/tmp/predictive_energy_results_pr64.json'))
print(f'{len(data)} results')
for r in data:
    assert r['test_mae'] is not None and r['test_mae'] > 0
    assert r['best_validation_mae'] is not None and r['best_validation_mae'] > 0
    assert r['runtime_seconds'] is not None and r['runtime_seconds'] > 0
    assert r['validation_test_gap'] is not None
    assert isinstance(r['selected_parameters'], dict)
    print(f'  {r[\"strategy\"]:16s} budget={r[\"budget\"]} seed={r[\"seed\"]} val_mae={r[\"best_validation_mae\"]:.2f} test_mae={r[\"test_mae\"]:.2f}')
print('All results valid')
"
```

## CI Check Status

| Check | Status |
|-------|--------|
| build | PENDING |
| lint | PENDING |
| typecheck | PENDING |
| test (3.10) | PENDING |
| test (3.11) | PENDING |
| test (3.12) | PENDING |
| test (3.13) | PENDING |
