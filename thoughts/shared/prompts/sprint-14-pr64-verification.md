# PR #64 Verification Commands

## PR Details

- **URL:** https://github.com/datablogin/causal-optimizer/pull/64
- **Branch:** `sprint-14/benchmark-tests-docs`
- **Final commit:** `734aa01` (verification doc update) — **merged 2026-03-25**
- **Previous commits:** `fbdf3c0` (isfinite fix), `78d239d` (gauntlet pass), `9a0600c` (polish)
- **Status:** MERGED

## Changes Since Previous Review

**Commit `fbdf3c0`** addresses follow-up review comment:

1. **Tightened smoke assertions to reject non-finite metrics** — `_assert_valid_result()` now checks `math.isfinite(result.test_mae)` and `math.isfinite(result.best_validation_mae)` alongside the existing `> 0` checks. `float('inf')` or `float('nan')` will now fail the smoke tests.

2. **Updated this verification doc** to the current reviewed commit.

## Files in This PR

1. **`tests/integration/test_predictive_energy_smoke.py`** — 4 smoke tests calling `run_strategy` on the 200-row fixture with budget=3. Tests each strategy and verifies `PredictiveBenchmarkResult` fields are populated, positive, and finite.

2. **`tests/regression/test_predictive_energy_reproducibility.py`** — 9 reproducibility tests (3 per strategy) marked `@pytest.mark.slow`. Runs each strategy twice with seed=42, budget=3 and asserts exact equality of `best_validation_mae`, `test_mae`, and `selected_parameters`.

3. **`thoughts/shared/docs/predictive-energy-benchmark.md`** — Benchmark documentation covering dataset contract, locked split, strategies, output artifact schema, example commands, key rule (test performance required for claims), and limitations.

4. **`thoughts/shared/plans/06-energy-predictive-benchmark-handoff.md`** — Updated handoff doc reflecting shipped API divergences from the original scaffold.

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

## Focused Verification for Follow-Up Fixes

```bash
# Run the smoke and reproducibility tests directly
uv run pytest tests/integration/test_predictive_energy_smoke.py tests/regression/test_predictive_energy_reproducibility.py -q
```

## New Tests Only

```bash
# Smoke tests (4 tests)
uv run pytest tests/integration/test_predictive_energy_smoke.py -v

# Reproducibility tests (9 tests, marked slow)
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

# Verify JSON artifact is valid, all fields present and finite
python3 -c "
import json, math
data = json.load(open('/tmp/predictive_energy_results_pr64.json'))
print(f'{len(data)} results')
for r in data:
    assert r['test_mae'] is not None and math.isfinite(r['test_mae'])
    assert r['best_validation_mae'] is not None and math.isfinite(r['best_validation_mae'])
    assert r['runtime_seconds'] is not None and r['runtime_seconds'] > 0
    assert r['validation_test_gap'] is not None
    assert isinstance(r['selected_parameters'], dict)
    print(f'  {r[\"strategy\"]:16s} budget={r[\"budget\"]} seed={r[\"seed\"]} val_mae={r[\"best_validation_mae\"]:.2f} test_mae={r[\"test_mae\"]:.2f}')
print('All results valid and finite')
"
```

## CI Check Status

| Check | Status |
|-------|--------|
| build | PASS |
| lint | PASS |
| typecheck | PASS |
| test (3.10) | PASS |
| test (3.11) | PASS |
| test (3.12) | PASS |
| test (3.13) | PASS |
