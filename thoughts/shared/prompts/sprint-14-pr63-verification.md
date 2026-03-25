# PR #63 Verification Commands

## PR Details

- **URL:** https://github.com/datablogin/causal-optimizer/pull/63
- **Branch:** `sprint-14/benchmark-runner`
- **Final commit:** `d075662` (follow-up review fixes) — **merged 2026-03-25**
- **Previous commits:** `fe9222e` (initial gauntlet pass), `49d82d8` (greptile fixes)
- **Status:** MERGED

## Change History

### Initial delivery (`fe9222e`)

- `scripts/energy_predictive_benchmark.py` — CLI entrypoint with argparse, `run_strategy()`, JSON artifact writer, summary table printer
- 19 unit tests for `parse_args`, `_sanitize_for_json`, `_fmt_mean_std`
- Gauntlet: greploop 4 iterations (7 comments), claudeloop 3 iterations (7 issues), check-pr pass

### Greptile follow-up (`49d82d8`)

- Added empty DataFrame guards in `ValidationEnergyRunner.__init__` and `evaluate_on_test` (prevents `ZeroDivisionError`)
- Added file format allowlist in `load_energy_frame` (rejects non `.csv`/`.parquet`)

### Human review follow-up (`d075662`)

1. **`runtime_seconds` now covers the full strategy run** — `time.perf_counter()` stop moved after `evaluate_on_test()` so runtime includes held-out test evaluation, not just the validation search loop.

2. **All-crash combinations are skipped, not serialized** — `run_strategy()` returns `None` when no valid result is produced. `main()` filters out `None` results so the JSON artifact contains only real benchmark records (no sentinel `inf`/`null` rows).

3. **New tests added (3):**
   - `test_runtime_includes_test_evaluation` — verifies `runtime_seconds > 0` on a real run
   - `test_returns_none_when_engine_all_crash` — mocks runner to always crash, asserts `run_strategy` returns `None`
   - `test_invalid_strategy_raises` — asserts `ValueError` on unknown strategy name

## Lint & Format

```bash
uv run ruff check scripts/energy_predictive_benchmark.py
uv run ruff format --check scripts/energy_predictive_benchmark.py
```

## Type Check

```bash
uv run mypy scripts/energy_predictive_benchmark.py
```

## Unit Tests (22 tests — 19 original + 3 follow-up)

```bash
uv run pytest tests/unit/test_energy_predictive_benchmark_script.py -v
```

## Full Test Suite

```bash
uv run pytest -m "not slow"
```

## Focused Verification for Follow-Up Fixes

```bash
# 1. Smoke test with all three strategies — verify JSON has no sentinel rows
uv run python scripts/energy_predictive_benchmark.py \
  --data-path tests/fixtures/energy_load_fixture.csv \
  --budgets 3 \
  --seeds 0 \
  --strategies random,surrogate_only,causal \
  --output /tmp/predictive_energy_results_pr63_all.json

# Verify no null/inf in the artifact
python3 -c "
import json
data = json.load(open('/tmp/predictive_energy_results_pr63_all.json'))
print(f'{len(data)} results')
for r in data:
    assert r['test_mae'] is not None, 'sentinel row found'
    assert r['best_validation_mae'] is not None, 'sentinel row found'
print('No sentinel rows — all results are valid')
"

# 2. Run the new skip-behavior tests directly
uv run pytest tests/unit/test_energy_predictive_benchmark_script.py::TestRunStrategy -v
```

## Manual Smoke Test

```bash
# Minimal: one strategy, one seed, tiny budget on fixture data
uv run python scripts/energy_predictive_benchmark.py \
  --data-path tests/fixtures/energy_load_fixture.csv \
  --budgets 3 \
  --seeds 0 \
  --strategies random

# Verify JSON artifact was created
cat predictive_energy_results.json

# Custom output path
uv run python scripts/energy_predictive_benchmark.py \
  --data-path tests/fixtures/energy_load_fixture.csv \
  --budgets 3 \
  --seeds 0,1 \
  --strategies random \
  --output /tmp/bench_results.json
```

## CI Checks (all passed)

| Check | Status |
|-------|--------|
| build | PASS |
| lint | PASS |
| typecheck | PASS |
| test (3.10) | PASS |
| test (3.11) | PASS |
| test (3.12) | PASS |
| test (3.13) | PASS |
