# Sprint 12b — Real-Data Adapter Hardening

## Context

Sprint 12 shipped `EnergyLoadAdapter` and `MarketingLogAdapter` (PRs #33-34). Post-merge audit found 4 items to fix. This sprint addresses all of them in a single PR.

**Branch:** `sprint-12b/adapter-hardening`

Read `CLAUDE.md` for project conventions. Use strict TDD — write failing tests first, then implement.

## Reference Files

Read these before starting:
- `CLAUDE.md`
- `causal_optimizer/domain_adapters/energy_load.py`
- `causal_optimizer/domain_adapters/marketing_logs.py`
- `causal_optimizer/diagnostics/models.py` (the `Recommendation` model: fields are `rank`, `rec_type`, `confidence`, `title`, `description`, `evidence`, `next_step`, `expected_info_gain`, `variables_involved`)
- `examples/energy_load.py`
- `examples/marketing_logs.py`
- `tests/unit/test_examples.py` (existing example test pattern using `_import_example()`)
- `tests/unit/test_energy_load_adapter.py`
- `tests/unit/test_marketing_log_adapter.py`

---

## Task 1 — Fix broken marketing_logs example

**File:** `examples/marketing_logs.py`, line 83-84

The `Recommendation` model fields are `rec_type`, `title`, `description` — but the example uses `recommendation_type`, `summary`, `rationale` which don't exist.

**Fix:**
```python
# BEFORE (line 83-84):
print(f"  [{rec.recommendation_type.value}] {rec.summary}")
print(f"    Rationale: {rec.rationale}")

# AFTER:
print(f"  [{rec.rec_type.value}] {rec.title}")
print(f"    {rec.description}")
```

**Verify** `examples/energy_load.py` lines 88-89 already use the correct fields (`rec.rec_type.value`, `rec.title`, `rec.description`).

---

## Task 2 — Add both real-data examples to the test surface

**File:** `tests/unit/test_examples.py`

Follow the existing pattern (`_import_example()` + `mod.main()`). Add two test classes at the bottom of the file:

```python
class TestEnergyLoadExampleRuns:
    """Test that energy_load.py runs to completion."""

    def test_energy_load_runs(self) -> None:
        mod = _import_example("energy_load")
        mod.main()


class TestMarketingLogsExampleRuns:
    """Test that marketing_logs.py runs to completion."""

    def test_marketing_logs_runs(self) -> None:
        mod = _import_example("marketing_logs")
        mod.main()
```

Run the tests and confirm both pass:
```bash
uv run pytest tests/unit/test_examples.py::TestEnergyLoadExampleRuns -v
uv run pytest tests/unit/test_examples.py::TestMarketingLogsExampleRuns -v
```

---

## Task 3 — Add Parquet loading support to both adapters

Both adapters currently only accept CSV via `pd.read_csv()`. Add support for `.parquet` files using `pd.read_parquet()`.

**No new dependencies required.** `pd.read_parquet()` works with `pyarrow` (which pandas can install as an optional backend). The adapters should attempt the read and let pandas raise if the parquet engine isn't installed — matching the project's graceful degradation pattern.

### Implementation

Add a shared helper method to each adapter's `__init__`, or just inline the logic. Detect by file extension:

```python
if data_path is not None:
    if data_path.endswith(".parquet"):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_csv(data_path)
```

Apply to both:
- `EnergyLoadAdapter.__init__()` (line 64)
- `MarketingLogAdapter.__init__()` (line 71)

### Tests (write first)

Add to `tests/unit/test_energy_load_adapter.py`:

```python
class TestParquetLoading:
    """Test that Parquet files can be loaded."""

    def test_load_from_parquet(self, fixture_df: pd.DataFrame, tmp_path: Path) -> None:
        parquet_path = tmp_path / "energy.parquet"
        fixture_df.to_parquet(parquet_path, index=False)
        adapter = EnergyLoadAdapter(data_path=str(parquet_path), seed=42)
        space = adapter.get_search_space()
        assert len(space.variables) == 7

    def test_parquet_produces_same_metrics_as_csv(
        self, fixture_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        parquet_path = tmp_path / "energy.parquet"
        fixture_df.to_parquet(parquet_path, index=False)
        csv_adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
        parquet_adapter = EnergyLoadAdapter(data_path=str(parquet_path), seed=42)
        params = {
            "model_type": "ridge",
            "lookback_window": 3,
            "use_temperature": True,
            "use_humidity": False,
            "use_calendar": True,
            "regularization": 1.0,
            "n_estimators": 50,
        }
        assert csv_adapter.run_experiment(params) == parquet_adapter.run_experiment(params)
```

Add equivalent tests to `tests/unit/test_marketing_log_adapter.py`.

Mark these tests with `pytest.importorskip("pyarrow")` at the top of the test class or use a fixture that skips if pyarrow is not installed:

```python
@pytest.fixture(autouse=True)
def _require_pyarrow(self) -> None:
    pytest.importorskip("pyarrow")
```

---

## Task 4 — Add timestamp sorting/validation to EnergyLoadAdapter

The adapter uses `pd.Series.shift()` for lag features, which is order-dependent. If data is passed unsorted, lag features will be silently wrong.

### Implementation

In `EnergyLoadAdapter._validate_data()` or at the end of `__init__()` after validation, add:

1. **Parse timestamps**: Convert the `timestamp` column to datetime if it isn't already.
2. **Sort by timestamp**: Sort the DataFrame by timestamp and reset the index.
3. **Check for duplicates**: Warn (via `logger.warning`) if duplicate timestamps exist.
4. **Check for gaps**: Compare the actual number of rows to the expected number based on the time range and inferred frequency. If there's a significant gap, log a warning — don't error, since the adapter handles missing values via forward-fill.

```python
# In __init__, after _validate_data():
self._data["timestamp"] = pd.to_datetime(self._data["timestamp"])
self._data = self._data.sort_values("timestamp").reset_index(drop=True)

n_dupes = self._data["timestamp"].duplicated().sum()
if n_dupes > 0:
    logger.warning("%d duplicate timestamps found; keeping first occurrence", n_dupes)
    self._data = self._data.drop_duplicates(subset=["timestamp"], keep="first").reset_index(
        drop=True
    )
```

### Tests (write first)

Add to `tests/unit/test_energy_load_adapter.py`:

```python
class TestTimestampHandling:
    """Timestamp sorting, validation, and gap handling."""

    def test_unsorted_data_is_sorted(self, fixture_df: pd.DataFrame) -> None:
        """Passing shuffled data should produce same results as sorted."""
        shuffled = fixture_df.sample(frac=1, random_state=99).reset_index(drop=True)
        adapter_sorted = EnergyLoadAdapter(data=fixture_df, seed=42)
        adapter_shuffled = EnergyLoadAdapter(data=shuffled, seed=42)
        params = {
            "model_type": "ridge",
            "lookback_window": 3,
            "use_temperature": True,
            "use_humidity": False,
            "use_calendar": True,
            "regularization": 1.0,
            "n_estimators": 50,
        }
        m1 = adapter_sorted.run_experiment(params)
        m2 = adapter_shuffled.run_experiment(params)
        assert m1["mae"] == pytest.approx(m2["mae"], rel=1e-6)

    def test_duplicate_timestamps_handled(self, fixture_df: pd.DataFrame) -> None:
        """Duplicate timestamps should be deduplicated with a warning."""
        duped = pd.concat([fixture_df, fixture_df.iloc[:5]], ignore_index=True)
        adapter = EnergyLoadAdapter(data=duped, seed=42)
        # Should succeed — duplicates dropped
        space = adapter.get_search_space()
        assert len(space.variables) == 7

    def test_timestamp_parsed_as_datetime(self, fixture_df: pd.DataFrame) -> None:
        """Timestamp column should be parsed to datetime."""
        adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
        assert pd.api.types.is_datetime64_any_dtype(adapter._data["timestamp"])
```

---

## Task 5 — Add leakage and support warning metrics (#41)

### EnergyLoadAdapter — add these metrics to `run_experiment()` return dict

| Metric | Computation | Purpose |
|--------|-------------|---------|
| `validation_set_size` | `float(len(x_val))` | Warns when validation shrinks due to large lookback |
| `nan_rows_dropped` | `float(n_original - len(df))` after NaN drop | Data quality signal |
| `train_val_ratio_actual` | `float(train_end / len(df))` after NaN drop | Drift from requested ratio |

Add these after the existing metric computation, before the return statement.

### MarketingLogAdapter — add these metrics to `run_experiment()` return dict

| Metric | Computation | Purpose |
|--------|-------------|---------|
| `propensity_clip_fraction` | `float((raw != clipped).mean())` comparing pre/post clip | How aggressively propensities are clipped |
| `max_ips_weight` | `float(weights.max())` | Largest individual IPS weight (fragile estimate signal) |
| `weight_cv` | `float(weights[weights > 0].std() / weights[weights > 0].mean())` if any positive weights, else 0.0 | Weight distribution stability |

### Tests (write first)

**EnergyLoadAdapter tests:**

```python
class TestWarningMetrics:
    """Warning/diagnostic metrics are present and valid."""

    def test_warning_metrics_present(
        self, adapter: EnergyLoadAdapter, default_params: dict
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert "validation_set_size" in metrics
        assert "nan_rows_dropped" in metrics
        assert "train_val_ratio_actual" in metrics

    def test_validation_set_size_is_positive(
        self, adapter: EnergyLoadAdapter, default_params: dict
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert metrics["validation_set_size"] > 0

    def test_nan_rows_dropped_nonnegative(
        self, adapter: EnergyLoadAdapter, default_params: dict
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert metrics["nan_rows_dropped"] >= 0

    def test_large_lookback_increases_nan_drops(self, fixture_df: pd.DataFrame) -> None:
        adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
        params_small = {**default_params, "lookback_window": 1}
        params_large = {**default_params, "lookback_window": 24}
        m_small = adapter.run_experiment(params_small)
        m_large = adapter.run_experiment(params_large)
        assert m_large["nan_rows_dropped"] >= m_small["nan_rows_dropped"]
```

Note: use a `default_params` fixture or inline dict — match the existing test pattern.

**MarketingLogAdapter tests:**

```python
class TestWarningMetrics:
    """Warning/diagnostic metrics are present and valid."""

    def test_warning_metrics_present(
        self, adapter: MarketingLogAdapter, default_params: dict
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert "propensity_clip_fraction" in metrics
        assert "max_ips_weight" in metrics
        assert "weight_cv" in metrics

    def test_all_warning_metrics_finite(
        self, adapter: MarketingLogAdapter, default_params: dict
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        for key in ["propensity_clip_fraction", "max_ips_weight", "weight_cv"]:
            assert np.isfinite(metrics[key]), f"{key} is not finite"

    def test_propensity_clip_fraction_bounded(
        self, adapter: MarketingLogAdapter, default_params: dict
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert 0.0 <= metrics["propensity_clip_fraction"] <= 1.0

    def test_aggressive_clip_increases_clip_fraction(
        self, fixture_df: pd.DataFrame
    ) -> None:
        adapter = MarketingLogAdapter(data=fixture_df, seed=42)
        params_tight = {**default_params, "min_propensity_clip": 0.01}
        params_wide = {**default_params, "min_propensity_clip": 0.45}
        m_tight = adapter.run_experiment(params_tight)
        m_wide = adapter.run_experiment(params_wide)
        assert m_wide["propensity_clip_fraction"] >= m_tight["propensity_clip_fraction"]
```

---

## Verification

After all changes, run:

```bash
# New and changed tests
uv run pytest tests/unit/test_energy_load_adapter.py tests/unit/test_marketing_log_adapter.py tests/unit/test_examples.py -v

# Lint and format
uv run ruff check causal_optimizer/domain_adapters/energy_load.py causal_optimizer/domain_adapters/marketing_logs.py examples/energy_load.py examples/marketing_logs.py
uv run ruff format causal_optimizer/domain_adapters/energy_load.py causal_optimizer/domain_adapters/marketing_logs.py examples/energy_load.py examples/marketing_logs.py

# Type check
uv run mypy causal_optimizer/domain_adapters/energy_load.py causal_optimizer/domain_adapters/marketing_logs.py

# Full regression
uv run pytest -m "not slow"
```

---

## Workflow

```
write failing tests → implement → verify → /polish → gh pr create → /gauntlet → report PR URL
```

Do NOT merge — leave PR open for human review.

---

## Files to modify

| File | Changes |
|------|---------|
| `examples/marketing_logs.py` | Fix lines 83-84: `recommendation_type` → `rec_type`, `summary` → `title`, `rationale` → `description` |
| `tests/unit/test_examples.py` | Add `TestEnergyLoadExampleRuns` and `TestMarketingLogsExampleRuns` classes |
| `causal_optimizer/domain_adapters/energy_load.py` | Add timestamp sorting/validation in `__init__`, add warning metrics in `run_experiment()`, add parquet loading |
| `causal_optimizer/domain_adapters/marketing_logs.py` | Add warning metrics in `run_experiment()`, add parquet loading |
| `tests/unit/test_energy_load_adapter.py` | Add `TestTimestampHandling`, `TestWarningMetrics`, `TestParquetLoading` classes |
| `tests/unit/test_marketing_log_adapter.py` | Add `TestWarningMetrics`, `TestParquetLoading` classes |

No new files. No new dependencies.
