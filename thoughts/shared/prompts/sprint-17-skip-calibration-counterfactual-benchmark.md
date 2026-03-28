# Sprint 17 — Skip Calibration + Semi-Synthetic Counterfactual Benchmark

## Context

Sprint 16 (PRs #70, #71, #73) delivered:

1. **Causal fallback differentiation** — `causal` and `surrogate_only` now produce
   different suggestions under the RF-surrogate path
2. **Second real benchmark** — ERCOT COAST + Houston IAH weather
3. **Multi-benchmark suite** — cross-benchmark acceptance rules with coverage validation

Sprint 16 suite verdict: **CONDITIONAL**. Causal is differentiated (0.1–0.4% gap
from surrogate_only) but does not beat random. The star-graph topology provides no
selective advantage — all 7 variables are direct parents of the objective.

This sprint completes the remaining two agents from
`thoughts/shared/plans/08-optimizer-improvement-briefs.md`:

- **Agent 4**: Anytime + skip calibration (make speed claims trustworthy)
- **Agent 5**: Semi-synthetic counterfactual benchmark (test causal validity
  on a problem where causal structure actually matters)

## Design Principle

From Plan 08: improve the optimizer in a domain-general way. Every change must be
judged across the existing multi-benchmark suite. Do not optimize for one dataset.

## Steps

Steps 1 and 2 are **independent** and should be run in parallel (separate
worktrees). Step 3 depends on both being merged first.

```
Phase A (parallel):
  Step 1: Skip Calibration Diagnostics      →  PR  →  human review  →  merge
  Step 2: Semi-Synthetic Counterfactual      →  PR  →  human review  →  merge

Phase B (after Phase A merges):
  Step 3: Suite Re-Run + Final Report        →  PR  →  human review  →  merge
```

---

## Step 1: Anytime + Skip Calibration Diagnostics

### GitHub Issue

Create issue titled:
**"Add anytime metrics and skip calibration diagnostics"**

Body:

```
## Problem

Engine-based strategies (surrogate_only, causal) run faster than random because
the off-policy predictor skips experiments it predicts will be poor. But we
cannot tell whether those skips are correct — a high skip rate with bad
calibration means the optimizer is just being reckless, not smart.

## What to do

Add instrumentation that tracks skip decisions and anytime learning curves so
we can tell whether speed gains are trustworthy.

## Acceptance

1. Skip diagnostics are emitted per-run: candidates considered, evaluated,
   skipped, skip confidence distribution
2. Anytime metrics at budgets 5/10/20/40/80 show learning curves
3. Optional audit mode occasionally force-evaluates a skipped candidate to
   estimate false-skip rate
4. Diagnostics integrate with the benchmark suite
5. Existing tests remain green
```

### Agent Prompt

```
Stand up one agent in an isolated worktree. Follow this exact workflow:

  /tdd → implement → /polish → gh pr create → /gauntlet → report PR URL

Read these before starting:
- `CLAUDE.md`
- `thoughts/shared/plans/08-optimizer-improvement-briefs.md` (Agent 4 brief)
- `causal_optimizer/predictor/off_policy.py` (OffPolicyPredictor, should_run_experiment)
- `causal_optimizer/engine/loop.py` (ExperimentEngine.step — where skips happen)
- `causal_optimizer/benchmarks/predictive_energy.py` (PredictiveBenchmarkResult)
- `scripts/energy_predictive_benchmark.py` (run_strategy — where engine is called)
- `scripts/energy_benchmark_suite.py` (suite runner)
- `thoughts/shared/docs/sprint-16-suite-report.md` (current evidence)

Branch name: `sprint-17/skip-calibration`

Feature:

### 1. Skip Diagnostics Dataclass

Create `causal_optimizer/diagnostics/skip_calibration.py` with:

```python
@dataclass
class SkipDiagnostics:
    """Per-run skip decision diagnostics."""
    candidates_considered: int     # total suggest() calls
    candidates_evaluated: int      # actually ran
    candidates_skipped: int        # predicted poor → skipped
    skip_ratio: float              # skipped / considered
    skip_confidences: list[float]  # confidence scores of skip decisions
    audit_results: list[AuditResult] | None  # if audit mode enabled

@dataclass
class AuditResult:
    """Result of force-evaluating a skipped candidate."""
    parameters: dict[str, Any]
    predicted_outcome: float
    actual_outcome: float
    was_correct_skip: bool  # actual was indeed worse than best
```

### 2. Engine Instrumentation

Modify `ExperimentEngine` to track skip decisions:

- Add `_skip_log: list[dict]` attribute that records each skip decision
  with parameters, predicted outcome, and confidence
- Add `skip_diagnostics() -> SkipDiagnostics` property that summarizes
  the log
- Add optional `audit_skip_rate: float = 0.0` parameter to `__init__`.
  When > 0, randomly force-evaluate that fraction of would-be-skipped
  candidates and record the result as `AuditResult`

Do NOT change the engine's decision logic — only add observation/recording.

### 3. Anytime Metrics

Add `causal_optimizer/diagnostics/anytime.py` with:

```python
@dataclass
class AnytimeMetrics:
    """Learning curve at checkpoints during optimization."""
    checkpoints: list[int]          # budget values where metrics were sampled
    best_objective_at: list[float]  # best objective at each checkpoint
    n_evaluated_at: list[int]       # experiments evaluated at each checkpoint
    n_skipped_at: list[int]         # experiments skipped at each checkpoint
```

Add a `anytime_metrics(checkpoints: list[int]) -> AnytimeMetrics` method
to `ExperimentEngine` that reads from the experiment log.

### 4. Benchmark Integration

Extend `PredictiveBenchmarkResult` with optional fields:

```python
skip_diagnostics: SkipDiagnostics | None = None
anytime_metrics: AnytimeMetrics | None = None
```

Update `run_strategy` in `scripts/energy_predictive_benchmark.py` to
populate these fields from the engine after each run.

Update `scripts/energy_benchmark_suite.py` to:
- Include skip diagnostics in per-benchmark summary
- Add a "Skip Calibration" section to the suite report showing:
  - Skip ratio by strategy (random should be 0, engine strategies > 0)
  - Mean skip confidence
  - Audit false-skip rate (if audit mode was enabled)

### 5. Tests

Add `tests/unit/test_skip_calibration.py`:

1. **test_skip_diagnostics_recorded**: Run engine with off-policy predictor
   and verify skip diagnostics are populated
2. **test_audit_mode_force_evaluates**: Set audit_skip_rate=1.0, verify
   all would-be-skipped candidates get audit results
3. **test_anytime_metrics_checkpoints**: Run engine for 20 steps, verify
   anytime metrics at checkpoints [5, 10, 20]
4. **test_skip_diagnostics_zero_for_random**: Random strategy should have
   zero skips

### Design Notes

- Skip diagnostics are per-run, not per-benchmark. The suite aggregates.
- Audit mode is optional and off by default (audit_skip_rate=0.0).
  It's expensive because it runs experiments the predictor would skip.
- Anytime checkpoints default to [5, 10, 20, 40, 80] but are configurable.
- Do not change the off-policy predictor's decision logic — only observe.

Conventions: from __future__ import annotations, type hints on all public
methods, ruff line length 100, mypy strict.

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Do NOT merge — leave PR open for human review
- Report: PR URL, polish summary, gauntlet summary
```

---

## Step 2: Semi-Synthetic Counterfactual Benchmark

### GitHub Issue

Create issue titled:
**"Semi-synthetic demand-response counterfactual benchmark"**

Body:

```
## Problem

The real predictive-energy benchmarks test model selection, not causal
reasoning. The causal graph is a star (all variables → objective), so
causal guidance provides no selective advantage. We need a benchmark
where causal structure matters — where knowing the graph gives a
real edge.

## What to do

Build a semi-synthetic demand-response benchmark using real ERCOT
covariates with known treatment effects and counterfactual ground truth.

## Acceptance

1. Counterfactual ground truth is explicit and verifiable
2. The benchmark is not tied to one exact real dataset row order
3. `causal` has a plausible path to outperform `random` and
   `surrogate_only` when the graph has non-trivial structure
4. Smoke test and reproducibility test pass
5. Integrates with the existing benchmark suite runner
```

### Agent Prompt

```
Stand up one agent in an isolated worktree. Follow this exact workflow:

  /tdd → implement → /polish → gh pr create → /gauntlet → report PR URL

Read these before starting:
- `CLAUDE.md`
- `thoughts/shared/plans/08-optimizer-improvement-briefs.md` (Agent 5 brief,
  and the "Recommended Semi-Synthetic Counterfactual Benchmark" section)
- `causal_optimizer/benchmarks/predictive_energy.py` (existing benchmark pattern)
- `causal_optimizer/benchmarks/runner.py` (BenchmarkRunner protocol)
- `causal_optimizer/domain_adapters/energy_load.py` (energy adapter pattern)
- `causal_optimizer/types.py` (CausalGraph, SearchSpace)
- `scripts/energy_predictive_benchmark.py` (runner pattern)
- `scripts/energy_benchmark_suite.py` (suite integration)
- `thoughts/shared/docs/sprint-16-suite-report.md` (why this benchmark is needed)

Branch name: `sprint-17/counterfactual-benchmark`

Feature:

### 1. Data Generation

Create `causal_optimizer/benchmarks/counterfactual_energy.py` with:

**DemandResponseScenario** — generates semi-synthetic data from real
ERCOT covariates:

1. **Covariates** (from real ERCOT Parquet):
   - temperature, humidity, hour_of_day, day_of_week, is_holiday
   - lagged load features (e.g., load_lag_1h, load_lag_24h)

2. **Treatment variable**: `demand_response_event` (binary)
   - Assignment: propensity depends on temperature and hour_of_day
     (more likely on hot afternoons)
   - Known propensity function for IPW-style evaluation

3. **Potential outcomes** with known structural rules:
   - Y(0) = base load (real target_load from ERCOT data)
   - Y(1) = base load - treatment_effect(temperature, hour_of_day)
   - treatment_effect: large on hot afternoons (temp > 90F, hour 14-18),
     near-zero on mild nights (temp < 70F or hour 0-6),
     moderate otherwise
   - Add a cost penalty for treatment: cost = fixed_cost_per_event

4. **Causal graph** (non-trivial structure — NOT a star):
   ```
   temperature → demand_response_event → load_reduction
   hour_of_day → demand_response_event → load_reduction
   temperature → load_reduction (direct effect too)
   humidity → base_load (but NOT → load_reduction — a non-parent)
   day_of_week → base_load (but NOT → load_reduction)
   ```
   This graph has non-parents (humidity, day_of_week don't affect the
   treatment effect), so causal guidance should help by focusing on
   the actual parents of the outcome.

5. **Oracle policy**: always treat when treatment_effect > cost.
   Known because we have counterfactual ground truth.

### 2. Benchmark Harness

Create `CounterfactualBenchmarkResult`:

```python
@dataclass
class CounterfactualBenchmarkResult:
    strategy: str
    budget: int
    seed: int
    policy_value: float          # avg outcome under learned policy
    oracle_value: float          # avg outcome under oracle policy
    regret: float                # oracle_value - policy_value
    treatment_effect_mae: float  # MAE of estimated vs true effects
    runtime_seconds: float
```

Create a runner script `scripts/counterfactual_benchmark.py` with:
- Same CLI pattern as `energy_predictive_benchmark.py`
- `--data-path` points to a real ERCOT Parquet (covariates source)
- `--budgets`, `--seeds`, `--strategies` flags
- The benchmark:
  1. Generates the semi-synthetic dataset from real covariates
  2. Splits into train/val/test by time
  3. For each strategy+budget+seed:
     - Runs the optimizer to find the best policy parameters
     - The search space includes treatment decision variables
       (threshold temperature, threshold hour, etc.)
     - Evaluates the learned policy on the test set using
       counterfactual ground truth
  4. Computes regret vs oracle

### 3. Search Space Design

The optimizer searches over policy parameters, NOT model hyperparameters:

```python
SearchSpace(variables=[
    Variable("treat_temp_threshold", VariableType.CONTINUOUS, lower=60.0, upper=100.0),
    Variable("treat_hour_start", VariableType.INTEGER, lower=0, upper=23),
    Variable("treat_hour_end", VariableType.INTEGER, lower=0, upper=23),
    Variable("treat_humidity_threshold", VariableType.CONTINUOUS, lower=0.0, upper=100.0),
    Variable("treat_day_filter", VariableType.CATEGORICAL,
             categories=["all", "weekday", "weekend"]),
])
```

The causal graph tells the optimizer which variables actually affect
the treatment outcome. Variables like `treat_humidity_threshold` and
`treat_day_filter` are in the search space but are NOT parents of the
outcome in the causal graph — they're noise dimensions that causal
guidance should learn to deprioritize.

This is the key design: the graph has genuine non-parents, so
`focus_variables` will be a strict subset, and the targeted
intervention candidates will focus on the right variables.

### 4. Suite Integration

Update `scripts/energy_benchmark_suite.py` to optionally accept
counterfactual benchmark results alongside predictive energy results,
OR create a parallel `scripts/counterfactual_benchmark_suite.py` that
follows the same pattern.

The simpler path: make the counterfactual benchmark produce its own
standalone report, and reference it from the suite. Do not force it
into the energy suite's acceptance rules (different metrics).

### 5. Tests

Add `tests/unit/test_counterfactual_benchmark.py`:

1. **test_scenario_generates_valid_data**: Verify generated dataset
   has expected columns, no NaN in outcomes, treatment assignment
   correlated with temperature/hour
2. **test_oracle_policy_is_optimal**: Verify oracle achieves the best
   possible policy value on the test set
3. **test_treatment_effect_varies_by_context**: Verify hot-afternoon
   effect > mild-night effect (the structural rule works)
4. **test_causal_graph_has_non_parents**: Verify the graph's
   `parents("load_reduction")` excludes humidity and day_of_week
5. **test_benchmark_smoke**: Run with budget=3, seed=0, verify
   result has expected fields and finite regret
6. **test_reproducibility**: Same seed produces same results

### Design Notes

- Use real ERCOT NORTH_C Parquet as the covariate source (already available)
- The treatment effect function must be deterministic given covariates
  (no additional noise) so counterfactual truth is exact
- The propensity function should create mild confounding (treatment more
  likely when it would be effective) so naive methods are biased
- Random seed controls treatment assignment randomness and optimizer
  randomness, not the covariate data
- Do not modify the existing energy benchmark harness

Conventions: from __future__ import annotations, type hints on all public
methods, ruff line length 100, mypy strict.

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Do NOT merge — leave PR open for human review
- Report: PR URL, polish summary, gauntlet summary
```

---

## Step 3: Suite Re-Run + Final Report

### Prerequisites

Both Step 1 and Step 2 must be merged to main before starting Step 3.

### GitHub Issue

Create issue titled:
**"Sprint 17 suite re-run with skip diagnostics and counterfactual benchmark"**

Body:

```
## Goal

Re-run the full benchmark suite with skip calibration enabled and
produce a combined report that includes:
1. Real benchmark results with skip diagnostics
2. Counterfactual benchmark results showing whether causal guidance
   helps when the graph has non-trivial structure
3. Updated promotion assessment

## Acceptance

1. Suite report includes skip diagnostics section
2. Counterfactual benchmark shows causal vs random vs surrogate_only
3. Combined findings update the promotion decision
4. All fast tests pass
```

### Agent Prompt

```
Stand up one agent in an isolated worktree. Rebase on main first (must
include Steps 1 and 2). Follow this exact workflow:

  Run benchmarks → write report → /polish → gh pr create → /gauntlet → report PR URL

Read these before starting:
- `CLAUDE.md`
- `thoughts/shared/plans/08-optimizer-improvement-briefs.md`
- `scripts/energy_benchmark_suite.py` (suite runner)
- `scripts/counterfactual_benchmark.py` (new counterfactual runner)
- `causal_optimizer/diagnostics/skip_calibration.py` (skip diagnostics)
- `causal_optimizer/diagnostics/anytime.py` (anytime metrics)
- `thoughts/shared/docs/sprint-16-suite-report.md` (previous report)

Branch name: `sprint-17/suite-rerun-final-report`

### Phase A: Real Benchmark Suite Re-Run

Run the existing suite with skip diagnostics enabled:

```bash
uv run python scripts/energy_benchmark_suite.py \
  --datasets /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet,/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet \
  --dataset-ids ercot_north_c,ercot_coast \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output-dir /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/suite_sprint17
```

### Phase B: Counterfactual Benchmark Run

Run the new counterfactual benchmark:

```bash
uv run python scripts/counterfactual_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/counterfactual_sprint17_results.json
```

### Phase C: Combined Report

Write `thoughts/shared/docs/sprint-17-combined-report.md` covering:

1. **Real benchmark results**: Same metrics as Sprint 16, now with skip
   diagnostics per strategy
2. **Skip calibration analysis**:
   - Skip ratio by strategy (random=0, engine > 0)
   - Are skip decisions trustworthy? (audit false-skip rate if available)
   - Does higher skip ratio correlate with better or worse outcomes?
3. **Counterfactual benchmark results**:
   - Regret by strategy and budget
   - Does causal beat random/surrogate_only on a non-trivial graph?
   - Treatment effect estimation quality
4. **Combined promotion assessment**:
   - Does causal guidance help when the graph has non-trivial structure?
   - Are skip decisions trustworthy?
   - Updated recommendation: PROMOTE / INVESTIGATE / REJECT
5. **What this means for the next sprint**

### Local-only outputs (do NOT commit):
- Real suite: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/suite_sprint17/`
- Counterfactual: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/counterfactual_sprint17_results.json`

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Do NOT merge — leave PR open for human review
- Report: PR URL, key findings, promotion assessment
```

---

## Execution Sequence

```
Phase A — run in parallel:
  Worktree 1: Step 1 prompt → agent delivers PR → human reviews → merge
  Worktree 2: Step 2 prompt → agent delivers PR → human reviews → merge

Phase B — after both Phase A PRs merge:
  Worktree 3: Step 3 prompt → agent delivers PR → human reviews → merge
```

Post-merge verification after all three:

```bash
# Unit tests still green
uv run pytest -m "not slow" -v

# Counterfactual benchmark smoke
uv run python scripts/counterfactual_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet \
  --budgets 3 \
  --seeds 0 \
  --strategies random,surrogate_only,causal \
  --output /tmp/counterfactual_smoke.json
```

## Acceptance Checklist

Sprint 17 is complete when:

1. Skip diagnostics are instrumented and recorded per-run
2. Anytime learning curves are available at configurable checkpoints
3. Audit mode can force-evaluate skipped candidates for calibration
4. Semi-synthetic counterfactual benchmark exists with known ground truth
5. The counterfactual graph has genuine non-parents (causal guidance
   should help)
6. Counterfactual benchmark results show whether causal beats random
   on a non-trivial graph
7. Combined report answers: are skip decisions trustworthy? does causal
   help when the graph matters?
8. All fast tests pass (`uv run pytest -m "not slow"`)
9. Plan 08 is fully complete (all 5 agents delivered)
