# Sprint 10 Prompt — Domain Adapters & Observational Integration

## Context

Sprints 1–9 complete. The engine has all core algorithms, Bayesian optimization, causal GP surrogates,
multi-objective + constrained optimization, SQLite persistence, CLI, and a research advisor diagnostics
module. Sprint 10 focuses on usability: making the two existing domain adapters (marketing, ML training)
actually runnable with realistic simulators, and deeply integrating observational estimation into the
diagnostics and decision-making pipeline.

**Two tracks**, non-overlapping by module:

| Track | Branch | Modules touched |
|-------|--------|----------------|
| A — Domain Adapters | `sprint-10/domain-adapters` | `domain_adapters/`, `examples/`, tests |
| B — Observational Integration | `sprint-10/observational-integration` | `diagnostics/`, `predictor/`, `engine/loop.py`, tests |

**Branch base:** `main` (after Sprint 9 — PR #27)

**Merge order after human approval:**
1. Either order — the two tracks are non-overlapping
2. Rebase the later PR on main after the first merges; re-run `uv run pytest -m "not slow"`

---

## Invocation

Stand up one agent per track. Each agent must follow this exact workflow:

```
/tdd → implement → /polish → gh pr create → /gauntlet → report PR URL
```

Rules:
- Do not skip `/polish` before creating the PR
- Do not skip `/gauntlet` after creating the PR
- Each agent works in an isolated worktree (`isolation: "worktree"`)
- Do **NOT** merge — leave PRs open for human review
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)
- Read `CLAUDE.md` for project conventions and `thoughts/shared/plans/01-what-to-build-first.md`
  for original architecture vision

---

## Track A — `sprint-10/domain-adapters`

### Problem

The marketing and ML training adapters exist with rich search spaces and prior causal graphs,
but their `run_experiment()` methods raise `NotImplementedError`. Users can't actually run them.
There's no example showing how to use a domain adapter end-to-end with the CLI.

### Goal

Make both adapters runnable out of the box with realistic simulators that respect the causal
structure in their prior graphs. Add CLI-runnable examples and integration tests that prove
the full pipeline works (explore → screen → optimize → diagnose).

---

### Task 1 — Marketing Adapter Simulator (`domain_adapters/marketing.py`)

Replace the `NotImplementedError` in `run_experiment()` with a simulator that respects the
prior causal graph structure. The marketing adapter already defines 13 directed edges and
2 bidirected edges — the simulator must follow these causal mechanisms.

**Prior graph edges (already defined):**
```
email_frequency → email_opens → website_visits → conversions
social_spend_pct → impressions → brand_awareness → search_volume → conversions
search_bid_multiplier → paid_clicks → conversions
creative_variant → ctr → conversions
retargeting_enabled → repeat_visits → conversions
Bidirected: purchase_intent ↔ (social_spend_pct, conversions)
Bidirected: seasonality ↔ (brand_awareness, search_volume)
```

**Implementation requirements:**

1. **Structural equations** — Each intermediate variable must be computed from its parents
   in the DAG, plus noise. Use realistic functional forms:
   - `email_opens = f(email_frequency)` — saturating (log or sqrt), not linear
   - `impressions = f(social_spend_pct)` — roughly linear with diminishing returns
   - `brand_awareness = f(impressions, U_seasonality)` — includes confounder
   - `conversions = f(website_visits, paid_clicks, ctr, repeat_visits, U_purchase_intent)` — additive

2. **Confounders** — The two bidirected edges represent unobserved confounders. Implement them
   as latent variables sampled per experiment:
   - `U_purchase_intent` ~ N(0, 1) → affects both `social_spend_pct` response and `conversions`
   - `U_seasonality` ~ N(0, 1) → affects both `brand_awareness` and `search_volume`

3. **Return metrics** — Must return at minimum:
   - `conversions` (the objective, to maximize)
   - `total_spend` (descriptor for MAP-Elites)
   - `channel_diversity` (descriptor for MAP-Elites)

4. **Noise scale** — Add Gaussian noise (configurable via `__init__` param, default σ=0.1)
   to make optimization non-trivial but solvable in 30-50 experiments.

5. **Known optimum** — Document the approximate optimal configuration in a docstring so tests
   can verify the optimizer gets close. The optimum should NOT be at a boundary — make the
   landscape have an interior optimum.

6. **Seed support** — Accept an optional `seed` parameter in `__init__` for reproducibility.

**Reference:** Look at `tests/integration/test_marketing_scenario.py::MarketingSimRunner` for
an existing simulation pattern. The new simulator should be more realistic (saturating effects,
interactions) but follow the same structure.

---

### Task 2 — ML Training Adapter Simulator (`domain_adapters/ml_training.py`)

Replace the `NotImplementedError` in `run_experiment()` with a simulator that models
hyperparameter optimization for a neural network training run.

**Prior graph edges (already defined):**
```
learning_rate → gradient_scale → training_stability → val_loss
batch_size → gradient_noise, throughput
n_layers, n_heads, hidden_dim → model_capacity → val_loss
dropout, weight_decay → regularization → val_loss
optimizer → gradient_scale
activation → model_capacity
Bidirected: hardware ↔ (throughput, memory_usage)
Bidirected: data_distribution ↔ (model_capacity, val_loss)
```

**Implementation requirements:**

1. **Structural equations** — Model realistic ML training dynamics:
   - `gradient_scale = f(learning_rate, optimizer)` — optimizer choice affects effective LR
   - `training_stability = f(gradient_scale, batch_size)` — too-high LR or too-small batch → unstable
   - `model_capacity = f(n_layers, n_heads, hidden_dim, activation)` — roughly proportional to param count
   - `regularization = f(dropout, weight_decay)` — combined regularization strength
   - `val_loss = f(training_stability, model_capacity, regularization, U_data)` — U-shaped in capacity (underfitting vs overfitting)

2. **Realistic failure modes:**
   - Very high learning rate → training diverges → val_loss = large constant (e.g., 10.0)
   - Very large model + no regularization → overfitting → val_loss increases
   - Tiny model → underfitting → val_loss stays high

3. **Return metrics:**
   - `val_loss` (the objective, to minimize)
   - `memory_usage` (descriptor, proportional to model size)
   - `model_capacity` (descriptor)

4. **Categorical handling** — The optimizer and activation are categorical variables.
   Map them to numeric effects:
   - `adamw` → effective_lr_scale = 1.0, `sgd` → 0.8, `muon` → 1.1, `lion` → 0.95
   - `gelu` → capacity_bonus = 1.0, `swiglu` → 1.1, `relu` → 0.9

5. **Seed support** — Accept optional `seed` parameter.

---

### Task 3 — CLI Examples (`examples/`)

Create two new example files that demonstrate the full pipeline using the domain adapters:

#### 3a. `examples/marketing_optimization.py`

```python
"""Marketing campaign optimization using the causal optimizer CLI.

Usage:
    # Run optimization
    uv run causal-optimizer run \
        --adapter causal_optimizer.domain_adapters.marketing:MarketingAdapter \
        --budget 40 --db marketing.db --maximize --id demo

    # View results with research recommendations
    uv run causal-optimizer report --id demo --db marketing.db --maximize --next

    # Or run programmatically:
    uv run python examples/marketing_optimization.py
"""
```

The programmatic version should:
1. Create `MarketingAdapter` and extract its configuration
2. Run `ExperimentEngine` for 40 experiments
3. Print best result
4. Call `engine.diagnose()` and print the summary
5. Show top 3 recommendations

#### 3b. `examples/ml_hyperparameter_tuning.py`

Same pattern but for ML training. Run 50 experiments (ML needs more due to 9-dim space).
Show how the optimizer discovers that learning_rate and regularization matter most.

---

### Task 4 — Integration Tests

#### 4a. `tests/integration/test_marketing_adapter.py`

Update or replace `tests/integration/test_marketing_scenario.py` to use the new built-in
simulator instead of the test-only `MarketingSimRunner`. Tests should verify:

1. **Full pipeline runs** — 30 experiments complete without crashes
2. **Phase transitions** — Engine transitions from exploration to optimization
3. **Causal focus** — Engine concentrates on causal ancestors of `conversions`
4. **POMIS pruning** — `compute_pomis()` on the prior graph yields < 2^5 = 32 intervention sets
5. **Optimization works** — Best `conversions` after 30 experiments is better than median of first 10
6. **Diagnostics work** — `engine.diagnose()` returns a valid `DiagnosticReport` with recommendations
7. **Reproducibility** — Same seed produces same results

#### 4b. `tests/integration/test_ml_training_adapter.py`

Update or replace `tests/integration/test_ml_training_scenario.py`. Tests should verify:

1. **Full pipeline runs** — 30 experiments without crashes
2. **Phase transitions** — Exploration to optimization
3. **Screening identifies key variables** — learning_rate and model capacity variables should
   appear as high-signal in diagnostics
4. **Optimization works** — Best `val_loss` improves over baseline
5. **Categorical handling** — Optimizer and activation choices don't cause crashes
6. **Diagnostics work** — `engine.diagnose()` produces valid report

#### 4c. Test performance

All integration tests must be marked `@pytest.mark.slow` and complete in < 60 seconds each.

---

### Task 5 — Update `__init__.py` Exports

Update `causal_optimizer/domain_adapters/__init__.py` to export both adapters:

```python
from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.domain_adapters.marketing import MarketingAdapter
from causal_optimizer.domain_adapters.ml_training import MLTrainingAdapter

__all__ = ["DomainAdapter", "MarketingAdapter", "MLTrainingAdapter"]
```

---

### Files to create/modify

| File | Action |
|------|--------|
| `causal_optimizer/domain_adapters/marketing.py` | Modify — add simulator |
| `causal_optimizer/domain_adapters/ml_training.py` | Modify — add simulator |
| `causal_optimizer/domain_adapters/__init__.py` | Modify — export adapters |
| `examples/marketing_optimization.py` | Create |
| `examples/ml_hyperparameter_tuning.py` | Create |
| `tests/integration/test_marketing_adapter.py` | Create or modify existing |
| `tests/integration/test_ml_training_adapter.py` | Create or modify existing |

### Verification

```bash
# Unit + fast tests
uv run pytest -m "not slow" -v

# Integration tests (slow)
uv run pytest tests/integration/test_marketing_adapter.py tests/integration/test_ml_training_adapter.py -v

# CLI smoke test
uv run causal-optimizer run \
    --adapter causal_optimizer.domain_adapters.marketing:MarketingAdapter \
    --budget 20 --db /tmp/test_marketing.db --maximize --id smoke-test
uv run causal-optimizer report --id smoke-test --db /tmp/test_marketing.db --maximize --next

# Lint + type check
uv run ruff check . && uv run ruff format --check .
uv run mypy causal_optimizer/
```

---

## Track B — `sprint-10/observational-integration`

### Problem

The `ObservationalEstimator` (DoWhy wrapper) is fully built and tested, but it's isolated.
It can be used via `effect_method="observational"` on the engine, but:

1. The **diagnostics advisor** never recommends using observational estimation
2. The **off-policy predictor** doesn't use observational estimates to improve predictions
3. There's no way to know **when** observational estimation would help vs. waste time
4. The **sensitivity validator** doesn't cross-check experimental results against observational estimates

This track integrates observational estimation into the diagnostics and decision-support pipeline
so the system can recommend when to trust cheap observational estimates vs. run expensive experiments.

### Goal

Make the research advisor aware of observational estimation capabilities. When a causal graph
is available and sufficient experimental data exists, the advisor should:
- Estimate which variables have identifiable causal effects via backdoor/frontdoor adjustment
- Compare observational estimates with experimental results to flag discrepancies
- Recommend using observational mode when appropriate (saving experiment budget)
- Warn when observational estimates are unreliable (weak instruments, narrow CI issues)

---

### Task 1 — Observational Signal Analysis (`diagnostics/observational.py`)

Create a new analysis module that assesses the feasibility and value of observational estimation.

```python
def analyze_observational(
    experiment_log: ExperimentLog,
    search_space: SearchSpace,
    objective_name: str,
    minimize: bool,
    causal_graph: CausalGraph | None = None,
) -> ObservationalAnalysis:
```

**`ObservationalAnalysis` model** (add to `diagnostics/models.py`):

```python
class ObservationalVariableReport(BaseModel):
    variable_name: str
    identifiable: bool              # Can DoWhy identify causal effect?
    identification_method: str | None  # "backdoor", "frontdoor", "iv", or None
    obs_estimate: float | None       # E[Y | do(X = best_value)]
    obs_ci: tuple[float, float] | None  # Confidence interval
    exp_estimate: float | None       # Experimental effect estimate (from variable_signal)
    agreement: float | None          # 0-1, how well obs and exp agree
    ci_width_ratio: float | None     # obs_ci_width / objective_range (tighter = more useful)

class ObservationalAnalysis(BaseModel):
    n_identifiable: int              # How many variables have identifiable effects
    n_variables: int                 # Total search space variables
    variables: list[ObservationalVariableReport]
    obs_experimental_agreement: float | None  # Average agreement across identifiable vars
    recommendation: str              # Summary: "use observational", "insufficient data", etc.
```

**Implementation:**

1. **Skip if no causal graph** — Return empty analysis with `recommendation="no causal graph available"`

2. **For each search space variable that is a causal ancestor of the objective:**
   a. Check identifiability: Try `ObservationalEstimator` with backdoor, then frontdoor, then IV.
      Record which method (if any) identifies the effect.
   b. If identifiable and sufficient data (≥10 KEEP experiments):
      - Estimate `E[Y | do(X = x_best)]` where `x_best` is the best-found value
      - Record the CI width
   c. If variable_signal analysis has an effect estimate for this variable, compute agreement:
      - `agreement = 1 - |obs_effect - exp_effect| / max(|obs_effect|, |exp_effect|, 1e-10)`
      - Clamp to [0, 1]

3. **Graceful degradation:**
   - DoWhy not installed → all variables `identifiable=False`, recommendation includes install hint
   - < 10 experiments → skip estimation, only check identifiability
   - Estimation fails → log warning, mark variable as `identifiable=True` but `obs_estimate=None`

4. **Summary recommendation logic:**
   - If n_identifiable == 0: `"No variables have identifiable causal effects from observational data"`
   - If agreement > 0.8: `"Observational estimates agree with experiments — consider effect_method='observational' to save budget"`
   - If agreement < 0.5: `"Observational and experimental estimates disagree — possible unmeasured confounding"`
   - If < 10 experiments: `"Insufficient data for observational estimation — run more experiments first"`

---

### Task 2 — Wire into ResearchAdvisor (`diagnostics/advisor.py`)

**Modify `analyze_from_log()`** to run the new observational analysis as a fifth analysis
(only when `causal_graph` is provided):

```python
def analyze_from_log(self, ..., causal_graph=None, ...) -> DiagnosticReport:
    signal = analyze_variable_signal(...)
    convergence = analyze_convergence(...)
    coverage = analyze_coverage(...)
    robustness = analyze_robustness(...)

    # NEW: observational analysis (only with causal graph)
    observational = None
    if causal_graph is not None:
        observational = analyze_observational(
            experiment_log, search_space, self.objective_name,
            self.minimize, causal_graph
        )

    recommendations = _synthesize_recommendations(
        signal, convergence, coverage, robustness, search_space,
        observational=observational,  # NEW parameter
    )

    return DiagnosticReport(
        ...,
        observational=observational,  # NEW field
    )
```

**Add `observational` field to `DiagnosticReport`** (optional, `None` when no graph):

```python
class DiagnosticReport(BaseModel):
    # ... existing fields ...
    observational: ObservationalAnalysis | None = None
```

**Update `DiagnosticReport.summary()`** to include observational section when available:

```
Observational Estimation:
  Identifiable variables: 3/5
  Obs-experimental agreement: 82%
  Recommendation: Consider effect_method='observational' to save budget
```

---

### Task 3 — Observational Recommendations (`diagnostics/advisor.py`)

Add new recommendation patterns to `_synthesize_recommendations()`:

| Pattern | Type | Score | Condition |
|---------|------|-------|-----------|
| Strong obs-exp agreement | EXPLOIT | 0.75 | agreement > 0.8 AND n_identifiable >= 2 |
| Obs-exp disagreement | PIVOT | 0.7 | agreement < 0.5 AND n_identifiable >= 2 |
| Identifiable but untested | EXPLORE | 0.8 | Variable identifiable but not in variable_signal |
| Tight obs CI on ancestor | DROP | 0.6 | obs CI shows near-zero effect on ancestor variable |

**Recommendation examples:**

```python
# Strong agreement — save budget
Recommendation(
    rec_type=RecommendationType.EXPLOIT,
    confidence=ConfidenceLevel.MEDIUM,
    title="Use observational estimation to save experiment budget",
    description=(
        f"Observational estimates agree with experimental results "
        f"({obs.obs_experimental_agreement:.0%} agreement across "
        f"{obs.n_identifiable} identifiable variables). "
        f"Consider switching to effect_method='observational'."
    ),
    next_step="Re-run with --effect-method observational to use DoWhy estimates",
    expected_info_gain=0.75,
)

# Disagreement — possible confounding
Recommendation(
    rec_type=RecommendationType.PIVOT,
    confidence=ConfidenceLevel.MEDIUM,
    title="Observational and experimental estimates disagree",
    description=(
        f"Observational estimates disagree with experiments "
        f"({obs.obs_experimental_agreement:.0%} agreement). "
        f"This may indicate unmeasured confounding in the causal graph."
    ),
    next_step="Review the causal graph for missing confounders or add bidirected edges",
    expected_info_gain=0.7,
)
```

---

### Task 4 — Observational-Enhanced Off-Policy Prediction (`predictor/off_policy.py`)

Add an optional observational estimation path to the off-policy predictor. When a causal graph
is available and the effect is identifiable, use observational estimates to tighten prediction
confidence intervals.

**Modify `OffPolicyPredictor.__init__()`:**

```python
def __init__(
    self,
    ...,
    causal_graph: CausalGraph | None = None,  # NEW
    objective_name: str = "objective",          # NEW
):
```

**Modify `predict()`:**

After the RF prediction, if `causal_graph` is available and DoWhy is installed, attempt an
observational estimate for the candidate parameters. If both RF and observational estimates
agree (within 1 std), tighten the confidence interval. If they disagree, widen it.

```python
def predict(self, parameters: dict[str, Any]) -> Prediction | None:
    rf_prediction = self._rf_predict(parameters)
    if rf_prediction is None:
        return None

    if self._causal_graph is not None and self._experiment_log is not None:
        obs_prediction = self._observational_predict(parameters)
        if obs_prediction is not None:
            rf_prediction = self._combine_predictions(rf_prediction, obs_prediction)

    return rf_prediction
```

**`_observational_predict()` method:**

1. For each variable in `parameters` that differs from the dataset mean by > 1 std:
   - Try `ObservationalEstimator.estimate_intervention()` with that variable's value
2. If multiple variables changed, use the variable with the tightest CI
3. Return `Prediction(expected_value=obs_estimate, uncertainty=ci_width/2, ...)`
4. Return `None` if DoWhy not installed, no graph, or estimation fails

**`_combine_predictions()` method:**

1. If RF and obs agree (|rf.expected - obs.expected| < rf.uncertainty):
   - Use RF expected value (it's trained on actual experimental data)
   - Tighten uncertainty: `combined_uncertainty = min(rf.uncertainty, obs.uncertainty)`
2. If they disagree:
   - Use RF expected value
   - Widen uncertainty: `combined_uncertainty = max(rf.uncertainty, obs.uncertainty) * 1.5`
3. Return updated `Prediction`

**Important:** This must be fully optional. If DoWhy is not installed or no causal graph
is provided, the predictor must work exactly as before (pure RF). Use try/except and
graceful degradation throughout.

---

### Task 5 — Wire Causal Graph into Predictor (`engine/loop.py`)

The engine already creates an `OffPolicyPredictor` in `__init__`. Pass the causal graph
and objective name so the predictor can use observational estimates:

```python
# In ExperimentEngine.__init__():
self._predictor = OffPolicyPredictor(
    ...,
    causal_graph=causal_graph,       # NEW
    objective_name=objective_name,    # NEW
)
```

Also pass the experiment log reference so the predictor can access historical data for
observational estimation. The predictor already receives the log via `fit()` — ensure
it stores a reference for use in `_observational_predict()`.

---

### Task 6 — Tests

#### 6a. `tests/unit/test_observational_diagnostics.py`

Test the new `analyze_observational()` function:

1. **No causal graph** → empty analysis, appropriate recommendation
2. **With graph, < 10 experiments** → identifiability check only, no estimates
3. **With graph, sufficient data, backdoor identifiable** → estimates computed, CI reported
4. **Non-identifiable variable** → marked as `identifiable=False`
5. **Agreement computation** → obs and exp estimates agree/disagree correctly
6. **DoWhy not installed** → graceful degradation (mock the import failure)
7. **Integration with advisor** → `analyze_from_log()` includes observational analysis
8. **Recommendations generated** → agreement/disagreement patterns produce correct rec types
9. **Summary output** → `report.summary()` includes observational section when available

#### 6b. `tests/unit/test_observational_predictor.py`

Test the observational-enhanced off-policy predictor:

1. **No causal graph** → predictor works exactly as before (pure RF)
2. **With graph, estimates agree** → tighter CI
3. **With graph, estimates disagree** → wider CI
4. **DoWhy not installed** → falls back to RF-only
5. **Estimation failure** → falls back to RF-only
6. **should_run_experiment() behavior** → tighter CI may skip more experiments

#### 6c. Update `tests/unit/test_diagnostics.py`

Add tests for the new `observational` field on `DiagnosticReport`:
1. `report.observational` is `None` when no graph provided
2. `report.observational` is populated when graph provided
3. `report.summary()` includes observational section

---

### Files to create/modify

| File | Action |
|------|--------|
| `causal_optimizer/diagnostics/observational.py` | Create — observational signal analysis |
| `causal_optimizer/diagnostics/models.py` | Modify — add ObservationalAnalysis models |
| `causal_optimizer/diagnostics/advisor.py` | Modify — wire in observational analysis + recommendations |
| `causal_optimizer/diagnostics/__init__.py` | Modify — export new models |
| `causal_optimizer/predictor/off_policy.py` | Modify — add observational prediction path |
| `causal_optimizer/engine/loop.py` | Modify — pass causal graph to predictor |
| `tests/unit/test_observational_diagnostics.py` | Create |
| `tests/unit/test_observational_predictor.py` | Create |
| `tests/unit/test_diagnostics.py` | Modify — add observational field tests |

### Verification

```bash
# All fast tests (including new ones)
uv run pytest -m "not slow" -v

# Specifically the new tests
uv run pytest tests/unit/test_observational_diagnostics.py tests/unit/test_observational_predictor.py -v

# Lint + type check
uv run ruff check . && uv run ruff format --check .
uv run mypy causal_optimizer/

# Manual verification — run with a causal graph and check diagnostics
uv run python -c "
from causal_optimizer import ExperimentEngine, SearchSpace, Variable, VariableType, CausalGraph
from unittest.mock import MagicMock

ss = SearchSpace(variables=[
    Variable(name='x', variable_type=VariableType.CONTINUOUS, lower=0, upper=10),
    Variable(name='z', variable_type=VariableType.CONTINUOUS, lower=0, upper=10),
])
graph = CausalGraph(edges=[('x', 'z'), ('z', 'objective')])
runner = MagicMock()
runner.run.return_value = {'objective': 1.0}
engine = ExperimentEngine(search_space=ss, runner=runner, causal_graph=graph)
engine.run_loop(15)
report = engine.diagnose()
print(report.summary())
if report.observational:
    print(f'Identifiable: {report.observational.n_identifiable}/{report.observational.n_variables}')
    print(f'Recommendation: {report.observational.recommendation}')
"
```

---

## Cross-Track Dependencies

None. The two tracks are fully independent:

- **Track A** modifies `domain_adapters/`, `examples/`, and integration tests
- **Track B** modifies `diagnostics/`, `predictor/`, and unit tests
- Neither touches the other's files

The only shared file is `engine/loop.py`, but:
- Track A does NOT modify `engine/loop.py`
- Track B adds one parameter pass-through in `__init__` (causal_graph to predictor)

---

## Definition of Done

### Track A
- [ ] `MarketingAdapter.run_experiment()` runs without error and returns realistic metrics
- [ ] `MLTrainingAdapter.run_experiment()` runs without error and returns realistic metrics
- [ ] Both adapters work with the CLI (`causal-optimizer run --adapter ...`)
- [ ] Integration tests verify full pipeline (explore → screen → optimize → diagnose)
- [ ] Examples run successfully and print meaningful output
- [ ] All fast tests pass; integration tests pass (marked slow)

### Track B
- [ ] `analyze_observational()` correctly identifies variables with identifiable causal effects
- [ ] `DiagnosticReport` includes observational analysis when causal graph is available
- [ ] Advisor generates observational recommendations (agreement/disagreement/identifiable-untested)
- [ ] Off-policy predictor uses observational estimates when available (tighter/wider CI)
- [ ] All changes are fully optional — no regressions when DoWhy is not installed
- [ ] All fast tests pass; new unit tests cover all observational paths
