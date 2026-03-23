# Sprint 7 Prompt

## Context

Sprint 6 complete. The engine has a working Ax/BoTorch optimizer with a soft POMIS prior
(post-hoc filter). Sprint 7 runs three independent tracks in parallel:

- **Track A (Research):** Causal GP surrogates per mechanism + custom causal acquisition kernel
- **Track B (Engineering):** Multi-objective optimization (Pareto front) + constrained optimization
- **Track C (Engineering):** SQLite persistence + experiment resumption + CLI

All three branches are independent and can merge in any order. Rebase each on main after the
previous merges.

**Branch base:** `main` (after Sprint 6 — PRs #17, #18)

**Merge order after human approval:**
1. Any order — all three are non-overlapping
2. Rebase later PRs on main after earlier ones merge; re-run `uv run pytest -m "not slow"`

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

---

## Track A — `sprint-7/causal-gp`

### Research context

The current Ax/BoTorch optimizer treats all variables as a single black-box GP.
The CBO (Causal Bayesian Optimization) paper (Aglietti et al., AISTATS 2020, NeurIPS 2021)
proposes fitting **separate GPs per causal mechanism** — one GP for each directed edge
in the causal graph — then composing them to predict the interventional distribution
P(Y | do(X=x)). This differs from a single GP over all variables because it:

1. Respects the factorization of the SCM (each mechanism is modeled independently)
2. Enables transfer when a mechanism is shared across experiments
3. Gives a principled acquisition function that reasons about do(X) rather than P(Y|X)

**What "mechanism" means in our context**: a mechanism is the conditional P(X_i | Pa(X_i))
for each node X_i in the graph. We model each mechanism as a GP with its own kernel and
hyperparameters. The interventional prediction for a target set S is:

```
E[Y | do(S=s)] = f_Y(s, E[non-S ancestors | graph])
```

For our purposes, a practical approximation: fit one GP per *node* (not per edge), conditioned
on the node's observed parents. Chain predictions through the graph using the posterior means.

### Task 1 — `optimizer/causal_gp.py`

Implement `CausalGPSurrogate`:

```python
class CausalGPSurrogate:
    """Separate GP per causal mechanism, composed for interventional prediction.

    This is an EXPERIMENTAL implementation of the CBO architecture.
    It requires botorch and gpytorch. Falls back gracefully if unavailable.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        causal_graph: CausalGraph,
        objective_name: str,
        minimize: bool = True,
        seed: int | None = None,
    ) -> None: ...

    def fit(self, experiment_log: ExperimentLog) -> None:
        """Fit one GP per node using observed parent values."""
        ...

    def predict_interventional(
        self,
        intervention: dict[str, float],
        n_samples: int = 100,
    ) -> tuple[float, float]:
        """Return (mean, std) of E[Y | do(intervention)] via graph composition."""
        ...

    def suggest(self, n_candidates: int = 100) -> dict[str, Any]:
        """Suggest next intervention using Expected Improvement over do(X)."""
        ...
```

Implementation rules:
- Use `botorch` and `gpytorch` for GP fitting (already available via `ax-platform` dep)
- One `SingleTaskGP` per node, trained on `(parent_values, node_value)` pairs
- Interventional prediction: for intervened variables, use the fixed intervention value;
  for others, propagate posterior means through the topological order
- Acquisition: Expected Improvement computed on the interventional prediction, not the
  observational prediction
- Graceful degradation: if `botorch` is not importable, raise `ImportError` with install hint
- Mark the class as `# experimental` in the docstring — it ships but is not the default strategy

### Task 2 — Wire into `optimizer/suggest.py`

Add `strategy="causal_gp"` as a new branch in `_suggest_optimization()`:

```python
if strategy == "causal_gp":
    return _suggest_causal_gp(
        search_space, experiment_log, causal_graph, minimize, objective_name
    )
```

- `strategy` parameter threads through from `ExperimentEngine.__init__()` (add it there too,
  default `"bayesian"` to preserve backward compatibility)
- Only activates in optimization phase and requires a causal graph; falls back to `"bayesian"`
  if no graph is provided
- Falls back to RF surrogate if neither `ax` nor `botorch` is available

### Task 3 — Tests

Create `tests/unit/test_causal_gp.py`:
1. `test_causal_gp_requires_botorch` — mock botorch as unimportable; assert `ImportError`
2. `test_causal_gp_fit_runs_without_crash` — fit on 10 ToyGraph observations; assert no exception
3. `test_causal_gp_predict_interventional_shape` — returns `(float, float)` tuple
4. `test_causal_gp_suggest_returns_valid_params` — suggest returns params within search space
5. `test_causal_gp_respects_graph_topology` — with a chain A→B→Y, predict do(A=hi) > do(A=lo)
   when the SCM is monotone increasing; assert mean prediction reflects this ordering
6. `test_causal_gp_in_engine_with_strategy` — run `ExperimentEngine(strategy="causal_gp")`
   for 15 steps on ToyGraph; assert no crashes

Create `tests/integration/test_causal_gp_vs_ax.py` (`@pytest.mark.slow`):
1. `test_causal_gp_matches_ax_on_toygraph` — run both strategies for 30 steps on ToyGraph
   with `n_seeds=3`; assert CausalGP mean final objective ≤ Ax mean × 1.15 (comparable, with
   15% tolerance since CausalGP has extra inductive bias)

### Acceptance criteria
- `uv run pytest tests/unit/test_causal_gp.py -v` all pass
- `uv run pytest -m "not slow"` still passes (causal_gp is an opt-in strategy)
- `pyproject.toml` `[project.optional-dependencies] bayesian` already includes botorch
  via ax-platform; verify and add `gpytorch>=1.11` if missing
- `uv run mypy causal_optimizer/` passes

---

## Track B — `sprint-7/multi-objective-constrained`

### Task 1 — Multi-objective support (Pareto front)

**Background**: The engine currently optimizes a single scalar objective. Many real experiments
have multiple objectives (e.g., maximize revenue AND minimize cost). Multi-objective optimization
returns a Pareto front — the set of non-dominated solutions.

**Types changes** (`types.py`):

Add `ObjectiveSpec`:
```python
class ObjectiveSpec(BaseModel):
    name: str
    minimize: bool = True
    weight: float = 1.0  # for scalarization fallback
```

Add `ParetoResult`:
```python
class ParetoResult(BaseModel):
    front: list[ExperimentResult]  # non-dominated results

    def dominated_by(self, other: ExperimentResult, objectives: list[ObjectiveSpec]) -> bool:
        """Return True if other dominates self on all objectives."""
        ...
```

Add `ExperimentLog.pareto_front(objectives: list[ObjectiveSpec]) -> list[ExperimentResult]`:
- Returns the non-dominated subset of KEEP results
- A result A dominates B if A is at least as good on all objectives and strictly better on one

**Engine changes** (`engine/loop.py`):

Add `objectives: list[ObjectiveSpec] | None` parameter to `ExperimentEngine.__init__()`:
- If `None` (default), existing single-objective behavior is unchanged (backward compatible)
- If set, `_evaluate_status()` uses Pareto dominance instead of scalar comparison:
  a new result KEEP if it is not dominated by any existing KEEP result

Add `engine.pareto_front` property returning `list[ExperimentResult]`.

**Suggest changes** (`optimizer/suggest.py`):

When `objectives` has >1 entry, scalarize for the surrogate using weighted sum
(weights from `ObjectiveSpec.weight`). This keeps suggestion logic simple while
enabling multi-objective evaluation.

### Task 2 — Constrained optimization

Add `Constraint` to `types.py`:
```python
class Constraint(BaseModel):
    metric_name: str       # must appear in experiment metrics
    upper_bound: float | None = None
    lower_bound: float | None = None
```

Add `constraints: list[Constraint] | None` parameter to `ExperimentEngine.__init__()`:
- KEEP only if all constraints are satisfied AND the result is an improvement
  (or non-dominated, if multi-objective)
- Violated results get status `DISCARD` with metadata `{"constraint_violated": True}`

### Task 3 — Benchmark: bi-objective ToyGraph

Add `ToyGraphBiObjective` to `benchmarks/toy_graph.py`:
- Same causal structure as ToyGraph
- Returns two metrics: `objective` (original) and `cost` (e.g., sum of absolute parameter values)
- `minimize = {"objective": True, "cost": True}`

### Task 4 — Tests

Create `tests/unit/test_multi_objective.py`:
1. `test_pareto_front_single_objective_same_as_best` — with one objective, pareto_front returns
   the single best KEEP result
2. `test_pareto_front_two_objectives_nondominated` — with 3 results [(1,3),(2,2),(3,1)],
   all 3 are non-dominated; assert len(pareto_front) == 3
3. `test_pareto_front_dominated_excluded` — result (2,3) is dominated by (1,2); assert excluded
4. `test_engine_multi_objective_keeps_nondominated` — run 20 steps on ToyGraphBiObjective;
   assert engine.pareto_front is non-empty and all results are non-dominated
5. `test_engine_single_objective_backward_compat` — engine with no `objectives` param behaves
   identically to current behavior

Create `tests/unit/test_constraints.py`:
1. `test_constraint_upper_bound_discards` — result with metric > upper_bound gets DISCARD
2. `test_constraint_lower_bound_discards` — result with metric < lower_bound gets DISCARD
3. `test_constraint_satisfied_keeps` — result within bounds gets KEEP (assuming improvement)
4. `test_constraint_metadata_tag` — discarded result has `{"constraint_violated": True}` in metadata
5. `test_engine_constraint_backward_compat` — engine with no `constraints` param unchanged

### Acceptance criteria
- `uv run pytest tests/unit/test_multi_objective.py tests/unit/test_constraints.py -v` all pass
- `uv run pytest -m "not slow"` still passes
- `uv run mypy causal_optimizer/` passes
- No breaking changes to existing engine API (all new params default to `None`)

---

## Track C — `sprint-7/persist-and-cli`

### Task 1 — SQLite persistence (`storage/sqlite.py`)

Implement `ExperimentStore`:

```python
class ExperimentStore:
    """SQLite-backed persistent store for experiment logs.

    Schema:
        experiments(id TEXT PRIMARY KEY, created_at TEXT, search_space_json TEXT)
        results(id TEXT, experiment_id TEXT, step INT, parameters_json TEXT,
                metrics_json TEXT, status TEXT, metadata_json TEXT, timestamp TEXT)
    """

    def __init__(self, path: str | Path) -> None:
        """Open or create a store at the given path. ':memory:' for in-memory."""
        ...

    def create_experiment(self, experiment_id: str, search_space: SearchSpace) -> None: ...
    def append_result(self, experiment_id: str, result: ExperimentResult, step: int) -> None: ...
    def load_log(self, experiment_id: str) -> ExperimentLog: ...
    def list_experiments(self) -> list[dict[str, Any]]: ...
    def delete_experiment(self, experiment_id: str) -> None: ...
```

Rules:
- Pure stdlib: `sqlite3`, `json`, `pathlib` — no new dependencies
- Thread-safe: use `check_same_thread=False` + a `threading.Lock`
- `ExperimentResult` round-trips losslessly: serialize via `.model_dump()`, deserialize via
  `ExperimentResult.model_validate()`
- `SearchSpace` serialized as JSON via `.model_dump()` / `SearchSpace.model_validate()`

### Task 2 — Engine persistence integration (`engine/loop.py`)

Add `store: ExperimentStore | None = None` and `experiment_id: str | None = None` to
`ExperimentEngine.__init__()`:
- If `store` is provided, call `store.create_experiment()` on init (no-op if already exists)
- After each `step()`, call `store.append_result()` with the step index
- Add `ExperimentEngine.resume(store, experiment_id, runner, **engine_kwargs)` classmethod:
  loads `ExperimentLog` from store, reconstructs engine state (phase, step count) from the log,
  returns a ready-to-continue engine

The resume classmethod signature:
```python
@classmethod
def resume(
    cls,
    store: ExperimentStore,
    experiment_id: str,
    runner: ExperimentRunner,
    **engine_kwargs: Any,
) -> ExperimentEngine:
    """Resume an interrupted experiment from a persistent store."""
    ...
```

Phase reconstruction from log: infer phase from `len(log.results)` using the same
thresholds as `_update_phase()` (exploration ≤ 10, optimization 11–50, exploitation 50+).

### Task 3 — CLI (`__main__.py` + `cli.py`)

Implement `causal_optimizer/cli.py` using `argparse` (stdlib only, no Click):

```
causal-optimizer run    --adapter <module:class> --budget <n> --db <path> [--id <str>] [--seed <int>]
causal-optimizer resume --adapter <module:class> --id <str> --db <path> [--budget <n>]
causal-optimizer report --id <str> --db <path> [--format table|json]
causal-optimizer list   --db <path>
```

- `run`: instantiate adapter from `module:ClassName`, create engine with store, run for budget steps
- `resume`: load from store, continue for additional budget steps
- `report`: print best result + summary stats (n_kept, n_discarded, n_crash, phases visited)
- `list`: print all experiment IDs with creation date and step count
- Adapter loading: `importlib.import_module(module)` then `getattr(module, classname)()`
- Domain adapters must implement `DomainAdapter` protocol (already defined in `domain_adapters/base.py`)

Add `causal_optimizer/__main__.py`:
```python
from causal_optimizer.cli import main
main()
```

Add to `pyproject.toml`:
```toml
[project.scripts]
causal-optimizer = "causal_optimizer.cli:main"
```

### Task 4 — Tests

Create `tests/unit/test_store.py`:
1. `test_store_create_and_load_empty` — create experiment, load_log returns empty ExperimentLog
2. `test_store_append_and_load` — append 3 results, load_log returns all 3 in order
3. `test_store_result_roundtrip` — ExperimentResult survives serialize/deserialize with all fields
4. `test_store_list_experiments` — list returns the experiment created
5. `test_store_delete` — after delete, load_log raises KeyError or returns empty
6. `test_store_in_memory` — `:memory:` path works for testing
7. `test_engine_with_store_persists_each_step` — run engine 5 steps with store; assert 5 rows in DB
8. `test_engine_resume_continues_from_step` — run 5 steps, create new engine via `resume()`,
   run 5 more; assert final log has 10 results

Create `tests/unit/test_cli.py`:
1. `test_cli_list_empty_db` — `causal-optimizer list --db :memory:` exits 0, prints headers
2. `test_cli_run_toy_adapter` — run 5 steps via CLI on a test adapter; assert DB has 5 rows
3. `test_cli_report_json` — report --format json returns valid JSON with `best_result` key
4. `test_cli_resume_adds_steps` — run 3, resume 3, assert 6 total steps

### Acceptance criteria
- `uv run pytest tests/unit/test_store.py tests/unit/test_cli.py -v` all pass
- `uv run pytest -m "not slow"` still passes
- `uv run mypy causal_optimizer/` passes
- `uv run causal-optimizer --help` exits 0
- No new required dependencies (sqlite3 is stdlib; argparse is stdlib)

---

## Key project context

```bash
# Install deps
uv sync --extra all --extra dev   # includes bayesian (ax, botorch, gpytorch)

# Run tests
uv run pytest -m "not slow"       # fast suite (used in CI)
uv run pytest -m slow             # slow regression tests only

# Lint / format / types
uv run ruff check .
uv run ruff format .
uv run mypy causal_optimizer/
```

- Python 3.10+, `from __future__ import annotations`
- Pydantic v2 for serializable models; `@dataclass` for `CausalGraph`
- Ruff line length 100, rules `E F I N W UP B SIM TCH`
- mypy strict mode — no `type: ignore` without a comment explaining why
- Graceful degradation: optional deps guarded by `try/except ImportError`
- `pytest.importorskip("ax")` / `pytest.importorskip("botorch")` for optional-dep tests
- Review: `./claude-review.sh <PR_NUMBER>` from repo root
- Greptile App NOT installed — greploop uses MCP tools (`mcp__greptile__query_repository`)

## Relevant files

| File | Relevance |
|------|-----------|
| `causal_optimizer/optimizer/bayesian.py` | `AxBayesianOptimizer` — reference for Track A GP structure |
| `causal_optimizer/optimizer/suggest.py` | `_suggest_bayesian()`, `_suggest_optimization()` — wire Track A strategy |
| `causal_optimizer/engine/loop.py` | `ExperimentEngine.__init__()` — add `strategy`, `objectives`, `constraints`, `store` params |
| `causal_optimizer/types.py` | `ExperimentResult`, `ExperimentLog`, `SearchSpace` — add Track B types |
| `causal_optimizer/domain_adapters/base.py` | `DomainAdapter` protocol — Track C CLI adapter loading |
| `causal_optimizer/benchmarks/toy_graph.py` | Add `ToyGraphBiObjective` for Track B |
| `pyproject.toml` | Add CLI script entry point (Track C); verify botorch/gpytorch in bayesian extra |

## Track A research note

The CausalGP implementation should follow this simplified CBO architecture:

1. **Topological order**: get nodes in topological sort from `CausalGraph`
2. **Fit phase**: for each node X_i, fit `SingleTaskGP(X=parent_values, y=node_values)`
   using only observed experiments where all parents are recorded
3. **Predict phase (interventional)**:
   - For intervened variables: set value = intervention value (do-operator)
   - For non-intervened variables: predict using their GP, conditioned on already-computed
     parent values (working in topological order)
   - Final prediction: evaluate objective GP at the propagated values
4. **Acquisition**: Expected Improvement using the objective node's posterior

This is a practical approximation — the true CBO acquisition marginalizes over the
uncertainty in all non-intervened nodes, but using posterior means is a reasonable
first implementation. Flag this simplification in the docstring.
