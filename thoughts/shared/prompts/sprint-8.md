# Sprint 8 Prompt — Hardening & Usability

## Context

Sprints 1–7 complete. The engine has all core algorithms wired, Bayesian optimization (Ax/BoTorch),
experimental causal GP surrogates (CBO), multi-objective + constrained optimization, SQLite
persistence, and a CLI. Sprint 8 focuses on hardening, usability, and closing integration gaps
identified in a post-Sprint 7 audit.

**Five independent tracks**, all non-overlapping, all can merge in any order.

**Branch base:** `main` (after Sprint 7 — PRs #19, #20, #21)

**Merge order after human approval:**
1. Any order — all five are non-overlapping
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
- Read `thoughts/shared/plans/01-what-to-build-first.md` for original architecture vision
  to ensure changes align with the project's design philosophy

---

## Track A — `sprint-8/cli-feature-parity`

### Problem

The CLI exposes only 3 of ~12 engine parameters. Users cannot access multi-objective mode,
constraints, strategy selection, discovery, or even basic settings like `--minimize` or
`--objective-name` from the command line.

### Task 1 — Extend `DomainAdapter` interface (`domain_adapters/base.py`)

Add optional configuration hooks with backward-compatible defaults:

```python
class DomainAdapter(ABC):
    # ... existing methods ...

    def get_objective_name(self) -> str:
        """Return the name of the primary objective metric."""
        return "objective"

    def get_minimize(self) -> bool:
        """Return True if the objective should be minimized."""
        return True

    def get_strategy(self) -> str:
        """Return the optimization strategy ('bayesian' or 'causal_gp')."""
        return "bayesian"

    def get_objectives(self) -> list[ObjectiveSpec] | None:
        """Return multi-objective specifications, or None for single-objective."""
        return None

    def get_constraints(self) -> list[Constraint] | None:
        """Return optimization constraints, or None for unconstrained."""
        return None

    def get_discovery_method(self) -> str | None:
        """Return causal discovery method, or None to disable."""
        return None
```

All return sensible defaults — no breaking changes to existing adapters.

### Task 2 — Extend CLI flags (`cli.py`)

Add flags to `run` and `resume` subcommands that override adapter defaults:

```
causal-optimizer run --adapter <spec> --budget <n> --db <path>
    [--id <str>] [--seed <int>]
    [--objective-name <str>]      # NEW: override adapter.get_objective_name()
    [--minimize / --maximize]     # NEW: override adapter.get_minimize()
    [--strategy <str>]            # NEW: override adapter.get_strategy()
    [--discovery-method <str>]    # NEW: override adapter.get_discovery_method()
```

Priority order: CLI flag > adapter method > engine default.

### Task 3 — Fix `report` hardcoded objective key

In `_cmd_report()` and `_cmd_run()`, the report hardcodes `best.metrics.get('objective', 'N/A')`.
Fix to use the actual objective name (store it in experiment metadata or detect from the
best result's metrics).

### Task 4 — Wire `_adapter_engine_kwargs` with all new hooks

Update `_adapter_engine_kwargs()` to pull all configuration from the adapter, then allow
CLI args to override. The engine should receive the full parameter surface.

### Task 5 — Tests

Create `tests/unit/test_cli_extended.py`:
1. `test_cli_minimize_flag` — `--minimize` sets `minimize=True` on engine
2. `test_cli_maximize_flag` — `--maximize` sets `minimize=False`
3. `test_cli_strategy_flag` — `--strategy causal_gp` propagates to engine
4. `test_cli_objective_name_flag` — `--objective-name cost` propagates
5. `test_cli_discovery_method_flag` — `--discovery-method correlation` propagates
6. `test_adapter_config_hooks_defaults` — default adapter methods return expected values
7. `test_cli_flag_overrides_adapter` — CLI flag takes precedence over adapter method
8. `test_report_uses_correct_objective` — report shows the right metric key

### Acceptance criteria
- `uv run pytest tests/unit/test_cli.py tests/unit/test_cli_extended.py -v` all pass
- `uv run pytest -m "not slow"` still passes
- `uv run mypy causal_optimizer/` passes
- `uv run causal-optimizer run --help` shows all new flags

---

## Track B — `sprint-8/variable-validation`

### Problem

`Variable(name="x", variable_type=VariableType.CONTINUOUS)` with no bounds is valid Pydantic
but crashes deep in the optimization loop when `_random_sample()` hits `assert var.lower is
not None`. Similarly, `CATEGORICAL` without `choices` crashes silently. These should fail
loudly at construction time.

### Task 1 — Add `@model_validator` to `Variable` (`types.py`)

```python
class Variable(BaseModel):
    # ... existing fields ...

    @model_validator(mode="after")
    def _validate_type_constraints(self) -> Variable:
        """Enforce type-specific invariants at construction time."""
        if self.variable_type in (VariableType.CONTINUOUS, VariableType.INTEGER):
            if self.lower is None or self.upper is None:
                raise ValueError(
                    f"{self.variable_type.value} variable {self.name!r} "
                    f"requires both 'lower' and 'upper' bounds"
                )
            if self.lower >= self.upper:
                raise ValueError(
                    f"Variable {self.name!r}: lower ({self.lower}) must be < upper ({self.upper})"
                )
        elif self.variable_type == VariableType.CATEGORICAL:
            if not self.choices:
                raise ValueError(
                    f"Categorical variable {self.name!r} requires non-empty 'choices'"
                )
        return self
```

### Task 2 — Fix any test fixtures that violate the new validation

Audit all test files and adapter definitions for `Variable` construction that would now fail.
Fix them to include proper bounds/choices. This is likely a small number of places.

### Task 3 — Tests

Create `tests/unit/test_variable_validation.py`:
1. `test_continuous_requires_bounds` — `Variable(name="x", variable_type=CONTINUOUS)` raises
2. `test_continuous_requires_lower_lt_upper` — `lower=5, upper=3` raises
3. `test_integer_requires_bounds` — same as continuous
4. `test_categorical_requires_choices` — `Variable(name="c", variable_type=CATEGORICAL)` raises
5. `test_categorical_empty_choices_raises` — `choices=[]` raises
6. `test_boolean_no_bounds_ok` — `Variable(name="b", variable_type=BOOLEAN)` is valid
7. `test_valid_continuous_passes` — `lower=0, upper=1` is valid
8. `test_valid_categorical_passes` — `choices=["a", "b"]` is valid

### Acceptance criteria
- `uv run pytest tests/unit/test_variable_validation.py -v` all pass
- `uv run pytest -m "not slow"` still passes (no fixtures broken)
- `uv run mypy causal_optimizer/` passes

---

## Track C — `sprint-8/seed-reproducibility`

### Problem

Several code paths create unseeded `np.random.default_rng()` instances, making results
non-reproducible even when `seed` is set on the engine:

- `optimizer/suggest.py:_random_sample()` — fallback sampling (line 466)
- `optimizer/suggest.py:_suggest_exploitation()` — perturbation (line 237)
- `engine/loop.py:suggest_next()` — MAP-Elites coin flip (line 398)
- `evolution/map_elites.py:sample_elite()` and `sample_diverse()` — archive sampling

### Task 1 — Thread `seed` through all random paths

The engine already stores `self._seed`. Thread it through:

1. **`suggest_parameters()`** — already accepts `seed` param; pass it to `_suggest_exploitation()`
   and `_random_sample()`
2. **`_random_sample(seed)`** — accept optional seed, create `rng = np.random.default_rng(seed)`
   Use a derived seed (e.g., `seed + hash("random_sample")`) to avoid identical sequences
3. **`_suggest_exploitation(seed)`** — same pattern
4. **`engine.suggest_next()`** — pass `self._seed` to the MAP-Elites coin flip:
   `rng = np.random.default_rng(self._seed + step_count)` so each step gets a different
   but reproducible flip
5. **`MAPElites.sample_elite(seed)` / `sample_diverse(seed)`** — accept optional seed

Use step-offset seeding: `derived_seed = seed + len(experiment_log.results)` so each call
gets a unique but deterministic seed. When `seed is None`, fall back to unseeded (current behavior).

### Task 2 — Tests

Create `tests/unit/test_seed_reproducibility.py`:
1. `test_engine_seeded_deterministic` — run engine with `seed=42` for 5 steps, run again
   with `seed=42`; assert identical results (same parameters, same metrics)
2. `test_engine_different_seeds_differ` — `seed=42` vs `seed=99` produce different parameters
3. `test_engine_unseeded_varies` — two unseeded runs may differ (non-deterministic assertion:
   run 5 times, assert not all identical)
4. `test_random_sample_seeded` — `_random_sample(search_space, seed=42)` returns same result
5. `test_exploitation_seeded` — `_suggest_exploitation(..., seed=42)` deterministic
6. `test_map_elites_sample_seeded` — `sample_elite(seed=42)` deterministic

### Acceptance criteria
- `uv run pytest tests/unit/test_seed_reproducibility.py -v` all pass
- `uv run pytest -m "not slow"` still passes
- `uv run mypy causal_optimizer/` passes
- Seeded engine runs are fully deterministic end-to-end

---

## Track D — `sprint-8/pyproject-and-packaging`

### Problem

1. `Development Status :: 2 - Pre-Alpha` — should be `3 - Alpha` after 7 sprints
2. `qdax>=0.5.0` is declared in `[evolution]` extra but never imported anywhere
3. No `Changelog`, `Documentation`, or `Bug Tracker` URLs
4. `engine/__init__.py` re-exports `ExperimentEngine` but `storage/__init__.py` doesn't
   export `ExperimentStore`; `types.py` doesn't get a public re-export from `__init__.py`

### Task 1 — Fix `pyproject.toml`

1. Update classifier: `"Development Status :: 3 - Alpha"`
2. Remove `qdax>=0.5.0` from `evolution` extra (it's unused; MAP-Elites is pure numpy)
3. Add URLs:
   ```toml
   [project.urls]
   Repository = "https://github.com/datablogin/causal-optimizer"
   Issues = "https://github.com/datablogin/causal-optimizer/issues"
   ```
4. If `evolution` extra becomes empty after removing qdax, remove the extra entirely and
   update the `all` extra to exclude it

### Task 2 — Add public re-exports in `__init__.py` files

- `causal_optimizer/__init__.py` — export key public API:
  ```python
  from causal_optimizer.engine.loop import ExperimentEngine
  from causal_optimizer.types import (
      CausalGraph, Constraint, ExperimentLog, ExperimentResult,
      ExperimentStatus, ObjectiveSpec, SearchSpace, Variable, VariableType,
  )
  ```
- `causal_optimizer/storage/__init__.py` — export `ExperimentStore`

### Task 3 — Tests

Create `tests/unit/test_public_api.py`:
1. `test_top_level_imports` — `from causal_optimizer import ExperimentEngine, SearchSpace, ...`
   all work
2. `test_storage_import` — `from causal_optimizer.storage import ExperimentStore` works
3. `test_engine_import` — `from causal_optimizer.engine import ExperimentEngine` works (existing)

### Acceptance criteria
- `uv run pytest tests/unit/test_public_api.py -v` all pass
- `uv run pytest -m "not slow"` still passes
- `uv run mypy causal_optimizer/` passes
- `uv sync --extra evolution` no longer fails on non-JAX platforms (qdax removed)
- `python -c "from causal_optimizer import ExperimentEngine"` works

---

## Track E — `sprint-8/examples-and-docs`

### Problem

Only one example (`quickstart.py`). The library's distinctive features — multi-objective
optimization, constrained optimization, CLI workflow, auto-discovery, causal GP — have no
runnable examples.

### Task 1 — Multi-objective example (`examples/multi_objective.py`)

A self-contained script demonstrating:
- Define two objectives (`ObjectiveSpec`)
- Run engine for 20 steps on `ToyGraphBiObjective`
- Print the Pareto front
- Show how non-dominated solutions trade off between objectives

Use the benchmark directly (no external deps). Include clear comments explaining each step.

### Task 2 — Constrained optimization example (`examples/constrained.py`)

A self-contained script demonstrating:
- Define constraints (`Constraint` with upper/lower bounds)
- Run engine with constraints
- Show that violated experiments are discarded with `constraint_violated` metadata
- Print summary of kept vs discarded

### Task 3 — CLI workflow example (`examples/cli_workflow.sh`)

A bash script demonstrating the full CLI lifecycle:
```bash
#!/bin/bash
# Create a simple adapter, run experiments, resume, and report
causal-optimizer run --adapter examples.demo_adapter:DemoAdapter --budget 10 --db demo.db --id demo-1
causal-optimizer report --id demo-1 --db demo.db
causal-optimizer resume --adapter examples.demo_adapter:DemoAdapter --id demo-1 --db demo.db --budget 5
causal-optimizer report --id demo-1 --db demo.db --format json
causal-optimizer list --db demo.db
```

Also create `examples/demo_adapter.py` — a minimal working `DomainAdapter` subclass
(e.g., Branin function) that the CLI example uses.

### Task 4 — Auto-discovery example (`examples/auto_discovery.py`)

A self-contained script demonstrating:
- Run engine **without** a prior causal graph, using `discovery_method="correlation"`
- Show the discovered graph after exploration phase
- Compare optimization with vs without discovery

### Task 5 — Tests

Create `tests/unit/test_examples.py`:
1. `test_quickstart_runs` — import and run `examples/quickstart.py` main logic (not as subprocess)
2. `test_multi_objective_runs` — import and run `examples/multi_objective.py`
3. `test_constrained_runs` — import and run `examples/constrained.py`
4. `test_auto_discovery_runs` — import and run `examples/auto_discovery.py`
5. `test_demo_adapter_works` — instantiate `DemoAdapter`, verify it implements the protocol

Note: Each example should be structured as a `main()` function so tests can import and call it.

### Acceptance criteria
- `uv run pytest tests/unit/test_examples.py -v` all pass
- `uv run pytest -m "not slow"` still passes
- All examples run standalone: `uv run python examples/multi_objective.py` etc.
- Examples are concise (< 80 lines each) with clear comments
- No new dependencies — use only what's already available

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
| `causal_optimizer/cli.py` | Track A: extend CLI flags and report |
| `causal_optimizer/domain_adapters/base.py` | Track A: add config hooks to DomainAdapter |
| `causal_optimizer/types.py` | Track B: add Variable model_validator |
| `causal_optimizer/optimizer/suggest.py` | Track C: thread seed through random paths |
| `causal_optimizer/engine/loop.py` | Track C: seed the MAP-Elites coin flip |
| `causal_optimizer/evolution/map_elites.py` | Track C: add seed to sample methods |
| `pyproject.toml` | Track D: fix classifiers, remove qdax, add URLs |
| `causal_optimizer/__init__.py` | Track D: public API re-exports |
| `examples/quickstart.py` | Track E: reference for example style |
| `causal_optimizer/benchmarks/toy_graph.py` | Track E: ToyGraphBiObjective for examples |

## Architecture context

Read `thoughts/shared/plans/01-what-to-build-first.md` for the original architecture vision.
The plan's "What We're NOT Building Yet" section listed CLI and persistent storage as premature —
those have since been built in Sprint 7. Sprint 8 is about making all these features properly
accessible and robust.
