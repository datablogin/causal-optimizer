# Sprint 6 Prompt

## Context

Phases 1 and 2 are complete. The engine is fully implemented and end-to-end tested (366 tests,
16 merged PRs). Sprint 6 closes the academic story with benchmark completeness, then upgrades
the optimizer from an RF surrogate to proper Bayesian optimization via Ax/BoTorch.

**Branch base:** `main` (after Sprint 5 merges — PRs #14, #15, #16)

**Merge order after human approval:**
1. `sprint-6/benchmark-completeness` (no new engine deps — merge first)
2. `sprint-6/ax-botorch` (adds new optional dep — merge second, rebase on main after #1)

---

## Invocation

Stand up one agent per feature. Each agent must follow this exact workflow:

```
/tdd → implement → /polish → gh pr create → /gauntlet → report PR URL
```

Rules:
- Do not skip `/polish` before creating the PR
- Do not skip `/gauntlet` after creating the PR
- Each agent works in an isolated worktree (`isolation: "worktree"`)
- Do **NOT** merge — leave PRs open for human review
- After each human-approved merge, next PR rebases on main and re-runs `uv run pytest`
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)

---

## Features

### `sprint-6/benchmark-completeness`

Close the two unverified success criteria from `thoughts/shared/plans/01-what-to-build-first.md`.

**Task 1 — CompleteGraph POMIS pruning benchmark**

Create `tests/regression/test_pomis_pruning.py` (`@pytest.mark.slow`):

The CompleteGraph benchmark from Aglietti et al. (AISTATS 2020):
- 7 observed variables (A–F, Y) + 2 unobserved confounders (U1 confounds A↔Y, U2 confounds B↔Y)
- Known POMIS = `[{B}, {D}, {E}, {B,D}, {D,E}]` — 5 sets
- Naive search space = 2^6 = 64 variable subsets

Tests to write:
1. `test_complete_graph_pomis_size` — assert `len(compute_pomis(graph, "Y")) == 5`
2. `test_complete_graph_pomis_members` — assert the 5 sets match expected exactly
3. `test_pomis_prunes_search_space` — run `BenchmarkRunner.compare(["causal", "random"], budget=30, n_seeds=3)` on a `CompleteGraphSCM`; assert causal explores ≤ 5 distinct variable subsets while random explores > 5
4. `test_pomis_pruning_ratio` — assert causal tries ≥ 10× fewer variable combinations than naive (64 → ≤ 6 unique subsets touched)

If `CompleteGraphSCM` doesn't exist in `benchmarks/scms.py`, implement it following the Aglietti equations. Use `numpy` for the SCM, `CausalGraph` with bidirected edges for U1 and U2.

**Task 2 — Interaction detection benchmark**

Create `tests/regression/test_interaction_detection.py` (`@pytest.mark.slow`):

Design: 5 variables where X1 and X2 interact — individually they hurt the objective,
together they help. Greedy hill-climbing discards both; factorial screening catches the interaction.

```python
# Interaction SCM:
# Y = -X1 - X2 + 3*X1*X2 + noise
# Optimal: X1=1, X2=1 → Y=1; X1=0,X2=0 → Y=0; X1=1,X2=0 → Y=-1
```

Tests to write:
1. `test_screening_detects_interaction` — run `ScreeningDesigner.screen()` on an `ExperimentLog` with 12 experiments sampling the interaction SCM; assert the `(X1, X2)` interaction appears in `screening_result.interactions`
2. `test_greedy_misses_interaction` — run a greedy (random strategy, no causal graph) `ExperimentEngine` for 20 steps on the SCM; assert it fails to find the X1=X2=1 optimum with >50% probability across 5 seeds
3. `test_causal_finds_interaction` — run `ExperimentEngine` with `discovery_method="correlation"` for 30 steps; assert the best result has X1≈1 and X2≈1

Add `tests/regression/test_pomis_pruning.py` and `tests/regression/test_interaction_detection.py`.
Update `tests/regression/conftest.py` docstring to reflect new test files.

**Acceptance criteria:**
- Both files pass (`uv run pytest tests/regression/ -m slow -v`)
- `tests/regression/helpers.py` extended if shared utilities needed
- Success criteria items 2 and 4 in `thoughts/shared/plans/01-what-to-build-first.md` updated to ✓

---

### `sprint-6/ax-botorch`

Replace the RF surrogate stub in the optimization phase with a real Ax/BoTorch Bayesian
optimizer. Keep the RF surrogate as a fallback when Ax is not installed.

**Background:** `optimizer/suggest.py::_suggest_bayesian()` currently raises
`NotImplementedError` or falls back immediately. The `bayesian` optional dep group
exists in `pyproject.toml` but may be empty or stubbed.

**Task 1 — Ax/BoTorch integration**

Implement `optimizer/bayesian.py`:

```python
class AxBayesianOptimizer:
    """Wraps Ax ServiceAPI for suggest/update loop."""

    def __init__(
        self,
        search_space: SearchSpace,
        objective_name: str,
        minimize: bool = True,
        focus_variables: list[str] | None = None,
        seed: int | None = None,
    ) -> None: ...

    def suggest(self) -> dict[str, Any]: ...
    def update(self, params: dict[str, Any], value: float) -> None: ...
    def best(self) -> dict[str, Any] | None: ...
```

Rules:
- Only operate on `focus_variables` if provided (POMIS/screening integration)
- Use `ax.service.ax_client.AxClient` with `GenerationStrategy` configured for
  `Sobol` (exploration) → `BoTorch` (optimization)
- Handle `ParameterType.FLOAT`, `ParameterType.INT`, `ParameterType.STRING`
  (categorical) and `ParameterType.BOOL` (fixed to FLOAT 0/1)
- Gracefully degrade: if `ax` is not importable, raise `ImportError` with a
  clear message pointing to `uv sync --extra bayesian`

Wire into `optimizer/suggest.py::_suggest_bayesian()`:
- Instantiate `AxBayesianOptimizer` on first call, cache on the `ExperimentEngine`
- Call `optimizer.update()` after each KEEP result
- Replace the `NotImplementedError` / RF fallback in the Bayesian branch

**Task 2 — Causal-aware acquisition weighting**

Add `pomis_prior: list[frozenset[str]] | None` parameter to `AxBayesianOptimizer`.
When set, add a soft constraint: candidates that only touch POMIS variables get a
0.2 bonus on the acquisition score (implemented as a post-hoc filter on Sobol
candidates, not a custom kernel — keep it simple).

**Task 3 — Tests**

Create `tests/unit/test_ax_optimizer.py`:
1. `test_ax_optimizer_suggest_returns_valid_params` — single suggest returns params
   within search space bounds
2. `test_ax_optimizer_update_improves_suggestion` — after 5 update()/suggest() cycles
   on a simple quadratic, final suggestion is closer to optimum than first
3. `test_ax_optimizer_focus_variables_respected` — with `focus_variables=["x1"]`,
   only `x1` varies; other params fixed at midpoint
4. `test_ax_optimizer_pomis_prior_biases_toward_pomis` — with pomis_prior set,
   over 10 suggestions, ≥70% touch only POMIS variables
5. `test_ax_optimizer_graceful_degradation` — mock `ax` as unimportable; assert
   `ImportError` is raised with helpful message
6. `test_ax_in_engine_optimization_phase` — run `ExperimentEngine` with
   `strategy="bayesian"` for 20 steps on ToyGraph; assert no crashes and
   `engine.phase` reaches `optimization`

Create `tests/integration/test_ax_pipeline.py`:
1. `test_ax_beats_rf_on_toygraph` — run both `strategy="bayesian"` (Ax) and
   `strategy="surrogate"` (RF) for 40 steps on ToyGraph with `n_seeds=3`;
   assert Ax mean final objective ≤ RF mean final objective × 1.1 (Ax is at
   least as good, with 10% tolerance)

**Acceptance criteria:**
- `uv run pytest tests/unit/test_ax_optimizer.py -v` all pass
- `uv run pytest tests/integration/test_ax_pipeline.py -v` passes
- `uv run pytest -m "not slow"` still passes (RF fallback still works without Ax)
- `uv sync --extra bayesian && uv run mypy causal_optimizer/` passes
- `pyproject.toml` `[project.optional-dependencies] bayesian` includes `ax-platform>=0.4`

---

## Key project context

```bash
# Install all deps including bayesian
uv sync --extra all --extra dev

# Run tests
uv run pytest                          # all tests
uv run pytest -m "not slow"            # skip regression tests
uv run pytest tests/unit/ -v           # unit only

# Lint / format / types
uv run ruff check .
uv run ruff format .
uv run mypy causal_optimizer/
```

- Python 3.10+, `from __future__ import annotations`
- Pydantic v2 for serializable models; `@dataclass` for `CausalGraph`
- Ruff line length 100, rules `E F I N W UP B SIM TCH`
- mypy strict mode — no `type: ignore` without a comment explaining why
- Graceful degradation pattern: optional deps guarded by `try/except ImportError`
- Review: `./claude-review.sh <PR_NUMBER>` from repo root
- Greptile App NOT installed — greploop uses MCP tools (`mcp__greptile__query_repository`)

## Relevant files

| File | Relevance |
|------|-----------|
| `causal_optimizer/optimizer/suggest.py` | `_suggest_bayesian()` to be wired |
| `causal_optimizer/optimizer/pomis.py` | `compute_pomis()` for pruning test |
| `causal_optimizer/benchmarks/scms.py` | Add `CompleteGraphSCM` here |
| `causal_optimizer/benchmarks/runner.py` | `BenchmarkRunner.compare()` |
| `causal_optimizer/designer/screening.py` | `ScreeningDesigner.screen()` |
| `tests/regression/helpers.py` | Shared assertion utilities |
| `thoughts/shared/plans/01-what-to-build-first.md` | Update success criteria ✓ after merge |
