# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync                        # core only
uv sync --extra all            # all optional deps (bayesian, doe, evolution, causal)
uv sync --extra dev            # dev tools (pytest, ruff, mypy)

# Run tests
uv run pytest                  # all tests
uv run pytest tests/unit/      # unit tests only
uv run pytest tests/unit/test_engine.py           # single file
uv run pytest tests/unit/test_engine.py::test_name -v  # single test

# Lint and type check
uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy causal_optimizer/  # type check (strict mode)
```

## Architecture

**Causally-informed experiment optimization engine.** Uses causal inference + Bayesian optimization + evolutionary strategies to decide what experiment to run next.

### Core loop (`engine/loop.py` → `ExperimentEngine`)

The engine runs a three-phase optimization loop:
1. **Exploration** (experiments 1–10): Latin Hypercube space-filling via `designer/factorial.py`
2. **Optimization** (11–50): Bayesian optimization (Ax/BoTorch) or RF surrogate, guided by causal graph ancestors. Screening runs at phase transition to identify important variables.
3. **Exploitation** (50+): Local perturbation around best configuration, with MAP-Elites diversity sampling.

Each `step()` call: suggest parameters → off-policy check (skip if predicted poor) → run experiment → evaluate (bootstrap CI or greedy) → update phase.

### Key modules

| Module | Entry point | Role |
|--------|------------|------|
| `types.py` | `CausalGraph`, `SearchSpace`, `ExperimentLog` | Core data models. `CausalGraph` supports bidirected edges for confounders, do-calculus surgery, c-components. |
| `optimizer/suggest.py` | `suggest_parameters()` | Routes to exploration/optimization/exploitation strategy. Uses `focus_variables` from causal graph ancestors + screening intersection. |
| `estimator/effects.py` | `EffectEstimator` | Bootstrap, difference, AIPW treatment effect estimation. |
| `predictor/off_policy.py` | `OffPolicyPredictor` | RF surrogate that predicts outcomes; `should_run_experiment()` gates expensive runs. |
| `designer/screening.py` | `ScreeningDesigner` | fANOVA-based variable importance + interaction detection. |
| `evolution/map_elites.py` | `MAPElites` | Diversity archive indexed by behavioral descriptors. |
| `domain_adapters/base.py` | `DomainAdapter` | Abstract base for domain-specific adapters (search space + runner + optional prior graph). |

### Data flow

`ExperimentEngine` owns an `ExperimentLog` (list of `ExperimentResult`). Each result has parameters, metrics, and status (KEEP/DISCARD/CRASH). The `suggest_parameters()` function reads the log + optional `CausalGraph` to propose the next experiment.

### Graceful degradation

Ax/BoTorch → RF surrogate. AIPW → bootstrap CI. pyDOE3 → built-in LHS. Core runs on numpy/pandas/scipy/scikit-learn only.

## PR Merge Policy

**NEVER merge a PR without completing ALL of the following:**

1. **claude-review.sh** — Run `./claude-review.sh <PR_NUMBER>` from repo root. Posts review as PR comment. Fix all issues found.
2. **greploop** — Run the `/greploop` skill on the PR. Iterate until Greptile gives 5/5 confidence with zero unresolved comments.
3. **Human approval** — Wait for explicit human sign-off before merging. Do not merge autonomously.

This is a hard requirement. No exceptions. PRs that skip any of these steps must be reverted or re-reviewed.

## Conventions

- Python 3.10+ with `from __future__ import annotations`
- Pydantic v2 `BaseModel` for serializable types; `@dataclass` for `CausalGraph` (needs mutable post-init)
- Ruff: line length 100, rules `E F I N W UP B SIM TCH`
- mypy strict mode enabled
- `ExperimentLog.best_result()` accepts `objective_name` and `minimize` parameters (defaults preserve backward compatibility)

## Known issues

None currently tracked.
