# Sprint 11 Prompt — PyPI Publishing Pipeline

## Context

Sprints 1–10 complete. Both packages (`causal-optimizer` and `causal-inference-marketing`)
are feature-complete but neither is publishable to PyPI. This sprint fixes that.

**Two repos, five parallel tracks, two sprints:**

| Sprint | Tracks | Depends on |
|--------|--------|------------|
| Sprint 11a | A, B, C (causal-optimizer) + D, E (causal-inference-marketing) | Nothing — all parallel |
| Sprint 11b | F (reconnect + publish prep) | Sprint 11a merged |

---

## Key decisions made

1. **CIM PyPI name:** `causal-inference-marketing` (available on PyPI)
2. **CIM import name:** Keep `causal_inference` (avoid 200-file rename). Note: the name
   `causal-inference` (v0.0.4) exists on PyPI with the same import — document in README
   that users should not install both.
3. **CIM scope:** Publish only the inner library (`libs/causal_inference/`), not services/shared.
4. **Publishing method:** GitHub Actions OIDC trusted publishers (no API tokens).
5. **causal-optimizer v0.1.0** ships without CIM dependency in `causal` extra (just `dowhy`).
   Sprint 11b reconnects it once CIM is published.

---

## Sprint 11a — All Tracks Parallel

### Track Layout

| Track | Repo | Branch | Modules touched |
|-------|------|--------|----------------|
| A — Package files | causal-optimizer | `sprint-11/pypi-package-files` | root files, `causal_optimizer/` |
| B — Dependency cleanup | causal-optimizer | `sprint-11/pypi-deps-cleanup` | `pyproject.toml`, `README.md` |
| C — Publish workflow | causal-optimizer | `sprint-11/pypi-publish-workflow` | `.github/workflows/` |
| D — CIM library prep | causal-inference-marketing | `sprint-11/pypi-library-prep` | `libs/causal_inference/pyproject.toml`, metadata |
| E — CIM cleanup + workflow | causal-inference-marketing | `sprint-11/pypi-cleanup` | `.bak` files, `.github/workflows/` |

### Invocation

Tracks A, B, C each get one agent working in an isolated worktree of **causal-optimizer**
(`/Users/robertwelborn/Projects/causal-optimizer`).

Tracks D, E each get one agent working in an isolated worktree of **causal-inference-marketing**
(`/Users/robertwelborn/Projects/causal-inference-marketing`).

Each agent must follow this exact workflow:

```
implement → /polish → gh pr create → /gauntlet → report PR URL
```

Rules:
- Each agent works in an isolated worktree (`isolation: "worktree"`)
- Do **NOT** merge — leave PRs open for human review
- Read `CLAUDE.md` for project conventions
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)

---

## Track A — Package Files (`causal-optimizer`)

### Task 1 — Create LICENSE file

Create `/LICENSE` with the standard MIT license text:

```
MIT License

Copyright (c) 2025 Robert Welborn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Task 2 — Create py.typed marker

Create an empty file at `causal_optimizer/py.typed` for PEP 561 type stub support.
This enables downstream type checkers (mypy, pyright) to use the package's inline type hints.

### Task 3 — Create CHANGELOG.md

Create `CHANGELOG.md` at the repo root. Use [Keep a Changelog](https://keepachangelog.com/)
format. Summarize sprints 1–10 under a single `## [0.1.0] - 2026-03-21` release:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-21

### Added

- Three-phase optimization loop: exploration (LHS), optimization (Bayesian/RF),
  exploitation (MAP-Elites diversity sampling)
- Causal graph integration: focus variables from DAG ancestors, POMIS pruning
  of intervention sets
- Bayesian optimization via Ax/BoTorch with POMIS-aware acquisition functions
- Causal GP surrogates for causally-informed surrogate modeling
- Multi-objective optimization with Pareto dominance and scalarization
- Constrained optimization with constraint satisfaction checking
- Effect estimation with bootstrap confidence intervals for keep/discard decisions
- Off-policy prediction with RF surrogate to skip predicted-poor experiments
- Observational estimation via DoWhy (backdoor/frontdoor/IV adjustment)
- Observational-enhanced off-policy predictions (tighter/wider CI based on agreement)
- fANOVA-based screening at phase transitions to identify important variables
- MAP-Elites archive for diversity tracking with behavioral descriptors
- Epsilon controller for observation-intervention tradeoff
- Causal discovery from experiment data (correlation, PC, NOTEARS methods)
- Sensitivity validation for causal effect robustness checking
- Research advisor diagnostics with four analyses + recommendation synthesis
  (EXPLOIT/EXPLORE/DROP/PIVOT) and observational signal analysis
- Domain adapters: marketing campaign optimization, ML hyperparameter tuning
  (both with realistic simulators, confounders, and failure modes)
- SQLite persistence with resume support
- CLI: run, resume, report, list commands
- 530+ tests across unit, integration, and regression suites
```

### Task 4 — Add license-files to pyproject.toml

Add `license-files` per PEP 639:

```toml
license-files = ["LICENSE"]
```

This goes in the `[project]` section, after `license = "MIT"`.

### Task 5 — Add author email

Update the authors field in `pyproject.toml`:

```toml
authors = [
    { name = "Robert Welborn", email = "robert@datablogin.com" },
]
```

**Note:** Check the CIM repo's pyproject.toml or git config for the correct email address.
If you can't determine it, use a placeholder and note it in your PR description.

### Task 6 — Add hatch build config

Add build target configuration to `pyproject.toml` to exclude tests, thoughts, and
examples from the wheel (they shouldn't ship in site-packages):

```toml
[tool.hatch.build.targets.wheel]
packages = ["causal_optimizer"]

[tool.hatch.build.targets.sdist]
exclude = [
    "/.github",
    "/tests",
    "/thoughts",
    "/.coverage",
    "/.claude",
]
```

### Task 7 — Add project URLs

Expand `[project.urls]` in `pyproject.toml`:

```toml
[project.urls]
Homepage = "https://github.com/datablogin/causal-optimizer"
Documentation = "https://github.com/datablogin/causal-optimizer#readme"
Repository = "https://github.com/datablogin/causal-optimizer"
Issues = "https://github.com/datablogin/causal-optimizer/issues"
Changelog = "https://github.com/datablogin/causal-optimizer/blob/main/CHANGELOG.md"
```

### Files to create/modify

| File | Action |
|------|--------|
| `LICENSE` | Create |
| `causal_optimizer/py.typed` | Create (empty) |
| `CHANGELOG.md` | Create |
| `pyproject.toml` | Modify — license-files, author email, hatch build, URLs |

### Verification

```bash
uv run pytest -m "not slow" -x -q
uv run ruff check . && uv run ruff format --check .
uv run mypy causal_optimizer/
```

---

## Track B — Dependency Cleanup (`causal-optimizer`)

### Task 1 — Remove git+https dependency

Replace the `causal` optional dependency group. Remove the `causal-inference-marketing`
git reference and keep only `dowhy`:

**Before:**
```toml
causal = [
    "causal-inference-marketing @ git+https://github.com/datablogin/causal-inference-marketing.git",
    "dowhy>=0.11",
]
```

**After:**
```toml
causal = [
    "dowhy>=0.11",
]
```

### Task 2 — Remove allow-direct-references

Delete this section from `pyproject.toml`:

```toml
[tool.hatch.metadata]
allow-direct-references = true
```

### Task 3 — Update README with pip install instructions

The README currently only shows `uv sync` installation. Add `pip install` instructions
for PyPI users. After the existing installation section, add:

```markdown
### From PyPI

```bash
pip install causal-optimizer                    # core only
pip install causal-optimizer[bayesian]          # + Ax/BoTorch
pip install causal-optimizer[causal]            # + DoWhy
pip install causal-optimizer[all]               # everything
```

### From source (development)

```bash
git clone https://github.com/datablogin/causal-optimizer.git
cd causal-optimizer
uv sync --extra dev
```
```

Also update the existing `uv sync` section to be under the "From source" heading.

### Task 4 — Verify build works

Run a local build to confirm the package builds cleanly:

```bash
uv build
# Check the wheel contents — should only have causal_optimizer/ package
unzip -l dist/*.whl | head -40
# Validate metadata
uv run python -m twine check dist/* || echo "twine not installed, skip"
```

### Files to modify

| File | Action |
|------|--------|
| `pyproject.toml` | Modify — remove git dep, remove allow-direct-references |
| `README.md` | Modify — add pip install instructions |

### Verification

```bash
uv sync  # verify deps still resolve without git reference
uv run pytest -m "not slow" -x -q
uv run ruff check . && uv run ruff format --check .
uv build  # verify package builds
```

---

## Track C — Publish Workflow (`causal-optimizer`)

### Task 1 — Create GitHub Actions publish workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  test-publish:
    needs: build
    runs-on: ubuntu-latest
    environment: testpypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish:
    needs: test-publish
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

**Notes for PR description:**
- The `pypi` and `testpypi` GitHub environments must be created in repo settings
  with required reviewers for manual approval.
- Trusted publisher must be configured on pypi.org and test.pypi.org:
  - Owner: `datablogin`
  - Repository: `causal-optimizer`
  - Workflow: `publish.yml`
  - Environment: `pypi` / `testpypi`

### Task 2 — Add build validation to CI

Add a build check job to the existing `.github/workflows/ci.yml` so every PR
validates that the package builds:

```yaml
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv build
      - run: |
          pip install twine
          twine check dist/*
```

### Files to create/modify

| File | Action |
|------|--------|
| `.github/workflows/publish.yml` | Create |
| `.github/workflows/ci.yml` | Modify — add build validation job |

### Verification

```bash
# Verify YAML syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yml'))" 2>/dev/null || echo "yaml module not available, visual check OK"
# Verify existing CI still parses
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" 2>/dev/null || echo "yaml module not available, visual check OK"
```

---

## Track D — CIM Library Prep (`causal-inference-marketing`)

**Repo:** `/Users/robertwelborn/Projects/causal-inference-marketing`

The publishable library lives at `libs/causal_inference/`. It has its own `pyproject.toml`,
test suite (88 files), and `py.typed` marker. We publish from this inner directory.

### Task 1 — Fix inner library pyproject.toml

Edit `libs/causal_inference/pyproject.toml`:

**Changes required:**

1. **Name:** Change from `causal-inference` (taken on PyPI) to `causal-inference-marketing`

2. **Version:** Set to `0.1.0` (reconcile with root)

3. **Description:** Add a proper one-liner:
   ```
   "Causal inference library for marketing analytics — AIPW, TMLE, G-computation, IPW, causal discovery, and 30+ estimators."
   ```

4. **README:** Add `readme = "README.md"` (create a lib-specific README, see Task 2)

5. **License:** Ensure `license = "MIT"` and `license-files = ["../../LICENSE"]`
   (or copy LICENSE into the inner lib directory)

6. **Authors:** Add email

7. **Classifiers:** Add full set:
   ```toml
   classifiers = [
       "Development Status :: 3 - Alpha",
       "Intended Audience :: Science/Research",
       "License :: OSI Approved :: MIT License",
       "Programming Language :: Python :: 3",
       "Programming Language :: Python :: 3.11",
       "Programming Language :: Python :: 3.12",
       "Programming Language :: Python :: 3.13",
       "Topic :: Scientific/Engineering",
       "Topic :: Scientific/Engineering :: Artificial Intelligence",
   ]
   ```

8. **Dependencies:** Move heavy dependencies to optional extras:

   **Core (keep as required):**
   ```toml
   dependencies = [
       "numpy>=1.24.0",
       "pandas>=2.0.0",
       "scipy>=1.11.0",
       "statsmodels>=0.14.0",
       "scikit-learn>=1.3.0",
       "pydantic>=2.0.0",
       "networkx>=3.1",
       "matplotlib>=3.7.0",
   ]
   ```

   **Optional (new extras):**
   ```toml
   [project.optional-dependencies]
   bayesian = ["pymc>=5.12.0", "arviz>=0.17.1"]
   optimization = ["cvxpy>=1.4.0"]
   ml = ["lightgbm>=4.0.0", "joblib>=1.3.0", "shap>=0.44.0"]
   all = ["causal-inference-marketing[bayesian,optimization,ml]"]
   dev = [
       "pytest>=7.0",
       "pytest-cov>=4.0",
       "ruff>=0.4.0",
       "mypy>=1.8.0",
       "hypothesis>=6.0.0",
   ]
   ```

   **Remove from core deps (only used by services/shared, not the library):**
   - `fastapi`, `uvicorn` — API service only
   - `sqlalchemy`, `alembic`, `asyncpg`, `aiosqlite` — database only
   - `prometheus-client` — observability only
   - `hypothesis` — testing only (move to dev)

9. **URLs:**
   ```toml
   [project.urls]
   Homepage = "https://github.com/datablogin/causal-inference-marketing"
   Repository = "https://github.com/datablogin/causal-inference-marketing"
   Issues = "https://github.com/datablogin/causal-inference-marketing/issues"
   ```

10. **Build config:** Ensure only the library ships:
    ```toml
    [tool.hatch.build.targets.wheel]
    packages = ["causal_inference"]

    [tool.hatch.build.targets.sdist]
    exclude = ["tests", "examples"]
    ```

### Task 2 — Create library-specific README

Create `libs/causal_inference/README.md` — a focused README for the PyPI page.
Pull content from the root README but focus on the library API, not services.

Key sections:
1. **Title + one-liner**
2. **Install:** `pip install causal-inference-marketing`
3. **Quick start** (3-4 code examples: AIPW, IPW, causal discovery)
4. **Features list** (estimators, discovery algorithms, diagnostics)
5. **Note:** Import name is `causal_inference` (different from PyPI name).
   Do not install alongside the unrelated `causal-inference` package.

### Task 3 — Copy LICENSE into inner lib

Copy the LICENSE file to `libs/causal_inference/LICENSE` so it ships with the sdist.

### Task 4 — Verify build

```bash
cd libs/causal_inference
uv build
unzip -l dist/*.whl | head -40
```

### Files to create/modify

| File | Action |
|------|--------|
| `libs/causal_inference/pyproject.toml` | Modify — name, version, deps, metadata |
| `libs/causal_inference/README.md` | Create — PyPI-focused README |
| `libs/causal_inference/LICENSE` | Create (copy from root) |

### Verification

```bash
cd libs/causal_inference
uv sync --extra dev
uv run pytest tests/ -x -q -m "not slow"
uv run ruff check causal_inference/ && uv run ruff format --check causal_inference/
uv build
```

---

## Track E — CIM Cleanup + Workflow (`causal-inference-marketing`)

**Repo:** `/Users/robertwelborn/Projects/causal-inference-marketing`

### Task 1 — Remove .bak files

Delete all backup files from the repository:

```
libs/causal_inference/causal_inference/core/base.py.bak
libs/causal_inference/causal_inference/core/bootstrap.py.bak
libs/causal_inference/causal_inference/discovery/base.py.bak
libs/causal_inference/causal_inference/ml/super_learner.py.bak
```

Stage the deletions with `git rm`.

### Task 2 — Reconcile version numbers

The root `pyproject.toml` says `version = "0.1.0"` and the inner lib says
`version = "0.2.0"`. Set both to `0.1.0` since this is the first PyPI release.

Also reconcile the Python version requirement:
- Root: `requires-python = ">=3.9"`
- Inner: `requires-python = ">=3.11"`

Keep both as-is (they serve different purposes — root for local dev, inner for published lib).
But if any library code actually requires 3.11+ features (like `Self` type or `ExceptionGroup`),
document why in a code comment.

### Task 3 — Create publish workflow

Create `.github/workflows/publish.yml` for the inner library:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: libs/causal_inference
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: libs/causal_inference/dist/

  test-publish:
    needs: build
    runs-on: ubuntu-latest
    environment: testpypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish:
    needs: test-publish
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

### Task 4 — Add build check to CI

If `ci-fast.yml` exists, add a build validation job that builds the inner library:

```yaml
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: libs/causal_inference
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv python install 3.12
      - run: uv build
```

### Files to create/modify

| File | Action |
|------|--------|
| `libs/causal_inference/causal_inference/core/base.py.bak` | Delete |
| `libs/causal_inference/causal_inference/core/bootstrap.py.bak` | Delete |
| `libs/causal_inference/causal_inference/discovery/base.py.bak` | Delete |
| `libs/causal_inference/causal_inference/ml/super_learner.py.bak` | Delete |
| `libs/causal_inference/pyproject.toml` | Modify — version to 0.1.0 |
| `.github/workflows/publish.yml` | Create |
| `.github/workflows/ci-fast.yml` | Modify — add build job |

### Verification

```bash
cd libs/causal_inference
uv build
# Verify no .bak files in the built wheel
unzip -l dist/*.whl | grep -i bak  # should be empty
```

---

## Sprint 11a — Merge Order

All 5 tracks are non-overlapping:

- **Tracks A, B, C** are in `causal-optimizer` and touch different files:
  - A: LICENSE, py.typed, CHANGELOG, pyproject.toml (metadata fields)
  - B: pyproject.toml (deps section), README.md
  - C: .github/workflows/

  **Potential conflict:** Tracks A and B both modify `pyproject.toml`. Track A adds
  `license-files`, author email, hatch build, and URLs. Track B removes the git dep
  and `allow-direct-references`. These are different sections, so merge either order
  and the second will need a trivial rebase.

- **Tracks D, E** are in `causal-inference-marketing` and touch different files:
  - D: inner lib pyproject.toml (major rewrite), README, LICENSE
  - E: .bak deletions, publish workflow, CI

  **Potential conflict:** Both touch `libs/causal_inference/pyproject.toml` (D rewrites
  it, E changes version). Merge D first, then E rebases trivially.

---

## Sprint 11b — Reconnect + Publish Prep

**Depends on:** All Sprint 11a PRs merged to both repos.

**Single track, single agent.**

**Branch:** `sprint-11/pypi-reconnect` (in `causal-optimizer`)

### Task 1 — Fix import name mismatches

In `causal-optimizer`, three imports reference names that don't exist in CIM:

| File | Current import | Correct import |
|------|---------------|----------------|
| `causal_optimizer/estimator/effects.py` | `from causal_inference.estimators.aipw import AIPW` | `from causal_inference.estimators.aipw import AIPWEstimator as AIPW` |
| `causal_optimizer/discovery/graph_learner.py` | `from causal_inference.discovery import NOTEARS` | `from causal_inference.discovery import NOTEARSAlgorithm as NOTEARS` |
| `causal_optimizer/discovery/graph_learner.py` | `from causal_inference.discovery import PCAlgorithm` | (correct, no change) |

**Important:** These are all inside try/except blocks for graceful degradation. The fix
is to change the import name while keeping the local alias. This way the rest of the code
that references `AIPW` and `NOTEARS` doesn't change.

### Task 2 — Reconnect CIM as PyPI dependency

Once `causal-inference-marketing` is published to PyPI, update `pyproject.toml`:

```toml
causal = [
    "causal-inference-marketing>=0.1.0",
    "dowhy>=0.11",
]
```

**Note:** Do NOT re-add `allow-direct-references`. The dependency is now a normal PyPI
package.

If CIM is not yet on PyPI when this task runs, use a version specifier anyway and note
in the PR that it will resolve once CIM is published. The `causal` extra is optional,
so the base install works regardless.

### Task 3 — Verify cross-package integration

```bash
# Install causal-optimizer with causal extra (once CIM is on PyPI/TestPyPI)
pip install causal-optimizer[causal]

# Or install from local if CIM isn't published yet
pip install -e /Users/robertwelborn/Projects/causal-inference-marketing/libs/causal_inference
pip install -e .[causal]

# Verify imports work
python -c "
from causal_inference.estimators.aipw import AIPWEstimator
from causal_inference.discovery import PCAlgorithm, NOTEARSAlgorithm
print('All imports successful')
"

# Run tests
uv run pytest -m "not slow" -x -q
```

### Files to modify

| File | Action |
|------|--------|
| `causal_optimizer/estimator/effects.py` | Modify — fix AIPW import |
| `causal_optimizer/discovery/graph_learner.py` | Modify — fix NOTEARS import |
| `pyproject.toml` | Modify — reconnect CIM dependency |

### Verification

```bash
uv run pytest -m "not slow" -x -q
uv run ruff check . && uv run ruff format --check .
uv run mypy causal_optimizer/
```

---

## Post-Sprint: Manual Publishing Steps

These require human action (repo settings, PyPI accounts) — not automatable by agents:

### 1. PyPI account setup
- Create accounts on [pypi.org](https://pypi.org) and [test.pypi.org](https://test.pypi.org)
- Enable 2FA on both accounts

### 2. Configure trusted publishers

On **test.pypi.org** and **pypi.org**, add pending publishers:

**For causal-optimizer:**
- Owner: `datablogin`
- Repository: `causal-optimizer`
- Workflow: `publish.yml`
- Environment: `testpypi` / `pypi`

**For causal-inference-marketing:**
- Owner: `datablogin`
- Repository: `causal-inference-marketing`
- Workflow: `publish.yml`
- Environment: `testpypi` / `pypi`

### 3. Create GitHub environments

In each repo's Settings > Environments:
- Create `testpypi` environment
- Create `pypi` environment (add required reviewer for production safety)

### 4. Publish order

1. **CIM first** — Create a GitHub release `v0.1.0` on `causal-inference-marketing`.
   The publish workflow fires automatically. Verify on test.pypi.org, then it proceeds
   to production pypi.org.

2. **causal-optimizer second** — After CIM is live on PyPI, create release `v0.1.0`
   on `causal-optimizer`. Same workflow.

3. **Verify end-to-end:**
   ```bash
   pip install causal-optimizer[all]
   python -c "from causal_optimizer import ExperimentEngine; print('Success')"
   ```

---

## Definition of Done

### Sprint 11a
- [ ] causal-optimizer has LICENSE, py.typed, CHANGELOG
- [ ] causal-optimizer pyproject.toml has no git deps, has license-files, author email, build config, URLs
- [ ] causal-optimizer has publish.yml workflow + CI build validation
- [ ] causal-optimizer builds cleanly with `uv build`
- [ ] CIM inner lib pyproject.toml is PyPI-ready (name, deps, metadata)
- [ ] CIM .bak files removed
- [ ] CIM has publish.yml workflow + CI build validation
- [ ] CIM inner lib builds cleanly with `uv build`
- [ ] All tests pass in both repos

### Sprint 11b
- [ ] Import mismatches fixed (AIPW, NOTEARS)
- [ ] CIM dependency reconnected as PyPI version specifier
- [ ] All tests pass
- [ ] Both packages ready for `gh release create v0.1.0`
