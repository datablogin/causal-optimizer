# Sprint 28 Optimizer-Path Provenance Report

## Metadata

- **Date**: 2026-04-09
- **Sprint**: 28 (Optimizer-Path Provenance)
- **Issue**: #146
- **Branch**: `sprint-28/optimizer-path-provenance`
- **Base commit**: `8a48966` (Sprint 27 crossover scorecard merged to main)

## 1. What Was Added

### 1a. New Function: `detect_optimizer_path()`

Added to `causal_optimizer/benchmarks/provenance.py`.  Probes the runtime
environment to determine which optimizer backend is available.

Returns a JSON-serializable dict with four fields:

| Field | Type | Values |
|-------|------|--------|
| `optimizer_path` | str | `"ax_botorch"` or `"rf_fallback"` |
| `ax_available` | bool | Whether `ax` is importable |
| `botorch_available` | bool | Whether `botorch` is importable |
| `fallback_reason` | str or null | `null` when Ax/BoTorch available; describes missing packages otherwise |

### 1b. Integration Into `collect_provenance()`

The existing `collect_provenance()` function now includes an
`optimizer_path` key in its output.  Since all benchmark CLI scripts
(counterfactual, null, dose-response) already call `collect_provenance()`,
the new field appears automatically in every benchmark JSON artifact
without any script changes.

### 1c. Example Output

When Ax/BoTorch are installed:
```json
"optimizer_path": {
    "optimizer_path": "ax_botorch",
    "ax_available": true,
    "botorch_available": true,
    "fallback_reason": null
}
```

When Ax/BoTorch are not installed:
```json
"optimizer_path": {
    "optimizer_path": "rf_fallback",
    "ax_available": false,
    "botorch_available": false,
    "fallback_reason": "ax-platform not installed; botorch not installed"
}
```

## 2. How a Run Distinguishes Ax/BoTorch From RF Fallback

The detection logic is simple and mechanical:

1. Try `import ax` — if it fails, record `ax_available: false`
2. Try `import botorch` — if it fails, record `botorch_available: false`
3. If both succeed: `optimizer_path: "ax_botorch"`, `fallback_reason: null`
4. If either fails: `optimizer_path: "rf_fallback"`, `fallback_reason` lists the missing packages

This matches the runtime behavior in `suggest.py`:
`_suggest_optimization()` tries `_suggest_bayesian()` first, catches
`ImportError`, and falls back to `_suggest_surrogate()`.

## 3. How Fallback Reason Is Represented

The `fallback_reason` field is:
- `null` when no fallback occurred (both packages available)
- A semicolon-separated string listing each missing package when fallback
  occurred (e.g., `"ax-platform not installed; botorch not installed"`)

This captures the most common fallback scenarios:
- Only `ax-platform` missing
- Only `botorch` missing
- Both missing (typical in lightweight CI or worktree environments)

## 4. What Ambiguity This Removes

### Removed

- **Backend inference from narrative**: Prior scorecards had to state
  "this run used RF fallback" as a caveat.  Now every JSON artifact
  self-documents its optimizer path.
- **Cross-run comparison ambiguity**: When comparing Sprint 25 (Ax/BoTorch)
  to Sprint 27 regression gate (RF fallback), readers had to check the
  report text.  Now they can programmatically filter artifacts by
  `optimizer_path.optimizer_path == "ax_botorch"`.
- **Scorecard hygiene**: Future scorecards can auto-verify that compared
  runs used the same backend, or flag when they didn't.

### Remaining

- **Within-run path switches**: The provenance captures what was available
  at the start of the run, not per-step.  If a package becomes unavailable
  mid-run (unlikely but possible in edge cases), the artifact would not
  reflect that.
- **GP fallback within Ax**: When Ax/BoTorch are available, the engine
  may still fall back from causal GP to standard Bayesian optimization
  within `_suggest_optimization()`.  This internal fallback is not captured
  in the provenance (it is logged at runtime, not persisted in artifacts).
- **Version-specific behavior**: Two runs with `optimizer_path: "ax_botorch"`
  but different Ax/BoTorch versions may produce different results.  The
  `package_versions` field (already present) covers this, but the
  `optimizer_path` field alone does not.

## 5. Scope of Change

- **1 function added**: `detect_optimizer_path()` in `provenance.py`
- **1 function modified**: `collect_provenance()` — added `optimizer_path` key
- **10 tests added**: `tests/unit/test_optimizer_path_provenance.py`
- **0 script changes**: all benchmark CLIs inherit the new field automatically
- **Backward compatible**: existing JSON readers that don't expect
  `optimizer_path` will silently ignore it
