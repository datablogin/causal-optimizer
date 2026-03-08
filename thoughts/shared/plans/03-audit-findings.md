# Codebase Audit Findings

Full audit of what's built, what's stubbed, and what's disconnected.

## Execution Flow: What Actually Happens in `run_loop()`

```
engine.run_loop(30)
  └─ for each experiment:
       1. suggest_next()
            └─ phase == "exploration" (n < 10)?
                 → FactorialDesigner.latin_hypercube(1)     ← WORKS
            └─ phase == "optimization" (10 ≤ n < 50)?
                 → try Ax/BoTorch                           ← IMPORT FAILS (not installed)
                 → except: _suggest_surrogate()             ← RF fallback WORKS
                 → _get_focus_variables() called            ← COMPUTED BUT RESULT IGNORED
            └─ phase == "exploitation" (n ≥ 50)?
                 → perturb best parameters                  ← WORKS

       2. run_experiment(parameters)
            └─ runner.run(parameters)                       ← DOMAIN-SPECIFIC
            └─ _evaluate_status(metrics)
                 └─ if better than best: KEEP               ← GREEDY, NO STATISTICS
                 └─ else: DISCARD

       3. _update_phase()
            └─ hardcoded thresholds: 10, 50                 ← NO FEEDBACK
```

### What's NOT Called

| Module | File | Status |
| --- | --- | --- |
| ScreeningDesigner | designer/screening.py | Built, never called |
| EffectEstimator | estimator/effects.py | Built, never called |
| OffPolicyPredictor | predictor/off_policy.py | Built, never called |
| MAPElites | evolution/map_elites.py | Built, never called |
| SensitivityValidator | validator/sensitivity.py | Built, never called |
| GraphLearner | discovery/graph_learner.py | Built, never called |

**Six fully-implemented modules are orphaned.** This is the #1 finding.

## Test Coverage

```
Tested (12 tests):
  - engine/loop.py         — 5 tests (basic flow, phases, crashes)
  - types.py               — 4 tests (validation, graph, log)
  - designer/factorial.py  — 3 tests (full factorial, LHS, categorical)

Untested (0 tests):
  - designer/screening.py
  - discovery/graph_learner.py
  - estimator/effects.py
  - evolution/map_elites.py
  - optimizer/suggest.py
  - predictor/off_policy.py
  - validator/sensitivity.py
  - domain_adapters/*

Integration tests: empty directory
```

## Domain Adapter Readiness

Both adapters define search spaces and prior causal graphs but raise `NotImplementedError`
on `run_experiment()`. They are templates, not working adapters.

- **MarketingAdapter**: 5 variables, 13-edge causal DAG, descriptors defined
- **MLTrainingAdapter**: 9 variables, 16-edge causal DAG, descriptors defined

## Overall Assessment

- **Feature breadth**: 80% (all major algorithms present)
- **Feature depth**: 40% (many built but not integrated)
- **Test coverage**: ~15% (12 smoke tests)
- **Production readiness**: 30%
- **The gap**: not missing algorithms, missing wiring
