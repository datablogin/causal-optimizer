# Sprint 19 — Harder Counterfactual Benchmark Variants

## Overview

Sprint 18's counterfactual benchmark was valid but "easy" — all strategies converged to
near-optimal at B80. This deliverable adds two harder variants designed to create realistic
conditions where causal graph knowledge provides a structural advantage.

## Variant 1: High-Dimensional Noise (`high_noise`)

**Mechanism:** Adds 10 irrelevant nuisance dimensions (`noise_var_0` through `noise_var_9`,
continuous [0, 1]) to the search space. The 3 true causal parents (temperature threshold,
hour start, hour end) are unchanged. Total search space: 15 dimensions.

**Why causal should help:** The causal graph excludes nuisance variables as ancestors of the
objective. Causal guidance focuses on 3 real dimensions; surrogate_only must search 15. The
advantage should grow at lower budgets where sample efficiency matters most.

**Oracle statistics (480-row synthetic covariates, seed=123):**

- Oracle treat rate: ~32% (within required 10-40% range)
- Nuisance variables confirmed zero-effect (permutation test: oracle unchanged)
- Search space dimensionality: 15 (>= 13 required)

**Smoke results (budget=3, seed=0):**

| Strategy | Policy Value | Regret |
|----------|-------------|--------|
| random | finite | >= 0 |
| surrogate_only | finite | >= 0 |
| causal | finite | >= 0 |

All strategies complete without crashes at budget=3.

## Variant 2: Confounded Treatment Assignment (`confounded`)

**Mechanism:** Hidden confounder "grid stress" ~ Beta(2, 5) affects both:
1. Treatment propensity: logit(p) += 1.5 * (stress - 0.5)
2. Base load: Y(0) += 500 * stress

This creates Simpson's paradox: treated hours have systematically higher base load (due
to grid stress), making the apparent treatment benefit larger than the true causal effect.

**Why causal should help:** The causal graph includes a bidirected edge
(`treat_temp_threshold` <-> `objective`) marking the confounding. POMIS-aware search
recognizes the hidden common cause and avoids chasing the biased outcome surface.

**Oracle statistics (480-row synthetic covariates, seed=123):**

- Oracle treat rate: ~32% (within required 10-40% range)
- Naive vs oracle disagreement: >5% (confirmed Simpson's paradox effect)
- Naive ATE bias: >10% relative to true ATE
- Bidirected edges in graph: 1 (marking the confounder)

**Smoke results (budget=3, seed=0):**

| Strategy | Policy Value | Regret |
|----------|-------------|--------|
| random | finite | >= 0 |
| surrogate_only | finite | >= 0 |
| causal | finite | >= 0 |

All strategies complete without crashes at budget=3.

## Runner Script

Updated `scripts/counterfactual_benchmark.py` with `--variant` flag:
- `--variant base` (default): original 5-dimensional benchmark
- `--variant high_noise`: 15-dimensional with nuisance variables
- `--variant confounded`: Simpson's paradox from hidden grid stress

Backward compatible — existing invocations produce identical results.

## Test Coverage

16 tests in `tests/unit/test_counterfactual_variants.py`:

| Test | Status |
|------|--------|
| Oracle treat rate 10-40% (high noise) | PASS |
| Nuisance vars zero effect (permutation) | PASS |
| Search space >= 13 dimensions | PASS |
| Graph excludes nuisance ancestors | PASS |
| Oracle differs from naive (confounded) | PASS |
| Confounder creates bias (>10% relative) | PASS |
| Oracle treat rate 10-40% (confounded) | PASS |
| Graph has bidirected edge | PASS |
| Smoke: high_noise random/surrogate/causal | PASS (3) |
| Smoke: confounded random/surrogate/causal | PASS (3) |
| Reproducibility: same seed same result | PASS (2) |

## Full Benchmark Results

**Not yet available.** Full benchmark runs (budgets 20,40,80 × seeds 0,1,2,3,4 × 3
strategies) were started but exceeded the 10-minute timeout per variant. These should be
run locally with extended timeout to get regret comparison numbers.

Recommended command:
```bash
uv run python scripts/counterfactual_benchmark.py \
  --data-path <path>/ercot_north_c_dfw_2022_2024.parquet \
  --variant high_noise --budgets 20,40,80 --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output artifacts/counterfactual_high_noise_results.json
```

## Assessment

Both variants are structurally valid positive controls:

1. **High noise** creates a clear dimensionality advantage for causal guidance (3 vs 15
   dimensions). This should separate strategies at low budget.

2. **Confounded** creates a bias trap where naive estimation overestimates treatment
   benefit. This should separate strategies that can handle confounding from those that
   cannot.

Whether these variants actually separate `causal` from `surrogate_only` depends on the
full benchmark results (pending). The structural mechanisms are sound — the question is
whether the optimizer code exploits them effectively.

## Limitations

- Full benchmark runs not completed (timeout) — need local execution
- Reduced grid (budgets 20,40, seeds 0,1,2) recommended if full grid is too slow
- Confounding strength (grid stress multiplier = 500, logit shift = 1.5) may need tuning
  based on full results
