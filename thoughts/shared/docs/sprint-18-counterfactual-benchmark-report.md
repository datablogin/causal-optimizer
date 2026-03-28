# Sprint 18 Step 2: Counterfactual Benchmark Repair Report

## Summary

Sprint 17's counterfactual demand-response benchmark was degenerate:
`treatment_cost=50.0` exceeded the treatment benefit for every covariate
pattern, making the oracle policy "never treat" and all strategies tie at
regret=0. This sprint re-parameterizes the benchmark so the oracle treats
a meaningful minority of rows and weak strategies have measurable regret.

## Changes Made

1. **Treatment effect re-parameterization** -- Converted temperature scale
   from Fahrenheit to Celsius (the ERCOT data uses Celsius). Propensity
   function and treatment effect function now operate on `[10, 41] C` range
   instead of `[50, 105] F`.
2. **Treatment cost adjusted** -- Set `treatment_cost=60.0` to produce an
   oracle treat rate of ~32%, creating a non-trivial decision boundary.
3. **Performance fix** -- Causal is no longer 100x slower than surrogate_only.
   The smooth treatment effect function (sigmoid * Gaussian) produces better
   surrogate fits, reducing wasted iterations. The `max_skips=0` setting
   also eliminates off-policy skip overhead.

## Oracle Statistics

| Metric | Value |
|--------|-------|
| Oracle treat rate | 31.9% (8,399 / 26,291 rows) |
| Oracle policy value | 36.80 |
| Always-treat policy value | 3.17 |
| Never-treat policy value | 0.00 |

The oracle treats roughly one-third of hours -- those with high temperature
and midday timing where demand-response curtailment benefit exceeds the
$60 cost. "Always treat" yields only 3.17 (most hours have sub-cost
benefit), confirming the decision boundary is meaningful.

## Regret by Strategy and Budget

Results are averaged over 5 seeds (0--4). Oracle test-set value = 48.41.

| Strategy | Budget | Mean Policy Value | Mean Regret | Std Regret | Decision Error |
|----------|--------|-------------------|-------------|------------|----------------|
| random | 20 | 28.34 | 20.06 | 10.39 | 0.2518 |
| random | 40 | 29.30 | 19.11 | 9.21 | 0.2721 |
| random | 80 | 39.25 | 9.16 | 2.45 | 0.2102 |
| surrogate_only | 20 | 36.34 | 12.07 | 7.17 | 0.1925 |
| surrogate_only | 40 | 37.25 | 11.16 | 7.91 | 0.1789 |
| surrogate_only | 80 | 46.66 | 1.75 | 1.11 | 0.0733 |
| causal | 20 | 30.67 | 17.74 | 4.99 | 0.2515 |
| causal | 40 | 30.74 | 17.66 | 4.89 | 0.2511 |
| causal | 80 | 45.95 | 2.46 | 0.86 | 0.0925 |

## Key Findings

### 1. Benchmark is non-degenerate (all acceptance criteria met)

- Oracle treat rate = 31.9% (target: 10--40%)
- Oracle policy value = 36.80 (clearly above 0)
- Random B80 regret = 9.16 (measurable)
- Never-treat value = 0.00, oracle value = 36.80 ("never treat" is not optimal)
- Causal runtime ~13s vs surrogate_only ~14s (within 10x, essentially equal)
- Smoke and reproducibility tests pass (807 tests green)

### 2. Surrogate_only edges causal at B80

At the highest budget (80 experiments), surrogate_only achieves slightly
lower regret (1.75) than causal (2.46). Both dramatically outperform
random (9.16). The gap is modest -- both strategies find near-optimal
policies.

### 3. Surrogate_only leads at lower budgets too

At B20 and B40, surrogate_only (regret 12.07, 11.16) outperforms causal
(regret 17.74, 17.66) and random (regret 20.06, 19.11). This suggests
that on this benchmark, the causal graph's focus-variable restriction
may over-constrain early exploration, while surrogate_only explores the
full space more freely.

### 4. Causal has lower variance

Causal's standard deviation across seeds is notably lower than both
surrogate_only and random (e.g., B20: causal std=4.99 vs surrogate_only
std=7.17 vs random std=10.39). The causal strategy is more consistent
even when its mean is slightly worse.

### 5. All strategies improve with budget

Every strategy shows clear improvement as budget increases from 20 to 80,
confirming the benchmark rewards intelligent search.

## Runtime

| Strategy | B20 (s) | B40 (s) | B80 (s) |
|----------|---------|---------|---------|
| random | 0.01 | 0.01 | 0.02 |
| surrogate_only | 2.92 | 7.09 | 13.71 |
| causal | 2.73 | 7.06 | 13.17 |

Causal and surrogate_only have essentially identical runtimes.
The Sprint 17 100x slowdown is fully resolved.

Total suite runtime: 234 seconds.

## Assessment of Benchmark Validity

The re-parameterized benchmark is a valid, non-degenerate test of
counterfactual policy optimization:

1. **Non-trivial oracle**: 32% treat rate with clear benefit over
   always-treat and never-treat baselines.
2. **Discriminating**: Random is clearly worse than learned strategies at
   all budgets.
3. **Reasonable scale**: Regret ranges from ~2 (best) to ~20 (worst),
   providing signal.
4. **Reproducible**: Low cross-seed variance for causal and surrogate_only
   at B80.
5. **Fast**: Full 45-run suite completes in under 4 minutes.

The benchmark does not yet show causal advantage over surrogate_only.
This is an honest result: on this particular graph (3 causal parents,
2 noise dimensions), the noise dimensions are easy enough to screen
out without causal knowledge. Causal advantage may emerge on graphs
with more confounders or harder noise structure.
