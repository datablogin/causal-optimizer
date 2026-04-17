# Sprint 33 Criteo Benchmark Report (Run 1: Degenerate Surface)

**Date:** 2026-04-17
**Issue:** #179
**PR:** #180
**Dataset:** Criteo Uplift v2.1 (13,979,592 rows), 1M-row fixed-seed subsample (seed 20260417)
**Backend:** Ax/BoTorch (primary, not RF fallback)
**Predecessor:** Sprint 31 Hillstrom benchmark (PR #176), Sprint 32 contract (#177)

## Summary

This is Run 1 of the Criteo uplift benchmark, evaluating the causal optimizer on
a 1M-row subsample of the Criteo Uplift v2.1 dataset under a degenerate
2-variable search surface. The benchmark evaluates three strategies (random,
surrogate_only, causal) at budgets 20, 40, and 80 with 10 seeds each, plus a
permuted-outcome null-control pass.

**Top-line result:** Both surrogate_only and causal converge to exactly the same
treat-everyone corner (eligibility_threshold=0.0, treatment_budget_pct=1.0) on
all 10 seeds at all 3 budgets, producing identical policy values of 0.048485.
Random is slightly lower (mean 0.0480--0.0482) due to not always finding the
optimal corner. The degenerate surface offers zero heterogeneity, so both
optimized strategies trivially converge.

**Run 1 verdict: NEAR-PARITY (exact tie).** Per Sprint 32 contract Section 5f,
the mandatory Run 2 (heterogeneous surface with synthesized f0 tertile segments)
is required before publishing the final Criteo verdict.

## Data Provenance

| Property | Value |
|----------|-------|
| Source dataset | Criteo Uplift v2.1 (13,979,592 rows) |
| Subsample | 1,000,000 rows, fixed seed 20260417 |
| SHA-256 (subsample) | `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc` |
| File size | 311,422,618 bytes |
| Treatment ratio | 0.8500 (85:15) |
| Visit rate | 0.0469 (4.69%) |
| Conversion rate | 0.002882 |
| Control count | 150,018 |
| Treated count | 849,982 |

## Configuration

- **Backend:** Ax/BoTorch (primary)
- **Propensity:** constant 0.85
- **Strategies:** random, surrogate_only, causal
- **Budgets:** 20, 40, 80
- **Seeds:** 0--9 (10 seeds per cell)
- **Active variables:** eligibility_threshold, treatment_budget_pct (2 variables)
- **Frozen params:** email_share=1.0, social_share_of_remainder=0.0, min_propensity_clip=0.01, regularization=1.0
- **Projected prior graph:** 5 edges
- **Objective:** policy_value (maximize)
- **Total runs:** 150 (90 real + 60 null-control)
- **Suite runtime:** 3565.7 seconds (59.4 minutes)

## Propensity Gate

**PASSED.** Maximum deviation 0.79pp from the expected constant propensity of
0.85 (threshold: 2pp). All f0 deciles fall within tolerance.

## Verdicts

| Budget | Causal vs Surrogate-Only | p-value (two-sided MWU) | Verdict | Direction |
|--------|--------------------------|-------------------------|---------|-----------|
| B20 | identical all 10 seeds | 1.000 | Near-parity (exact tie) | -- |
| B40 | identical all 10 seeds | 1.000 | Near-parity (exact tie) | -- |
| B80 | identical all 10 seeds | 1.000 | Near-parity (exact tie) | -- |

| Budget | Causal vs Random | p-value (two-sided MWU) | Verdict | Direction |
|--------|------------------|-------------------------|---------|-----------|
| B20 | causal wins all seeds | 0.0001 | Certified | causal > random |
| B40 | causal wins all seeds | 0.0001 | Certified | causal > random |
| B80 | causal wins all seeds | 0.0001 | Certified | causal > random |

| Budget | Surrogate-Only vs Random | p-value (two-sided MWU) | Verdict | Direction |
|--------|--------------------------|-------------------------|---------|-----------|
| B20 | s.o. wins all seeds | 0.0001 | Certified | s.o. > random |
| B40 | s.o. wins all seeds | 0.0001 | Certified | s.o. > random |
| B80 | s.o. wins all seeds | 0.0001 | Certified | s.o. > random |

Abbreviation: **s.o.** = `surrogate_only` throughout this report.

**Overall Run 1 verdict: NEAR-PARITY (exact tie).** Both optimized strategies
converge to exactly the same corner on every seed at every budget. The surface
is degenerate and uninformative about the relative value of causal guidance.

## Summary Statistics

| Strategy | Budget | Mean | Std (ddof=0) |
|----------|--------|------|--------------|
| random | 20 | 0.047985 | 0.000322 |
| surrogate_only | 20 | 0.048485 | 0.000000 |
| causal | 20 | 0.048485 | 0.000000 |
| random | 40 | 0.048106 | 0.000206 |
| surrogate_only | 40 | 0.048485 | 0.000000 |
| causal | 40 | 0.048485 | 0.000000 |
| random | 80 | 0.048176 | 0.000176 |
| surrogate_only | 80 | 0.048485 | 0.000000 |
| causal | 80 | 0.048485 | 0.000000 |

The zero standard deviation for surrogate_only and causal confirms that all 10
seeds converge to the identical optimal policy at all three budgets.

## Per-Seed Detail Table (B80)

All three budgets produce the same pattern; B80 is shown as representative.

| Seed | random | surrogate_only | causal | Winner |
|------|--------|----------------|--------|--------|
| 0 | 0.048480 | 0.048485 | 0.048485 | tie (s.o. = causal) |
| 1 | 0.047855 | 0.048485 | 0.048485 | tie (s.o. = causal) |
| 2 | 0.048175 | 0.048485 | 0.048485 | tie (s.o. = causal) |
| 3 | 0.048220 | 0.048485 | 0.048485 | tie (s.o. = causal) |
| 4 | 0.048310 | 0.048485 | 0.048485 | tie (s.o. = causal) |
| 5 | 0.048100 | 0.048485 | 0.048485 | tie (s.o. = causal) |
| 6 | 0.048050 | 0.048485 | 0.048485 | tie (s.o. = causal) |
| 7 | 0.048200 | 0.048485 | 0.048485 | tie (s.o. = causal) |
| 8 | 0.048170 | 0.048485 | 0.048485 | tie (s.o. = causal) |
| 9 | 0.048200 | 0.048485 | 0.048485 | tie (s.o. = causal) |

The optimal corner is eligibility_threshold=0.0, treatment_budget_pct=1.0
(treat everyone). On the degenerate surface, where segment is omitted and
channel is constant, all observations receive uniform uplift scores of 0.5.
Any policy with eligibility_threshold below 0.5 and treatment_budget_pct at 1.0
produces the same outcome. Random search occasionally misses this corner,
leading to slightly lower values on some seeds.

## ESS Diagnostics

| Strategy | ESS | weight_cv | max_weight |
|----------|-----|-----------|------------|
| surrogate_only | 849,982 | 0.000 | 1.176 |
| causal | 849,982 | 0.000 | 1.176 |
| random (median) | 751K--771K | ~0.3 | 6.667 |

- **surrogate_only and causal:** ESS equals the full treated count (849,982).
  Both strategies found the treat-everyone policy, where all observations are
  policy-matched on the treated arm. The maximum weight of 1.176 = 1/0.85,
  corresponding to the inverse propensity weight for treated observations.
- **random:** ESS median 751K--771K. Random policies sometimes treat fewer
  users, creating control-arm matched observations with weight 1/(1-0.85) =
  6.667.
- **No zero-support events** on any run across all strategies and seeds.
- All ESS values vastly exceed the 100-observation threshold. The IPS stack is
  stable at 85:15 imbalance.

## Null Control

| Cell | Mean Policy Value | Null Baseline mu | 5% Band |
|------|-------------------|------------------|---------|
| B20 random | 0.047108 | 0.046879 | [0.044535, 0.049223] |
| B20 s.o. | 0.047166 | 0.046879 | [0.044535, 0.049223] |
| B20 causal | 0.047160 | 0.046879 | [0.044535, 0.049223] |
| B40 random | 0.047140 | 0.046879 | [0.044535, 0.049223] |
| B40 s.o. | 0.047191 | 0.046879 | [0.044535, 0.049223] |
| B40 causal | 0.047192 | 0.046879 | [0.044535, 0.049223] |

Null baseline mu = 0.046879 (raw visit rate on the unshuffled frame).
5% tolerance band: [0.044535, 0.049223].

**All 6 strategy-budget cells pass within the 5% band.** Null control: PASS
(clean).

## Interpretation

1. **The degenerate surface collapses to a single corner.** With only two active
   variables (eligibility_threshold, treatment_budget_pct) and no segment
   heterogeneity, the response surface is flat everywhere except at the boundary.
   The optimal policy is trivially "treat everyone" because all observations have
   identical uplift scores. Both optimized strategies find this corner on every
   seed, leaving zero room for differentiation.

2. **The engine runs cleanly on a 1M-row, 85:15 imbalanced, binary-outcome
   Criteo dataset under Ax/BoTorch.** This is the first Ax-primary marketing
   benchmark. Hillstrom ran on RF fallback; Criteo Run 1 confirms the full
   Ax/BoTorch backend is operational on real marketing log data at scale.

3. **The IPS stack is stable at 85:15 imbalance.** ESS of 849,982 with zero
   weight coefficient of variation and no zero-support events demonstrates that
   the inverse propensity scoring pipeline handles heavy treatment imbalance
   without variance pathologies.

4. **The propensity gate confirms constant propensity is valid.** All f0 deciles
   fall within 0.79pp of the expected 0.85, well inside the 2pp threshold.

5. **The null control passes cleanly.** All 6 cells fall within the 5% tolerance
   band around the raw visit rate baseline. This validates the estimator on
   permuted outcomes for this dataset and surface configuration.

6. **This is a surface problem, not an engine problem.** The degenerate surface
   offers no structure for the causal graph to exploit. The tie is the expected
   result under the contract, not a failure of the causal path.

## What This Shows

1. The engine runs cleanly on a 1M-row, 85:15 imbalanced, binary-outcome Criteo
   dataset under Ax/BoTorch.
2. The IPS stack is stable at 85:15 imbalance (ESS ~850K, no variance
   pathologies).
3. The null control passes cleanly within the 5% band.
4. The propensity gate confirmed constant propensity is valid.
5. The degenerate 2-variable surface collapses to a single corner, as the
   contract predicted.
6. No strategy can differentiate on a surface with zero heterogeneity.

## What This Does NOT Show

1. Whether causal guidance adds value when the search surface has structure
   (requires Run 2 with synthesized f0 tertile segments).
2. Whether the result changes under a wider search space (3+ active variables).
3. Whether auto-discovered graphs would help on Criteo.
4. Absolute incrementality levels (Criteo is non-uniformly subsampled).

## Run 2 Status

Run 2 (synthesized f0 tertile segments: "high_value" / "medium" / "low") is
required by the Sprint 32 contract before the final Criteo verdict can be
published. Run 2 introduces genuine uplift heterogeneity across segments,
creating a non-degenerate surface where the causal graph has the opportunity to
provide differential value.

Per contract Section 8e, the combined Criteo verdict depends on the Run 2
outcome:

- If Run 2 shows a causal win: **Conditional Criteo win** (causal guidance adds
  value on structured surfaces but is inert on degenerate ones).
- If Run 2 shows near-parity or surrogate-only advantage: **Criteo near-parity**
  (neither path dominates on binary uplift policy problems under Criteo).

## Reproducibility

- **Subsample seed:** 20260417 (fixed)
- **SHA-256:** `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- **File size:** 311,422,618 bytes (uncompressed subsample)
- **Source:** Criteo Uplift v2.1, available from the Criteo AI Lab

## Attribution

Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2018). "A Large Scale
Benchmark for Uplift Modeling." *AdKDD 2018 Workshop, KDD London.*
