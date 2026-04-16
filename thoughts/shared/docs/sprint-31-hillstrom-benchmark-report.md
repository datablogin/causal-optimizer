# Sprint 31 Hillstrom Benchmark Report

**Date:** 2026-04-16
**Issue:** #170
**Branch:** `sprint-31/hillstrom-full-benchmark`
**Dataset:** MineThatData Hillstrom E-Mail (64,000 rows)
**Backend:** RF surrogate (Ax/BoTorch not available in this environment)

## Summary

This is the first non-energy real-data benchmark for the causal optimizer. The
Hillstrom dataset is a three-arm randomized marketing trial. The benchmark
evaluates two binary slices: primary (Womens E-Mail vs No E-Mail) and pooled
(Any E-Mail vs No E-Mail), across three strategies (random, surrogate_only,
causal) at budgets 20, 40, and 80 with 10 seeds each. A permuted-outcome
null-control pass is included for the primary slice.

**Top-line result:** Surrogate-only outperforms causal at low and medium budgets
on both slices. At B80 on the primary slice, causal shows a mean advantage
(0.8644 vs 0.8293) driven by 3 seeds that discover a high-policy-value region,
but the high variance makes the result statistically non-significant (p=0.817).
At B80 on the pooled slice, surrogate-only reliably finds the optimal corner
(eligibility_threshold=0.0, treatment_budget_pct=1.0) and outperforms causal
at certified significance (p=0.019, two-sided MWU).

## Verdicts

| Slice | Budget | Causal vs Surrogate-Only | p-value (two-sided MWU) | Verdict | Direction |
|-------|--------|--------------------------|-------------------------|---------|-----------|
| Primary | B20 | s.o. wins 8/10 seeds | 0.060 | Trending | s.o. > causal |
| Primary | B40 | s.o. wins 10/10 seeds | 0.0001 | Certified | s.o. > causal |
| Primary | B80 | s.o. wins 7/10 seeds | 0.817 | Near-parity (bimodal) | causal mean > s.o. mean |
| Pooled | B20 | s.o. wins 8/10 seeds | 0.017 | Certified | s.o. > causal |
| Pooled | B40 | s.o. wins 9/10 seeds | 0.002 | Certified | s.o. > causal |
| Pooled | B80 | s.o. wins 7/10 seeds | 0.019 | Certified | s.o. > causal |

Abbreviation: **s.o.** = `surrogate_only` throughout this report.

**Overall Hillstrom verdict: surrogate-only advantage.** The causal path does
not show a consistent advantage over surrogate-only on this dataset under the
RF backend. This is the expected null result for a domain where the projected
prior graph has limited predictive leverage over a pure surrogate approach on a
narrow 3-variable search space.

## Configuration

- **Slices:** primary (Womens E-Mail vs No E-Mail, 42,693 rows, propensity=0.5), pooled (Any E-Mail vs No E-Mail, 64,000 rows, propensity=2/3)
- **Strategies:** random, surrogate_only, causal
- **Budgets:** 20, 40, 80
- **Seeds:** 0--9 (10 seeds per cell)
- **Null control:** primary slice, budgets 20 and 40, permuted outcomes
- **Frozen params:** email_share=1.0, social_share_of_remainder=0.0, min_propensity_clip=0.01
- **Active search space:** eligibility_threshold, regularization, treatment_budget_pct (3 variables)
- **Projected prior graph:** 7 edges over active nodes + intermediates
- **Objective:** policy_value (maximize)
- **Suite runtime:** 1302.8 seconds (21.7 minutes)

## Per-Seed Detail Tables

### Primary Slice (Real)

#### Budget=20

| Seed | random | surrogate_only | causal | Winner |
|------|--------|----------------|--------|--------|
| 0 | 0.8270 | 0.8293 | 0.8288 | s.o. |
| 1 | 0.8267 | 0.8294 | 0.8288 | s.o. |
| 2 | 0.8290 | 0.8293 | 0.8288 | s.o. |
| 3 | 0.8287 | 0.8294 | 0.8288 | s.o. |
| 4 | 0.8291 | 0.8293 | 0.8288 | s.o. |
| 5 | 0.8239 | 0.8294 | 0.8288 | s.o. |
| 6 | 0.8290 | 0.8228 | 0.8240 | causal |
| 7 | 0.8289 | 0.8228 | 0.8231 | causal |
| 8 | 0.8269 | 0.8287 | 0.8228 | s.o. |
| 9 | 0.8268 | 0.8292 | 0.8275 | s.o. |

#### Budget=40

| Seed | random | surrogate_only | causal | Winner |
|------|--------|----------------|--------|--------|
| 0 | 0.8290 | 0.8294 | 0.8288 | s.o. |
| 1 | 0.8288 | 0.8294 | 0.8288 | s.o. |
| 2 | 0.8290 | 0.8294 | 0.8288 | s.o. |
| 3 | 0.8287 | 0.8294 | 0.8288 | s.o. |
| 4 | 0.8291 | 0.8294 | 0.8288 | s.o. |
| 5 | 0.8290 | 0.8294 | 0.8288 | s.o. |
| 6 | 0.8290 | 0.8293 | 0.8289 | s.o. |
| 7 | 0.8289 | 0.8291 | 0.8287 | s.o. |
| 8 | 0.8270 | 0.8293 | 0.8270 | s.o. |
| 9 | 0.8291 | 0.8292 | 0.8275 | s.o. |

#### Budget=80

| Seed | random | surrogate_only | causal | Winner |
|------|--------|----------------|--------|--------|
| 0 | 0.8290 | 0.8294 | 0.8293 | s.o. |
| 1 | 0.8288 | 0.8294 | 0.8293 | s.o. |
| 2 | 0.8293 | 0.8294 | 0.8293 | s.o. |
| 3 | 0.8287 | 0.8294 | 0.8294 | s.o. |
| 4 | 0.8291 | 0.8294 | 0.8294 | s.o. |
| 5 | 0.8290 | 0.8294 | 0.8294 | s.o. |
| 6 | 0.8290 | 0.8293 | 0.8292 | s.o. |
| 7 | 0.8289 | 0.8291 | 0.9255 | causal |
| 8 | 0.8270 | 0.8293 | 0.9474 | causal |
| 9 | 0.8291 | 0.8292 | 0.9656 | causal |

Note: Seeds 7--9 at B80 find a qualitatively different region with
eligibility_threshold=0.0, producing policy values 12--16% above the
plateau where all other runs cluster (~0.829). These seeds have enough
budget to explore past the exploitation phase transition and discover a
corner solution. The surrogate-only path never finds this region because
it converges too tightly on the local plateau.

### Pooled Slice (Real)

#### Budget=20

| Seed | random | surrogate_only | causal | Winner |
|------|--------|----------------|--------|--------|
| 0 | 0.9507 | 0.9599 | 0.9309 | s.o. |
| 1 | 0.9282 | 0.9520 | 0.9229 | s.o. |
| 2 | 0.9232 | 0.8936 | 0.9574 | causal |
| 3 | 0.9095 | 0.9596 | 0.9105 | s.o. |
| 4 | 0.9554 | 0.9373 | 0.9459 | causal |
| 5 | 0.9477 | 0.9596 | 0.9231 | s.o. |
| 6 | 0.9455 | 0.9596 | 0.9446 | s.o. |
| 7 | 0.9447 | 0.9596 | 0.9492 | s.o. |
| 8 | 0.9307 | 0.9520 | 0.9492 | s.o. |
| 9 | 0.9017 | 0.9642 | 0.9492 | s.o. |

#### Budget=40

| Seed | random | surrogate_only | causal | Winner |
|------|--------|----------------|--------|--------|
| 0 | 0.9642 | 0.9643 | 0.9309 | s.o. |
| 1 | 0.9596 | 0.9640 | 0.9229 | s.o. |
| 2 | 0.9232 | 0.8940 | 0.9574 | causal |
| 3 | 0.9506 | 0.9642 | 0.9105 | s.o. |
| 4 | 0.9554 | 0.9642 | 0.9520 | s.o. |
| 5 | 0.9515 | 0.9643 | 0.9231 | s.o. |
| 6 | 0.9455 | 0.9643 | 0.9446 | s.o. |
| 7 | 0.9447 | 0.9643 | 0.9492 | s.o. |
| 8 | 0.9307 | 0.9643 | 0.9492 | s.o. |
| 9 | 0.9489 | 0.9643 | 0.9492 | s.o. |

#### Budget=80

| Seed | random | surrogate_only | causal | Winner |
|------|--------|----------------|--------|--------|
| 0 | 0.9642 | 1.2496 | 1.2496 | tie |
| 1 | 0.9596 | 1.2496 | 0.9571 | s.o. |
| 2 | 0.9521 | 0.9641 | 0.9596 | s.o. |
| 3 | 0.9506 | 0.9642 | 1.2496 | causal |
| 4 | 0.9554 | 0.9642 | 0.9641 | s.o. |
| 5 | 0.9641 | 1.2496 | 1.2496 | tie |
| 6 | 0.9594 | 1.2496 | 0.9595 | s.o. |
| 7 | 0.9535 | 1.2496 | 0.9521 | s.o. |
| 8 | 0.9519 | 1.2496 | 0.9521 | s.o. |
| 9 | 0.9489 | 1.2496 | 0.9521 | s.o. |

Note: The optimal corner (eligibility_threshold=0.0, treatment_budget_pct=1.0)
yields policy_value=1.2496 on the pooled slice, which exceeds the null baseline
of 1.0509. Surrogate-only finds this corner in 7/10 seeds at B80; causal finds
it in only 3/10 seeds (1 win + 2 ties). Seed 2 is a consistently difficult
seed for surrogate-only on the pooled slice (worst s.o. value at B20, B40,
and B80), suggesting a seed-specific initialization that steers the surrogate
away from the optimal corner.

## Summary Statistics

### Primary Slice

| Strategy | Budget | Mean | Std (ddof=0) | Min | Max |
|----------|--------|------|--------------|-----|-----|
| random | 20 | 0.8276 | 0.0016 | 0.8239 | 0.8291 |
| surrogate_only | 20 | 0.8280 | 0.0026 | 0.8228 | 0.8294 |
| causal | 20 | 0.8270 | 0.0025 | 0.8228 | 0.8288 |
| random | 40 | 0.8288 | 0.0006 | 0.8270 | 0.8291 |
| surrogate_only | 40 | 0.8293 | 0.0001 | 0.8291 | 0.8294 |
| causal | 40 | 0.8285 | 0.0006 | 0.8270 | 0.8289 |
| random | 80 | 0.8288 | 0.0006 | 0.8270 | 0.8293 |
| surrogate_only | 80 | 0.8293 | 0.0001 | 0.8291 | 0.8294 |
| causal | 80 | 0.8644 | 0.0543 | 0.8292 | 0.9656 |

### Pooled Slice

| Strategy | Budget | Mean | Std (ddof=0) | Min | Max |
|----------|--------|------|--------------|-----|-----|
| random | 20 | 0.9337 | 0.0173 | 0.9017 | 0.9554 |
| surrogate_only | 20 | 0.9497 | 0.0200 | 0.8936 | 0.9642 |
| causal | 20 | 0.9383 | 0.0145 | 0.9105 | 0.9574 |
| random | 40 | 0.9474 | 0.0118 | 0.9232 | 0.9642 |
| surrogate_only | 40 | 0.9572 | 0.0211 | 0.8940 | 0.9643 |
| causal | 40 | 0.9389 | 0.0149 | 0.9105 | 0.9574 |
| random | 80 | 0.9560 | 0.0052 | 0.9489 | 0.9642 |
| surrogate_only | 80 | 1.1640 | 0.1308 | 0.9641 | 1.2496 |
| causal | 80 | 1.0445 | 0.1343 | 0.9521 | 1.2496 |

## Null Baseline

| Slice | Null baseline mu |
|-------|------------------|
| Primary | 0.8654 |
| Pooled | 1.0509 |

The null baseline is `μ = mean(spend)` on the unshuffled reshaped frame,
computed before any optimization or policy conditioning.

Most runs on the primary slice fall below the null baseline (0.8654), with the
exception of causal B80 seeds 7--9 (0.926--0.966), which discover a higher-value
region. On the pooled slice, surrogate-only B80 reliably exceeds the null
baseline (mean=1.164 vs baseline=1.051).

## Null Control (Primary Slice, Permuted Outcomes)

### Budget=20

| Seed | random | surrogate_only | causal | baseline |
|------|--------|----------------|--------|----------|
| 0 | 0.8368 | 0.8490 | 0.8423 | 0.8654 |
| 1 | 0.8716 | 0.8729 | 0.8729 | 0.8654 |
| 2 | 0.8786 | 0.8806 | 0.8785 | 0.8654 |
| 3 | 0.9468 | 0.9504 | 0.9504 | 0.8654 |
| 4 | 0.9055 | 0.9032 | 0.8926 | 0.8654 |
| 5 | 0.8317 | 0.8045 | 0.8022 | 0.8654 |
| 6 | 0.9077 | 0.9033 | 0.9029 | 0.8654 |
| 7 | 0.9182 | 0.9181 | 0.8864 | 0.8654 |
| 8 | 0.9317 | 0.9329 | 0.9324 | 0.8654 |
| 9 | 0.8813 | 0.8697 | 0.8684 | 0.8654 |

### Budget=40

| Seed | random | surrogate_only | causal | baseline |
|------|--------|----------------|--------|----------|
| 0 | 0.8433 | 0.8490 | 0.8423 | 0.8654 |
| 1 | 0.8716 | 0.8729 | 0.8729 | 0.8654 |
| 2 | 0.8786 | 0.8807 | 0.8785 | 0.8654 |
| 3 | 0.9468 | 0.9504 | 0.9504 | 0.8654 |
| 4 | 0.9055 | 0.9036 | 0.8926 | 0.8654 |
| 5 | 0.8317 | 0.8296 | 0.8022 | 0.8654 |
| 6 | 0.9122 | 0.9033 | 0.9029 | 0.8654 |
| 7 | 0.9214 | 0.9219 | 0.8864 | 0.8654 |
| 8 | 0.9320 | 0.9330 | 0.9351 | 0.8654 |
| 9 | 0.8813 | 0.8853 | 0.8684 | 0.8654 |

### Null Control Summary

| Strategy | Budget | Mean | Std (ddof=0) | Exceed baseline |
|----------|--------|------|--------------|-----------------|
| random | 20 | 0.8910 | 0.0362 | 8/10 |
| surrogate_only | 20 | 0.8885 | 0.0405 | 8/10 |
| causal | 20 | 0.8829 | 0.0400 | 8/10 |
| random | 40 | 0.8925 | 0.0357 | 8/10 |
| surrogate_only | 40 | 0.8930 | 0.0354 | 8/10 |
| causal | 40 | 0.8832 | 0.0403 | 8/10 |

The null-control pass shows that all three strategies produce policy values that
exceed the null baseline in 8/10 permuted-outcome seeds, with similar means and
standard deviations. **This means high policy values can still arise after
permuting outcomes (breaking the treatment-outcome link).** The optimizer finds
different parameter regions on the permuted data than on the real data —
selected parameters diverge substantially between real and null runs — so the
null pass does not show that the optimizer converges to the same solution
regardless of signal. What it does show is that the IPS-weighted objective can
be optimized above baseline even on noise, due to some combination of
finite-sample IPS variance, multiple-comparison effects across the search
trajectory, and allocation-policy mechanics. The primary-slice gains should
therefore not be interpreted as clean treatment-effect evidence on their own
without additional analysis to separate these factors.

## Secondary Outcomes (Full-Slice Arm Aggregates)

| Slice | Treated Visit Rate | Control Visit Rate | Treated Conversion Rate | Control Conversion Rate |
|-------|--------------------|--------------------|-------------------------|-------------------------|
| Primary | 15.14% | 10.62% | 0.88% | 0.57% |
| Pooled | 16.70% | 10.62% | 1.07% | 0.57% |

These are in-sample treated/control-arm means on the reshaped frame, not
policy-conditioned. The Womens E-Mail treatment shows a 4.5pp lift in visit
rate and a 0.31pp lift in conversion rate over control.

## Causal vs Random

| Slice | Budget | p-value (two-sided MWU) | Verdict | Direction |
|-------|--------|-------------------------|---------|-----------|
| Primary | B20 | 0.566 | Near-parity | random > causal |
| Primary | B40 | 0.014 | Certified | random > causal |
| Primary | B80 | 0.0004 | Certified | causal > random |
| Pooled | B20 | 0.623 | Near-parity | causal > random |
| Pooled | B40 | 0.241 | Near-parity | random > causal |
| Pooled | B80 | 0.161 | Near-parity | causal > random |

At B80 on the primary slice, the causal path significantly outperforms
random (p=0.0004, two-sided MWU), with 9/10 causal wins. This confirms
that the causal path does provide value relative to random search at high
budget, even though it underperforms surrogate-only at low and medium budgets.

## Interpretation

1. **The primary slice has a very flat response surface.** Most runs on the
   primary slice converge to policy values near 0.829, with less than 1% spread
   across strategies and seeds at B20--B40. The surrogate-only path is slightly
   better at finding this plateau (0.8293 vs 0.8285 for causal at B40), leading
   to a statistically significant but practically tiny advantage.

2. **The causal path occasionally discovers a higher-value region at B80.**
   Three seeds (7, 8, 9) on the primary slice at B80 find solutions with
   eligibility_threshold=0.0 that produce 12--16% higher policy values. This
   suggests the causal graph's focus on the regularization-to-policy_value
   edge helps the optimizer escape the local plateau when given enough budget.
   However, the discovery is seed-dependent (3/10 seeds), making the mean
   advantage noisy and not statistically significant.

3. **On the pooled slice, the optimal corner is easier to find without causal
   guidance.** The best pooled-slice policy (eligibility_threshold=0.0,
   treatment_budget_pct=1.0) gives policy_value=1.2496, which is 19% above
   the null baseline. Surrogate-only finds this corner in 7/10 seeds at B80;
   causal finds it in only 3/10 seeds. The causal graph's variable-focusing
   behavior may be narrowing exploration away from the corner.

4. **No strategy consistently beats the null baseline on the primary slice.**
   The primary slice has a weak treatment effect on spend, and all strategies
   produce policy values mostly below the null baseline (0.865). The optimizer
   is finding good policies within the search space but the treatment effect
   is not strong enough for any policy to consistently improve over the no-
   treatment baseline.

5. **The RF backend is the active backend.** Ax/BoTorch was not available in
   this environment, so all optimization runs used the RF surrogate fallback.
   The causal path's behavior may differ under the Ax/BoTorch backend, which
   was the primary backend for the energy benchmark wins. A follow-up run with
   Ax/BoTorch would be needed to determine whether the causal advantage
   pattern changes with a stronger surrogate.

## Verdict Per Slice

**Primary slice:** Trending s.o. advantage at B20, certified s.o. advantage
at B40, near-parity at B80. The causal path shows an interesting B80 tail where 3/10
seeds find a qualitatively better region, but this is not statistically
significant. Overall verdict: **NEAR-PARITY WITH CAUSAL B80 TAIL**.

**Pooled slice:** Certified surrogate-only advantage at all three budgets.
Surrogate-only more reliably finds the optimal corner at B80. Overall verdict:
**SURROGATE-ONLY ADVANTAGE**.

## What This Does Not Show

1. This is a single non-energy dataset. It does not generalize the Hillstrom
   result to all marketing datasets or all intervention domains.
2. The benchmark ran on the RF fallback backend, not Ax/BoTorch. The certified
   energy-domain causal wins were all Ax-primary. The Hillstrom result under
   Ax/BoTorch is unknown.
3. The 3-variable active search space is narrow compared to the energy
   benchmarks. The causal graph's value may scale with dimensionality.
4. This report does not claim a product-level generality win or loss from
   Hillstrom alone. It is one data point in a broader generalization research
   program.

## Reproducibility

The benchmark script is committed at `scripts/hillstrom_benchmark.py`. The
Hillstrom CSV is available from the official MineThatData source:
<https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>

The full Hillstrom dataset is local-only (not committed). The committed fixture
at `tests/fixtures/hillstrom_fixture.csv` is a small synthetic,
Hillstrom-shaped file for CI smoke tests — it does not reproduce benchmark
results. The JSON artifact from this run is stored locally at
`$ARTIFACTS_DIR/sprint-31-hillstrom-benchmark/hillstrom_results.json`.

## Commands Run

```bash
uv run python scripts/hillstrom_benchmark.py \
    --data-path $DATA_DIR/hillstrom.csv \
    --slices primary,pooled \
    --budgets 20,40,80 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --strategies random,surrogate_only,causal \
    --null-control \
    --output $ARTIFACTS_DIR/sprint-31-hillstrom-benchmark/hillstrom_results.json
```

Suite runtime: 1302.8 seconds (21.7 minutes), 240 total runs (180 real + 60
null control).

## Runtime Detail

| Slice | Budget | Strategy | Mean Runtime (s) |
|-------|--------|----------|------------------|
| primary | B20 | random | 0.10 |
| primary | B20 | surrogate_only | 2.58 |
| primary | B20 | causal | 2.51 |
| primary | B40 | random | 0.20 |
| primary | B40 | surrogate_only | 7.49 |
| primary | B40 | causal | 7.96 |
| primary | B80 | random | 0.38 |
| primary | B80 | surrogate_only | 14.39 |
| primary | B80 | causal | 15.72 |
| pooled | B20 | random | 0.15 |
| pooled | B20 | surrogate_only | 2.67 |
| pooled | B20 | causal | 2.88 |
| pooled | B40 | random | 0.29 |
| pooled | B40 | surrogate_only | 7.29 |
| pooled | B40 | causal | 10.75 |
| pooled | B80 | random | 0.57 |
| pooled | B80 | surrogate_only | 14.28 |
| pooled | B80 | causal | 20.07 |
