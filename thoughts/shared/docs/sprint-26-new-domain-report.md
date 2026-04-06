# Sprint 26: Dose-Response Clinical Benchmark Report

## Summary

Added a **dose-response clinical trial benchmark** -- a fully synthetic, new-domain
benchmark that evaluates optimization strategies on a treatment protocol selection
problem.  The benchmark is cheap to run (~3.5 minutes for 45 runs), fully
reproducible, and requires no external data files.

## Benchmark Design

### Domain

Clinical dose-response optimization.  An optimizer searches for the best treatment
protocol: what dose level to administer, and which patients to select for treatment
based on biomarker and severity thresholds.

### Causal Structure

The true data-generating process follows an **Emax (Hill equation) dose-response
model** with biomarker-mediated efficacy:

- **Dose level**: Sigmoid dose-response curve (ED50 = 0.3, Hill coefficient = 2.5)
- **Biomarker**: Amplifies max effect from 20 to 80 symptom-score points
- **Severity**: Modifier from 0.3x to 1.0x

Three noise dimensions (BMI threshold, age threshold, comorbidity threshold) have
**zero causal effect** on treatment benefit.  The causal graph encodes this:

```
dose_level ──────────> objective
biomarker_threshold ──> objective
severity_threshold ──-> objective
bmi_threshold ──────-> patient_risk  (NOT connected to objective)
age_threshold ──────-> patient_risk
comorbidity_threshold -> patient_risk
```

Causal knowledge should help by pruning the 3 noise dimensions, focusing search
on the 3 real parameters.

### Evaluation Protocol

- **N patients**: 1000 (synthetic, Beta-distributed covariates)
- **Split**: 80/20 opt/test by position
- **Oracle**: Treats exactly the patients whose effect at reference dose exceeds
  the treatment cost (15.0 symptom-score units)
- **Oracle treat rate**: ~25-40% (non-degenerate)
- **Oracle value**: 9.03 (test set)
- **Metric**: Policy value = average net benefit (symptom reduction minus cost)
- **Regret**: Oracle value minus policy value (non-negative)

### Non-Degeneracy Check

- Oracle value is positive (9.03) and represents meaningful benefit
- Oracle treat rate is ~32%, not 0% or 100%
- Random strategy at B20 achieves regret ~8.6 (far from oracle), confirming
  the problem is not trivially solvable

## Results

| Strategy        | Budget | Regret (mean +/- std) | Policy Value (mean +/- std) |
|-----------------|--------|-----------------------|-----------------------------|
| random          |     20 |      8.62 +/- 0.33    |          0.41 +/- 0.33      |
| random          |     40 |      8.37 +/- 0.49    |          0.65 +/- 0.49      |
| random          |     80 |      7.73 +/- 0.57    |          1.30 +/- 0.57      |
| surrogate_only  |     20 |      8.72 +/- 0.38    |          0.30 +/- 0.38      |
| surrogate_only  |     40 |      6.83 +/- 1.57    |          2.20 +/- 1.57      |
| surrogate_only  |     80 |  **1.32 +/- 2.08**    |      **7.71 +/- 2.08**      |
| causal          |     20 |      8.61 +/- 0.29    |          0.42 +/- 0.29      |
| causal          |     40 |      8.55 +/- 0.38    |          0.48 +/- 0.38      |
| causal          |     80 |      6.51 +/- 1.83    |          2.52 +/- 1.83      |

5 seeds per cell (0-4).  Oracle value = 9.03.

## Analysis

### Key Finding: Surrogate-Only Dominates on This Domain

The surrogate-only strategy dramatically outperforms both causal and random at
B80, achieving regret 1.32 vs 6.51 (causal) and 7.73 (random).  This is a
clean, real result -- not a measurement artifact.

**Why this happens:**

1. **Smooth, all-continuous landscape.**  The dose-response benchmark has 6
   continuous variables with no categorical dimensions and a smooth Emax
   objective surface.  The RF surrogate in the optimization phase can model
   this surface efficiently.

2. **Moderate dimensionality.**  At 6 dimensions (3 real + 3 noise), the noise
   burden is manageable for surrogate modeling.  Compare to the high-noise
   energy benchmark (15 dimensions), where noise dimensions create a genuine
   needle-in-a-haystack problem.

3. **Causal pruning may over-constrain.**  The causal strategy's focus-variable
   mechanism restricts the search to ancestors of the objective, which are the
   3 real parameters.  While this is correct in principle, it may cause the
   engine's exploration phase to miss good initial configurations.  The
   surrogate-only strategy explores all 6 dimensions freely and converges
   faster on the smooth landscape.

### Comparison to Energy Benchmarks

| Domain         | Dims | Noise Dims | Causal Advantage? | Why                              |
|----------------|------|------------|-------------------|----------------------------------|
| Energy (base)  |    5 |          2 | Yes (B80)         | Categorical noise + confounding  |
| Energy (high)  |   15 |         12 | Yes (B40, B80)    | High noise burden overwhelms RF  |
| Energy (confounded) | 5 | 2       | No (all misled)   | Bidirected edges insufficient    |
| **Clinical**   |    6 |          3 | **No (B80)**      | Smooth landscape, RF excels      |

This confirms that causal advantage is **domain-dependent**: it helps most when
the search space is high-dimensional with many irrelevant variables, or when
categorical variables create discrete barriers that the surrogate cannot smooth
over.  On a smooth, moderate-dimensional landscape, the RF surrogate's
data-driven approach is more efficient.

### Causal Improvement Path (B40 -> B80)

The causal strategy does show improvement from B40 to B80 (regret 8.55 -> 6.51),
but the rate of improvement is slower than surrogate_only (8.72 -> 1.32).  This
suggests the causal strategy's exploration phase is not generating enough diverse
initial points in the productive region of the search space.

## Provenance

- **Branch**: `sprint-26/new-domain-benchmark`
- **Seeds**: 0, 1, 2, 3, 4
- **Budgets**: 20, 40, 80
- **Strategies**: random, surrogate_only, causal
- **N patients**: 1000
- **Treatment cost**: 15.0
- **Suite runtime**: ~213 seconds

## Verdict: USEFUL

The dose-response benchmark is a clean, reproducible, new-domain test that
produces interpretable results.  Its value is:

1. **Negative control for causal advantage.**  It shows that causal guidance
   is not universally superior -- on smooth, moderate-dimensional landscapes,
   surrogate-only wins.  This prevents over-claiming.

2. **Different domain semantics.**  Clinical dose-response is structurally
   different from energy demand-response: continuous dose levels, biomarker
   interactions, patient selection thresholds.

3. **Cheap and self-contained.**  No external data dependency, ~3.5 minutes
   for a full sweep.

4. **Non-degenerate oracle.**  Oracle treat rate ~32%, oracle value 9.03,
   well above the floor.

The benchmark is ready for routine reruns and inclusion in the regression gate
suite.
