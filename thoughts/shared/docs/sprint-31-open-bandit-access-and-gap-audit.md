# Sprint 31: Open Bandit Dataset Access and Adapter-Gap Audit

**Date**: 2026-04-16
**Sprint**: 31 (Generalization Research)
**Issue**: #172
**Branch**: `sprint-31/open-bandit-audit`

## 1. Official Source and Download Path

**Dataset name**: Open Bandit Dataset (OBD)

**Publisher**: ZOZO, Inc. (operator of ZOZOTOWN, Japan's largest fashion e-commerce platform)

**Official page**: https://research.zozo.com/data.html

**Direct download**: `https://research.zozo.com/data_release/open_bandit_dataset.zip`

**Companion library**: Open Bandit Pipeline (OBP), https://github.com/st-tech/zr-obp

**Reference paper**: Saito et al., "Open Bandit Dataset and Pipeline: Towards Realistic
and Reproducible Off-Policy Evaluation," NeurIPS Datasets and Benchmarks 2021.
arXiv: https://arxiv.org/abs/2008.07146

**Registration requirements**: None. The zip file is a direct HTTP download from the
ZOZO Research page. No account, API key, or form submission is required.

**Small demo version**: The OBP repository (`zr-obp/obd/`) ships a small 10,000-record-per-slice
demo version that can be installed via `pip install obp`. This is sufficient for
schema exploration and smoke tests but not for meaningful OPE benchmarks.

## 2. License and Usage Restrictions

**Practical conclusion**: the causal-optimizer is an open-source research tool
with no commercial redistribution of the data. Usage for non-commercial
benchmarking is clearly within both the stated intent and the formal license
terms. If the project ever redistributes a fixture subset, it should include
attribution to Saito et al. (2021) and ZOZO, Inc.

**Stated license**: CC BY 4.0

**ZOZO's stated intent**: The dataset page and paper describe the data as available
for "non-commercial research purposes." This framing is narrower than what
CC BY 4.0 technically allows.

**Actual CC BY 4.0 terms**: CC BY 4.0 permits copying, redistribution, adaptation,
and commercial use, provided attribution is given. The non-commercial language
reflects ZOZO's preferred use, not a formal license restriction.

**OBP library license**: Apache 2.0. No restrictions on using the library code.

## 3. Approximate Dataset Size and Local Setup Cost

### Row counts by campaign and behavior policy

The three campaigns (All, Men, Women) are separate, independently run
experiments on different item catalogs within ZOZOTOWN, not nested subsets.
Each campaign has its own item pool and was run concurrently during the same
7-day collection period. Row counts from Saito et al. (2021), Table 1:

| Campaign | Uniform Random | Bernoulli TS | Total |
|----------|---------------|-------------|-------|
| All      | 1,374,327     | 12,168,084  | 13,542,411 |
| Men      | 452,949       | 4,077,727   | 4,530,676 |
| Women    | 864,585       | 7,765,497   | 8,630,082 |
| **Total** | **2,691,861** | **24,011,308** | **~26.7M** |

### Column count and context dimensions

| Campaign | Items (actions) | Context dimensions |
|----------|----------------|--------------------|
| All      | 80             | 84                 |
| Men      | 34             | 38                 |
| Women    | 46             | 50                 |

Context dimensions include user features (anonymized categorical attributes
like age and gender), user-item affinity scores (derived from past click
history), and item features (4 per item: price, brand, product category,
all hashed for privacy).

### Disk space estimate

The full dataset is a zip of CSV files. With 26.7M rows and roughly 90+ columns
of mixed numeric and categorical data, the compressed zip is estimated at 2-4 GB
and the uncompressed CSVs at 5-10 GB. The small demo version in the OBP repo
is negligible (a few MB).

### Download method

**Preferred**: use the OBP library's `OpenBanditDataset` class (`pip install obp`),
which handles download, path configuration, and schema loading. This is more
resilient to URL changes than hardcoded links.

**Fallback**: direct HTTP download of `open_bandit_dataset.zip` from the ZOZO
Research page. Research dataset URLs tend to move; the OBP library is the more
stable access path. The raw CSVs are usable without OBP once downloaded.

### Local setup cost

- Download: single HTTP fetch, a few minutes on broadband
- Unzip: straightforward
- Loading: pandas can handle individual campaign CSVs (1-12M rows each)
  but loading all 26M rows into memory simultaneously requires 8+ GB RAM
- For benchmarking purposes: a single campaign slice (e.g., Men/Random at
  ~453K rows) is sufficient as a starting point

## 4. Action Structure

**This is not a binary treatment dataset.** This is the most important
architectural distinction between Open Bandit and the datasets currently
supported by causal-optimizer.

### Current engine assumption

The `MarketingLogAdapter` and all current benchmarks assume **binary treatment**:
a single treatment column with values {0, 1}. The engine's IPS-weighted policy
evaluation, causal graph, and search space all operate on this binary framing.

### Open Bandit action structure

- **Multi-action**: 34 to 80 discrete items (actions) depending on campaign
- **Positional**: each impression shows 3 items in positions (left, center, right)
- **Combinatorial**: the full action space is item x position, not just item selection
- **Logged policy**: each row records one (item, position) pair with its propensity

### What this means for the adapter

Open Bandit requires reasoning about which of 34-80 items to recommend, not
whether to treat or not treat. The binary treatment column does not exist.
Instead, the `item_id` column encodes a categorical action drawn from a
multi-action policy.

This is a fundamental mismatch with the current `MarketingLogAdapter`, which:
- validates that treatment is binary {0, 1}
- computes IPS weights as 1/p(T=1) for treated and 1/(1-p) for control
- defines a policy as a threshold/budget rule over a binary treatment decision

None of these operations transfer directly to a multi-action setting.

## 5. Propensity and Logging-Policy Availability

**Propensities are available.** This is one of the dataset's strongest features
for OPE research.

### How propensities were generated

Two behavior policies were run during a 7-day A/B test in late November 2019:

1. **Uniform Random**: each item is equally likely at each position.
   Propensity = 1/n_items for each position.
2. **Bernoulli Thompson Sampling (BTS)**: propensities computed via Monte Carlo
   simulation from the beta-distribution parameters used during data collection.

The propensity scores are stored in the `action_prob` column of each CSV file.
They represent the probability that the logged policy selected the observed
(item, position) pair.

### Why this matters

Having true propensities (not estimated ones) is rare in real-world logged data.
It enables:
- unbiased IPW/IPS estimation
- ground-truth validation of OPE estimators
- clean comparison between Random and BTS policy data

### Propensity characteristics

Because the action space is large (34-80 items across 3 positions), individual
propensity scores are small. Under the Random policy, propensity per item per
position is approximately 1/n_items (e.g., 1/80 = 0.0125 for the All campaign).
Under BTS, propensities are concentrated on high-performing items, making them
even more variable.

Small propensities lead to large IPS weights, which is a known variance challenge
for this dataset. The OBP paper reports that self-normalized estimators and
DRos (Doubly Robust with Optimistic Shrinkage) significantly outperform
vanilla IPW for this reason.

## 6. Outcome and Reward Definition

### Primary reward

- **click**: binary (0/1), indicating whether the user clicked the recommended item
- This is the only reward signal in the dataset

### Click-through rates

Empirical CTR values from Saito et al. (2021), Table 1. Intervals are 95%
confidence intervals in percentage points (e.g., 0.35% +/- 0.010 pp means
the true CTR is estimated between 0.340% and 0.360%):

| Campaign | Random CTR             | BTS CTR                |
|----------|------------------------|------------------------|
| All      | 0.35% (+/-0.010 pp)    | 0.50% (+/-0.004 pp)    |
| Men      | 0.51% (+/-0.021 pp)    | 0.67% (+/-0.008 pp)    |
| Women    | 0.48% (+/-0.014 pp)    | 0.64% (+/-0.056 pp)    |

### What is absent

- **No revenue or conversion signal.** Only click, not purchase or spend.
- **No cost column.** Unlike the `MarketingLogAdapter`, which requires a cost
  column and computes `total_cost`, Open Bandit has no per-observation cost.
- **No continuous outcome.** The reward is strictly binary, unlike the
  continuous `outcome` column in the marketing log fixture.

### Implications for adapter design

The current `MarketingLogAdapter` returns `policy_value` as a self-normalized
IPS-weighted average of a continuous outcome, and `total_cost` as an IPS-weighted
sum of a cost column. Open Bandit would need:
- `policy_value` redefined as IPS-weighted CTR (mean of binary clicks)
- `total_cost` either dropped or replaced with a different metric
- new diagnostic metrics appropriate for recommendation (e.g., precision@k,
  nDCG, or mean reward per position)

## 7. Adapter Compatibility Assessment

### Can it work with the current stack via a small wrapper?

**No.** The `MarketingLogAdapter` cannot be adapted to Open Bandit with minor
changes. The mismatches are structural, not cosmetic:

| Feature | MarketingLogAdapter | Open Bandit Requirement |
|---------|-------------------|------------------------|
| Treatment | Binary {0, 1} | Categorical {0, ..., 79} |
| IPS formula | 1/p(T=1) and 1/(1-p) | 1/p(action) for logged action only |
| Policy | threshold + budget rule | item-selection policy (score each item) |
| Outcome | Continuous revenue | Binary click |
| Cost column | Required | Does not exist |
| Position | Not modeled | 3 positions, integral to action |
| Search space | 6 continuous policy knobs | Must parameterize an item-scoring policy |

### Does it need a new adapter?

**Yes.** A new `BanditLogAdapter` or `RecommendationLogAdapter` would be needed.
This adapter would need to:

1. Accept multi-action logged data (item_id, position, action_prob, click)
2. Define a parameterized item-scoring policy (not a binary threshold)
3. Compute IPS-weighted CTR using multi-action propensities
4. Handle the position dimension (either marginalize or model explicitly)
5. Report appropriate diagnostics (ESS per action, support coverage, etc.)

### Does it need a new multi-action OPE stack?

**Likely yes, for production quality.** The current engine's IPS implementation
in `MarketingLogAdapter.run_experiment()` is binary-specific. A multi-action
OPE stack would need:

- **Multi-action IPW**: weight = 1/p(a|x) for the observed action, where a is
  one of 34-80 possible actions
- **Self-normalized IPW (SNIPW)**: critical for variance control with small
  propensities
- **Doubly Robust or DRos estimators**: the OBP paper shows these are
  30-60% more accurate than IPW on this dataset
- **Action embedding support**: for large action spaces, marginal IPS or
  embedding-based estimators may be needed to control variance

The OBP library (`pip install obp`) already implements all of these estimators.
The integration question is whether to:
- (a) wrap OBP's estimators behind the adapter contract, or
- (b) implement a minimal multi-action IPW in-house

Option (a) is faster and more correct. Option (b) keeps the dependency footprint
small but requires reimplementing non-trivial OPE estimators.

## 8. Biggest Support and Action-Space Risks

### Risk 1: Positivity violations from large action spaces

With 80 items and 3 positions, the effective action space is large. Even under
the Random policy, per-action propensities are small (~0.0125). Under BTS,
propensities for low-performing items can be near zero, creating extreme IPS
weights.

**Mitigation**: use self-normalized estimators (SNIPW, SNDR) or DRos.
Clipping alone (as the current `MarketingLogAdapter` does) is insufficient
when propensities are structurally small rather than occasionally small.

### Risk 2: Very low reward signal

Click-through rates are 0.3-0.7%. This means the vast majority of observations
have zero reward. Policy differences are measured in fractions of a percentage
point. This is a much harder signal-to-noise environment than the current
marketing fixture (which has continuous outcomes with substantial variance).

**Implication**: the optimizer would need many more observations per policy
evaluation to detect meaningful differences, and bootstrap CIs will be wide.

### Risk 3: Combinatorial policy space

The search space for a recommendation policy is fundamentally different from
the current 6-variable continuous policy space. A useful Open Bandit adapter
would need to parameterize a scoring function over items, not a threshold
over a single treatment decision.

Options:
- Linear scoring: weight each item feature, recommend top-k
- Contextual scoring: weight user-item affinity features
- Simple rule: temperature parameter over BTS, epsilon for exploration

Even the simplest parameterization is architecturally different from what
`suggest_parameters()` currently optimizes.

### Risk 4: Position bias confounding

Items in position 1 (left) may receive more clicks simply due to position,
not item quality. The dataset includes position information but does not
de-bias for position effects. An adapter would need to either:
- model position as a confounding variable in the causal graph
- marginalize over positions in the IPS estimator
- restrict evaluation to a single position

### Risk 5: Benchmark runtime

Saito et al. (2021), Section 5.1, report that OPE benchmark experiments with
T=200 bootstrap iterations on a 2019-era MacBook Pro (2.4 GHz Intel Core i9,
64 GB RAM) required 221-750 minutes per campaign at n=300,000. At the smaller
n=10,000 setting, runtimes were 22-48 minutes per campaign. Modern hardware
(Apple Silicon or current-gen x86) would likely reduce these times by 3-5x,
but a full 10-seed benchmark at scale would still require careful subsetting
or meaningful compute time.

## 9. Recommendation for When It Should Enter the Benchmark Queue

### Not before Sprint 34. Prerequisites must be met first.

Open Bandit is a harder architectural step than either Hillstrom or Criteo
Uplift. Both of those datasets have binary treatment structure that fits
the current engine with small-to-moderate adapter changes. Open Bandit
requires a new multi-action policy evaluation paradigm.

### Prerequisite 1: Hillstrom benchmark passing (Sprint 31-32)

The Hillstrom binary marketing benchmark should be the first non-energy
real-data proof point. It tests whether the engine transfers at all,
using the existing adapter contract with minimal changes.

### Prerequisite 2: Criteo Uplift benchmark passing (Sprint 32-33)

Criteo Uplift is a large-scale binary uplift dataset that stress-tests
the engine at scale while still using binary treatment semantics. It is the
natural second step before changing the treatment paradigm.

### Prerequisite 3: Multi-action adapter contract design (Sprint 33-34)

Before implementing an Open Bandit adapter, the project needs:

1. A `MultiActionLogAdapter` base class or contract extension that supports
   categorical actions, multi-action IPS, and position-aware evaluation
2. A decision on whether to depend on OBP or reimplement core estimators
3. A search-space design for item-scoring policies that works with the
   existing `suggest_parameters()` infrastructure
4. A causal graph design that encodes item features, position effects,
   and user context as intervention variables

### Prerequisite 4: Decision on OBP dependency

The OBP library provides production-quality OPE estimators (IPW, SNIPW, DR,
DRos, Switch-DR) that have been validated on this exact dataset. Using OBP
would save weeks of implementation time but adds a dependency. This decision
should be made explicitly, not deferred to implementation time.

### Recommended timeline

| Sprint | Milestone |
|--------|-----------|
| 31-32  | Hillstrom binary benchmark: first non-energy real-data evidence |
| 32-33  | Criteo Uplift: scale test with binary treatment |
| 33     | Multi-action adapter contract RFC |
| 34     | Open Bandit adapter implementation + smoke test |
| 34-35  | Open Bandit benchmark: first multi-action OPE evidence |

### What Open Bandit would prove if it passes

1. The engine can optimize recommendation policies, not just binary treatments
2. Causal guidance (graph-based focus on item features and user context) helps
   in a large action space
3. The project's claim to "domain-agnostic causal research assistant" extends
   beyond A/B-test-style interventions

### What Open Bandit would prove if it fails

1. Whether the failure is in the multi-action OPE estimation (fixable)
2. Whether the failure is in the search space parameterization (fixable)
3. Whether the engine's optimization loop fundamentally assumes binary
   treatment (architectural limit that would require redesign)

## 10. Minimum Viable Feasibility Check

Before committing to a full adapter implementation in Sprint 34, a 2-4 hour
spike can validate the data access path and confirm the dataset matches
expectations. This spike does not require any engine changes.

### Steps

1. `pip install obp` and load the Men/Random slice (~453K rows) via
   `OpenBanditDataset(behavior_policy="random", campaign="men")`
2. Confirm schema: verify `action`, `position`, `reward`, `pscore`, `context`,
   and `action_context` keys are present in the bandit feedback dict
3. Compute vanilla IPW CTR using OBP's built-in `InverseProbabilityWeighting`
   estimator
4. Compare the estimated CTR to the paper's reported value (0.51% +/-0.021)
5. Repeat with the BTS slice to confirm cross-policy estimation works

### What this proves

- Data is accessible and loadable without issues
- Schema matches documentation
- OBP estimators produce values consistent with published results
- The project has a working local copy before committing to adapter design

### What this does not prove

- That the engine can optimize over this data (requires adapter work)
- That a search space parameterization exists (requires design work)
- That the causal graph provides value in a multi-action setting (requires
  the full benchmark)

## Summary

Open Bandit is the strongest open dataset for offline policy evaluation in a
multi-action recommendation setting. It has real propensities, a well-studied
OPE benchmark, and an active companion library. It is also a substantially
harder architectural step than the binary-treatment datasets in the current
queue.

The dataset is freely downloadable, licensed under CC BY 4.0, and requires
no registration. The full dataset is approximately 26.7 million rows across
6 campaign-policy slices, with 34-80 discrete actions and binary click rewards.

The current `MarketingLogAdapter` cannot accommodate Open Bandit. A new
multi-action adapter, multi-action IPS estimator, and item-scoring search
space would all be required. This is real architecture work, not a reshaping
exercise.

**Recommendation**: queue Open Bandit for Sprint 34-35, after Hillstrom and
Criteo Uplift have validated that the engine transfers to non-energy domains
within the existing binary treatment contract. Use the intervening sprints to
design the multi-action adapter contract and decide on the OBP dependency
question.
