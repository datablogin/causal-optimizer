# Sprint 35 Open Bandit Benchmark Report (Men/Random)

**Date:** 2026-04-20
**Sprint:** 35.C (first Men/Random Open Bandit benchmark)
**Issue:** [#187](https://github.com/datablogin/causal-optimizer/issues/187)
**Branch:** `sprint-35/open-bandit-benchmark` (stacked on `sprint-35/open-bandit-bridge`)
**Dataset:** Open Bandit Dataset, ZOZOTOWN Men campaign, uniform-random logger (full slice, 452,949 rows)
**Backend:** Ax/BoTorch (primary; RF fallback not used — see `provenance.optimizer_path`)
**Predecessors:** Sprint 34 Open Bandit contract ([PR #184](https://github.com/datablogin/causal-optimizer/pull/184)), Sprint 35.A adapter ([PR #189](https://github.com/datablogin/causal-optimizer/pull/189)), Sprint 35.B OPE stack ([PR #188](https://github.com/datablogin/causal-optimizer/pull/188)), Sprint 35 bridge ([PR #191](https://github.com/datablogin/causal-optimizer/pull/191)).

## Summary

This is the first Men/Random Open Bandit benchmark for the causal optimizer. Both optimized strategies (`surrogate_only` and `causal`) converge to the same softmax-over-affinity item-scoring policy on every seed at every budget, producing a clean certified separation from the `random` baseline. All five Sprint 34 Section 7 support gates pass cleanly. `causal` and `surrogate_only` tie exactly because the first-run `BanditLogAdapter` returns `get_prior_graph() -> None` (Section 4e defers the multi-action prior graph to Sprint 36+), so the `causal` strategy operates without causal knowledge on this slice and is behaviorally identical to `surrogate_only`.

**Verdict: clean Men/Random row, NEAR-PARITY between `causal` and `surrogate_only` (exact tie), both CERTIFIED over `random` at every budget (p = 0.0002, two-sided MWU). All Section 7 gates PASS.**

## Verdict Row

| Slice | Budget | Causal vs Surrogate-Only | p-value (two-sided MWU) | Verdict | Direction |
|-------|--------|--------------------------|-------------------------|---------|-----------|
| Men/Random (full) | B20 | identical all 10 seeds | 1.000 | Near-parity (exact tie) | — |
| Men/Random (full) | B40 | identical all 10 seeds | 1.000 | Near-parity (exact tie) | — |
| Men/Random (full) | B80 | identical all 10 seeds | 1.000 | Near-parity (exact tie) | — |

| Slice | Budget | Causal vs Random | p-value (two-sided MWU) | Verdict | Direction |
|-------|--------|------------------|-------------------------|---------|-----------|
| Men/Random (full) | B20 | causal wins 10/10 seeds | 0.0002 | Certified | causal > random |
| Men/Random (full) | B40 | causal wins 10/10 seeds | 0.0002 | Certified | causal > random |
| Men/Random (full) | B80 | causal wins 10/10 seeds | 0.0002 | Certified | causal > random |

| Slice | Budget | Surrogate-Only vs Random | p-value (two-sided MWU) | Verdict | Direction |
|-------|--------|--------------------------|-------------------------|---------|-----------|
| Men/Random (full) | B20 | s.o. wins 10/10 seeds | 0.0002 | Certified | s.o. > random |
| Men/Random (full) | B40 | s.o. wins 10/10 seeds | 0.0002 | Certified | s.o. > random |
| Men/Random (full) | B80 | s.o. wins 10/10 seeds | 0.0002 | Certified | s.o. > random |

Abbreviation: **s.o.** = `surrogate_only`.

**Overall Men/Random verdict: EXACT-TIE between `causal` and `surrogate_only`, both CERTIFIED over `random`.** Per Sprint 34 contract Section 12 this is the "least valuable but acceptable" outcome (clean diagnostics, no separation between causal and surrogate-only) and carries the same weight as the Criteo near-parity row. The tie is mechanical rather than substantive — no causal graph was supplied for Sprint 35, so the `causal` path has no additional information over `surrogate_only`.

## Data Provenance

| Property | Value |
|----------|-------|
| Source | ZOZOTOWN Open Bandit Dataset, Men campaign, uniform-random logger |
| Download | `https://research.zozo.com/data_release/open_bandit_dataset.zip` |
| Release version | `08/18/2020 Open Bandit Dataset 1.0` (inside zip `VERSION` file) |
| Slice path on disk | `{data_path}/random/men/men.csv` and `{data_path}/random/men/item_context.csv` |
| men.csv size | 151,946,449 bytes |
| men.csv SHA-256 | `c4b6f65e62bf2c683914703ab6b875cc3e1b4ef0403a5779f548f5578cc34d6d` |
| item_context.csv size | 4,287 bytes |
| n_rounds (rows after loader) | **452,949** (matches Saito et al. 2021 Table 1 for Men/Random) |
| n_actions (distinct item_id) | 34 |
| n_positions (distinct) | 3 |
| Position convention | 0-indexed contiguous integers (loader applies `scipy.stats.rankdata(..., 'dense') - 1`; raw CSV positions are 1/2/3) |
| `propensity_score` empirical mean | 0.029411764706 (= 1 / 34) |
| `click` empirical mean (μ_null) | 0.005124197205 |
| Propensity schema | **conditional** `P(item \| position) = 1 / n_items` (Sprint 34 contract Section 5c; confirmed by Sprint 35.A smoke test and by the 7d sanity gate with relative deviation < 2e-15) |
| Propensity floor (`min_propensity_clip`) | `1 / (2 · n_actions · n_positions) = 1 / (2 · 34 · 3) = 4.9019608e-03` (Sprint 34 contract Section 5c; frozen, not tuned) |
| `obp.__version__` | 0.4.1 |
| Subsample | **none** — full Men/Random slice (`--data-path` points at the unzipped dataset root) |
| git SHA | `847636b9c168e8150cba049f9d51045475a97eb9` |
| Python version | 3.13.12 |
| Run timestamp (UTC) | 2026-04-20T05:13:41 |

## Configuration

- **Backend:** Ax/BoTorch (primary). `provenance.optimizer_path = {'optimizer_path': 'ax_botorch', 'ax_available': True, 'botorch_available': True, 'fallback_reason': None}`. RF surrogate fallback was NOT exercised on any cell.
- **Logger:** uniform-random (ZOZO's Men/Random campaign).
- **Strategies:** `random`, `surrogate_only`, `causal` (Sprint 34 contract Section 6a).
- **Budgets:** 20, 40, 80 (Sprint 34 contract Section 6e; B80 is the verdict budget).
- **Seeds:** 0..9 (10 seeds per cell).
- **Total runs:** 180 (90 real + 90 null-control; every cell completed with no skipped seeds).
- **Suite runtime:** 2,880.4 seconds (48.0 minutes); dataset load: 0.9 s.
- **Search space (6 variables, Sprint 34 contract Section 4c):**
  - `tau` continuous `[0.1, 10.0]` — softmax temperature
  - `eps` continuous `[0.0, 0.5]` — uniform-exploration mix
  - `w_item_feature_0` continuous `[-3.0, 3.0]` — per-item continuous feature weight
  - `w_user_item_affinity` continuous `[-3.0, 3.0]` — per-(row, item) affinity weight
  - `w_item_popularity` continuous `[-3.0, 3.0]` — per-item popularity prior weight
  - `position_handling_flag` categorical `{"marginalize", "position_1_only"}`
- **Estimators:** SNIPW (primary), DM (secondary), DR (secondary; in-module Dudík et al. 2011 DR on a per-action empirical-CTR reward model).
- **Permutation seed (null control):** 20260419 (one fixed seed per benchmark, Sprint 34 contract Section 7a).
- **Causal prior graph:** `None`. The Sprint 35.A adapter returns `get_prior_graph() -> None` (Sprint 34 contract Section 4e defers a multi-action prior graph to Sprint 36+). Consequently the `causal` strategy operates without causal knowledge on Men/Random and is indistinguishable from `surrogate_only`.

## Per-Budget Outcome Tables (SNIPW — primary)

Population std (ddof=0). All 10 seeds per cell completed cleanly.

| Strategy | Budget | n | Mean SNIPW | Std (ddof=0) | Min | Max |
|----------|--------|---|------------|--------------|-----|-----|
| random | 20 | 10 | 0.005213 | 0.000084 | 0.005137 | 0.005407 |
| surrogate_only | 20 | 10 | 0.005805 | 0.000277 | 0.005344 | 0.006174 |
| causal | 20 | 10 | 0.005805 | 0.000277 | 0.005344 | 0.006174 |
| random | 40 | 10 | 0.005318 | 0.000236 | 0.005147 | 0.005987 |
| surrogate_only | 40 | 10 | 0.006180 | 0.000009 | 0.006156 | 0.006189 |
| causal | 40 | 10 | 0.006180 | 0.000009 | 0.006156 | 0.006189 |
| random | 80 | 10 | 0.005431 | 0.000221 | 0.005244 | 0.005987 |
| surrogate_only | 80 | 10 | **0.006182** | 0.000008 | 0.006158 | 0.006189 |
| causal | 80 | 10 | **0.006182** | 0.000008 | 0.006158 | 0.006189 |

SNIPW at B80 (verdict budget per Section 6e): `surrogate_only = causal = 0.006182`, `random = 0.005431`. Absolute lift of optimized strategies over random at B80 is `0.000751` (≈14% relative over random, ≈20% relative over the logged-policy μ = 0.005124). At the tightest budget (B20), `surrogate_only` and `causal` already beat `random` on every seed.

`surrogate_only` and `causal` return identical best-of-seed policy values on every seed at every budget. On inspection of the selected parameters, both strategies converge to the same softmax policy each seed (e.g. seed 0 B80: `tau≈0.285, eps=0.0, w_user_item_affinity=3.0, w_item_popularity=-2.461, w_item_feature_0≈0.231, position_handling_flag="position_1_only"`). This confirms the tie is driven by the missing causal prior, not by the optimizer state.

## Secondary Estimators (DM and DR)

Population means over 10 seeds, real data only.

| Strategy | Budget | Mean DM | Mean DR |
|----------|--------|---------|---------|
| random | 20 | 0.005122 | 0.005219 |
| surrogate_only | 20 | 0.005112 | 0.005812 |
| causal | 20 | 0.005112 | 0.005812 |
| random | 40 | 0.005115 | 0.005324 |
| surrogate_only | 40 | 0.005140 | 0.006192 |
| causal | 40 | 0.005140 | 0.006192 |
| random | 80 | 0.005108 | 0.005436 |
| surrogate_only | 80 | 0.005143 | 0.006194 |
| causal | 80 | 0.005143 | 0.006194 |

DM is tightly clustered across all strategies (0.00511–0.00514) because the first-run reward model is the zero-context per-action mean; DM is therefore an almost-policy-agnostic lower bound in this configuration and is quoted only as a sanity check against blatant SNIPW failure, not as a second verdict signal. DR tracks SNIPW closely on every (strategy, budget) cell (maximum relative divergence across all seeds: 0.48%) — the 7e cross-check is well within tolerance.

## Section 7 Support Gates

| Gate | Threshold | Status | Observed |
|------|-----------|--------|----------|
| 7a null control | every strategy-budget cell ≤ 1.05 × μ_null | **PASS** | 6/6 cells within band; max ratio = 1.0154 |
| 7b ESS floor | median ESS ≥ max(1000, n_rows / 100) = 4,530 (marginalize) | **PASS** | median ESS (B80, all strategies) = 49,867 |
| 7c zero-support fraction | best-of-seed ≤ 10% | **PASS** | best = 0.0 on every B80 cell |
| 7d propensity sanity | empirical mean within 10% relative of 1/n_items = 0.029412 | **PASS** | empirical 0.029412, relative deviation ≈ 2e-15 |
| 7e DR/SNIPW cross-check | per-seed relative divergence ≤ 25% | **PASS** | max observed divergence = 0.48%; 0 offending seeds |

**All five Section 7 gates PASS.** The Sprint 35 benchmark meets the Sprint 34 contract's verdict-publication bar.

## Gate Details

### 7a Null control (PASS, permutation seed = 20260419)

μ_null = 0.005124 (raw mean of the permuted reward column on the full 452,949-row slice). Threshold = 1.05 × μ_null = 0.005380. All 9 strategy-budget cells (3 strategies × 3 budgets) on permuted data satisfy `policy_value ≤ threshold`.

| Strategy | Budget | Mean Policy Value (null) | Ratio | Within 5% band |
|----------|--------|---------------------------|-------|----------------|
| random | 20 | 0.005137 | 1.0026 | yes |
| surrogate_only | 20 | 0.005157 | 1.0063 | yes |
| causal | 20 | 0.005157 | 1.0063 | yes |
| random | 40 | 0.005146 | 1.0042 | yes |
| surrogate_only | 40 | 0.005187 | 1.0123 | yes |
| causal | 40 | 0.005187 | 1.0123 | yes |
| random | 80 | 0.005150 | 1.0051 | yes |
| surrogate_only | 80 | 0.005203 | 1.0154 | yes |
| causal | 80 | 0.005203 | 1.0154 | yes |

The highest observed ratio on permuted data is 1.0154 at `surrogate_only / causal B80`, versus the 1.05 band limit — a clear pass. Optimized strategies collapse toward μ_null once the reward-to-row association is destroyed, exactly as the Section 7a gate requires.

### 7b Effective Sample Size (PASS)

- B80 per-seed ESS across all 30 (strategy, seed) cells: min = 4,724; median = 50,551; max = 144,247; mean = 57,591.
- Per-strategy B80 medians: `random = 94,624`; `surrogate_only = 49,146`; `causal = 49,146`.
- Floor (Sprint 34 contract Section 7b): `max(1000, 452,949 / 100) = 4,530` — the minimum observed ESS (4,724) is still above the floor, and the medians are an order of magnitude above it.

### 7c Zero-support fraction (PASS)

Best-of-seed zero-support fraction at every B80 cell is 0.0. Every optimized policy assigns strictly-positive mass to every logged `(row, action)` pair. Threshold is 10%; observed is 0%.

### 7d Propensity sanity (PASS)

- Schema: **conditional** `P(item | position)` (Men/Random stores per-row `propensity_score ≈ 1/34`).
- Target under conditional schema: 1 / 34 ≈ 0.029411764706.
- Empirical mean: 0.029411764706.
- Relative deviation: ≈ 1.89e-15 (numerical noise); tolerance is 10% relative.

This confirms the Sprint 35.A smoke-test finding: OBD Men/Random's `propensity_score` is conditional, not joint.

### 7e DR / SNIPW cross-check (PASS)

- Maximum per-seed relative divergence between DR and SNIPW across 30 B80 (strategy, seed) cells: 0.48%.
- Number of seeds where divergence > 25% tolerance: 0.
- Per-seed divergence remains < 1% across every seed-strategy combination.

This gives strong joint evidence that the SNIPW primary estimate is not a variance artefact: a completely independent DR path tracks it within 1% on every seed.

### 7f Backend recording

`optimizer_path = "ax_botorch"`, Ax/BoTorch available, no fallback reason recorded. Every verdict cell in this report is Ax-primary; there is no mixing with RF-fallback results (Sprint 28/33 provenance policy).

## Per-Seed Detail Table (B80, SNIPW)

| Seed | random | surrogate_only | causal |
|------|--------|----------------|--------|
| 0 | 0.005452 | 0.006185 | 0.006185 |
| 1 | 0.005646 | 0.006187 | 0.006187 |
| 2 | 0.005286 | 0.006180 | 0.006180 |
| 3 | 0.005255 | 0.006183 | 0.006183 |
| 4 | 0.005987 | 0.006158 | 0.006158 |
| 5 | 0.005407 | 0.006185 | 0.006185 |
| 6 | 0.005280 | 0.006187 | 0.006187 |
| 7 | 0.005244 | 0.006184 | 0.006184 |
| 8 | 0.005288 | 0.006182 | 0.006182 |
| 9 | 0.005468 | 0.006189 | 0.006189 |

`surrogate_only` and `causal` agree bit-identically on every seed; both beat `random` on every seed.

## ESS Diagnostics (B80 verdict cells)

| Strategy | Median ESS | Mean ESS | Mean weight_cv | Mean max_weight | Mean zero_support | Mean n_effective_actions |
|----------|-----------:|---------:|---------------:|----------------:|------------------:|-------------------------:|
| random | 94,624 | 91,010 | 1.099 | 27.64 | 0.000 | 27.42 |
| surrogate_only | 49,146 | 40,881 | 1.965 | 34.00 | 0.000 | 24.91 |
| causal | 49,146 | 40,881 | 1.965 | 34.00 | 0.000 | 24.91 |

Optimized strategies concentrate more mass on preferred actions (higher `weight_cv`, higher `max_weight` — pegged at `1 / min_pscore = 34`), which is expected and safe — the 7b ESS floor still has a 10× margin. `random`'s more diffuse policy yields higher ESS but lower SNIPW, as expected.

## Null Control Detail

Full table in Section "7a Null control (PASS)" above. Takeaways: the null-control pass collapses every optimized cell toward the logged CTR μ_null = 0.005124 (maximum ratio 1.0154), which is well under the 1.05 band. There is no strategy-budget cell that inflated dangerously on permuted rewards.

## Runtime

- Dataset load: 0.9 s (single-pass pandas read of `men.csv` and `item_context.csv`; no OBP loader path — OBP 0.4.1's `pre_process` uses a positional `DataFrame.drop` signature that modern pandas rejects, so the loader reads the CSVs directly).
- Suite wall-clock: 2,880.4 seconds (48.0 minutes) on the dev laptop, macOS 25.2.0, darwin arm64, Python 3.13.12, single-process.
- Largest per-cell cost: causal / surrogate_only at B80 (~28 s per cell); random at B20 (~1 s per cell).

## Artifacts

- Full JSON artifact (local only, **not committed**): `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/sprint-35-open-bandit-benchmark/men_random_results.json` (173,002 bytes, 180 result rows, all 5 gate reports, provenance dict).
- Full run log (local only): `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/sprint-35-open-bandit-benchmark/benchmark.log` (8.2 MB).
- Benchmark CLI script: `scripts/open_bandit_benchmark.py` (this branch).
- Benchmark runner module: `causal_optimizer/benchmarks/open_bandit_benchmark.py` (this branch).
- Report-draft helper: `scripts/_open_bandit_report_helper.py` (this branch).
- Adapter: `causal_optimizer/domain_adapters/bandit_log.py` (Sprint 35.A, merged in PR #189).
- OPE stack: `causal_optimizer/benchmarks/open_bandit.py` (Sprint 35.B, merged in PR #188).
- Bridge (stacked base): `sprint-35/open-bandit-bridge` / PR #191 (Sprint 35 three bridge seams).

## Reproducibility

Exact run command:

```bash
uv run python scripts/open_bandit_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/open_bandit/open_bandit_dataset \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --strategies random,surrogate_only,causal \
  --null-control \
  --permutation-seed 20260419 \
  --output /.../artifacts/sprint-35-open-bandit-benchmark/men_random_results.json
```

## Interpretation

1. **The engine runs cleanly on a 452K-row, 34-action, 3-position logged bandit feedback dataset under Ax/BoTorch.** This is the project's first real multi-action benchmark; it confirms that the existing `ExperimentEngine` loop + Ax backend handles the 6-variable item-scoring surface without any modifications.
2. **All five Section 7 gates pass on the first run.** Null control, ESS, zero-support, propensity sanity, and DR/SNIPW cross-check are all clean with comfortable margins — the verdict is trustworthy.
3. **`surrogate_only` and `causal` tie exactly because the Sprint 35 adapter does not ship a multi-action prior graph.** Per Sprint 34 contract Section 4e, the first-run adapter is allowed to return `get_prior_graph() -> None`, which is exactly what Sprint 35.A did. The two strategies are therefore behaviorally identical on this slice — `causal` reduces to `surrogate_only` whenever the prior is null. Distinguishing `causal` from `surrogate_only` on Open Bandit is a Sprint 36+ conversation that requires someone to actually write down a bandit-log-compatible multi-action prior graph.
4. **Both optimized strategies beat `random` at certified significance on every seed at every budget (p = 0.0002, two-sided MWU).** The surface is learnable — the softmax-over-affinity policy family discovered by Ax/BoTorch gives a ~14% relative lift over the uniform-random baseline at B80. The optimizer consistently converges to a low-temperature, no-exploration softmax that weights `w_user_item_affinity` heavily and uses the `position_1_only` default, which matches the Sprint 34 contract Section 4c first-run rationale.
5. **Causal vs surrogate-only comparison on Open Bandit is deferred to Sprint 36+.** The exact tie seen here is the expected null result under a null prior graph. Per the Sprint 34 contract Section 12, this is the "least valuable but acceptable" first-run outcome: it carries the same weight as the Criteo near-parity row and preserves the Sprint 33 closure verdict (`GENERALITY IS REAL BUT CONDITIONAL`). A meaningful Open Bandit causal-vs-surrogate comparison needs a Sprint 36 multi-action prior graph, not a rerun of Sprint 35.

## Scope Boundaries (what this report does NOT claim)

Per Sprint 34 contract Section 9, this report does not claim any of the following — they are explicit out-of-scope for the first Open Bandit run and are intentionally omitted:

- Women or All campaigns; cross-campaign aggregation
- Bernoulli Thompson Sampling as primary logger
- Online learning, bandit training from scratch, or non-offline evaluation
- Slate-level / ranking-aware OPE
- Action embeddings, MIPS, Switch-DR, or DRos-primary
- Continuous-action OPE
- Multi-objective optimization (click + revenue, click + diversity, etc.)
- Deep or tree-based item-scoring inside `suggest_parameters()`
- Position-bias causal modeling beyond `"marginalize"` / `"position_1_only"`
- Auto-discovered causal graphs on OBD
- Multiple permutation seeds for the 7a null control
- A second dataset (MovieLens, Outbrain, Yahoo! R6)
- Any online-decisioning or A/B-test claim
- Revenue or cost metrics

Re-opening any of these is a Sprint 36+ conversation.

## Attribution

The Open Bandit Dataset and Open Bandit Pipeline are owned and published by ZOZO, Inc. (Saito, Aihara, Matsutani, and Narita, "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation," NeurIPS Datasets and Benchmarks 2021, [arXiv:2008.07146](https://arxiv.org/abs/2008.07146)). The dataset is CC BY 4.0; OBP is Apache 2.0. This benchmark redistributes neither; it only reads the locally-unzipped dataset under `--data-path`.
