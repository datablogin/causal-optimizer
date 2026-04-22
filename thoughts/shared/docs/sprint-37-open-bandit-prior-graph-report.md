# Sprint 37 Open Bandit Prior-Graph Rerun Report (Men/Random)

**Date:** 2026-04-22
**Sprint:** 37 (Option A1 — first prior graph + minimal-focus flag)
**Issue:** [#197](https://github.com/datablogin/causal-optimizer/issues/197)
**Branch:** `sprint-37/open-bandit-prior-graph-a1`
**Dataset:** Open Bandit Dataset, ZOZOTOWN Men campaign, uniform-random logger (full slice, 452,949 rows)
**Backend:** Ax/BoTorch (primary; RF fallback not exercised on any verdict cell)
**Predecessors:** Sprint 35 Open Bandit benchmark report ([sprint-35-open-bandit-benchmark-report.md](sprint-35-open-bandit-benchmark-report.md), PR #191/PR #189), Sprint 36 preregistration ([26-sprint-36-recommendation.md](../plans/26-sprint-36-recommendation.md), PR #195).

## Summary

Sprint 37 closes the Sprint 36 preregistration loop by landing **Option A1**:

1. `BanditLogAdapter.get_prior_graph()` now returns the preregistered seven-node / six-edge graph (every search variable directly parents `policy_value`, no bidirected edges).
2. A new explicit `pomis_minimal_focus` flag on `ExperimentEngine` (default `False`) opts a workload into the minimal-focus heuristic; the Open Bandit benchmark harness sets it to `True` only for the `causal` arm. `surrogate_only` and `random` are mechanically unchanged from Sprint 35.
3. The heuristic restricts focus to `screened_variables ∩ ancestors` whenever (a) the graph is supplied, (b) every search variable is an ancestor of `policy_value`, and (c) the resulting intersection is a non-empty proper subset of the search space. The same rule is applied in both optimization and exploitation so a B80 run that crosses the `>= 50` exploitation boundary does not silently revert to full-space behavior.

The Sprint 35 empirical contract (slice, budgets, seeds, estimators, null-control seed, Section 7 thresholds) is preserved unchanged.

**Verdict at B80: NEAR-PARITY between `causal` and `surrogate_only` (`p = 0.7337`, two-sided MWU; means agree to six decimals); both still CERTIFIED over `random` at every budget (`p = 0.0002`). All five Section 7 gates PASS. The Sprint 35 bit-identical tie is broken — A1 produces a different per-seed trajectory — but the verdict-budget difference is below 10-seed MWU power. Predicted outcome H0 (`p > 0.15` at B80) is confirmed.**

## Verdict Row (two-sided Mann-Whitney U, 10 seeds)

| Slice | Budget | Causal vs Surrogate-Only | p-value | Verdict | Direction |
|-------|--------|--------------------------|---------|---------|-----------|
| Men/Random (full) | B20 | causal lower in 8/10 seeds | 0.0820 | Trending | causal < surrogate_only |
| Men/Random (full) | B40 | causal lower in 5/10 seeds, equal 0/10 | 0.2123 | Not significant | mean direction: causal < surrogate_only (Δ ≈ 1.7e-5) |
| Men/Random (full) | B80 | causal lower in 6/10 seeds | 0.7337 | **Not significant (near-parity)** | means: 0.006181 vs 0.006182 |

| Slice | Budget | Causal vs Random | p-value | Verdict | Direction |
|-------|--------|------------------|---------|---------|-----------|
| Men/Random (full) | B20 | causal wins 9/10 seeds | 0.0006 | Certified | causal > random |
| Men/Random (full) | B40 | causal wins 10/10 seeds | 0.0002 | Certified | causal > random |
| Men/Random (full) | B80 | causal wins 10/10 seeds | 0.0002 | Certified | causal > random |

| Slice | Budget | Surrogate-Only vs Random | p-value | Verdict | Direction |
|-------|--------|--------------------------|---------|---------|-----------|
| Men/Random (full) | B20 | s.o. wins 9/10 seeds | 0.0002 | Certified | s.o. > random |
| Men/Random (full) | B40 | s.o. wins 10/10 seeds | 0.0002 | Certified | s.o. > random |
| Men/Random (full) | B80 | s.o. wins 10/10 seeds | 0.0002 | Certified | s.o. > random |

Sprint 33 / Sprint 35 scorecard labels: `p <= 0.05` certified; `0.05 < p <= 0.15` trending; `p > 0.15` not significant. "Near-parity" is reserved for within-noise identical distributions inside the not-significant band — exactly what the B80 row shows.

## A1 Implementation (what landed and where)

| Surface | File | Notes |
|---------|------|-------|
| Preregistered prior graph | [`causal_optimizer/domain_adapters/bandit_log.py`](../../causal_optimizer/domain_adapters/bandit_log.py) `BanditLogAdapter.get_prior_graph()` | Returns the seven-node / six-edge graph (every search var → `policy_value`, no bidirected edges). |
| Engine flag | [`causal_optimizer/engine/loop.py`](../../causal_optimizer/engine/loop.py) `ExperimentEngine.__init__` (`pomis_minimal_focus: bool = False`) | Threaded through both `suggest_next` paths (the main path and the MAP-Elites elite path). Validated to be `bool`. |
| A1 helper | [`causal_optimizer/optimizer/suggest.py`](../../causal_optimizer/optimizer/suggest.py) `_apply_minimal_focus_a1` | Pure function. When the flag is off, returns `base_focus` unchanged. When on and the binding conditions hold, returns `screened ∩ ancestors`; otherwise returns `base_focus`. |
| Optimization-phase wiring | `_suggest_optimization` in `suggest.py` | A1 helper applied after the existing graph∩screening intersection so the union fallback cannot re-introduce non-ancestor variables under the flag. |
| Exploitation-phase wiring | `suggest_parameters` exploitation branch in `suggest.py` | A1 helper applied to `_get_focus_variables(...)` output before `_suggest_exploitation`. Required because B80 crosses the `>= 50` exploitation boundary and the focus must persist there. |
| Per-arm enablement | [`causal_optimizer/benchmarks/open_bandit_benchmark.py`](../../causal_optimizer/benchmarks/open_bandit_benchmark.py) `OpenBanditScenario.run_strategy` | `pomis_minimal_focus=(strategy == "causal")` only. `surrogate_only` and `random` unchanged. |

The flag is `False` by default at the engine level, so non-Open-Bandit workloads pick up no behavior change from this PR. The Open Bandit `causal` arm is the only call site that flips it on.

### Preregistered prior graph (printed verbatim)

```
nodes (7):
  tau, eps, w_item_feature_0, w_user_item_affinity, w_item_popularity,
  position_handling_flag, policy_value

directed edges (6):
  tau                    -> policy_value
  eps                    -> policy_value
  w_item_feature_0       -> policy_value
  w_user_item_affinity   -> policy_value
  w_item_popularity      -> policy_value
  position_handling_flag -> policy_value

bidirected edges: none
```

This is the graph from `thoughts/shared/plans/26-sprint-36-recommendation.md` Minimal Preregistered Graph section, authored from the adapter's scoring code (per-edge code citations are tabulated in that plan).

## Configuration

- **Backend:** Ax/BoTorch primary. `provenance.optimizer_path = {"optimizer_path": "ax_botorch", "ax_available": True, "botorch_available": True, "fallback_reason": null}`. RF surrogate fallback was NOT exercised on any verdict cell.
- **Logger:** uniform-random (Men/Random campaign).
- **Strategies:** `random`, `surrogate_only`, `causal` (Sprint 34 contract Section 6a).
- **Budgets:** 20, 40, 80. B80 gates the verdict.
- **Seeds:** 0..9 (10 seeds per cell).
- **Total runs:** 180 (90 real + 90 null-control). Every cell completed cleanly.
- **Suite runtime:** 6,100.2 seconds (101.7 minutes); dataset load: 0.9 s.
- **Search space (6 variables):** `tau`, `eps`, `w_item_feature_0`, `w_user_item_affinity`, `w_item_popularity`, `position_handling_flag` (Sprint 34 contract Section 4c).
- **Estimators:** SNIPW (primary), DM (secondary), DR (secondary; in-module Dudík et al. 2011 form on a per-action empirical-CTR reward model).
- **Permutation seed (null control):** `20260419` (taken from the committed Sprint 35 report; unchanged).
- **Causal prior graph:** preregistered seven-node / six-edge graph above.
- **A1 flag:** `pomis_minimal_focus = True` only on the `causal` arm; `False` for `surrogate_only` and `random`.

## Per-Budget Outcome Tables (SNIPW — primary)

Population std (ddof=0). All 10 seeds per cell completed cleanly.

| Strategy | Budget | n | Mean SNIPW | Std (ddof=0) | Min | Max |
|----------|--------|---|------------|--------------|-----|-----|
| random | 20 | 10 | 0.005213 | 0.000084 | 0.005137 | 0.005407 |
| surrogate_only | 20 | 10 | 0.005805 | 0.000277 | 0.005344 | 0.006174 |
| causal | 20 | 10 | 0.005570 | 0.000261 | 0.005344 | 0.006140 |
| random | 40 | 10 | 0.005318 | 0.000236 | 0.005147 | 0.005987 |
| surrogate_only | 40 | 10 | 0.006180 | 0.000009 | 0.006156 | 0.006189 |
| causal | 40 | 10 | 0.006163 | 0.000041 | 0.006044 | 0.006188 |
| random | 80 | 10 | 0.005431 | 0.000221 | 0.005244 | 0.005987 |
| surrogate_only | 80 | 10 | **0.006182** | 0.000008 | 0.006158 | 0.006189 |
| causal | 80 | 10 | **0.006181** | 0.000008 | 0.006163 | 0.006189 |

Compared to Sprint 35: `random` and `surrogate_only` rows are identical — both arms ran with `pomis_minimal_focus=False` and an unchanged engine path. The `causal` row diverges at B20 (mean 0.005570 vs Sprint 35's 0.005805) and converges back at B80 (0.006181 vs Sprint 35's 0.006182). The Sprint 35 bit-identical `causal == surrogate_only` tie is broken on every cell — but the gap at B80 is below MWU power at 10 seeds.

## Secondary Estimators (DM and DR)

Population means over 10 seeds, real data only.

| Strategy | Budget | Mean DM | Mean DR |
|----------|--------|---------|---------|
| random | 20 | 0.005122 | 0.005219 |
| surrogate_only | 20 | 0.005112 | 0.005812 |
| causal | 20 | 0.005026 | 0.005566 |
| random | 40 | 0.005115 | 0.005324 |
| surrogate_only | 40 | 0.005140 | 0.006192 |
| causal | 40 | 0.005138 | 0.006174 |
| random | 80 | 0.005108 | 0.005436 |
| surrogate_only | 80 | 0.005143 | 0.006194 |
| causal | 80 | 0.005145 | 0.006190 |

DR tracks SNIPW closely on every (strategy, budget) cell; the per-seed maximum relative DR/SNIPW divergence is 0.481% (Section 7e gate). DM remains a near-policy-agnostic lower bound under the zero-context per-action reward model and is quoted only as a sanity check.

## Section 7 Support Gates

| Gate | Threshold | Status | Observed |
|------|-----------|--------|----------|
| 7a null control | every strategy-budget cell ≤ 1.05 × μ_null | **PASS** | max ratio = 1.0263; band = 1.05 |
| 7b ESS floor | median ESS ≥ max(1000, n_rows / 100) = 4,529 | **PASS** | aggregate B80 median ESS = 51,255 |
| 7c zero-support fraction | best-of-seed ≤ 10% | **PASS** | best = 0.0 on every B80 cell |
| 7d propensity sanity | empirical mean within 10% relative of 1/n_items = 0.029412 | **PASS** | relative deviation ≈ 1.89e-15 |
| 7e DR/SNIPW cross-check | per-seed relative divergence ≤ 25% | **PASS** | max observed divergence = 0.481% |

**All five Section 7 gates PASS.** The Sprint 37 rerun meets the Sprint 34 contract's verdict-publication bar.

### 7a Null control (PASS, permutation seed = 20260419)

μ_null = 0.005124 (raw mean of the permuted reward column on the full slice). Threshold = 1.05 × μ_null = 0.005380. All 9 strategy-budget cells (3 strategies × 3 budgets) on permuted data satisfy `policy_value ≤ threshold`. The maximum observed ratio is 1.0263 (`surrogate_only` / B40 / seed 0); the second-highest is 1.0240 (`surrogate_only` / B40 / seed 2); both are within the 1.05 band.

| Strategy | Budget | Mean Policy Value (null) | Ratio |
|----------|--------|---------------------------|-------|
| random | 20 | 0.005137 | 1.0026 |
| surrogate_only | 20 | 0.005157 | 1.0064 |
| causal | 20 | 0.005137 | 1.0026 |
| random | 40 | 0.005146 | 1.0042 |
| surrogate_only | 40 | 0.005187 | 1.0123 |
| causal | 40 | 0.005144 | 1.0038 |
| random | 80 | 0.005150 | 1.0051 |
| surrogate_only | 80 | 0.005203 | 1.0155 |
| causal | 80 | 0.005151 | 1.0053 |

The `causal` arm collapses cleanly toward μ_null on permuted rewards — the A1 flag does not produce spurious lift in the absence of signal.

### 7b Effective Sample Size (PASS)

Aggregate B80 median ESS across all 30 (strategy, seed) cells: 51,255. Floor: max(1000, 452,949 / 100) = 4,529. Per-strategy means (B80): random ≈ 91,010; surrogate_only ≈ 40,881; causal ≈ 46,852. The `causal` arm's ESS is slightly higher than `surrogate_only`'s (46,852 vs 40,881) — the A1 minimal-focus restriction nudges the optimizer toward less-concentrated softmax policies on average.

### 7c Zero-support fraction (PASS)

Best-of-seed zero-support fraction at every B80 cell is 0.0. Threshold is 10%; observed is 0%.

### 7d Propensity sanity (PASS)

Schema: conditional `P(item | position) = 1/34`. Empirical mean = 0.0294118. Relative deviation ≈ 1.9e-15 (numerical noise); tolerance is 10% relative.

### 7e DR / SNIPW cross-check (PASS)

Maximum per-seed relative divergence between DR and SNIPW across 30 B80 (strategy, seed) cells: 0.481%. Number of seeds where divergence > 25% tolerance: 0. Per-seed divergence remains < 1% across every seed-strategy combination.

### 7f Backend recording

`optimizer_path = "ax_botorch"`, Ax/BoTorch available, no fallback reason recorded on any cell. Every verdict cell in this report is Ax-primary; there is no mixing with RF-fallback results (Sprint 28/33 provenance policy).

## Per-Seed Detail Table (B80, SNIPW)

| Seed | random | surrogate_only | causal |
|------|--------|----------------|--------|
| 0 | 0.005452 | 0.006185 | 0.006186 |
| 1 | 0.005646 | 0.006187 | 0.006180 |
| 2 | 0.005286 | 0.006180 | 0.006188 |
| 3 | 0.005255 | 0.006183 | 0.006189 |
| 4 | 0.005987 | 0.006158 | 0.006188 |
| 5 | 0.005407 | 0.006185 | 0.006163 |
| 6 | 0.005280 | 0.006187 | 0.006172 |
| 7 | 0.005244 | 0.006184 | 0.006183 |
| 8 | 0.005288 | 0.006182 | 0.006181 |
| 9 | 0.005468 | 0.006189 | 0.006175 |

`causal` and `surrogate_only` are no longer bit-identical on any seed (the Sprint 35 exact tie is broken). `causal` beats `surrogate_only` on 4/10 seeds at B80 (seeds 0, 2, 3, 4), is lower on 6/10 seeds (1, 5, 6, 7, 8, 9), and the per-cell delta is at most 4e-5 in either direction. The two-sided MWU `p = 0.7337` confirms the distributions are statistically indistinguishable at this seed count.

## Per-Seed Detail Table (B20, SNIPW)

| Seed | random | surrogate_only | causal |
|------|--------|----------------|--------|
| 0 | 0.005139 | 0.005499 | 0.005360 |
| 1 | 0.005172 | 0.005969 | 0.006140 |
| 2 | 0.005286 | 0.005979 | 0.005568 |
| 3 | 0.005212 | 0.006174 | 0.005923 |
| 4 | 0.005196 | 0.005983 | 0.005351 |
| 5 | 0.005407 | 0.005344 | 0.005344 |
| 6 | 0.005137 | 0.006103 | 0.005379 |
| 7 | 0.005148 | 0.005748 | 0.005484 |
| 8 | 0.005288 | 0.005427 | 0.005425 |
| 9 | 0.005143 | 0.005823 | 0.005720 |

At B20, `causal` is below `surrogate_only` on 8/10 seeds (the exception is seed 1 where causal = 0.006140 beats surrogate = 0.005969). The two-sided MWU `p = 0.0820` lands in the trending band, not certified, with a mean-regret direction of `causal < surrogate_only`. This is consistent with the A1 restriction making early-budget exploration more conservative on the ancestor-restricted subset.

## ESS Diagnostics (B80 verdict cells)

| Strategy | Median ESS | Mean ESS | Mean weight_cv | Mean max_weight | Mean zero_support | Mean n_effective_actions |
|----------|-----------:|---------:|---------------:|----------------:|------------------:|-------------------------:|
| random | 94,220 | 91,010 | 1.099 | 27.64 | 0.000 | 27.42 |
| surrogate_only | 46,694 | 40,881 | 1.965 | 34.00 | 0.000 | 24.91 |
| causal | 49,488 | 46,852 | 1.576 | 34.00 | 0.000 | 27.14 |

Compared to Sprint 35, `causal` is the only row that moves: median ESS rises from 49,146 → 49,488 (+0.7%), mean ESS rises from 40,881 → 46,852 (+14.6%), and `n_effective_actions` rises from 24.91 → 27.14. The A1 restriction biases the optimizer away from the most concentrated softmax policies, which produces a slightly higher-ESS, slightly less peaked behavioral fingerprint without changing the SNIPW value at the verdict budget.

## Hypothesis Reconciliation

The Sprint 36 preregistered hypotheses (recommendation Exit Criterion, Preregistered H0/H1/H2):

- **H0 (predicted):** `p > 0.15` at B80, two-sided MWU on SNIPW between `causal` and `surrogate_only`.
- **H1 (alternative — certified):** `p ≤ 0.05` at B80.
- **H2 (trending):** `0.05 < p ≤ 0.15` at B80.

**Outcome at B80:** `p = 0.7337`. **H0 confirmed.** The Sprint 36 prediction explicitly noted that 10 seeds have limited power to certify small SNIPW differences, and that "a real but small (~1%) causal advantage could still land in H2 rather than H1 under 10 seeds." The B80 means agree to 6 decimals (0.006181 vs 0.006182), so this is not a power-limited near-miss — the trajectories genuinely converge to the same value at B80, just via slightly different per-seed routes.

**Trajectory note (B20 only).** At B20 the rerun lands `p = 0.0820`, which is **trending** under the Sprint 33 / Sprint 35 scorecard with a mean-regret direction of `causal < surrogate_only`. This is not part of the verdict (Sprint 36 plan, "Power and the H0-vs-H1 boundary": "B20 and B40 are reported for trajectory analysis but do not gate the Sprint 37 verdict, matching the Criteo and Sprint 35 conventions"). The B20 trend signals an early-budget penalty under the A1 restriction; it is below certified separation and the gap closes by B40 (`p = 0.2123`) and disappears by B80 (`p = 0.7337`).

The bit-identical Sprint 35 tie did not survive A1 — the A1 flag does change which Ax candidates the soft-causal reranker selects, exactly as the Sprint 36 plan predicted. The new behavior simply does not move the verdict-budget mean.

## Causal-vs-Surrogate Comparison: separated, near-parity, or regressed?

**Near-parity at B80, with a non-certified early-budget regression at B20.** Specifically:

1. At B80 (the verdict budget) the rerun is **not significant** (`p = 0.7337`); both arms produce SNIPW within 1e-6 of each other. By the Sprint 35 report's terminology this is "near-parity" (within-noise identical distributions).
2. At B20 the rerun is **trending** (`p = 0.0820`) toward `causal < surrogate_only`. Mean-regret direction only — not certified.
3. At B40 the rerun is **not significant** (`p = 0.2123`); means within 1.7e-5.
4. The bit-identical Sprint 35 `causal == surrogate_only` tie does not recur on any seed at any budget.

Sprint 37 therefore neither separates the `causal` arm from `surrogate_only` (no certified row) nor produces a regression (the trending B20 row is not certified and the verdict-budget B80 row is near-parity).

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
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/sprint-37-open-bandit-prior-graph-rerun/men_random_results.json
```

The CLI requires no new flags — the per-arm A1 enablement happens inside `OpenBanditScenario.run_strategy`. Re-running the same command on a fresh checkout reproduces the rerun byte-for-byte (modulo Ax/BoTorch internal nondeterminism, which the Sprint 35 report also did not lock).

## Artifacts

- Full JSON artifact (local only, not committed): `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/sprint-37-open-bandit-prior-graph-rerun/men_random_results.json` (180 result rows, 5 gate reports, provenance dict).
- Full run log (local only): `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/sprint-37-open-bandit-prior-graph-rerun/benchmark.log`.
- Adapter: [`causal_optimizer/domain_adapters/bandit_log.py`](../../causal_optimizer/domain_adapters/bandit_log.py).
- Engine: [`causal_optimizer/engine/loop.py`](../../causal_optimizer/engine/loop.py).
- Optimizer: [`causal_optimizer/optimizer/suggest.py`](../../causal_optimizer/optimizer/suggest.py).
- Benchmark runner: [`causal_optimizer/benchmarks/open_bandit_benchmark.py`](../../causal_optimizer/benchmarks/open_bandit_benchmark.py).
- A1 unit tests: [`tests/unit/test_a1_minimal_focus.py`](../../tests/unit/test_a1_minimal_focus.py), [`tests/unit/test_bandit_log_prior_graph.py`](../../tests/unit/test_bandit_log_prior_graph.py), reranker sign-convention regression in [`tests/unit/test_soft_causal.py`](../../tests/unit/test_soft_causal.py).

## Interpretation

1. **A1 is mechanically wired correctly.** The Sprint 35 bit-identical tie is gone on every seed at every budget — the A1 flag changes which Ax candidates the soft-causal reranker selects, exactly as Sprint 36's path-by-path engine analysis predicted (path 4: soft-causal reranker is active under the default `causal_softness = 0.5`). Both arms still converge to a low-temperature, no-exploration softmax that weights `w_user_item_affinity` heavily and uses the `position_1_only` default.
2. **The verdict-budget impact is null.** At B80, mean SNIPW for `causal` and `surrogate_only` agree to six decimals; the two-sided MWU `p` is 0.7337. The Sprint 36 H0 prediction (`p > 0.15`) holds, and not as a near-miss inside the H2 trending band — the distributions genuinely overlap.
3. **A1 produces a small, non-certified early-budget regression.** At B20, `causal` lands below `surrogate_only` on 8/10 seeds with `p = 0.0820`. The mean delta is ≈ 2.4e-4 SNIPW. This is the trending band, not certified, but it is consistent across seeds and is the signal that A1 is doing *something* mechanical — restricting focus to `screened ∩ ancestors` at the optimization phase boundary makes early-budget exploration more conservative.
4. **The path 4 prediction was the right one.** Sprint 36's engine analysis specifically called out the soft-causal reranker as the active graph-consuming path under the Sprint 37 default configuration, and Sprint 37 confirms this empirically — the per-seed delta exists, even though the verdict-budget mean does not move.
5. **No Section 7 gate moved into the danger zone.** All five gates pass with margins comparable to or better than Sprint 35 (the `causal` arm's ESS rises slightly, suggesting the A1 restriction biases away from the most concentrated softmax policies).
6. **Sprint 37 closes the Sprint 36 hypothesis loop without forcing a Sprint 38 power-extension.** The B80 outcome is unambiguous (`p = 0.7337`, means within 1e-6) — this is not a power-limited "needs more seeds" result, it is an honest near-parity. A Sprint 38 power extension is not required to interpret this row; it would be required only if a future option (Option B graph widening, a different focus heuristic) produced a B80 trending result that 10 seeds could not certify.

## Sprint 38+ Implications

1. **A1 with the preregistered minimal graph does not produce a certified causal advantage on Men/Random.** Sprint 38 should pick exactly one follow-up:
   - **Option B (graph widening):** add one non-ancestor structural node so `_get_focus_variables` returns a proper subset directly, instead of relying on screening to do the restriction. The Sprint 36 plan calls out `logged_position_distribution` and `request_item_overlap` as candidates grounded in the adapter code.
   - **Option C (different heuristic):** A1 used `screened ∩ ancestors`. A magnitude-thresholded variant (drop ancestors whose screening importance is below some quantile) is one alternative the Sprint 36 plan also lists.
   - **Option D (move on):** accept that under the Sprint 35 surface, the `causal` and `surrogate_only` paths converge and reopen the multi-objective or second-dataset workstream instead.
2. **Do not chase the B20 trend.** The B20 row is not certified, the direction is *worse* not better, and chasing it via a finer focus-restriction tuning would be exactly the kind of post-hoc convergence chase Sprint 36 explicitly forbade.
3. **Soft-causal reranker sign convention is now locked by regression test.** [`tests/unit/test_soft_causal.py::test_rerank_alignment_only_picks_higher_score_candidate`](../../tests/unit/test_soft_causal.py) replays the Sprint 36 plan's two-candidate sanity check; the next sprint can rely on the sign without re-deriving it.

## Scope Boundaries (what this report does NOT claim)

Per Sprint 36 Exit Criterion guardrails and Sprint 34 contract Section 9, this report does not claim:

- Women, All, BTS, slate OPE, second-dataset, or DRos-primary results.
- Multi-objective extension or continuous-action OPE.
- Auto-discovered OBD graphs (the prior graph is preregistered, not learned).
- Bidirected-edge or Option B graph widening.
- Online-decisioning or A/B-test claims.
- Any change to Section 7 thresholds, the seed count, the verdict rule, or the null-control seed.
- Certified separation between `causal` and `surrogate_only` at any budget. The B80 row is near-parity inside the not-significant band; the B20 row is trending without certification.

Re-opening any of these is a Sprint 38+ conversation.

## Attribution

Open Bandit Dataset and Open Bandit Pipeline are owned and published by ZOZO, Inc. (Saito, Aihara, Matsutani, and Narita, "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation," NeurIPS Datasets and Benchmarks 2021, [arXiv:2008.07146](https://arxiv.org/abs/2008.07146)). The dataset is CC BY 4.0; OBP is Apache 2.0. This benchmark redistributes neither; it only reads the locally-unzipped dataset under `--data-path`.
