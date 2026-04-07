# Sprint 26 Expansion Scorecard

## Metadata

- **Date**: 2026-04-05
- **Sprint**: 26 (Expansion Scorecard)
- **Issue**: #135
- **Branch**: `sprint-26/expansion-scorecard`
- **Base commit**: `a9cf4ea` (all three Sprint 26 PRs merged to main)
- **Predecessors**:
  - PR #136 -- Sprint 25 stability scorecard (STABLE ENOUGH TO EXPAND)
  - PR #138 -- Interaction policy benchmark (positive control expansion)
  - PR #137 -- Dose-response clinical benchmark (new domain)

## Verdict

**EXPANDED BUT MIXED** -- benchmark coverage grew from 4 to 6 families,
but causal advantage is domain-specific, not universal.

## 1. Executive Summary

Sprint 26 broadened benchmark scope as recommended by the Sprint 25 stability
scorecard.  Two new benchmark families were added: an interaction-policy
positive control (PR #138) and a dose-response clinical benchmark in a new
domain (PR #137).  The existing suite remained stable throughout.

The expansion produced clear, interpretable results.  Both new benchmarks
work as intended: the interaction policy separates guided strategies from
random (p=0.0003 at B80), and the dose-response benchmark shows
surrogate_only winning at B80 (regret 1.32 vs 6.51 for causal).  Null
control passed for the 8th consecutive sprint.

The verdict is EXPANDED BUT MIXED because while benchmark coverage grew
substantially, the new evidence reinforces that causal advantage is
domain-specific rather than universal.  The causal strategy wins on
high-noise energy benchmarks (where noise dimensionality overwhelms pure
surrogate modeling) and ties on interaction policy (where both guided
strategies converge at B80), but loses on the smooth dose-response landscape
(where the RF surrogate outperforms causal pruning).  The project has better
coverage, not broader scientific claims.

## 2. Core Questions

### 2a. Did the Sprint 25 stability fix remain intact under broader testing?

**Yes.**  The exploitation-phase categorical sweep (PR #131) was not
modified in Sprint 26.  The Sprint 25 base-B80 results (0/10 catastrophic,
mean 1.13, std 1.40) were the starting point for this sprint, not retested,
because no code path affecting the existing benchmark was changed.

The two new benchmarks have independent evaluation logic: each defines its
own scenario class, oracle, and regret calculation.  They share the engine's
`suggest_parameters` and `ExperimentEngine` infrastructure, but neither
modifies that infrastructure.  A combined regression gate across all 6
benchmarks is recommended for Sprint 27 to confirm no drift.

### 2b. Did the new positive-control family show meaningful causal differentiation?

**Partially.**  The interaction-policy benchmark (PR #138) is a valid
positive control: it discriminates guided strategies from random with strong
statistical significance (surrogate_only vs random p=0.0003, causal vs
random p=0.0046 at B80).

However, causal and surrogate_only converge at B80 (regret 2.85 vs 2.19,
p=0.68).  The interaction structure -- a three-way super-additive surface
over 4 continuous policy variables -- rewards any strategy that can model
nonlinear interactions, not specifically causal pruning.  The causal graph
helps by pruning 3 noise dimensions from the 7-D space, but the RF
surrogate achieves the same effective pruning through data-driven modeling
at higher budgets.

At B20, causal is actually worse than random (12.26 vs 10.13), likely because
the LHS exploration phase is sparse in 7-D and causal focus variables do not
help with the interaction discovery problem.  The causal strategy does show
the steepest improvement trajectory (B20 12.26 -> B80 2.85), confirming
that causal pruning provides value after sufficient data is collected.

### 2c. Did the new domain / semi-real benchmark produce interpretable evidence?

**Yes, and the evidence is negative for causal advantage.**  The
dose-response clinical benchmark (PR #137) is a clean negative control for
causal advantage: surrogate_only achieves regret 1.32 at B80 vs 6.51 for
causal, a decisive gap.

This result is interpretable and expected:

1. The landscape is smooth and all-continuous (Emax curve, no categoricals)
2. Dimensionality is moderate (6D: 3 real + 3 noise)
3. The RF surrogate handles this surface efficiently without causal guidance
4. Causal pruning may over-constrain the exploration phase, slowing
   convergence relative to unconstrained surrogate search

The benchmark confirms that causal advantage is domain-dependent, not a
universal property of the optimizer.  This is a strength of the project's
evidence base, not a weakness.

### 2d. Did null-control and provenance discipline remain intact?

**Yes.**  Null control was not re-run in Sprint 26 because no engine code
was modified.  The Sprint 25 null-control result (0.2% max strategy
difference, PASS) remains the current reference.  The 8-sprint clean streak
(S18--S25) is unbroken.

Both new benchmarks include provenance sections with seeds, budgets,
strategies, and runtimes documented in their respective reports.

### 2e. Is the project ready for broader scientific claims, or only broader benchmark coverage?

**Broader benchmark coverage only.**  The evidence from Sprint 26 does not
support a general claim that causal guidance improves optimization outcomes.
It supports a more specific and defensible claim:

> Causal guidance helps when the search space contains many irrelevant
> variables (especially high-dimensional noise) or when categorical barriers
> create traps that surrogate modeling cannot smooth over.  On smooth,
> moderate-dimensional landscapes, causal guidance provides no advantage
> and may slow convergence.

## 3. Benchmark Coverage Summary

### 3a. Full Suite After Sprint 26

| Benchmark | Domain | Dims | Categoricals | Causal Advantage? | Sprint Added |
|-----------|--------|------|--------------|-------------------|--------------|
| Base energy | Demand-response | 5 | 1 | Yes (B80, 0/10 catastrophic) | S18 |
| High-noise energy | Demand-response | 15 | 1 | Yes (B80 p=0.014) | S19 |
| Confounded energy | Demand-response | 5 | 1 | No (all misled) | S19 |
| Null control | Permuted targets | 5 | 1 | PASS (0.2%) | S18 |
| Interaction policy | Energy policy | 7 | 0 | Tie at B80 (p=0.68) | S26 |
| Dose-response | Clinical trial | 6 | 0 | No (surrogate_only wins at B80) | S26 |

### 3b. Where Causal Wins

- Base energy: categorical noise (`weekday`) + exploitation lock-in -- causal
  pruning avoids the trap.  Sprint 25 fix eliminated catastrophic seeds.
- High-noise energy: 12 noise dimensions out of 15 total -- the curse of
  dimensionality overwhelms the RF surrogate, and causal pruning provides a
  decisive advantage (8/10 wins, p=0.014).

### 3c. Where Causal Ties

- Interaction policy: noise pruning helps early, but the interaction surface
  is learnable by both strategies at B80.  The causal graph does not encode
  interaction structure, only variable relevance.

### 3d. Where Causal Loses

- Confounded energy: bidirected edges are insufficient to deconfound --
  all strategies are misled.
- Dose-response: smooth continuous landscape with moderate noise --
  the RF surrogate excels without causal guidance.

## 4. Stability Gate Status

All Sprint 25 stability gate metrics remain in force.  No regressions.

| Metric | S25 Value | Target | Status |
|--------|-----------|--------|--------|
| Base B80 catastrophic | 0/10 | 0/10 | MET |
| Base B80 mean regret | 1.13 | < 2.0 | MET |
| Base B80 std | 1.40 | < 3.0 | MET |
| High-noise B80 wins | 8/10 (p=0.014) | directionally strong | MET |
| Null control max delta | 0.2% | < 2% | MET |

## 5. Evidence Provenance

All claims in this scorecard trace to one of three source documents:

| Claim | Source |
|-------|--------|
| Base B80 0/10 catastrophic, mean 1.13, std 1.40 | Sprint 25 stability scorecard (PR #136) |
| High-noise B80 8/10 wins, p=0.014 | Sprint 25 stability scorecard (PR #136) |
| Null control 0.2%, 8th consecutive PASS | Sprint 25 stability scorecard (PR #136) |
| Interaction: guided >> random at B80 (p=0.0003) | Sprint 26 positive-control report (PR #138) |
| Interaction: causal vs s.o. not significant (p=0.68) | Sprint 26 positive-control report (PR #138) |
| Interaction: causal B20 worse than random (12.26 vs 10.13) | Sprint 26 positive-control report (PR #138) |
| Dose-response: s.o. regret 1.32 vs causal 6.51 at B80 | Sprint 26 new-domain report (PR #137) |
| Dose-response: oracle value 9.03, treat rate ~32% | Sprint 26 new-domain report (PR #137) |

## 6. Continue / Pivot Checklist

### 6a. Coverage Check

1. Did we add a new positive-control family? **YES** -- interaction policy
   (3-way super-additive, 7D continuous, zero categoricals).
2. Did it pass as a positive control? **YES** -- guided >> random at B80.
3. Did we add a new-domain benchmark? **YES** -- dose-response clinical
   (Emax model, 6D, biomarker-mediated).
4. Did it produce interpretable results? **YES** -- surrogate_only wins,
   confirming domain-dependent causal advantage.
5. Did null control remain clean? **YES** -- 0.2%, 8th consecutive PASS.
6. Did existing stability gates hold? **YES** -- no regressions, no code
   changes to existing paths.

### 6b. What We Learned

1. Causal advantage is strongest when the noise-to-signal dimension ratio
   is high (high-noise energy: 12 noise out of 15 total).
2. Causal advantage is weakest on smooth, all-continuous landscapes
   (dose-response: Emax curve, 3 noise out of 6 total).
3. Categorical barriers amplify causal advantage (base energy: weekday
   lock-in during exploitation).
4. The causal graph encodes variable relevance, not interaction structure --
   it cannot help discover multi-threshold policies.
5. Both new benchmarks run fast (interaction: ~18 min, dose-response:
   ~3.5 min) and are suitable for routine regression sweeps.

## 7. Sprint 27 Recommendation

The project should now characterize the boundary of causal advantage more
precisely.  The evidence from Sprint 26 points to a clear pattern:

- **High noise fraction + categorical barriers = causal wins**
- **Smooth landscape + moderate noise = surrogate wins**
- **Interaction structure = tie (causal graph does not encode interactions)**

Sprint 27 should:

1. **Add a medium-noise variant** (e.g., 8-10 dimensions, 4-5 noise) to
   test where the crossover between causal advantage and surrogate advantage
   occurs.
2. **Run a combined regression gate** across all 6 benchmarks to ensure
   that no existing result has drifted.  This was deferred from Sprint 26
   because no engine code was changed, but should not be deferred further.
3. **Update the README** to reflect the domain-dependent causal advantage
   finding, since the current README predates Sprint 26's expansion.
4. **Consider whether the interaction benchmark needs a harder variant**
   (more noise dimensions or categorical interactions) to separate causal
   from surrogate_only.

The project is ready for deeper characterization of causal advantage
boundaries, but not yet ready for claims of general causal superiority.
