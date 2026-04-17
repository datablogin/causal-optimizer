# Sprint 34 Recommendation

Updated: 2026-04-17

## Sprint Theme

**General Causal Autoresearch: Multi-Action Expansion After Binary Marketing Boundary-Mapping**

Sprint 33 closed with a clear empirical answer on the current binary
marketing line:

1. Hillstrom gave the first non-energy real-data result and favored
   `surrogate_only`
2. Criteo extended the test to a larger Ax-primary marketing dataset and ended
   in near-parity rather than a causal win
3. together, those two results suggest the current binary-treatment marketing
   contract is now better understood than underexplored

That means the next high-value step is not another immediate binary rerun.
It is the next problem class: logged multi-action recommendation / policy data.

## Goal

Sprint 34 should define the first executable Open Bandit benchmark contract and
the minimum architecture required to support it.

The sprint should end with:

1. a merged Open Bandit contract / architecture brief
2. a concrete implementation issue for the first multi-action benchmark harness
3. a clear decision on whether to depend on OBP directly or keep the first pass
   internal

Sprint 34 should answer:

1. what is the smallest honest Open Bandit benchmark the project can run?
2. what new adapter, search-space, and estimator interfaces are required?
3. what can remain shared with the current binary-treatment harness?
4. what should count as a trustworthy null control and support diagnostic in
   the multi-action setting?

## Why This Sprint Now

The project has now learned something specific from three real-data lanes:

1. ERCOT showed real-world differentiation, but only in one domain
2. Hillstrom showed a clean non-energy boundary where `surrogate_only` was
   stronger
3. Criteo showed near-parity even after the mandatory heterogeneous follow-up

That combination makes Open Bandit the right next move:

1. it tests generalization on a new intervention structure, not just a new
   binary uplift table
2. it pushes the project toward the "causal autoresearch" identity rather than
   deeper specialization inside one adapter family
3. it is explicitly the next queued dataset family after Hillstrom and Criteo
   in the earlier audit

## Main Workstream

### 1. Open Bandit Contract And Architecture Brief

This is the Sprint 34 critical path.

The work should:

1. define the first Open Bandit benchmark scope
2. choose one initial data slice / policy family to keep the first run narrow
3. specify the minimum multi-action adapter interface
4. specify the minimum OPE stack for the first honest run
5. define null-control, support, and estimator-stability gates
6. define what should remain out of scope for the first implementation sprint

The output should be one authoritative contract document that future
implementation work can execute without reopening the design.

## Recommended First Scope

Sprint 34 should prefer a narrow first Open Bandit contract:

1. one campaign-policy slice
2. offline evaluation only
3. one primary reward metric
4. one baseline random / logging-policy comparison
5. a limited action-space exposure path if the full item set is too large for a
   first pass

The sprint should avoid over-scoping into:

1. full recommender-system architecture
2. counterfactual ranking stacks
3. online-learning claims
4. generalized multi-objective policy optimization

## Parallel Lane

### 2. Sprint 33 Closure And Restart Sync

This should run in parallel with Sprint 34 planning, not ahead of it.

The closure lane should:

1. publish the Sprint 33 synthesis scorecard
2. update `07-benchmark-state.md`
3. update `handoff.md`
4. update README only if the merged Sprint 33 result materially changes the
   public framing

This lane is important, but it is not the next scientific frontier. Treat it as
documentation / memory synchronization work, not as the main research sprint.

## Success Criteria

Sprint 34 is successful if:

1. the project has a concrete Open Bandit contract rather than only an audit
2. the first multi-action benchmark issue can be opened without unresolved
   architectural ambiguity
3. the contract clearly distinguishes:
   - adapter requirements
   - evaluator requirements
   - benchmark claims
   - out-of-scope future work
4. the team does not accidentally reopen the binary marketing question instead
   of advancing the next problem class

## What A Good Outcome Looks Like

Best case:

1. the contract defines a small but honest first Open Bandit run
2. the needed interface changes are limited and well-bounded
3. the next implementation sprint is obvious

Still valuable:

1. Sprint 34 concludes that a first Open Bandit run needs deeper estimator work
2. the blocker is specific enough to choose the next technical task
3. the project still leaves the sprint with a clear architectural map

## What Not To Do

1. do not reopen Hillstrom or Criteo as the main lane before documenting the
   Sprint 33 result
2. do not start coding a multi-action adapter before the contract is written
3. do not claim Open Bandit is a drop-in fit for `MarketingLogAdapter`
4. do not mix contract design with implementation guesses that have not been
   validated against the actual dataset / loader APIs

## Recommended Order

1. run Sprint 33 closure in parallel as a short documentation lane
2. draft the Open Bandit contract / architecture brief
3. decide the first benchmark slice, evaluator, and support gates
4. open the first Open Bandit implementation issue at the end of Sprint 34

## Suggested Sprint 34 Issues

1. publish the Sprint 33 closure scorecard and sync restart docs
2. define the Open Bandit benchmark contract and multi-action architecture brief

## Exit Criterion

At the end of Sprint 34, we should know:

1. what the trusted post-Sprint 33 project position is
2. what the first honest Open Bandit benchmark should look like
3. what code needs to exist before that benchmark can run
