Work on Issue #182: Sprint 34 Open Bandit benchmark contract and multi-action architecture brief.

Primary contract:
- /Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/24-sprint-34-recommendation.md

Read first:
1. /Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-31-open-bandit-access-and-gap-audit.md
2. /Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-31-hillstrom-lessons-learned.md
3. /Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-33-criteo-benchmark-report.md
4. /Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-33-generalization-scorecard.md
5. /Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-30-general-causal-portability-brief.md
6. /Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/16-agentic-science-architecture.md
7. /Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/07-benchmark-state.md
8. /Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/handoff.md
9. /Users/robertwelborn/Projects/causal-optimizer/CLAUDE.md

Goal:
- Convert the Open Bandit audit into a first executable contract for the next
  problem class: multi-action logged policy data.
- Do not implement the adapter or evaluator in this issue. This is a contract /
  architecture sprint.

Produce:
1. thoughts/shared/docs/sprint-34-open-bandit-contract.md

Optionally update only if the new contract truly changes project direction:
1. thoughts/shared/plans/07-benchmark-state.md
2. thoughts/shared/docs/handoff.md

The contract must settle:
1. the first Open Bandit data slice / campaign scope
2. the minimum multi-action adapter interface
3. the minimum OPE stack for the first trustworthy run
4. the benchmark objective / primary reward metric
5. null-control and support diagnostics for the multi-action setting
6. what remains out of scope for the first implementation sprint
7. whether the first implementation should depend directly on OBP or stay internal

Important standards:
1. Do not pretend Open Bandit is a drop-in extension of MarketingLogAdapter.
2. Distinguish clearly between contract requirements and future nice-to-haves.
3. Keep the first implementation scope narrow enough to be believable.
4. Separate architecture work from benchmark verdict work.
5. Use the audit as input, but make the new document executable as a real next-sprint contract.
6. Sprint 33 closure is already merged. Do not spend this issue re-litigating Hillstrom or Criteo.
7. Do not merge anything. Stop after PR creation / update and deliver back for
   human review.

Execution plan:
1. Create a branch for the Sprint 34 Open Bandit contract issue.
2. Read the Open Bandit audit and the post-Hillstrom / post-Criteo lessons.
3. Write the first executable Open Bandit contract / architecture brief.
4. If helpful, update restart docs so future agents know Open Bandit is the
   next frontier after Sprint 33 closure.
5. Open or update a PR with the contract doc once the scope is explicit enough
   for implementation planning.

Required workflow:
1. Use the `tdd` skill first.
2. Implement the work.
3. Run the `polish` skill before opening the PR.
4. Push the branch and open or update the GitHub PR.
5. Run the `gauntlet` skill on the PR.
6. Deliver back the PR URL and current status.
7. Do not merge. Wait for human approval.

Deliver back:
1. PR URL
2. branch
3. head commit
4. final recommendation
5. one-paragraph case for the recommendation
6. exact files changed
7. whether gauntlet is clean or what remains
