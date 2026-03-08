# Causal Bandit Algorithms

## The Connection to Optimization

Multi-armed bandits formalize the exploration-exploitation tradeoff: each "arm" is a candidate
action, and pulling an arm reveals a reward. The goal is to maximize cumulative reward (or
equivalently, minimize regret — the gap between your choices and the best arm).

In our setting, each candidate experiment configuration is an arm. Pulling an arm means running
the experiment. The reward is the objective metric improvement.

## Key Papers

### Bandits with Unobserved Confounders (Bareinboim, Forney, Pearl, NeurIPS 2015)

The foundational paper. Shows that in the presence of unobserved confounders, a rational agent
needs *both* observational and interventional data to achieve low regret.

**Key insight**: if you only use interventional data (standard bandit), you ignore useful
information from observational data. If you only use observational data, confounders bias your
estimates. The optimal strategy combines both.

This directly motivates our observation-intervention tradeoff.

### Structural Causal Bandits: Where to Intervene? (Lee & Bareinboim, NeurIPS 2018)

The critical follow-up. Introduces **POMIS** (Possibly-Optimal Minimal Intervention Sets) for
bandits:

**Problem**: with N variables, there are 2^N possible intervention subsets. Which should the
agent explore?

**Solution**: use the causal graph to identify POMIS — the minimal subsets that could possibly
be optimal. This is a *graphical* criterion (computable from the DAG structure alone).

**Result**: orders of magnitude faster convergence than causal-insensitive bandit strategies.

**Code**: https://github.com/sanghack81/SCMMAB-NIPS2018

### Causal Bandits without Prior Knowledge (Kroon et al., 2022)

Removes the requirement for knowing the full causal graph. Uses separating sets instead.
Relevant for settings where we're learning the graph as we go.

### Contextual MAB for Causal Marketing (Amazon, 2018)

Applied causal bandits to targeting "persuadable" customers. Key idea: optimize the *causal
treatment effect* (uplift), not the raw outcome. Uses doubly-robust estimators.

This is directly relevant to the marketing adapter: target customers where the marketing
*causes* the most incremental lift, not where outcomes are highest (which may be due to
pre-existing purchase intent).

## How This Maps to causal-optimizer

| Bandit Concept | Our Implementation |
| --- | --- |
| Arms | Candidate experiment configurations |
| Pulling an arm | Running an experiment (`ExperimentRunner.run()`) |
| Reward | Objective metric (e.g., val_bpb, conversion rate) |
| Regret | Experiments wasted on suboptimal configurations |
| POMIS | Variables/subsets worth testing (from causal graph) |
| Observational data | Experiment log history |
| Interventional data | New experiment results |

## Implementation Considerations

### POMIS Computation

POMIS is computed from the DAG structure alone. The algorithm:
1. Identify all possible intervention targets (manipulable variables)
2. For each subset, check if it's a MIS (minimal set that uniquely determines the
   interventional distribution on the target)
3. Filter to POMIS (subsets that could be optimal under some SCM parameterization)

This is a graph-theoretic computation — we can implement it over our `CausalGraph` type
using networkx.

### Regret Bounds

Causal bandits with POMIS achieve regret bounds that scale with |POMIS| rather than 2^N.
In practice, |POMIS| is often O(N) or smaller, giving exponential improvement over
causal-agnostic approaches.

### Integration with Bayesian Optimization

CBO (paper 02) is essentially "causal bandits + Bayesian optimization." The POMIS pruning
from causal bandits combines with the surrogate model + acquisition function from BO.
Our `optimizer/suggest.py` should eventually implement this full combination.

## References

- Bareinboim, Forney, Pearl (2015). "Bandits with Unobserved Confounders." NeurIPS.
- Lee, Bareinboim (2018). "Structural Causal Bandits: Where to Intervene?" NeurIPS.
- Kroon et al. (2022). "Causal Bandits without Prior Knowledge." CLeaR.
- Amazon (2018). "Contextual Multi-Armed Bandits for Causal Marketing."
