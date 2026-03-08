# Origin: Why This Project Exists

## The Spark

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) (March 2026) demonstrated
a compelling idea: give an AI agent a training script and let it experiment autonomously overnight.
Modify code, train for 5 minutes, check if the metric improved, keep or discard, repeat. You wake
up to ~100 experiments and (hopefully) a better model.

The approach works, but it's fundamentally **greedy hill-climbing over code**. It tests one change
at a time, uses a simple keep/discard heuristic, and has no mechanism for:

- Detecting interactions between changes
- Distinguishing signal from noise
- Reasoning about *why* a change helped
- Deciding which experiment to run next based on what's been learned

## The Insight

Causal inference provides a mature framework for exactly these problems. The field has spent
decades developing methods to answer "what would happen if I intervened?" from observational and
experimental data. These methods — propensity scores, doubly-robust estimation, structural causal
models, do-calculus — are well-understood theoretically and battle-tested in medicine, economics,
and marketing.

Meanwhile, Bayesian optimization and evolutionary strategies provide principled frameworks for
sequential experiment design. But they treat the system as a black box.

**The gap**: no existing system combines causal reasoning with automated experiment optimization.
The causal inference community builds tools for *analyzing* experiments. The optimization community
builds tools for *selecting* experiments. This project connects the two.

## The Key Papers

The most directly relevant academic work:

1. **Causal Bayesian Optimization** (Aglietti et al., AISTATS 2020) — extends Bayesian
   optimization with causal graphs. Uses POMIS (Possibly-Optimal Minimum Intervention Sets) to
   prune the search space and an observation-intervention tradeoff to decide when to use existing
   data vs. run new experiments. Applied to Apache Spark configuration tuning.

2. **Structural Causal Bandits** (Lee & Bareinboim, NeurIPS 2018) — uses causal graphs to
   identify which subsets of interventions could possibly be optimal, achieving orders-of-magnitude
   faster convergence than causal-insensitive strategies.

3. **AlphaEvolve** (DeepMind, May 2025) — LLM-guided program optimization using MAP-Elites for
   diversity. Not causal, but demonstrates the value of population-based search over greedy
   hill-climbing. Achieved 0.7% worldwide compute recovery at Google.

4. **Bandits with Unobserved Confounders** (Bareinboim, Forney, Pearl, NeurIPS 2015) — showed
   that in the presence of confounders, rational agents need *both* observational and
   interventional data.

## The Opportunity

No production-ready system combines:

- Causal discovery (learn the DAG from data)
- Experimental design (factorial, not one-at-a-time)
- Doubly-robust estimation (was the improvement real?)
- Causal Bayesian optimization (what to try next, given the DAG)
- Evolutionary diversity (avoid local optima)
- Off-policy evaluation (predict without running)
- Sensitivity analysis (is this robust?)

This project builds that system — domain-agnostic, with adapters for marketing, ML training,
manufacturing, and drug discovery.
