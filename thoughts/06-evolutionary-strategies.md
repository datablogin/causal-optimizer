# Evolutionary Strategies and MAP-Elites

## Why Greedy Optimization Gets Stuck

Greedy hill-climbing converges to the nearest local optimum. In high-dimensional, non-convex
landscapes (which is most real-world optimization), this is rarely the global optimum.

Standard Bayesian optimization mitigates this via the exploration-exploitation tradeoff
(acquisition functions encourage exploring uncertain regions), but still maintains a single
best solution and optimizes toward it.

## MAP-Elites: Quality-Diversity Optimization

MAP-Elites (Mouret & Clune, 2015) takes a fundamentally different approach: maintain an
*archive* of diverse high-quality solutions.

### How It Works

1. Define **behavioral descriptors** — dimensions that characterize *how* a solution works
   (not just how good it is)
2. Discretize descriptors into a grid of cells
3. Each cell stores the **best solution** with that behavioral profile
4. New solutions compete only within their cell, not globally

### Example: ML Training Optimization

Behavioral descriptors: (model_size_bin, memory_usage_bin)

```
                    Memory Usage
                Low    Med    High
Model    Small  [sol1] [sol2] [sol3]
Size     Med    [sol4] [sol5] [sol6]
         Large  [sol7] [sol8] [sol9]
```

Each cell contains the best-performing solution with that size/memory profile. The archive
preserves solutions like "best small model under 4GB" even if "best large model at 40GB"
has a better absolute metric.

### Why This Matters

1. **Avoids local optima**: diversity pressure prevents premature convergence
2. **Discovers stepping stones**: a mediocre solution in one cell may mutate into a great
   solution in another — a path that greedy optimization would never explore
3. **Provides options**: the user gets a menu of solutions across tradeoff dimensions
4. **Enables cross-pollination**: combining strategies from different cells can produce
   novel solutions

## AlphaEvolve: The State of the Art

DeepMind's AlphaEvolve (May 2025) is the most sophisticated LLM-guided program optimization
system. Key architectural features:

### Dual LLM Strategy
- **Gemini Flash** (fast, cheap): generates diverse mutations for breadth
- **Gemini Pro** (slow, smart): generates sophisticated reasoning for depth
- 80% Flash, 20% Pro — most mutations should be diverse; a few should be clever

### MAP-Elites + Island Model
- Multiple independent MAP-Elites populations ("islands")
- Periodic migration of elites between islands
- Prevents all populations from converging to the same region

### Program-Level Evolution
- Evolves entire codebases, not just numerical parameters
- LLMs generate code mutations (not random mutations)
- Context includes: current code, behavioral descriptors, evaluation results

### Results
- 0.7% worldwide compute recovery at Google
- First improvement over Strassen's matrix multiplication algorithm in 56 years
- 23% kernel speedups over hand-tuned implementations

## Integration with Causal Reasoning

AlphaEvolve is purely evolutionary — no causal reasoning. Adding causal structure would:

1. **Guide mutations**: instead of random LLM-generated changes, use the causal graph to
   identify which components causally affect the objective. Mutate ancestors, not descendants.
2. **Interpret crossover**: when combining solutions from different cells, the causal graph
   indicates which components can be independently swapped vs. which are causally entangled.
3. **Attribute improvement**: when a mutation improves performance, causal analysis can
   identify *which part* of the mutation helped, enabling more targeted follow-up mutations.

This causal + evolutionary combination is novel — no existing system does it.

## Our Implementation

`evolution/map_elites.py` implements the core MAP-Elites archive:

- `add(result, fitness, descriptors)` — add a solution, competing within its cell
- `sample_elite()` — sample a random elite for mutation
- `sample_diverse(n)` — sample from different regions for cross-pollination
- `coverage` — fraction of the archive filled (exploration progress)

### Future Work

- Island model (multiple archives with migration)
- Descriptor auto-discovery (learn descriptors from data instead of specifying them)
- LLM-guided mutations for code optimization domains
- Causal-guided mutation strategy (mutate causal ancestors preferentially)

## References

- Mouret, Clune (2015). "Illuminating search spaces by mapping elites." arXiv:1504.04909.
- DeepMind (2025). "AlphaEvolve: A Gemini-powered coding agent for designing advanced
  algorithms." https://deepmind.google/blog/alphaevolve/
- OpenEvolve: https://huggingface.co/blog/codelion/openevolve
