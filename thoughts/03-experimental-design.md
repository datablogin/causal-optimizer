# Design of Experiments for Function Optimization

## Why One-at-a-Time Testing Fails

Greedy hill-climbing (autoresearch-style) tests one change at a time. This has a fundamental
blind spot: **interaction effects**.

Consider two candidate changes:
- Change A: switch optimizer from AdamW to Muon
- Change B: double the learning rate

Testing A alone: performance gets worse (Muon needs different LR)
Testing B alone: performance gets worse (AdamW diverges at 2x LR)
Testing A+B together: performance improves significantly (Muon + higher LR is optimal)

A greedy optimizer will discard both A and B individually and *never discover* that A+B helps.
This is not a contrived example — it's common in ML hyperparameter tuning, marketing channel
interactions, and drug combination therapy.

## Factorial Designs

### Full Factorial
- Test all combinations of k factors at L levels: L^k experiments
- For k=5 binary factors: 32 experiments
- Estimates all main effects and all interactions
- Expensive for large k but definitive

### Fractional Factorial
- Trade off higher-order interactions for fewer experiments
- Resolution III (2^(k-p), p generators): main effects estimable, confounded with 2-way
  interactions. Example: 5 factors in 8 runs instead of 32.
- Resolution IV: main effects clear, 2-way interactions confounded with each other
- Resolution V: main effects and 2-way interactions all estimable
- The sparsity-of-effects principle: most real systems are dominated by main effects and
  low-order interactions. Higher-order interactions are rare and small.

### When to Use
- **Screening phase**: many candidate changes, need to identify which matter
- **Interaction detection**: suspect synergies between changes
- **Early exploration**: before committing to Bayesian optimization

## Latin Hypercube Sampling (LHS)

- Space-filling design for continuous parameters
- Guarantees uniform marginal coverage with fewer samples than grid search
- Each parameter value appears exactly once in each "row" and "column"
- Good for initial exploration before model-based optimization
- Available via `scipy.stats.qmc.LatinHypercube`

## Response Surface Methodology (RSM)

- Fits polynomial models (usually quadratic) to experimental data
- Central Composite Design (CCD) or Box-Behnken designs
- Useful after screening to find the optimum of the important factors
- Less powerful than Bayesian optimization for complex landscapes but much simpler

## Integration with Causal Optimizer

The `designer/` module implements:

1. **Full factorial** (`FactorialDesigner.full_factorial`) — for small search spaces or when
   comprehensive interaction data is needed
2. **Fractional factorial** (`FactorialDesigner.fractional_factorial`) — for screening many
   variables efficiently
3. **Latin Hypercube** (`FactorialDesigner.latin_hypercube`) — for continuous space exploration
4. **Screening analysis** (`ScreeningDesigner.screen`) — fANOVA-style variable importance +
   pairwise interaction detection via random forest

### The screening → optimization handoff

1. Start with fractional factorial or LHS covering all variables
2. Run `ScreeningDesigner.screen()` to identify important variables and interactions
3. Reduce the search space to important variables only
4. Switch to Bayesian optimization or CBO on the reduced space
5. This two-phase approach is more efficient than full-space BO from the start

## Key Tools

- **pyDOE3**: Python library for factorial, Plackett-Burman, LHS, and other designs
  https://github.com/relf/pyDOE3
- **scipy.stats.qmc**: Latin Hypercube and other quasi-Monte Carlo samplers
- **fANOVA**: functional ANOVA for hyperparameter importance
  https://github.com/automl/fanova

## References

- Fisher, R.A. (1935). "The Design of Experiments." — the original
- Montgomery, D.C. (2017). "Design and Analysis of Experiments." 9th ed. — the textbook
- Hutter et al. (2014). "An Efficient Approach for Assessing Hyperparameter Importance."
  ICML. (fANOVA paper)
