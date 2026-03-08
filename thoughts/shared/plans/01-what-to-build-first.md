# What to Build First

## The Problem

The codebase audit reveals a surprising finding: **we built seven sophisticated modules but
only wired one of them into the engine loop.** The loop currently runs as naive hill-climbing
with phase transitions. Six fully-implemented modules sit unused:

| Module | Built | Integrated into loop | Tested |
| --- | --- | --- | --- |
| Engine loop | Yes | Yes (it IS the loop) | Yes (5 tests) |
| Factorial/LHS design | Yes | Yes (exploration phase) | Yes (3 tests) |
| Screening (fANOVA) | Yes | **No** | No |
| Effect estimation | Yes | **No** | No |
| Off-policy prediction | Yes | **No** | No |
| MAP-Elites | Yes | **No** | No |
| Sensitivity validation | Yes | **No** | No |
| Causal discovery | Yes | **No** | No |
| Causal graph → focus vars | Yes | **Computed but ignored** | No |

The engine accepts a causal graph but barely uses it. The `_get_focus_variables()` function
computes ancestor variables from the DAG but the result is never applied to filter suggestions.
The observation-intervention tradeoff is implemented in `predictor/off_policy.py` but never
called. The effect estimator could replace the greedy keep/discard heuristic with statistical
testing, but it's never invoked.

**We have the parts. We need to connect them.**

## Priority Order

### Priority 1: Wire the loop (Week 1)

**Why first**: This is pure integration work — no new algorithms needed. It transforms the
system from "demo with disconnected modules" to "working causal optimization engine." Every
subsequent feature builds on a properly integrated loop.

#### 1a. Integrate effect estimation into keep/discard decisions

**Current behavior** (greedy, no statistical rigor):
```python
def _evaluate_status(self, metrics):
    if current_objective < best_objective:
        return KEEP
    else:
        return DISCARD
```

**Target behavior**: Use the EffectEstimator to compute bootstrap confidence intervals.
Only KEEP if the improvement is statistically significant (or if we have too few samples
for a meaningful test). This prevents keeping noise and discarding real improvements with
high variance.

**Implementation**: ~30 lines in `engine/loop.py`. The EffectEstimator already supports
bootstrap CIs. Wire it into `_evaluate_status()` when we have enough history (≥5 experiments).

#### 1b. Integrate off-policy prediction (observation-intervention tradeoff)

**Current behavior**: Every suggested experiment is executed.

**Target behavior**: Before running an experiment, ask `OffPolicyPredictor.should_run_experiment()`.
If the surrogate model is confident about the outcome AND the predicted outcome isn't promising,
skip the experiment. This saves expensive computation.

**Implementation**: ~20 lines in `engine/loop.py`. The predictor is already built. Add a
`fit()` call after each experiment and a `should_run_experiment()` check before each execution.

#### 1c. Integrate causal graph into suggestion strategy

**Current behavior**: `_get_focus_variables()` computes ancestor variables from the DAG
but the result is never used to filter suggestions.

**Target behavior**: In the optimization phase, only suggest changes to variables that are
causal ancestors of the objective. Pass `focus_variables` to the surrogate model to reduce
the search space dimensionality.

**Implementation**: ~15 lines in `optimizer/suggest.py`. The `_get_focus_variables()` function
already works. Thread its output into `_suggest_surrogate()` to filter the candidate space.

#### 1d. Integrate screening into the exploration→optimization transition

**Current behavior**: Phase transitions at hardcoded thresholds (n=10, n=50).

**Target behavior**: At the end of the exploration phase, run `ScreeningDesigner.screen()`
to identify important variables and interactions. Use the results to:
1. Set `focus_variables` (even without a prior causal graph)
2. Log which variables matter and which interactions exist
3. Inform whether to extend exploration or transition to optimization

**Implementation**: ~25 lines. The ScreeningDesigner is fully built.

#### 1e. Integrate MAP-Elites for diversity tracking

**Current behavior**: Only the single best result is tracked.

**Target behavior**: Maintain a MAP-Elites archive alongside the experiment log. When
suggesting in the exploitation phase, sample from diverse elites (not just perturb the best).
This prevents getting stuck in local optima.

**Implementation**: ~30 lines in `engine/loop.py`. Requires defining behavioral descriptors,
which domain adapters already provide via `get_descriptor_names()`.

**Total for Priority 1**: ~120 lines of integration code. No new algorithms. Transforms
the system from a demo to a working engine.

---

### Priority 2: POMIS computation (Week 2)

**Why second**: POMIS is the core theoretical contribution that makes this project novel.
It's what allows the optimizer to prune which variable subsets are worth experimenting with,
giving exponential speedup over naive search. Without POMIS, the causal graph only provides
"focus on ancestors" — useful but not transformative.

**Algorithm** (from Lee & Bareinboim, NeurIPS 2018):

The reference implementation at `github.com/sanghack81/SCMMAB-NIPS2018` (file `npsem/where_do.py`)
provides a compact ~100-line implementation:

```
POMISs(G, Y):
  1. Restrict graph to ancestors of Y
  2. Compute MUCT_IB(G, Y):
     - MUCT = Minimal Unobserved Confounder's Territory: starting from {Y},
       expand by finding c-components (connected via bidirected/confounding edges)
       and their descendants, until fixed point
     - IB = Interventional Border: parents of MUCT not in MUCT
  3. Apply intervention: H = G.do(IB)[MUCT | IB]
  4. Recursively enumerate via subPOMISs(H, Y, causal_order)
  5. Union with {frozenset(IB)}
```

**Required graph operations** (all standard, networkx suffices):
- Ancestors, Descendants, Parents, Children
- C-components (connected components via bidirected edges) — **requires extending CausalGraph
  to support bidirected edges for unobserved confounders**
- `do()` operator (remove incoming edges to intervened variables)
- Induced subgraph
- Topological ordering

**Key design decision**: Our `CausalGraph` type currently only stores directed edges.
POMIS requires bidirected edges to represent unobserved confounders. We need to extend
`CausalGraph` to support `bidirected_edges: list[tuple[str, str]]`.

**Deliverables**:
1. Extend `CausalGraph` with bidirected edge support
2. Implement POMIS computation in `optimizer/pomis.py`
3. Integrate POMIS into `_suggest_optimization()`: only suggest interventions on POMIS members
4. Update domain adapter causal graphs to include bidirected edges where confounders exist
5. Unit tests with known POMIS from the CBO paper benchmarks

**Estimated complexity**: Medium. ~150 lines of new code, mostly porting from the reference
implementation. The tricky part is the MUCT fixed-point computation and handling bidirected
edges cleanly.

---

### Priority 3: Benchmarks that prove causal reasoning helps (Week 3)

**Why third**: After wiring the loop (Priority 1) and implementing POMIS (Priority 2), we
need to prove the system actually works better than causal-agnostic optimization. Without
benchmarks, we're making claims we can't substantiate.

**Benchmark suite** (from the CBO and SCB papers):

#### 3a. ToyGraph (chain: X → Z → Y)

From Aglietti et al. (AISTATS 2020):
```python
X = epsilon_0                          # exogenous
Z = exp(-X) + epsilon_1
Y = cos(Z) - exp(-Z/20) + epsilon_2
```
- POMIS = [{Z}] — intervening on Z alone is sufficient
- A causal-agnostic optimizer wastes experiments varying X directly
- **What this proves**: POMIS correctly identifies that X only matters through Z

#### 3b. CompleteGraph (confounded, 7 variables)

From Aglietti et al.:
- 7 observed variables (A-F, Y) + 2 unobserved confounders (U1, U2)
- U1 confounds A↔Y; U2 confounds B↔Y
- POMIS = [{B}, {D}, {E}, {B,D}, {D,E}]
- Naive optimizer searches 2^6 = 64 subsets; POMIS reduces to 5
- **What this proves**: POMIS gives exponential search space reduction

#### 3c. High-dimensional sparse graph

Custom benchmark:
- 50 observed variables, only 3 are causal ancestors of Y
- Standard BO must search 50-dimensional space
- CBO with POMIS searches 3-dimensional space
- **What this proves**: Causal reasoning dominates in high-dimensional sparse settings

#### 3d. Interaction benchmark

Custom benchmark designed to show factorial screening beats one-at-a-time:
- 5 variables, two of which interact (help together, hurt alone)
- Greedy optimizer discards both; factorial design discovers the interaction
- **What this proves**: DoE-based screening catches what hill-climbing misses

**Deliverables**:
1. `benchmarks/` directory with synthetic SCMs
2. `benchmarks/runner.py` — runs causal-optimizer vs. baselines (random search, vanilla BO)
3. Convergence plots: experiments-to-optimum for each method
4. Integration test: `test_causal_beats_naive.py`

---

### Priority 4: Epsilon controller (observation-intervention tradeoff) (Week 4)

**Why fourth**: After the loop is wired and POMIS works, the epsilon controller is the next
multiplier. It decides when to use existing data (free) vs. run a new experiment (expensive).

**Algorithm** (from Aglietti et al., file `CBO.py`):
```python
coverage = volume(ConvexHull(observed_data)) / volume(full_domain)
rescale = n_observed / n_max
epsilon = coverage / rescale

if random() < epsilon:
    OBSERVE  # estimate from data
else:
    INTERVENE  # run experiment
```

**Implementation**: ~30 lines. Requires `scipy.spatial.ConvexHull` for volume computation.
This replaces the simple uncertainty threshold in `OffPolicyPredictor.should_run_experiment()`
with the theoretically grounded epsilon controller.

**Integration**: The epsilon controller works alongside POMIS — for each POMIS member, decide
independently whether to observe or intervene.

---

### Priority 5: Causal discovery from experiment logs (Week 5)

**Why fifth**: Priorities 1-4 assume a prior causal graph (provided by domain adapters). But
the real power comes from *learning* the graph from data. This enables the system to work
without domain knowledge.

**Approach**: Use the **PC algorithm** via the `causal-learn` library.

Why PC over NOTEARS:
- PC detects **latent confounders** (outputs bidirected edges), which is essential for POMIS
- NOTEARS only outputs directed edges — no confounder detection
- PC is better understood theoretically

**Implementation**:
1. Add `causal-learn` as an optional dependency
2. Update `discovery/graph_learner.py` to use `causal-learn`'s PC implementation
3. Wire into the engine loop: after exploration phase, learn graph from experiment log
4. Use learned graph for POMIS computation in optimization phase

**The pipeline becomes**:
```
Exploration (LHS, no graph) → Learn graph (PC) → Compute POMIS → Optimize (CBO)
```

---

### Priority 6: Do-calculus for observational estimation (Week 6)

**Why last among the critical features**: This is the most complex component and benefits
from all prior work being in place. Do-calculus allows estimating interventional effects
from observational data — the "observe" side of the observation-intervention tradeoff.

**Approach**: Wrap **DoWhy** rather than implement from scratch.

DoWhy handles the common cases (backdoor adjustment, front-door adjustment) and implements
the full ID algorithm for arbitrary graphs. Implementing do-calculus from scratch would be
a multi-month effort; wrapping DoWhy is ~50 lines.

```python
import dowhy
model = dowhy.CausalModel(data=df, treatment='X', outcome='Y', graph=dag)
identified = model.identify_effect()
estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
```

**Integration**: Use DoWhy estimates as prior mean functions for the GP surrogates (when
using Ax/BoTorch) or as predicted values in the RF surrogate.

---

## Summary: The Build Order

```
Week 1: Wire the loop ──────────── Integration only, no new algorithms
         ├─ Effect estimation in keep/discard
         ├─ Off-policy prediction before execution
         ├─ Causal focus variables in suggestions
         ├─ Screening at phase transitions
         └─ MAP-Elites for diversity

Week 2: POMIS computation ──────── Core novel algorithm (~150 lines)
         ├─ Extend CausalGraph with bidirected edges
         ├─ Port POMIS from reference implementation
         └─ Integrate into suggestion strategy

Week 3: Benchmarks ─────────────── Prove it works
         ├─ ToyGraph, CompleteGraph from CBO paper
         ├─ High-dimensional sparse (custom)
         ├─ Interaction detection (custom)
         └─ Convergence comparison vs. baselines

Week 4: Epsilon controller ─────── Observation-intervention tradeoff (~30 lines)

Week 5: Causal discovery ──────── Learn graphs from data (wrap causal-learn)

Week 6: Do-calculus ────────────── Observational estimation (wrap DoWhy)
```

## Why This Order

The ordering follows two principles:

**1. Maximize value per effort.** Week 1 is pure integration (~120 lines) that transforms
the system from a demo to a working engine. Everything after builds on that foundation.

**2. Each week's work makes the next week's work more testable.** POMIS (Week 2) needs the
wired loop (Week 1) to test. Benchmarks (Week 3) need POMIS (Week 2) to compare. The epsilon
controller (Week 4) needs benchmarks (Week 3) to validate. Discovery (Week 5) feeds into
POMIS. Do-calculus (Week 6) feeds into the epsilon controller.

## What We're NOT Building Yet

- Ax/BoTorch integration (the RF surrogate fallback works fine for now)
- Island model for MAP-Elites (single archive is sufficient)
- Persistent storage / async execution (premature for pre-alpha)
- Additional domain adapters (marketing and ML training are enough to prove the concept)
- LLM-guided mutations (only relevant for code optimization adapter)
- CLI / API / dashboard (no users yet)

## Dependencies to Add

```toml
[project.optional-dependencies]
causal = [
    "causal-inference-marketing @ git+https://github.com/datablogin/causal-inference-marketing.git",
    "dowhy>=0.11",        # do-calculus, backdoor/frontdoor adjustment
    "causal-learn>=0.1",  # PC algorithm for graph learning
]
```

## Key Design Decisions

### 1. Bidirected edges in CausalGraph

POMIS requires knowing about unobserved confounders. We need to extend `CausalGraph`:

```python
@dataclass
class CausalGraph:
    edges: list[tuple[str, str]]              # directed: X → Y
    bidirected_edges: list[tuple[str, str]]    # confounding: X ↔ Y (unobserved U → X, U → Y)
    nodes: list[str] = field(default_factory=list)
```

This is a breaking change to the type, but the only consumers are domain adapters (which we
control) and tests.

### 2. When to learn vs. use prior graphs

The engine should support three modes:
- **Prior only**: domain adapter provides graph, used as-is (current behavior)
- **Learned only**: no prior, learn from experiment data after exploration phase
- **Hybrid**: start with prior, refine from data (best but most complex)

Start with prior-only (Week 1-2) and learned-only (Week 5). Hybrid can come later.

### 3. Statistical threshold for keep/discard

The effect estimator needs a threshold. Options:
- **Fixed alpha** (e.g., p < 0.05): simple, standard
- **Adaptive alpha**: start permissive (keep more), tighten as experiments accumulate
- **Bayesian**: compute posterior probability of improvement

Start with fixed alpha = 0.1 (permissive, avoids discarding real improvements early).
Tighten to 0.05 after 20+ experiments.

### 4. POMIS vs. full search space

When POMIS is available, the optimizer should ONLY suggest interventions on POMIS members.
This is the key theoretical insight — exploring non-POMIS subsets is provably suboptimal.

However, if the causal graph is wrong, POMIS will be wrong. The sensitivity validator should
flag when results are fragile.

## Success Criteria

After Week 3 (benchmarks), we should be able to show:

1. **On ToyGraph**: causal-optimizer converges in ~5 experiments; random search takes ~20
2. **On CompleteGraph**: causal-optimizer explores 5 POMIS members; naive explores 64 subsets
3. **On high-dim sparse**: causal-optimizer finds optimum in O(k) experiments where k is
   number of causal ancestors; naive takes O(d) where d >> k
4. **On interaction benchmark**: factorial screening detects the A×B interaction; greedy misses it

These results would constitute a publishable contribution.
