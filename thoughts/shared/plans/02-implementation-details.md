# Implementation Details: Key Algorithms

Reference material for building the critical components.

## 1. POMIS Algorithm (Port from Lee & Bareinboim)

Source: `github.com/sanghack81/SCMMAB-NIPS2018`, file `npsem/where_do.py`

### Required Graph Operations

```
An(V)     — ancestors of V (all nodes with directed path to V)
De(V)     — descendants of V
Pa(V)     — parents of V (direct predecessors)
Ch(V)     — children of V (direct successors)
C-comp(V) — c-component of V: nodes connected via bidirected edges
do(X)     — graph surgery: remove all incoming edges to X
G[S]      — induced subgraph on node set S
```

### MUCT and IB

```
MUCT(G, Y):
    T = {Y}
    repeat until convergence:
        T_new = T
        for each node v in T:
            T_new = T_new ∪ C-comp(v)      # add nodes connected by bidirected edges
        T_new = T_new ∪ De(T_new)           # add all descendants
        if T_new == T: break
        T = T_new
    return T

IB(G, Y):
    T = MUCT(G, Y)
    return Pa(T) \ T                        # parents of MUCT that aren't in MUCT

MUCT_IB(G, Y):
    return (MUCT(G, Y), IB(G, Y))
```

### POMIS Algorithm

```
POMISs(G, Y):
    G = G[An(Y)]                            # restrict to ancestors of Y
    (Ts, Xs) = MUCT_IB(G, Y)
    H = G.do(Xs)[Ts ∪ Xs]                  # intervene on IB, take subgraph
    order = causal_order(Ts \ {Y})          # topological sort
    result = subPOMISs(H, Y, order)
    result.add(frozenset(Xs))               # add the IB itself
    return result

subPOMISs(G, Y, order):
    if not order:
        return {frozenset()}                # empty intervention
    result = set()
    for i, Xi in enumerate(order):
        G_i = G.do(Xi)                      # intervene on Xi
        (Ts_i, Xs_i) = MUCT_IB(G_i, Y)
        remaining = [x for x in order[i+1:] if x in Ts_i]
        H_i = G_i[Ts_i ∪ Xs_i]
        sub_results = subPOMISs(H_i, Y, remaining)
        for s in sub_results:
            result.add(s | {Xi} | frozenset(Xs_i))
    return result
```

### Verification Test Cases

**ToyGraph** (X → Z → Y):
- An(Y) = {X, Z, Y}
- MUCT({Y}) = {Y} (no bidirected edges)
- IB = Pa({Y}) \ {Y} = {Z}
- POMIS = [{Z}]

**With confounder** (X → Z → Y, X ↔ Y via U):
- MUCT({Y}) = {X, Y} (X and Y share bidirected edge, plus descendants)
- IB = Pa({X, Y}) \ {X, Y} = {Z} (Z is parent of Y, not in MUCT... wait, need to trace)
- Actually: C-comp(Y) = {X, Y} via bidirected edge. De({X,Y}) includes Z and Y.
  So MUCT = {X, Y, Z}. IB = Pa({X,Y,Z}) \ {X,Y,Z} = {} (X has no parents outside).
  This would mean POMIS = [{}] — observational only, which is wrong.

Need to trace more carefully. The reference implementation handles this correctly.
**Port the code, don't re-derive.**

---

## 2. Epsilon Controller (from CBO)

Source: `github.com/VirgiAgl/CausalBayesianOptimization`, file `CBO.py`

### Algorithm

```python
def compute_epsilon(observed_data, domain_bounds, n_current, n_max):
    """Decide whether to observe (use existing data) or intervene (run experiment).

    Returns probability of choosing observation over intervention.
    """
    if len(observed_data) < 3:
        return 0.0  # not enough data to estimate coverage

    try:
        hull = ConvexHull(observed_data)
        coverage = hull.volume
    except QhullError:
        coverage = 0.0  # degenerate (collinear points, etc.)

    # Total volume of the intervention domain
    total_volume = np.prod([ub - lb for lb, ub in domain_bounds])

    if total_volume == 0:
        return 0.0

    coverage_ratio = coverage / total_volume
    rescale = n_current / n_max

    if rescale == 0:
        return 0.0

    epsilon = coverage_ratio / rescale
    return min(epsilon, 1.0)


def should_observe(observed_data, domain_bounds, n_current, n_max):
    """Returns True if we should observe (skip experiment), False if intervene."""
    epsilon = compute_epsilon(observed_data, domain_bounds, n_current, n_max)
    return np.random.random() < epsilon
```

### Integration Point

Replace `OffPolicyPredictor.should_run_experiment()` with:

```python
def should_run_experiment(self, parameters, experiment_log, n_max):
    # Phase 1: use simple uncertainty threshold (current impl)
    # Phase 2: use epsilon controller (this implementation)

    epsilon = compute_epsilon(
        observed_data=experiment_log.to_dataframe()[var_names].values,
        domain_bounds=[(v.lower, v.upper) for v in search_space.variables],
        n_current=len(experiment_log.results),
        n_max=n_max,
    )

    if np.random.random() < epsilon:
        # OBSERVE: use surrogate prediction instead of running
        prediction = self.predict(parameters)
        return prediction is None or prediction.uncertainty > self.uncertainty_threshold
    else:
        # INTERVENE: run the experiment
        return True
```

---

## 3. Benchmark SCMs

### ToyGraph (Aglietti et al.)

```python
class ToyGraphRunner:
    """X → Z → Y. POMIS = [{Z}]."""

    def __init__(self, noise_scale=0.1):
        self.noise_scale = noise_scale

    def run(self, parameters):
        x = parameters.get("x", 0.0)
        z = parameters.get("z", np.exp(-x))  # if z not intervened, use structural eq

        noise = np.random.normal(0, self.noise_scale)
        y = np.cos(z) - np.exp(-z / 20) + noise

        return {"objective": -y}  # negate because we minimize; optimal Y is maximized

    @staticmethod
    def search_space():
        return SearchSpace(variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="z", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=20.0),
        ])

    @staticmethod
    def causal_graph():
        return CausalGraph(
            edges=[("x", "z"), ("z", "objective")],
            bidirected_edges=[],
        )

    @staticmethod
    def known_pomis():
        return [frozenset({"z"})]
```

### CompleteGraph (Aglietti et al.)

```python
class CompleteGraphRunner:
    """7 variables + 2 unobserved confounders. POMIS = [{B},{D},{E},{B,D},{D,E}]."""

    def run(self, parameters):
        f = parameters.get("f", np.random.uniform(-4, 4))
        u1 = np.random.normal(0, 1)  # unobserved
        u2 = np.random.normal(0, 1)  # unobserved

        a = f**2 + u1 + np.random.normal(0, 0.1)
        b = parameters.get("b", u2 + np.random.normal(0, 0.1))
        c = np.exp(-b) + np.random.normal(0, 0.1)
        d = parameters.get("d", np.exp(-c) / 10 + np.random.normal(0, 0.1))
        e = parameters.get("e", np.cos(a) + c / 10 + np.random.normal(0, 0.1))
        y = np.cos(d) - d/5 + np.sin(e) - e/4 + u1 + np.exp(-u2) + np.random.normal(0, 0.1)

        return {"objective": -y}

    @staticmethod
    def causal_graph():
        return CausalGraph(
            edges=[
                ("f", "a"), ("a", "e"), ("b", "c"), ("c", "d"),
                ("c", "e"), ("d", "objective"), ("e", "objective"),
            ],
            bidirected_edges=[
                ("a", "objective"),  # confounded via U1
                ("b", "objective"),  # confounded via U2
            ],
        )

    @staticmethod
    def known_pomis():
        return [
            frozenset({"b"}),
            frozenset({"d"}),
            frozenset({"e"}),
            frozenset({"b", "d"}),
            frozenset({"d", "e"}),
        ]
```

### Interaction Benchmark (custom)

```python
class InteractionRunner:
    """Two variables that hurt alone but help together.
    Greedy hill-climbing fails; factorial design succeeds."""

    def __init__(self):
        self.baseline = 10.0

    def run(self, parameters):
        a = parameters.get("use_a", False)
        b = parameters.get("use_b", False)
        c = parameters.get("c_value", 0.5)

        result = self.baseline

        if a and not b:
            result += 2.0   # A alone hurts (higher = worse for minimization)
        elif b and not a:
            result += 1.5   # B alone hurts
        elif a and b:
            result -= 3.0   # A+B together helps significantly

        result += c * 0.1   # C has a small linear effect
        result += np.random.normal(0, 0.2)

        return {"objective": result}
```

---

## 4. PC Algorithm Integration (via causal-learn)

```python
# discovery/graph_learner.py — updated _learn_pc method

def _learn_pc(self, df: pd.DataFrame) -> CausalGraph:
    """PC algorithm via causal-learn library."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError:
        logger.warning("causal-learn not installed, falling back to correlation")
        return self._learn_correlation(df)

    # Run PC algorithm
    cg = pc(
        data=df.values,
        alpha=0.05,
        indep_test='fisherz',
        stable=True,
        uc_rule=0,  # UCS rule for orienting edges
        uc_priority=2,  # prioritize existing edges
    )

    # Parse adjacency matrix
    # cg.G.graph encoding: [i,j]==-1 and [j,i]==1 means i→j
    # [i,j]==1 and [j,i]==1 means i↔j (bidirected / unresolved)
    adj = cg.G.graph
    col_names = df.columns.tolist()

    directed_edges = []
    bidirected_edges = []

    for i in range(len(col_names)):
        for j in range(i + 1, len(col_names)):
            if adj[i, j] == -1 and adj[j, i] == 1:
                directed_edges.append((col_names[i], col_names[j]))
            elif adj[i, j] == 1 and adj[j, i] == -1:
                directed_edges.append((col_names[j], col_names[i]))
            elif adj[i, j] == 1 and adj[j, i] == 1:
                # Undirected or bidirected — treat as potential confounder
                bidirected_edges.append((col_names[i], col_names[j]))

    return CausalGraph(
        edges=directed_edges,
        bidirected_edges=bidirected_edges,
        nodes=col_names,
    )
```

---

## 5. DoWhy Integration for Observational Estimation

```python
# estimator/do_calculus.py — new file

def estimate_interventional_effect(
    data: pd.DataFrame,
    causal_graph: CausalGraph,
    treatment: str,
    outcome: str,
    treatment_value: float,
) -> float:
    """Estimate E[Y | do(X = x)] from observational data using DoWhy."""
    try:
        import dowhy
    except ImportError:
        raise ImportError("dowhy is required for do-calculus estimation")

    # Convert CausalGraph to DOT format for DoWhy
    dot_edges = "; ".join(f"{u} -> {v}" for u, v in causal_graph.edges)
    dot_bidirected = "; ".join(
        f"{u} -> {v} [style=dashed]; {v} -> {u} [style=dashed]"
        for u, v in causal_graph.bidirected_edges
    )
    graph_str = f"digraph {{ {dot_edges}; {dot_bidirected} }}"

    model = dowhy.CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=graph_str,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        target_units="ate",
        treatment_value=treatment_value,
        control_value=0,
    )

    return estimate.value
```

## New Dependencies Summary

```toml
# Add to pyproject.toml [project.optional-dependencies]
discovery = [
    "causal-learn>=0.1",
]
causal = [
    "causal-inference-marketing @ git+https://github.com/datablogin/causal-inference-marketing.git",
    "dowhy>=0.11",
    "causal-learn>=0.1",
]
```
