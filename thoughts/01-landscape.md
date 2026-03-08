# Landscape: Existing Tools and Where They Fall Short

## Bayesian Optimization Tools

### Ax (Meta) — Production-ready
- Uses BoTorch/GPyTorch for Gaussian Process surrogate models
- Acquisition functions: Expected Improvement, Knowledge Gradient
- Supports complex search spaces (categorical, conditional, constrained)
- Multi-objective optimization, noisy observations
- Used by thousands of developers at Meta
- **Limitation**: treats the system as a black box. No causal reasoning.
- https://github.com/facebook/Ax

### Optuna — Production-ready
- Tree-structured Parzen Estimators (TPE) as default sampler
- Pruning strategies (Successive Halving, Hyperband) for early stopping
- Lightweight, Pythonic API
- **Limitation**: same black-box limitation as Ax.
- https://github.com/optuna/optuna

### SMAC3 (AutoML Freiburg) — Production-ready
- Random forests as surrogate models (scales better than GPs to high dimensions)
- CASH formulation (Combined Algorithm Selection and Hyperparameter optimization)
- **Limitation**: no causal structure.
- https://github.com/automl/SMAC3

### BOHB — Research/Production
- Combines Bayesian optimization (TPE variant) with Hyperband
- Strong anytime performance via multi-fidelity
- **Limitation**: no causal reasoning.

### fANOVA — Diagnostic tool
- Functional ANOVA over random forest predictions
- Decomposes performance variation across hyperparameters
- Key finding: most variation comes from just a few parameters
- Conceptually related to causal attribution but purely correlational
- https://github.com/automl/fanova

## Causal Inference Frameworks

### DoWhy (Microsoft/PyWhy) — Production-ready
- Full causal inference pipeline: identify, estimate, refute
- DoWhy-GCM: structural causal models, root cause analysis, counterfactuals
- `gcm.counterfactual_samples` enables "what-if" analysis
- Used at Amazon/AWS for supply chain root cause analysis (KDD 2025)
- **Not designed for optimization** — answers "what was the effect?" not "what should I try?"
- https://github.com/py-why/dowhy

### EconML (Microsoft/PyWhy) — Production-ready
- Heterogeneous treatment effect estimation (CATE) via ML
- Double ML, causal forests, meta-learners
- Policy learning module for optimal treatment assignment
- **Analysis-focused**, not optimization-focused
- https://github.com/py-why/EconML

### CausalML (Uber) — Production-ready
- Meta-learners (S, T, X-learner) for uplift modeling
- Focused on marketing incrementality
- **Same gap**: analysis, not sequential optimization
- https://github.com/uber/causalml

### CausalNex (QuantumBlack/McKinsey) — Mature
- Bayesian network-based causal modeling
- What-if analysis for business decisions
- Natural fit for configuration modeling but no optimization loop
- https://github.com/quantumblacklabs/causalnex

### causal-inference-marketing (our library) — Alpha
- From-scratch implementations: G-computation, IPW, AIPW, TMLE, DML, IV, DiD, RDD,
  synthetic control, causal forests, meta-learners, Bayesian CI
- Causal discovery: PC, GES, NOTEARS
- Policy optimization: greedy top-k, budget-constrained ILP, off-policy evaluation
- Sensitivity analysis: E-values, Rosenbaum bounds, Oster's method
- Target trial emulation, interference, transportability
- FastAPI service layer, Docker, Prometheus
- https://github.com/datablogin/causal-inference-marketing

## Causal Bayesian Optimization (Research)

### CBO (Aglietti et al., AISTATS 2020)
- Foundational paper for causal + BO integration
- Key concepts: POMIS, observation-intervention tradeoff, separate GP surrogates
- Applied to Apache Spark tuning via causal graphs from static code analysis
- Reference implementation: https://github.com/VirgiAgl/CausalBayesianOptimization
- **Research prototype** — not production-hardened

### Extensions
- **Dynamic CBO (DCBO)** — NeurIPS 2021, time-varying causal relationships
- **Constrained CBO (cCBO)** — ICML 2023, intervention constraints
- **Adversarial CBO** — ICLR 2024, uncontrollable external factors
- **CBO with Unknown Graphs** — 2025, removes need for pre-specified DAG
- **Contextual CBO** — 2023, context-dependent interventions

### CausalTune (PyWhy) — Mature
- AutoML for causal inference (estimator selection + tuning)
- Uses FLAML for hyperparameter optimization of causal models
- Not optimization-oriented per se, but demonstrates causal + AutoML integration
- https://github.com/py-why/causaltune

## Program Optimization Systems

### AlphaEvolve (DeepMind, May 2025) — Production (internal)
- LLM-guided program optimization
- Dual LLM strategy: Gemini Flash (breadth) + Gemini Pro (depth)
- MAP-Elites + island model for diversity
- Evolves entire codebases, not just single functions
- 0.7% worldwide compute recovery at Google
- First improvement over Strassen's algorithm in 56 years
- **Not open-sourced**

### OpenEvolve — Community reimplementation
- Open-source approximation of AlphaEvolve
- https://huggingface.co/blog/codelion/openevolve

### FunSearch (DeepMind, Nature 2023)
- LLMs in evolutionary loop for mathematical function discovery
- Population-based, not greedy
- Predecessor to AlphaEvolve

## Ablation Tools

### Agentic Ablation — Early prototype
- LLM-powered multi-agent system for automated ablation studies
- Uses LangGraph with code generation, execution, reflection, analysis agents
- Users mark `#ABLATABLE_COMPONENT` in code
- https://github.com/AmirLayegh/agentic-ablation

### AbGen (ACL 2025)
- Benchmark for evaluating LLMs' ability to design ablation studies
- 2,000 expert-annotated examples from 677 NLP papers

## The Gap

| Capability | Ax/Optuna | DoWhy/EconML | CBO | AlphaEvolve | **causal-optimizer** |
| --- | --- | --- | --- | --- | --- |
| Bayesian optimization | Yes | No | Yes | No | Yes |
| Causal graph support | No | Yes | Yes | No | Yes |
| Experimental design (DoE) | No | No | No | No | Yes |
| Doubly-robust estimation | No | Yes | No | No | Yes |
| Population diversity | No | No | No | Yes | Yes |
| Off-policy evaluation | No | Yes | Partial | No | Yes |
| Sensitivity analysis | No | Yes | No | No | Yes |
| Domain adapters | No | No | No | No | Yes |
| Production-ready | Yes | Yes | No | Internal | Building |

No existing system covers all columns. That's the opportunity.
