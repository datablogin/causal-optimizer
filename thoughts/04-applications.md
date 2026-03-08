# Applications Beyond Code Optimization

The architecture is domain-agnostic. Any system where you pull levers, measure outcomes, and
want to find the best combination with the fewest experiments.

## 1. Marketing & Advertising

**Problem**: Which combination of channel, creative, audience, timing, and budget maximizes
incremental conversions?

**Why causal optimization matters**:
- Marketing has massive confounding (people who see ads were going to buy anyway)
- Channel interactions are real (social drives search, email drives direct)
- Experiments (A/B tests) are expensive (real ad spend, real customers)
- Massive observational data exists (click logs, conversion data) — the observation-intervention
  tradeoff can save significant budget

**Causal graph example**:
```
email_frequency → email_opens → site_visits → conversions
social_spend → impressions → brand_awareness → search_volume → search_clicks → site_visits
creative_variant → click_through_rate → site_visits
retargeting → return_visits → conversions
```

**Our existing tools**: causal-inference-marketing provides AIPW, TMLE, causal forests for
incrementality measurement. This project adds the experiment sequencing layer.

## 2. Drug Discovery & Clinical Trials

**Problem**: Which molecular modifications, dosing regimens, and patient populations maximize
efficacy while minimizing toxicity?

**Why causal optimization matters**:
- Wet-lab experiments cost $1K-$100K per compound
- Observation-intervention tradeoff is extremely valuable: computational predictions (QSAR,
  molecular dynamics) are cheap; synthesis + assay is expensive
- Drug interactions are common (combination therapy)
- Patient subgroup effects are critical (CATE estimation)

**Causal graph example**:
```
molecular_weight → solubility → bioavailability → efficacy
functional_group → binding_affinity → efficacy
functional_group → metabolism_rate → toxicity
dosage → plasma_concentration → efficacy
dosage → plasma_concentration → toxicity
```

**MAP-Elites value**: maintain diverse compounds across (efficacy, toxicity) tradeoff space

## 3. Manufacturing & Process Engineering

**Problem**: Optimize temperature, pressure, catalyst concentration, feed rate for yield/quality.

**Why causal optimization matters**:
- DoE was literally invented for this domain (Fisher, 1935; Taguchi methods)
- Process variables have known causal structure from chemistry/physics
- Experiments require production downtime — minimizing experiment count is high-value
- Sensor data is abundant — strong observational baseline for surrogate models

**Causal graph example**:
```
temperature → reaction_rate → yield
temperature → side_reaction_rate → impurity
pressure → solubility → reaction_rate
catalyst_concentration → reaction_rate
feed_rate → residence_time → conversion
```

**Key advantage over current practice**: most factories use DoE for screening but then switch
to ad-hoc optimization. CBO provides principled sequential experiment selection post-screening.

## 4. Personalized Medicine / Treatment Optimization

**Problem**: Given a patient's features, which treatment sequence maximizes outcomes?

**Why causal optimization matters**:
- Treatment effects are heterogeneous (different patients respond differently)
- Causal forests and meta-learners estimate individualized treatment effects (CATE)
- Target trial emulation bridges observational EHR data to experimental conclusions
- E-values quantify how strong unmeasured confounding would need to be to nullify findings

**Relevant modules from causal-inference-marketing**:
- Causal forests for individual-level CATE
- Meta-learners (S, T, X, R-learner)
- Target trial emulation
- Sensitivity analysis (E-values, Rosenbaum bounds)
- Time-varying treatments (sequential strategies)

## 5. Supply Chain & Operations

**Problem**: Which combination of supplier, inventory policy, routing, and pricing maximizes
margin while meeting SLAs?

**Why causal optimization matters**:
- Combinatorial explosion of SKU × warehouse × supplier × policy
- POMIS dramatically prunes which combinations are worth testing
- Amazon/AWS already uses DoWhy-GCM for supply chain root cause analysis (KDD 2025)
- Counterfactual estimation: "what would have happened with supplier B last quarter?"

## 6. Education & Adaptive Learning

**Problem**: Which combination of content format, difficulty progression, feedback timing, and
practice spacing maximizes learning outcomes?

**Why causal optimization matters**:
- Student outcomes are confounded by ability (need causal estimation, not just A/B testing)
- Pedagogical interventions interact (spaced repetition + immediate feedback)
- Per-student treatment effects (adaptive learning requires CATE)
- Ethical constraints on experimentation (can't give students bad curricula for too long)

## 7. Agriculture & Crop Optimization

**Problem**: Optimize seed variety, fertilizer mix, irrigation schedule, planting density.

**Why causal optimization matters**:
- Split-plot and strip-plot factorial designs are standard in agronomy
- Spatial interference: one plot's treatment affects neighbors (our spillover module)
- Long experiment cycles (one growing season per experiment) — must maximize information
  per experiment
- Weather confounding requires robust estimation

## Highest-Value Domains

Ranked by the value of causal optimization (experiment cost × available observational data):

1. **Drug discovery** — highest experiment cost, vast historical data
2. **Marketing** — real money at stake, massive click/conversion logs
3. **Manufacturing** — downtime is expensive, sensor data is abundant
4. **Clinical trials** — patient risk, extensive EHR data
5. **Supply chain** — margin impact, transaction history
