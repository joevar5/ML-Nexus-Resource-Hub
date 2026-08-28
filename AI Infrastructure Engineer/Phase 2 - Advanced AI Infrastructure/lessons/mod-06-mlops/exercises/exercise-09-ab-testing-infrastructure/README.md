# Exercise 09: A/B Testing Infrastructure for ML

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 08

## Objective

Build a proper A/B testing infrastructure for ML: assignment (consistent hashing), tracking, exposure logging, downstream metric attribution, and statistical analysis. Run a synthetic experiment to demonstrate the full flow.

## Why this matters

Half of "A/B tests" in industry are flawed: inconsistent assignment, unmeasured downstream metrics, naive statistical analysis. A working A/B platform turns model rollouts from "we think it's better" to "p=0.003, +3.2% conversion".

## Requirements

1. **Consistent assignment**: same user always gets same variant.
2. **Exposure log**: every prediction tagged with variant + user_id + timestamp.
3. **Downstream metric attribution**: link the prediction → downstream event (click, purchase) via a request_id.
4. **Statistical analysis**: proper two-sample test with confidence interval.
5. **Live dashboard** showing experiment state, traffic split, current significance.
6. **Sample size calculator** to set required runtime before starting.

## Step-by-step

### Step 1 — Sample size calculator (30 min)
```python
# sample_size.py
from scipy.stats import norm

def sample_size_per_arm(p_baseline, mde, alpha=0.05, power=0.80):
    """p_baseline: current conversion rate; mde: minimum detectable effect (relative)"""
    p1 = p_baseline
    p2 = p_baseline * (1 + mde)
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta  = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = ((z_alpha * (2 * p_bar * (1 - p_bar)) ** 0.5
        + z_beta * (p1*(1-p1) + p2*(1-p2)) ** 0.5) ** 2) / (p2 - p1) ** 2
    return int(n)

# Example
n = sample_size_per_arm(0.10, mde=0.05)
print(f"need ~{n:,} per arm for +5% lift at baseline 10%")
```

### Step 2 — Consistent assignment (30 min)
```python
# assignment.py
import hashlib

def assign(user_id: str, experiment: str, weights: dict[str, float]) -> str:
    """Deterministic assignment. Same user_id+experiment always returns same variant."""
    h = int(hashlib.md5(f"{user_id}:{experiment}".encode()).hexdigest(), 16)
    p = (h % 10_000) / 10_000     # uniform [0, 1)
    cumulative = 0
    for variant, w in weights.items():
        cumulative += w
        if p < cumulative:
            return variant
    return list(weights.keys())[-1]

# Usage
v = assign("user-123", "ranker-v2-test", {"control": 0.5, "treatment": 0.5})
```

### Step 3 — Exposure log (30 min)
Every prediction logs:
```python
import time, uuid

def predict(user_id, features):
    request_id = uuid.uuid4().hex
    variant = assign(user_id, "ranker-v2-test", {"control": 0.5, "treatment": 0.5})
    
    if variant == "treatment":
        pred = model_v2.predict(features)
    else:
        pred = model_v1.predict(features)
    
    exposure_log.write({
        "ts": time.time(),
        "request_id": request_id,
        "user_id": user_id,
        "experiment": "ranker-v2-test",
        "variant": variant,
        "model_version": "v2" if variant == "treatment" else "v1",
    })
    return {"prediction": pred, "request_id": request_id}
```

### Step 4 — Downstream event attribution (30 min)
The frontend / app passes `request_id` back when a user clicks / purchases:
```python
def record_event(user_id, event_type, request_id):
    event_log.write({
        "ts": time.time(),
        "user_id": user_id,
        "event_type": event_type,
        "request_id": request_id,  # links back to the prediction
    })
```

### Step 5 — Analysis script (45 min)
```python
# analyze.py
import pandas as pd
from scipy import stats

exposures = pd.read_parquet("exposures.parquet")
events    = pd.read_parquet("events.parquet")

# Join by request_id
joined = exposures.merge(events[events.event_type == "click"][["request_id"]],
                          on="request_id", how="left", indicator=True)
joined["converted"] = (joined["_merge"] == "both").astype(int)

# Per-variant metrics
summary = joined.groupby("variant").agg(
    exposures=("user_id", "count"),
    conversions=("converted", "sum"),
    rate=("converted", "mean"),
)
print(summary)

# Two-proportion z-test
control = joined[joined.variant == "control"]
treat   = joined[joined.variant == "treatment"]
zstat, pvalue = stats.ttest_ind(control.converted, treat.converted)
ci_lo, ci_hi = stats.norm.interval(0.95,
    loc=treat.converted.mean() - control.converted.mean(),
    scale=((treat.converted.std()**2 / len(treat)) + (control.converted.std()**2 / len(control))) ** 0.5)

print(f"p-value: {pvalue:.4f}, 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
```

### Step 6 — Live dashboard (45 min)
Grafana panels:
- Exposure count per variant (real-time)
- Conversion rate per variant (running)
- Conversion rate diff with 95% CI
- p-value over time
- "Stop early?" indicator if p < 0.01 with sufficient samples

## Deliverables

1. Library (`abtest/`) with the 6 components.
2. Synthetic test data + analysis output.
3. Grafana dashboard JSON.
4. `EXPERIMENT_PLAYBOOK.md`: how to design and run an A/B test on your team.

## Validation

- [ ] Same user_id consistently gets same variant across multiple calls.
- [ ] Sample size calculator returns reasonable numbers for varied baselines.
- [ ] Analysis produces correct p-value (verify against scipy directly with known inputs).
- [ ] Dashboard shows live data.

## Stretch goals

- Add **multi-arm bandits** (Thompson sampling, UCB) for adaptive allocation.
- Add **CUPED** variance reduction.
- Build a **guardrail metric** system: experiment auto-halts if a critical metric (error rate, latency) regresses > X%.

## Common pitfalls

- **Random assignment per request** — Same user gets different variants each call; ruins comparison.
- **Selection bias from delayed exposure** — Only logging exposures that lead to events skews results. Log every exposure.
- **Multiple comparisons** — Running 10 experiments simultaneously → 50% chance of false positive at p<0.05. Use Bonferroni or sequential testing.
- **Peeking** — Checking p-value every hour and stopping when significant inflates false positive rate. Pre-commit to sample size; use sequential tests if you need early stopping.
