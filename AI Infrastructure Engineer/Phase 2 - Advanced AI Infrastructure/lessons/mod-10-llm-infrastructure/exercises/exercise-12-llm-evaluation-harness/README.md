# Exercise 12: LLM Evaluation Infrastructure

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 10

## Objective

Build a reproducible scoring harness for LLM applications using LLM-as-judge and reference-based metrics. Track scores over time; alert on regression.

## Why this matters

LLM apps can silently degrade — a model upgrade, a prompt change, a retrieval drift. Without continuous scoring, you discover the regression from customer complaints. With it, you catch it in CI.

## Requirements

1. Scoring set with 100+ examples covering 5 categories.
2. Multiple scoring approaches: reference-based (BLEU, exact match), LLM-as-judge, regex assertions.
3. Scoring runs on every PR + nightly on prod.
4. Score persisted; trend dashboard.
5. Regression alert when score drops > 2pp.

## Step-by-step

### Step 1 — Scoring set design (45 min)
```yaml
# scoring/cases.yaml
- id: greet_001
  category: greeting
  input: "Hello"
  expected_contains: ["hello", "hi", "greetings"]
  expected_json_keys: ["answer"]
  expected_needs_human: false

- id: refund_simple_001
  category: support
  input: "How do I get a refund?"
  reference_answer: "Visit the order page, click 'Request Refund', and select the item."
  llm_judge_criteria: "Does the response correctly explain the refund process?"

- id: escalate_001
  category: support
  input: "I want to speak to a manager about my account being hacked"
  expected_needs_human: true
  llm_judge_criteria: "Does the response escalate appropriately?"
```

Build at least 100 examples balanced across happy path, edge cases, adversarial, ambiguity.

### Step 2 — Scoring methods (60 min)
```python
def score_contains(response, expected):
    text = response["answer"].lower()
    return all(s in text for s in expected)

def score_json_schema(response, expected_keys):
    return set(response.keys()) >= set(expected_keys)

def score_llm_judge(response, criteria):
    judge = OpenAI()
    r = judge.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are an evaluator. Reply with only YES or NO."},
            {"role":"user","content":f"Criteria: {criteria}\nResponse: {response}\nDoes the response meet the criteria?"},
        ],
        max_tokens=5, temperature=0,
    )
    return r.choices[0].message.content.strip().upper() == "YES"

def score_case(app, case):
    resp = app(case["input"])
    scores = {}
    if "expected_contains" in case:
        scores["contains"] = score_contains(resp, case["expected_contains"])
    if "expected_json_keys" in case:
        scores["json_schema"] = score_json_schema(resp, case["expected_json_keys"])
    if "llm_judge_criteria" in case:
        scores["llm_judge"] = score_llm_judge(resp, case["llm_judge_criteria"])
    return {"case_id": case["id"], "scores": scores, "passed": all(scores.values())}
```

### Step 3 — Scoring runner (30 min)
```python
def run_scoring(app, scoring_set_path):
    cases = yaml.safe_load(open(scoring_set_path))
    results = [score_case(app, c) for c in cases]
    pass_rate = sum(r["passed"] for r in results) / len(results)
    return {"timestamp": time.time(), "pass_rate": pass_rate, "n": len(results), "details": results}
```

### Step 4 — Persist results (15 min)
Store each run in S3 or a database:
```python
import boto3
s3.put_object(Bucket="scoring-runs", Key=f"runs/{run_id}.json", Body=json.dumps(result))
```

### Step 5 — Trend dashboard (30 min)
Grafana with Athena over S3:
```sql
SELECT date_trunc('day', from_unixtime(timestamp)) AS day,
       AVG(pass_rate) AS pass_rate,
       category
FROM scoring_runs
GROUP BY 1, category
ORDER BY 1 DESC;
```

### Step 6 — CI integration (15 min)
```yaml
- name: Run scoring
  run: |
    pass_rate=$(python run_scoring.py | jq .pass_rate)
    if (( $(echo "$pass_rate < 0.85" | bc -l) )); then
      echo "Scoring pass rate $pass_rate < 0.85; blocking."
      exit 1
    fi
```

### Step 7 — Nightly + regression alerts (15 min)
Scheduled job runs scoring against prod. If 7-day-rolling pass_rate drops by > 2pp, alert.

## Deliverables

1. Scoring set (100+ cases).
2. Scoring runner + scoring functions.
3. CI integration.
4. Nightly job.
5. Trend dashboard.
6. `SCORING_PHILOSOPHY.md`: what we measure, what we don't, when to extend the set.

## Validation

- [ ] Scoring runs reproducibly produce same score on same code.
- [ ] CI blocks PR with regressing scoring.
- [ ] Dashboard shows trend over weeks.
- [ ] Alert fires on synthetic regression.

## Common pitfalls

- **LLM-as-judge bias** — Same model evaluating itself overscores. Use a different model for judging.
- **Scoring set bias** — Built from easy cases; misses production weirdness. Mine from real logs.
- **Pass rate as single number** — Hides per-category collapse. Track segments.
- **Non-deterministic LLM responses** — Use temperature=0; even then, expect 1-2% variance.
