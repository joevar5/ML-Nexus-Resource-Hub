# Exercise 10: Prompt Management and Versioning

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 09

## Objective

Build a prompt management system: prompts in version-controlled storage, environment-specific values, A/B testing of prompt variants, eval-based gating, deployment without code change.

## Why this matters

Prompts are config, not code. Treating them as config (version, review, A/B test, rollback) is how mature LLM apps avoid the prompt-changes-broke-prod-but-nobody-noticed antipattern.

## Requirements

1. Prompts in Git (YAML).
2. Per-env values (dev: verbose, prod: concise).
3. A/B test framework selecting variant per request.
4. Eval suite running on every prompt change.
5. Deploy gated by eval score.

## Step-by-step

### Step 1 — Prompt schema (15 min)
```yaml
# prompts/customer_support_v3.yaml
id: customer-support
version: 3
system: |
  You are a helpful customer support agent for {{ company_name }}.
  Always reply in JSON: {"answer": str, "needs_human": bool}.
user_template: |
  Question: {{ question }}
  Context: {{ context | default("") }}
metadata:
  owner: support-team
  reviewed_by: alice@
  reviewed_at: 2026-05-22
```

### Step 2 — Loader (15 min)
```python
import jinja2, yaml
class PromptLoader:
    def __init__(self, repo_path): self.repo = repo_path
    def load(self, prompt_id: str, version: int | str = "latest") -> dict:
        # version "latest" → highest version
        path = f"{self.repo}/{prompt_id}_v{version}.yaml" if version != "latest" else self._latest(prompt_id)
        return yaml.safe_load(open(path))
    def render(self, p, **vars):
        env = jinja2.Environment()
        return {
            "system": env.from_string(p["system"]).render(**vars),
            "user":   env.from_string(p["user_template"]).render(**vars),
        }
```

### Step 3 — A/B test framework (45 min)
```python
ABTESTS = {
    "customer-support": [
        {"version": 3, "weight": 0.5},
        {"version": 4, "weight": 0.5},
    ],
}

def pick_variant(prompt_id, user_id):
    h = int(hashlib.md5(f"{user_id}:{prompt_id}".encode()).hexdigest(), 16) / 16**32
    cum = 0
    for variant in ABTESTS[prompt_id]:
        cum += variant["weight"]
        if h < cum: return variant["version"]
    return ABTESTS[prompt_id][-1]["version"]

def chat(prompt_id, user_id, **vars):
    version = pick_variant(prompt_id, user_id)
    p = loader.load(prompt_id, version)
    rendered = loader.render(p, **vars)
    # log: prompt_id, version, user_id for downstream attribution
    response = call_llm(rendered)
    log_attribution(prompt_id, version, user_id, response)
    return response
```

### Step 4 — Eval suite (45 min)
```python
# evals/customer_support_v4.yaml
- name: refund_simple
  inputs: { question: "How do I get a refund?" }
  expected:
    must_contain: ["refund"]
    must_not_contain: ["I don't know"]
    json_valid: true
    needs_human_when_unclear: false

- name: escalate_unclear
  inputs: { question: "asdf qwerty zxcv" }
  expected:
    needs_human_when_unclear: true
```

```python
def run_evals(prompt_id, version):
    p = loader.load(prompt_id, version)
    results = []
    for case in load_evals(prompt_id):
        rendered = loader.render(p, **case["inputs"])
        resp = call_llm(rendered)
        results.append(check(resp, case["expected"]))
    return {"pass_rate": mean(r["passed"] for r in results), "details": results}
```

### Step 5 — CI gate (15 min)
```yaml
- name: Evaluate prompt changes
  run: |
    for prompt in $(git diff --name-only main HEAD -- prompts/); do
      pass_rate=$(python evaluate.py $prompt | jq .pass_rate)
      if (( $(echo "$pass_rate < 0.9" | bc -l) )); then
        echo "Prompt $prompt eval pass rate $pass_rate < 0.9; blocking."
        exit 1
      fi
    done
```

### Step 6 — Production deploy (15 min)
Merging a new prompt version makes it available; A/B test config (`ABTESTS`) determines whether traffic flows to it. Promote by editing weights:
```yaml
- {version: 3, weight: 0.0}
- {version: 4, weight: 1.0}     # full rollout
```

## Deliverables

1. Prompt repo with at least 2 prompts and 2 versions each.
2. Loader + A/B test framework.
3. Eval suite with at least 10 cases.
4. CI gate.
5. `PROMPT_LIFECYCLE.md`: how to add, test, promote, retire.

## Validation

- [ ] A/B variants visible in request logs.
- [ ] Eval gate fails a deliberately bad prompt.
- [ ] Rolling out v4 to 100% is one PR (weights).
- [ ] Eval pass rates tracked over time.

## Common pitfalls

- **Prompts in code strings** — Buried in Python literals; no review, no version. Always external.
- **No eval suite** — "Looks better" is subjective. Quantify.
- **Skipping the gate for hotfixes** — A "quick fix" introduces a regression. Always run evals.
