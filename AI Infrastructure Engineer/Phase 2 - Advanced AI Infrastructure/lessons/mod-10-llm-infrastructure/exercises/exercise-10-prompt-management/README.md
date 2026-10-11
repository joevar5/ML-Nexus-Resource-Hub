# Exercise 10: Prompt Management and Versioning

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** [Exercise 09](../exercise-09-llm-observability/README.md) and [Lesson 09: Production LLM Best Practices](../../09-production-llm-best-practices.md)
**Hardware:** None beyond a model endpoint (local or hosted)

## Goal

Treat prompts like the production assets they are. Store them in Git with versions, test every change automatically, roll changes out to a slice of traffic first, and be able to roll back in minutes.

## Why it matters

A prompt change can break production as badly as a code change, and it is much easier to ship by accident. Someone tweaks one sentence, it "looks better" on three examples, and a week later support tickets rise. Teams that manage prompts well can answer three questions at any time: what is running right now, who changed it and why, and did the change make things better or worse?

## Before you start

Know the two common ways to do this:

- **Prompts in Git, loaded by your app.** Simple, reviewable, works with the CI you already have. This is what you will build.
- **A prompt registry or platform,** such as Langfuse, MLflow's prompt registry, or PromptLayer. They add a UI, labels like `production` and `staging`, and change history, so non-engineers can edit prompts safely.

Learn the Git version first. It teaches the concepts that every platform wraps. At the end, write down when you would move to a platform.

## Steps

### Step 1: Prompt files (15 min)

One file per prompt version. Keep the template, the model settings it was tested with, and who owns it, together.

```yaml
# prompts/customer-support/v3.yaml
id: customer-support
version: 3
model: { name: "your-model", temperature: 0.2, max_tokens: 400 }
system: |
  You are a customer support agent for {{ company_name }}.
  Reply in JSON: {"answer": str, "needs_human": bool}.
user: |
  Question: {{ question }}
  Context: {{ context | default("none") }}
owner: support-team
changelog: "Tell the model to set needs_human when the question is unclear."
```

Treat a prompt as including its model settings. The same text behaves differently at another temperature or on another model.

### Step 2: Loader and renderer (20 min)

Render with a **sandboxed** Jinja environment, and fail loudly when a variable is missing.

```python
import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

env = SandboxedEnvironment(undefined=StrictUndefined)

def load(prompt_id: str, version: int) -> dict:
    with open(f"prompts/{prompt_id}/v{version}.yaml") as f:
        return yaml.safe_load(f)

def render(p: dict, **variables) -> list[dict]:
    return [
        {"role": "system", "content": env.from_string(p["system"]).render(**variables)},
        {"role": "user",   "content": env.from_string(p["user"]).render(**variables)},
    ]
```

Add one more safeguard: user text goes into **variables**, never into the template itself, so that a user cannot rewrite your prompt.

### Step 3: Rollout config and A/B routing (45 min)

Keep "which version gets what share of traffic" in a small config file, separate from the prompt files. Then rolling out or rolling back is a one-line change.

```yaml
# rollout.yaml
customer-support:
  - { version: 3, weight: 90 }
  - { version: 4, weight: 10 }
```

Assign each user to a variant by hashing their ID, so the same user always sees the same version:

```python
import hashlib

def pick_version(prompt_id: str, user_id: str, rollout: dict) -> int:
    bucket = int(hashlib.sha256(f"{prompt_id}:{user_id}".encode()).hexdigest(), 16) % 100
    total = 0
    for v in rollout[prompt_id]:
        total += v["weight"]
        if bucket < total:
            return v["version"]
    return rollout[prompt_id][-1]["version"]
```

Log the **prompt id and version** with every request and trace (see Exercise 09). Without that you cannot compare versions afterwards.

### Step 4: Evaluation suite (45 min)

An eval suite is a set of cases with a way to score each one. Write at least 10 cases, including awkward ones: empty input, a very long input, a request in another language, and an attempt to override the instructions.

```yaml
# evals/customer-support.yaml
- name: refund_simple
  inputs: { question: "How do I get a refund?" }
  checks: [ json_valid, { contains: "refund" }, { field_equals: { needs_human: false } } ]
- name: gibberish_escalates
  inputs: { question: "asdf qwerty zxcv" }
  checks: [ json_valid, { field_equals: { needs_human: true } } ]
```

Use cheap, exact checks (valid JSON, required fields, forbidden phrases) wherever you can. Add an LLM-as-judge for qualities you can't check by rule, such as tone or helpfulness. Tools like promptfoo, DeepEval, and Langfuse's evals do this for you, so look at one of them before writing your own.

Because models are not fully deterministic, run each case a few times (or at temperature 0) and report a pass rate, not a single pass or fail.

### Step 5: CI gate (20 min)

Run the evals on every pull request that changes a prompt. Fail the build if quality drops.

The most useful gate compares the **new version against the current production version** on the same cases, not against a fixed 90% bar. A new prompt that scores 92% when the old one scored 97% is a regression, even though it clears 90%.

```yaml
- name: Evaluate changed prompts
  run: python evaluate.py --changed --compare-to production --fail-on-regression
```

### Step 6: Roll out and roll back (15 min)

Change the weights in `rollout.yaml`: 10% → 50% → 100%, watching your quality and cost metrics between each step. Rolling back is the same edit in the other direction, and it must not need a code deploy.

For a hotfix, keep the same path but make it fast. Skipping the evals is how regressions return.

## Deliverables

1. A prompt repo with at least two prompts and two versions each.
2. The loader, A/B router, and rollout config.
3. An eval suite with at least 10 cases per prompt.
4. A CI gate that compares new against current.
5. `PROMPT_LIFECYCLE.md`: how to propose, test, roll out, roll back, and retire a prompt, and when you would switch to a prompt platform.

## Check yourself

- [ ] The same user always gets the same version.
- [ ] Every logged request shows which prompt version produced it.
- [ ] A deliberately bad prompt fails the CI gate.
- [ ] You can roll back to the previous version without a code deploy.

## Common pitfalls

- **Prompts hidden in code.** A string buried in a Python file gets no review, no history, and no tests.
- **No evals.** "It looks better" is not a measurement, and three examples are not a test set.
- **Gating against a fixed number only.** Compare with the version currently in production.
- **Changing the model and the prompt together.** You will not know which one caused the change. Change one at a time.
- **User text inside the template.** Pass it as a variable, so it cannot change your instructions.
- **Leaving old versions around forever.** Retire versions you no longer serve, and keep the history in Git.
