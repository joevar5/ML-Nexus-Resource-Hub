# Exercise 12: LLM Evaluation Harness

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** [Exercise 10](../exercise-10-prompt-management/README.md) and [Lesson 09: Production LLM Best Practices](../../09-production-llm-best-practices.md)
**Hardware:** A model endpoint for the app, and a second, different model for judging

## Goal

Build a repeatable way to answer "is this version of our LLM app better or worse than the last one?" You will create a test set, score it in several ways, run it automatically on every change and every night, and get an alert when quality drops.

## Why it matters

LLM apps get worse without anyone touching them: a provider updates a model, retrieval drifts, a prompt gets edited. Without scoring, you hear about it from customers. With it, you hear about it in CI.

Evaluation is also how you make every other decision in this module, such as which model to use, whether quantization hurt, and whether a new prompt helps. A good harness is the most reusable thing you will build.

## Before you start

Know the three kinds of checks, from cheapest to most expensive:

1. **Rule checks.** Valid JSON, required fields, forbidden phrases, exact match. Fast, free, and reliable. Use them wherever they fit.
2. **Reference-based metrics.** Compare to a known answer, such as exact match or F1. Older text-overlap metrics like BLEU and ROUGE correlate poorly with quality for chat answers, so use them only where wording really must match.
3. **LLM-as-judge.** A model grades the answer against a rubric. It handles open-ended answers, but it has biases and costs money, so you must check the judge too.

Several open-source tools do much of this already (promptfoo, DeepEval, Ragas for RAG, Langfuse and Phoenix for logging and experiments, and `lm-evaluation-harness` for model benchmarks). Skim one before you start, then build a small version yourself so you understand what it does. Decide at the end whether to adopt a tool.

## Steps

### Step 1: Design the test set (45 min)

Write at least 100 cases across at least five categories. Mix easy cases with the ones that are likely to break:

| Category | Examples |
|---|---|
| Happy path | The questions most users ask |
| Edge cases | Empty input, very long input, typos, other languages |
| Adversarial | Attempts to override instructions or leak the prompt |
| Ambiguous | Cases where the right answer is to ask or escalate |
| Regressions | Every bug you have ever fixed, as a new case |

Mine real traffic for cases (with personal data removed), because invented questions are always too clean.

```yaml
- id: refund_001
  category: support
  input: "How do I get a refund?"
  checks:
    - json_valid
    - { field_equals: { needs_human: false } }
    - { judge: "Does the answer correctly explain how to request a refund, without inventing policy?" }
```

Split the set into a **dev set** you tune against and a **locked set** you only report on.

### Step 2: Scoring functions (60 min)

Write each check as a small function that returns pass or fail, plus a reason. A reason makes failures easy to debug.

```python
import json

def check_json_valid(resp: str):
    try:
        json.loads(resp); return True, ""
    except ValueError as e:
        return False, f"invalid JSON: {e}"

def check_judge(question: str, resp: str, rubric: str):
    prompt = (f"Question: {question}\nAnswer: {resp}\nRubric: {rubric}\n"
              'Reply as JSON: {"reason": "<one sentence>", "pass": true or false}')
    out = judge.chat.completions.create(model=JUDGE_MODEL, temperature=0, messages=[
        {"role": "user", "content": prompt}],
        response_format={"type": "json_object"})
    verdict = json.loads(out.choices[0].message.content)
    return verdict["pass"], verdict["reason"]
```

Make the judge better:
- Use a **different model** from the one being tested. A model scores its own answers too kindly.
- Ask for the **reason first**, then the verdict.
- Prefer a **specific rubric** ("states the 30-day limit") over a vague one ("is helpful").
- **Calibrate it**: have a person label 30–50 answers, then measure how often the judge agrees. If agreement is low, fix the rubric before trusting any scores.
- For comparing two versions, show both answers and swap their order on half the cases. Judges tend to prefer the first one.

### Step 3: The runner (30 min)

Run each case several times (3 is a good start) because outputs vary even at temperature 0. Report the pass rate and a confidence interval. Run cases in parallel with a concurrency limit.

Store, for every run: the git commit, the prompt version, the model and its settings, the test set version, and per-case results. A score without that information cannot be reproduced.

### Step 4: Store results and show trends (45 min)

Put results in a database table (Postgres, ClickHouse, or even SQLite to start), with one row per case per run. A Grafana or Metabase dashboard on top gives you the trend. Break the pass rate down **by category**, because one overall number hides a category that has collapsed.

### Step 5: CI gate (15 min)

On every pull request, run the suite and compare it to the main branch. Block the merge if the pass rate drops by more than the noise level you measured in Step 3, or if any critical case fails. A fixed global bar like "85%" lets a real regression through when the current score is 97%.

### Step 6: Nightly run on production (15 min)

Run the suite against the live system every night, since models and data can change under you. Alert when the 7-day average drops by more than 2 percentage points, or when any category drops sharply. Test the alert by making a deliberately bad prompt and checking that it fires.

### Step 7: Evaluate real traffic too (15 min)

A fixed test set eventually goes stale. Sample a small share of production conversations, grade them with your judge, and track the score. Add the failures you find to the test set. This loop is how the set stays useful.

## Deliverables

1. A test set of 100+ cases, split into dev and locked.
2. The runner and the scoring functions, with a judge you have calibrated.
3. A CI job and a nightly job.
4. A dashboard showing score by category over time.
5. `EVALUATION_POLICY.md`: what you measure, what you do not, how a case gets added, and how the judge was validated.

## Check yourself

- [ ] Running the same code twice gives scores that agree within the noise you measured.
- [ ] You know how often your judge agrees with a human.
- [ ] CI blocks a deliberately worse prompt.
- [ ] A synthetic regression triggers the nightly alert.

## Common pitfalls

- **A judge that likes itself.** Use a different model for judging.
- **A test set of easy cases.** Production is stranger than anything you invent. Mine real logs.
- **One overall number.** It hides a category that has collapsed. Always break it down.
- **Treating noise as signal.** A 1% change on 100 cases is within noise. Measure the spread before setting thresholds.
- **Tuning on the locked set.** Once you fix things while looking at it, it is no longer a fair test.
- **Never updating the set.** Add every real failure as a new case.
