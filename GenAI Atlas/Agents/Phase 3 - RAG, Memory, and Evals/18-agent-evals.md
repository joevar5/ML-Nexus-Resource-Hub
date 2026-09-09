# Day 18: Agent Evals

## Concept

You cannot improve what you can't measure, and "the demo looked good" is not measurement. Agent evals exist to answer, repeatably and quantitatively: does this agent do the task correctly, and did a given change make it better or worse?

Evals differ from unit tests in one key way: agent output is often not a single correct string, so pass/fail isn't always a straight equality check. Three levels of eval, from cheapest/coarsest to most expensive/precise:

- **Outcome-level evals.** Did the task succeed, by some checkable criterion? For a coding agent: do the tests pass. For a booking agent: was the correct flight actually booked. These are binary, cheap, and don't require judging the *reasoning*, only the *result* — prefer these whenever a task has a checkable outcome.
- **Trajectory-level evals.** Did the agent take a reasonable path to get there — right tools, right order, no wasted or dangerous steps? Matters when *how* the agent got the answer matters as much as the answer (e.g. did it check permissions before acting, not just "did the action eventually succeed").
- **LLM-as-judge evals.** For open-ended outputs with no checkable ground truth (is this summary good?), have a separate LLM call score the output against a rubric. Weakest signal of the three — subject to the judge's own biases and inconsistency — but sometimes the only option for genuinely open-ended tasks. Should be validated against human judgment on a sample before being trusted at scale (Day 19 covers this in depth).

A concrete eval suite structure:

```python
test_cases = [
    {"input": "...", "expected_outcome_check": lambda result: "flight_booked" in result.actions},
    # ...
]

def run_eval_suite(agent_fn, test_cases):
    results = [{"case": tc, "passed": tc["expected_outcome_check"](agent_fn(tc["input"]))} for tc in test_cases]
    return {"pass_rate": sum(r["passed"] for r in results) / len(results), "results": results}
```

The point of building this early (Day 18, not after the agent ships) is that agent behavior is genuinely non-deterministic run to run — without a fixed eval suite you're changing a prompt or a tool and just hoping it's better, with no way to catch a regression on the 3 cases that used to work.

## Coding Problem

Write `run_eval_suite(agent_fn, test_cases) -> dict` matching the pattern above, but make it resilient: if `agent_fn(tc["input"])` raises an exception, count that case as failed (don't let one crashing case abort the whole suite), and include the exception message in that case's result entry under `"error"`. Return `{"pass_rate": float, "results": [...]}` where each result includes `"case"`, `"passed"`, and optionally `"error"`.

## Quiz

### Question 1: Outcome vs. Trajectory Evals

**A coding agent's tests pass, but it took 40 unnecessary tool calls and read the same file 10 times to get there. What would an outcome-level eval miss that a trajectory-level eval would catch?**

A) Nothing — outcome-level evals capture everything relevant
B) The inefficiency and redundant steps — outcome-level evals only check the final result, not whether the path taken was reasonable
C) Trajectory-level evals cannot be applied to coding agents
D) The tests passing means the trajectory was necessarily optimal

**Answer**: B

**Explanation**: Outcome-level evals are binary on the end result — they say nothing about the path. A trajectory-level eval specifically checks whether the steps taken were sensible (no redundant calls, correct ordering), which is exactly the kind of inefficiency an outcome-only check can't see.

### Question 2: When to Use LLM-as-Judge

**Why is LLM-as-judge described as the "weakest signal" of the three eval levels, and when is it still worth using?**

A) It's always wrong and should never be used
B) It's subject to the judge model's own biases and inconsistency, but it's sometimes the only option for genuinely open-ended outputs with no checkable ground truth (e.g. "is this summary good?")
C) It's the strongest signal and should always be preferred over outcome checks
D) LLM-as-judge only works for numerical outputs

**Answer**: B

**Explanation**: Outcome and trajectory checks are objective when a checkable criterion exists. LLM-as-judge is used when no such criterion exists — but because it relies on another model's subjective judgment, it's noisier and should be validated against human judgment before being trusted at scale.

### Question 3: Why Build Evals Early

**Why does this lesson argue for building an eval suite before an agent ships, rather than after?**

A) Eval suites are required by law for any AI product
B) Without a fixed, repeatable suite, agent behavior's non-determinism makes it impossible to tell whether a prompt or tool change actually improved things or silently broke previously-working cases
C) Evals are only useful for debugging crashes, not correctness
D) There's no real benefit to building evals before shipping

**Answer**: B

**Explanation**: Agent outputs vary run to run even with no code changes. A fixed eval suite is what lets you distinguish "this change improved things" from "this change happened to work on the one example I tried" — and it catches regressions on cases that used to pass, which ad hoc testing won't reliably do.

## Interview Practice

**1.** You're asked to build an eval suite for an agent with no existing tests. Walk through how you'd choose your first 10 test cases — what makes a case worth including versus redundant with one already there.

**2.** A manager wants a single "quality score" for the agent, like a percentage. Explain why collapsing outcome-level, trajectory-level, and judge-based evals into one number can hide more than it reveals, and what you'd report instead.

**3.** Your eval pass rate improved from 80% to 92% after a prompt change. Describe what you'd check before trusting that number as a genuine improvement rather than an artifact of the eval suite itself.
