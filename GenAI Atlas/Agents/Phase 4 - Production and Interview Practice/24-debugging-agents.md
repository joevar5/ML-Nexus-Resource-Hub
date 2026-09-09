# Day 24: Debugging Agents

## Concept

Debugging an agent starts from the trace (Day 23) — without one, there's nothing to debug except vibes. Given a trace of a failed run, a systematic approach beats staring at the whole thing at once:

**1. Localize: model decision, tool execution, or data problem?** Walk the trace step by step and find the first point where behavior diverges from what a correct run should look like. Was it the *n*-th LLM call that decided on the wrong action given correct information (a reasoning failure), or did an earlier tool call return bad/misleading data that a *reasonable* decision was then based on (a data problem upstream of the model)? These need completely different fixes — prompt/reasoning failures get fixed by prompt changes; data failures get fixed at the tool/retrieval layer.

**2. Reproduce in isolation.** Once you've localized the divergence, extract just that step — the specific prompt and context that led to the bad decision — and re-run it standalone, outside the full agent loop. This turns a multi-step, expensive-to-rerun agent failure into a single, cheap, directly inspectable LLM call you can iterate on.

**3. Check if it's systematic or a one-off.** Run the isolated repro against your eval suite (Day 18) or a batch of similar past cases. A single input that jailbreaks a single edge case is a different fix (patch the specific case) than a pattern that fails 30% of a whole input category (fix the underlying prompt/tool design).

```python
def localize_divergence(trace_events: list[dict], expected_trajectory: list[str]) -> int | None:
    actual_actions = [e["name"] for e in trace_events if e["type"] == "tool_call"]
    for i, (actual, expected) in enumerate(zip(actual_actions, expected_trajectory)):
        if actual != expected:
            return i  # index of first divergence
    return None if len(actual_actions) == len(expected_trajectory) else len(expected_trajectory)
```

A common mistake: trying to fix a debugging session by rewriting the whole system prompt based on one bad trace. A single failure is one data point — the localize-and-reproduce-in-isolation steps above exist precisely so you can test a targeted fix against the specific failure *and* the existing eval suite before concluding it's actually better, rather than trading one fixed case for several newly broken ones.

## Coding Problem

Write `localize_divergence(trace_events, expected_trajectory)` as above, but handle a subtlety: if `expected_trajectory` allows for **either of two acceptable next actions** at a given step (represented as a `set` instead of a `str` in the expected list, e.g. `{"search", "search_v2"}`), treat a match against any element of that set as non-divergent. Return the index of the first true divergence, or `None` if the whole trajectory matches.

## Quiz

### Question 1: Localizing the Failure

**An agent's final answer is wrong. Why is it important to determine whether the failure came from a bad model decision vs. bad data from an earlier tool call, before attempting a fix?**

A) It doesn't matter — all agent failures are fixed the same way
B) The two failure types require different fixes — a reasoning failure needs a prompt change, a data failure needs a fix at the tool/retrieval layer, and applying the wrong one won't actually fix the root cause
C) Only model decisions can ever be the cause of a wrong answer
D) Tool calls can never be a source of failure

**Answer**: B

**Explanation**: A model reasoning correctly over bad input still produces a bad output — fixing the prompt won't help if the underlying data was wrong. Correctly localizing which layer diverged from expected behavior determines which layer actually needs the fix.

### Question 2: Reproduce in Isolation

**Why extract and re-run just the diverging step standalone, rather than re-running the entire multi-step agent to test a fix?**

A) Isolated re-runs are always inaccurate and shouldn't be trusted
B) It turns an expensive, multi-step, hard-to-inspect failure into a single cheap LLM call that can be directly iterated on and tested
C) The full agent loop cannot be re-run more than once
D) There's no benefit — always test against the full pipeline

**Answer**: B

**Explanation**: Re-running the full loop for every fix attempt is slow and noisy (other steps can vary too). Isolating just the diverging prompt/context turns it into a fast, directly testable single call, which makes iterating on a fix far more efficient.

### Question 3: Systematic vs. One-Off

**After finding a fix for a bad trace, why run it against the full eval suite instead of just confirming it fixes the one case you found?**

A) The eval suite is only relevant during initial development, not for fixes
B) A fix targeted at one bad trace can accidentally break other previously-working cases — checking against the eval suite catches regressions before considering the fix "done"
C) Running the eval suite has no bearing on whether a fix is correct
D) One passing case is always sufficient proof a fix works

**Answer**: B

**Explanation**: A prompt or logic change made to fix one observed failure can easily shift behavior on other cases in unintended ways. Validating against the full eval suite (not just the originally failing case) is what catches "fixed one thing, broke three others" before it ships.

## Interview Practice

**1.** Walk through your debugging process, step by step, for an agent that occasionally produces a correct-looking but factually wrong answer, with no exceptions or errors in the trace.

**2.** A colleague's instinct on seeing any agent bug is to rewrite the system prompt. Explain when that's the right fix and when it isn't, using the localize-first framework from this lesson.

**3.** Describe how you'd tell the difference between a one-off failure (safe to patch narrowly) and a systematic failure (needs a broader fix), given only a handful of bad traces to start from.
