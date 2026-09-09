# Day 3: Goals and Task Breakdown

## Concept

Hand an agent a vague goal ("improve our onboarding flow") and it will either stall or hallucinate a plan that looks reasonable but skips the actual constraints. Agents perform better when the goal is decomposed into a sequence of concrete, checkable subtasks before execution starts.

Two decomposition strategies, in order of complexity — and if these sound familiar, they should: this is the same **Plan-and-Execute vs. ReAct** tradeoff from [Day 2](02-the-agent-loop.md#3-picking-a-loop-pattern), just viewed from the angle of "how do I break this goal down" rather than "how does the loop run."

**1. Single-pass planning** (the decomposition-side view of **Plan-and-Execute**). The model produces a full task list up front, then executes it step by step. Cheap and predictable, but brittle — if step 3 reveals new information, the rest of the plan may no longer make sense, and a static plan won't adapt.

```
Goal: "Summarize this repo's open issues by severity"
Plan:
  1. List all open issues
  2. Classify each by severity using labels or content
  3. Group and count by severity
  4. Write summary
```

**2. Interleaved planning (plan-act-replan)** (the decomposition-side view of **ReAct**). The model plans one or two steps ahead, executes, observes the result, and re-plans. More expensive per task (more LLM calls) but handles surprises — an API returning unexpected data, a tool failing, a subtask turning out to be two subtasks.

The failure mode to watch for either way: **goal drift**. Across many steps, an agent's sense of the original objective can degrade as history fills with tool output. Keeping the original goal string verbatim and re-injecting it into every prompt (not just relying on it being "somewhere in history") measurably reduces drift.

## Coding Problem

Write `decompose_goal(goal: str, max_subtasks: int) -> list[str]` that calls an LLM (mocked in your implementation) to produce a numbered subtask list, then validates the output: reject and re-request if the model returns more than `max_subtasks` items, returns 0 items, or returns items that are not strings. Cap retries at 2; on the third failure, return a single-item list `[goal]` as a fallback (treat the whole goal as one subtask rather than crashing).

## Quiz

### Question 1: Decomposition Strategies

**What is the main tradeoff of single-pass planning vs. interleaved plan-act-replan?**

A) Single-pass is always more accurate
B) Single-pass is cheaper and simpler but can't adapt when a step reveals new information
C) Interleaved planning never uses more LLM calls
D) There is no meaningful difference between the two

**Answer**: B

**Explanation**: Single-pass planning produces the full plan upfront in one call, which is cheap but static. Interleaved planning re-plans after each step, costing more LLM calls but adapting to surprises the static plan couldn't anticipate.

### Question 2: Goal Drift

**What causes "goal drift" in a long-running agent, and what mitigates it?**

A) It's caused by the model being too small; only a larger model fixes it
B) It's caused by the original goal getting buried in growing tool-output history; re-injecting the goal verbatim in every prompt mitigates it
C) It's caused by using too many tools; removing tools fixes it
D) It's an unfixable property of all LLMs

**Answer**: B

**Explanation**: As history accumulates tool outputs, the model's attention to the original objective (stated once, early) can weaken. Explicitly re-including the goal in every prompt keeps it salient regardless of how much history has piled up.

### Question 3: Handling Failure

**In the coding problem, why cap decomposition retries and fall back to `[goal]` instead of retrying forever?**

A) Retrying forever is fine as long as the API is free
B) An unbounded retry loop can hang the whole agent on a single malformed response; a bounded fallback guarantees forward progress
C) LLMs never return malformed output more than once
D) The fallback is only for testing and should be removed in production

**Answer**: B

**Explanation**: Any loop without a bound is a liveness risk. Falling back to treating the goal as one subtask after a small number of retries guarantees the agent makes progress instead of hanging on a single bad LLM response.

## Interview Practice

**1.** A user gives an agent the goal "clean up our AWS spend." Walk through how you'd decompose this into subtasks, and identify at which point you'd switch from single-pass planning to interleaved plan-act-replan, and why.

**2.** Describe a concrete scenario where goal drift caused an agent to complete a task that technically matched its recent history but no longer matched the user's original intent. What single change would have prevented it?

**3.** Compare single-pass planning and plan-act-replan on cost, latency, and robustness to surprises. If you had to pick one default for a new agent and only revisit the choice if it caused problems, which would you pick and why?
