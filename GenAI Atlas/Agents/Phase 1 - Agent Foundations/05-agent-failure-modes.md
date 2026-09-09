# Day 5: Agent Failure Modes

## Concept

Agents fail differently than single-shot LLM calls — a bad single call gives one bad answer, but a bad step inside a loop can compound, cascade, or run away. Five failure modes come up repeatedly enough to name:

- **Looping.** The agent repeats the same (or a cycle of) tool calls without making progress — e.g. searching, not finding what it needs, searching the same query again. Detect with a repeated-action check (Day 2); fix by forcing a different strategy after N identical attempts.
- **Hallucinated tool calls.** The model invokes a tool that doesn't exist, or calls a real tool with arguments that don't match its schema. This is a parsing/validation problem — always validate tool calls against the actual available tool list and schema before executing, never execute blind.
- **Silent partial failure.** A tool call "succeeds" (no exception) but returns data that doesn't actually satisfy the request — an empty result set, a truncated response — and the agent proceeds as if it got what it needed. Requires the tool layer to distinguish "ran successfully" from "produced a useful result."
- **Compounding errors.** An early wrong assumption propagates through every later step because nothing re-validates it. E.g. the agent misreads a file path on step 2 and every subsequent file operation targets the wrong path. Mitigated by periodic self-checks against ground truth, not just forward momentum.
- **Runaway cost/steps.** No termination condition triggers, and the agent burns through step budget or, worse, real-world side effects (API calls with cost, emails sent) without ever reaching a stopping point. Mitigated by hard step/cost ceilings enforced outside the model's control — never rely on the model to decide to stop.

The common thread: agents fail when a wrong intermediate state gets treated as ground truth and nothing catches it. Every mitigation above is some form of validation — checking a tool call's shape, checking a tool result's usefulness, checking an assumption against reality — inserted somewhere the model itself doesn't naturally have a reason to look back.

## Coding Problem

Write `detect_failure_mode(step_history: list[dict]) -> str | None` where each entry in `step_history` is `{"tool": str, "args": dict, "result": Any}`. Return `"looping"` if the last 3 entries have identical `tool` and `args`. Return `"runaway"` if `len(step_history) > 50`. Return `None` otherwise. Keep the checks in that priority order and return on the first match.

## Quiz

### Question 1: Hallucinated Tool Calls

**An agent emits a tool call for `search_database(query="...")`, but the actual available tool is named `db_search`. What's the correct handling?**

A) Execute it anyway — the model probably meant the right thing
B) Validate the tool name against the actual available tool list before execution, and reject/re-prompt if it doesn't match
C) Silently rename the call to the closest matching tool name
D) Crash the entire agent process

**Answer**: B

**Explanation**: Never execute an unvalidated tool call. Checking the tool name and argument schema against what's actually registered — and feeding a rejection back to the model to retry — prevents hallucinated calls from causing real side effects.

### Question 2: Silent Partial Failure

**A search tool returns an empty result list (no exception raised) because the query matched nothing, and the agent's next step assumes it has the data it asked for. What failure mode is this, and what's the general fix?**

A) Looping; fix with a repeated-call check
B) Silent partial failure; the tool layer needs to signal "no useful result" distinctly from "call succeeded," so the agent can react instead of assuming success
C) Runaway cost; fix with a step ceiling
D) This is not a failure — empty results are always fine to proceed with

**Answer**: B

**Explanation**: A tool call without an exception isn't the same as a tool call that produced something usable. Distinguishing "technically succeeded" from "returned what was needed" lets the agent branch (retry with a different query, ask for clarification) instead of silently building on nothing.

### Question 3: Runaway Agents

**Why should step/cost ceilings be enforced outside the model's control rather than relying on the model to decide when to stop?**

A) Models are contractually forbidden from stopping early
B) A model that's stuck in a bad loop has no internal signal telling it it's stuck — an external hard limit is the only guaranteed backstop
C) It's cheaper to let the model decide
D) Step ceilings are only a UI feature, not a safety feature

**Answer**: B

**Explanation**: A model reasoning inside a failing loop generally can't reliably detect that it's failing — that's exactly the condition being guarded against. An externally enforced ceiling (step count, cost, wall-clock time) provides a guarantee that doesn't depend on the model's own judgment.

## Interview Practice

**1.** Walk through how you'd distinguish "looping" from "compounding errors" by looking only at an agent's trace log, without knowing in advance which failure occurred. What specific pattern in the log would tell you which one you're looking at?

**2.** A teammate says "silent partial failure isn't a real risk — if the tool call didn't throw an exception, it worked." Explain why this reasoning is wrong, using a concrete example.

**3.** Design a monitoring rule that would catch a runaway-cost agent in production before it burns through a full day's budget, without needing to know in advance which specific task caused the runaway behavior.
