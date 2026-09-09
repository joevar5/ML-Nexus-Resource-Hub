# Day 23: Observability

## Concept

An agent that fails in production with no trace of what it did is unfixable — you can't debug what you can't see. Observability for agents means capturing enough structured detail about every run to reconstruct exactly what happened, not just whether it succeeded or failed.

What a minimal agent trace needs, per run:

- **Every LLM call**: the full prompt sent, the raw response, token counts, latency, model/version used. Prompt and response, not a summary — you need to see exactly what the model saw and said to debug a specific failure.
- **Every tool call**: name, arguments, result (or error), latency. This is what lets you distinguish "the model made a bad decision" from "the model made a reasonable decision but the tool returned bad data."
- **The step sequence**: an ordered log tying LLM calls and tool calls together into the actual loop trajectory, so you can see the full path the agent took, not just isolated events.
- **A trace/run ID** that ties all of the above together and correlates with any downstream system logs (if a tool call triggered a database write, that write's log should be correlatable back to this specific agent run).

```python
class TraceLogger:
    def __init__(self, run_id):
        self.run_id = run_id
        self.events = []

    def log_llm_call(self, prompt, response, latency_ms):
        self.events.append({"type": "llm_call", "prompt": prompt, "response": response,
                             "latency_ms": latency_ms, "run_id": self.run_id})

    def log_tool_call(self, name, args, result, latency_ms):
        self.events.append({"type": "tool_call", "name": name, "args": args,
                             "result": result, "latency_ms": latency_ms, "run_id": self.run_id})
```

The distinction that matters in practice: **logging is for humans reading after the fact; observability is for systems querying at scale.** A print statement is logging. A structured event with consistent fields across every run, queryable by "show me all runs where tool X failed in the last hour" or "show me the p95 latency of the search tool," is observability — and that requires consistent structure decided up front, not ad hoc strings appended when convenient.

## Coding Problem

Write `TraceLogger` as above, plus a method `to_summary() -> dict` that computes, from `self.events`: total LLM calls, total tool calls, total latency (sum of all `latency_ms`), and a list of tool names that had at least one call with `result` containing the key `"error"`. This summary is what a dashboard would show per run without needing to render every individual event.

## Quiz

### Question 1: What a Trace Needs

**Why should an agent trace log the full prompt and response for each LLM call, rather than just a summary like "LLM call succeeded"?**

A) Full logging is required by every LLM provider's API
B) Debugging a specific failure requires seeing exactly what the model was shown and exactly what it said — a summary discards the detail needed to understand why a particular decision was made
C) Summaries are always more useful than full text for debugging
D) There's no practical difference for debugging purposes

**Answer**: B

**Explanation**: When something goes wrong, the question is almost always "what did the model actually see, and what did it actually say" — a coarse summary can't answer that. Full prompt/response capture is what makes a specific failure reproducible and diagnosable after the fact.

### Question 2: Tool Call vs. Model Decision

**Why does logging tool call arguments and results (not just that a call happened) matter for debugging?**

A) It doesn't matter — only whether the call succeeded is relevant
B) It lets you distinguish "the model made a reasonable decision but the tool returned bad data" from "the model made a bad decision" — two very different root causes that look identical without this detail
C) Tool call arguments are never useful for debugging
D) This information is automatically available without explicit logging

**Answer**: B

**Explanation**: An agent failure can originate from the model's reasoning or from the data a tool returned. Without the actual arguments and results logged, these two very different failure sources are indistinguishable from the outside — you'd be debugging blind.

### Question 3: Logging vs. Observability

**What distinguishes "observability" from simple logging, per this lesson?**

A) Observability just means using a bigger log file
B) Observability requires consistent, structured fields across every run so the data can be queried at scale (e.g. "show me all runs where tool X failed"), decided up front rather than ad hoc
C) Logging and observability are exactly the same thing
D) Observability means never writing anything to disk

**Answer**: B

**Explanation**: Ad hoc log strings are readable by a human scanning them one at a time, but they can't be systematically queried. Observability requires deciding on a consistent event schema in advance so the resulting data supports aggregate queries and dashboards, not just individual-run inspection.

## Interview Practice

**1.** Design the event schema you'd log for every agent run in a production system. What fields are non-negotiable, and what would you leave out to control storage cost?

**2.** A teammate says "we already have application logs, we don't need separate agent tracing." Explain what's missing from generic application logs for debugging a specific bad agent decision.

**3.** You need to answer "which tool call is our biggest latency contributor across all runs this week" without re-reading every trace by hand. Describe the query/dashboard you'd need, and what that implies about how traces must be structured.
