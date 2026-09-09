# Day 11: Multi-Step Tool Use

## Concept

A single tool call answers a single sub-question. Real tasks usually need a chain: search for a file, read it, edit it, run tests, read the failure, edit again. The hard part isn't calling multiple tools — most frameworks support that trivially — it's **managing dependencies and ordering between calls** correctly.

Three patterns show up repeatedly:

- **Sequential chaining.** Output of tool A feeds directly into tool B's arguments. E.g. `search()` returns a file path, which becomes the argument to `read_file()`. The agent (or an orchestration layer) must correctly extract the relevant value from A's result rather than re-deriving it from scratch or hallucinating it.
- **Parallel/independent calls.** Some steps don't depend on each other — checking weather in 3 cities, reading 5 independent files. Many APIs support returning multiple tool calls in a single response for exactly this case; executing them concurrently instead of one-by-one materially cuts latency for I/O-bound tools.
- **Conditional branching.** The next tool call depends on a previous result's *content*, not just its existence — if `check_inventory()` returns 0, call `notify_supplier()`; otherwise call `process_order()`. This is where the agent loop's "think" step earns its keep: a fixed pipeline can't branch, but a loop that re-prompts the model with the result can.

The failure to watch for is **argument fabrication between steps** — if tool B needs a value that should come from tool A's result, but the model doesn't have A's actual result in context (truncated history, or it never actually called A), it will often generate a plausible-looking value rather than surface that it's missing information. Explicitly including full, recent tool results in context — not summaries — for any value about to be used as another tool's argument avoids this.

## Coding Problem

Write `plan_execution_order(calls: list[dict]) -> list[list[dict]]` that batches a list of tool calls (each `{"name": str, "depends_on": list[str]}`, where `depends_on` names other calls by `"name"`) into ordered groups that can run in parallel — a topological sort into levels. Calls with no unmet dependencies go in level 0, then level 1 contains calls whose dependencies are all satisfied by level 0, and so on. Raise `ValueError("cycle detected")` if dependencies can't all be resolved.

## Quiz

### Question 1: Sequential Dependency

**A `search()` call returns a file path, and the next step calls `read_file()` with that path. What's the risk if the model's context doesn't actually contain `search()`'s real output?**

A) `read_file()` will simply fail with a clear "no path given" error every time
B) The model may fabricate a plausible-looking file path rather than surface that the information is missing
C) There is no risk — models always retain perfect memory of prior tool calls
D) The API will automatically inject the correct value

**Answer**: B

**Explanation**: If the actual prior result isn't visible in context, the model still has to produce *something* for the argument — and it will often generate a plausible guess instead of recognizing and reporting the gap, which is a subtle and dangerous failure since the resulting call looks legitimate.

### Question 2: Parallel Execution

**When should independent tool calls (e.g. checking weather for 3 unrelated cities) be executed in parallel rather than sequentially?**

A) Never — sequential execution is always safer
B) When the calls are I/O-bound and don't depend on each other's results, parallel execution reduces total latency with no correctness cost
C) Only when using exactly 2 tools
D) Parallel execution is not supported by any LLM API

**Answer**: B

**Explanation**: Calls with no data dependency between them gain nothing from being serialized — running them concurrently cuts wall-clock latency, and several major APIs support returning multiple tool calls in one response specifically to enable this.

### Question 3: Conditional Branching

**Why can't a fixed, pre-written pipeline (call A, then B, then C) handle "if inventory is 0, notify supplier; otherwise process order"?**

A) Fixed pipelines can handle any logic without modification
B) A fixed pipeline has no mechanism to inspect a result's content and choose a different next step — that requires the loop's think/decide step to re-evaluate based on the actual returned value
C) Conditional logic is impossible for any software system
D) This has nothing to do with agent design

**Answer**: B

**Explanation**: A static pipeline's steps are fixed at write-time. Branching on a result's actual content requires the "think" step of the agent loop to see that result and choose the next action accordingly — which is precisely the capability a hardcoded sequence lacks.

## Interview Practice

**1.** Design a multi-step tool-use flow for an agent that needs to check inventory across 3 warehouses and then decide where to fulfill an order from. Identify which steps can run in parallel and which must be sequential, and why.

**2.** An agent's chain of tool calls produced a wrong final answer, but each individual tool call's arguments look reasonable in isolation. Walk through how you'd determine whether this is argument fabrication (Day 11) versus a different failure mode from Day 5.

**3.** A teammate wants to always run every tool call sequentially "to keep things simple and predictable." Argue for when parallel execution is worth the added complexity, and when sequential really is the right call.
