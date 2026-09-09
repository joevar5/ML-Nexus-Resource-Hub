# Day 25: Cost and Latency

## Concept

Agents are more expensive and slower than single LLM calls by construction — every additional loop iteration is another full LLM call, and tool calls add their own latency on top. A 10-step agent task isn't 10x the cost of a single call in a rough sense — it's exactly that, plus growing context (Day 4) means later calls in the loop cost more per-call than earlier ones, since each includes more accumulated history.

Where the cost/latency actually goes, and the corresponding lever:

- **Input tokens grow with history.** Every tool result appended to context gets re-sent on every subsequent call. Mitigation: the state-management strategies from Day 4 (rolling summaries, structured state instead of full transcripts) directly reduce cost, not just context-window risk.
- **Redundant LLM calls.** An agent that re-derives something it already computed, or re-reads a file it already read this run, pays for that call again. Mitigation: caching — store tool results within a run (and across runs for stable data) so identical calls don't re-execute or re-pay.
- **Model choice per step.** Not every step needs the most capable (and most expensive) model. A classification or routing decision ("is this a billing or technical question") often works fine on a smaller/cheaper model, reserving the expensive model for the step that actually needs deep reasoning. This is model routing, and it's one of the highest-leverage cost optimizations available.
- **Sequential vs. parallel tool calls.** Latency (not cost) specifically benefits from executing independent tool calls concurrently (Day 11) rather than one at a time — this doesn't reduce total compute, but reduces wall-clock time, which matters for user-facing latency even when it doesn't change the bill.

```python
def estimate_run_cost(trace_events: list[dict], price_per_1k_input: float, price_per_1k_output: float) -> float:
    total = 0.0
    for e in trace_events:
        if e["type"] == "llm_call":
            total += (e["input_tokens"] / 1000) * price_per_1k_input
            total += (e["output_tokens"] / 1000) * price_per_1k_output
    return total
```

The practical discipline: cost and latency should be tracked per run (via the observability layer from Day 23) from the start, not discovered as a surprise bill after launch — a change that improves accuracy by 2% but doubles average steps-per-task needs to be evaluated against that cost, not just against the accuracy number in isolation.

## Coding Problem

Write `estimate_run_cost(trace_events, price_per_1k_input, price_per_1k_output)` as above, extended to also return a **cache hit savings estimate**: if any `llm_call` event has `"cached": True` (representing a prompt-cache hit on repeated context), compute its cost at a reduced rate (assume cached input tokens cost 10% of `price_per_1k_input`) instead of the full rate, and return `{"total_cost": float, "estimated_savings": float}` where `estimated_savings` is what the cost would have been without caching, minus the actual cost.

## Quiz

### Question 1: Why Agent Cost Grows Non-Linearly

**Why does a 10-step agent task typically cost more than 10x a single LLM call, not exactly 10x?**

A) LLM providers charge a flat fee per agent regardless of steps
B) Each step's context includes the accumulated history from all previous steps, so later calls in the loop have larger input token counts than earlier ones — cost per call increases across the run, not just call count
C) Agent calls are always priced identically regardless of context size
D) This claim is false — agent cost scales exactly linearly with step count

**Answer**: B

**Explanation**: Because context accumulates (Day 4), each subsequent call in the loop carries more tokens than the last. Total cost is the sum of increasingly large calls, not N copies of a fixed-size call — which is exactly why state management strategies double as cost optimizations.

### Question 2: Model Routing

**Why is using a smaller/cheaper model for a simple routing decision (e.g. "is this billing or technical?") a high-leverage cost optimization?**

A) Smaller models are always more accurate
B) Simple classification tasks often don't require the most capable model's reasoning ability, so routing them to a cheaper model saves cost on high-frequency, low-complexity steps while reserving the expensive model for steps that actually need it
C) Model choice has no effect on cost
D) Routing decisions cannot be made by any model smaller than the largest available

**Answer**: B

**Explanation**: Not all steps in an agent's workflow are equally hard. Matching model capability (and cost) to the actual difficulty of each step — cheap models for simple routing/classification, expensive models for complex reasoning — is one of the most effective levers for reducing average cost per task.

### Question 3: Latency vs. Cost

**Running 3 independent tool calls in parallel instead of sequentially reduces latency. Does it also reduce total compute cost?**

A) Yes, parallel execution is always cheaper in total compute
B) Not necessarily — parallelism reduces wall-clock time by overlapping the calls, but the total amount of compute/API usage across all 3 calls is roughly the same either way
C) Parallel execution always doubles the cost
D) Parallel execution is only possible for LLM calls, never for tool calls

**Answer**: B

**Explanation**: Parallelizing independent calls changes when the work happens (concurrently vs. sequentially), not how much work happens. It's a latency optimization, distinct from cost optimizations like reducing token count or routing to cheaper models — worth doing, but for a different reason.

## Interview Practice

**1.** Your agent's per-task cost tripled after adding a new capability, but accuracy only improved by 2%. Walk through how you'd decide whether that tradeoff is worth keeping, and what you'd measure to make the case either way.

**2.** Design a model-routing strategy for a multi-step agent — which steps get the cheap model, which get the expensive one, and what's your reasoning for the split.

**3.** A stakeholder asks why the agent gets slower and more expensive the longer a single conversation runs, even though each individual step seems fast. Explain the mechanism, and propose one concrete fix.
