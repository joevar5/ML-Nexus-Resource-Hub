# Day 12: Mini Project — Research Assistant

## Brief

Build an agent that answers a research question by searching, reading, and synthesizing — applying everything from Days 6–11 in one system.

**Goal:** given a question like "What are the tradeoffs between LoRA and full fine-tuning?", the agent should:

1. Call a `search(query: str) -> list[{title, url, snippet}]` tool (stub this — return canned results for a fixed set of test queries).
2. Decide which results are worth reading, and call `fetch(url: str) -> str` to get full text (also stub, returning canned page bodies keyed by URL).
3. Decide whether it has enough information or needs another search (conditional branching, Day 11).
4. Produce a final structured answer: `{"answer": str, "sources": list[str]}` (structured output, Day 10).

## Requirements

- Use the agent loop pattern from Day 2 — a `run(goal, max_steps)` function that alternates LLM calls and tool execution.
- Validate every tool call against its schema before executing (Day 7).
- Wrap tool execution in error handling that returns structured failures, not exceptions (Day 8).
- Cap total tool calls at 6. If the budget runs out before the model produces a final answer, return the best partial answer with whatever sources were gathered — not an error.
- Detect and break out of a repeated-identical-search loop (Day 5) — if the same query is searched twice, force the next step to either fetch a result or finalize, rather than searching again.
- The final answer must cite only URLs that were actually fetched during the run — validate this before returning (don't trust the model to have not cited an unfetched source).

## Suggested Structure

```
research_agent/
  tools.py       # search(), fetch() stubs + registry
  agent.py       # run() loop, calls LLM, dispatches tools
  schemas.py     # tool schemas + final-answer schema
  validate.py    # tool-call validation, citation validation
  test_cases.py  # 3-4 fixed questions with canned search/fetch data
```

## Self-Check

Before considering this done, verify:

- [ ] Running with a question that has a clean answer in 2 search + 2 fetch calls produces a correctly-cited answer.
- [ ] Running with a question where the first search returns irrelevant results forces a second, different search (not a repeat of the first).
- [ ] Forcing the step budget to 2 (too low to finish normally) still returns a non-crashing partial answer.
- [ ] A citation validation test: manually construct a final answer citing a URL that was never fetched, and confirm your validation step catches it.

This project doesn't require a real search API — the point is the control flow (loop, branching, budget enforcement, citation grounding), which stub tools exercise identically to real ones.

## Interview Practice

**1.** Walk through how your research assistant decides it has "enough" information to stop searching and produce a final answer. What signal drives that decision, and what happens if the model is wrong about having enough?

**2.** A reviewer asks why you capped tool calls at 6 instead of letting the agent search until it's satisfied. Defend the design choice, including what would go wrong without a cap.

**3.** Describe how you'd extend this project to handle a question that genuinely requires information from two independent sub-topics (e.g. "compare X and Y"). Would your current control flow handle it, or does it need to change?
