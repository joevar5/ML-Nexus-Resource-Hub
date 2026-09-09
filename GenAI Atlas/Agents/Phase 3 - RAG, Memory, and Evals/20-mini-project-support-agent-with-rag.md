# Day 20: Mini Project — Support Agent with RAG

## Brief

Build a customer-support agent that answers questions grounded in a knowledge base, remembers facts about the user across turns, and is measured by an eval suite — combining Days 13–19.

**Goal:** given a support question and a fixed knowledge base (stub as a list of `{id, text}` docs), the agent should retrieve relevant docs, answer with citations, use/update simple user memory (e.g. "this user is on the Pro plan"), and be scored by an eval suite you write.

## Requirements

- **Retrieval (Day 13-14):** implement `retrieve(query, docs, k)` using any similarity proxy you like (real embeddings, or a simple keyword-overlap score — the scoring logic matters more than the retrieval method's sophistication for this project).
- **Grounded, cited answers (Day 15):** the agent's answer must cite which doc(s) it used, and must say "I don't have information about that" when retrieval returns nothing sufficiently relevant, rather than answering from general knowledge.
- **Memory (Day 16-17):** maintain a small per-user fact store (e.g. `{"plan": "Pro", "past_issue": "billing dispute on 2026-02-01"}`). Inject only facts relevant to the current question into the prompt (not the entire fact store every time), and update the store when the conversation reveals a new fact.
- **Eval suite (Day 18-19):** write at least 5 test cases covering: (a) a question the KB clearly answers, (b) a question the KB doesn't cover at all (expect the "don't have information" response, not a hallucinated one), (c) a question where the correct answer depends on a remembered user fact (e.g. "what's my refund policy" depends on plan tier), (d) a case testing that an irrelevant stored fact isn't injected into an unrelated question, (e) at least one rubric-scored (LLM-as-judge or manual rubric) case for answer tone/conciseness.

## Suggested Structure

```
support_agent/
  knowledge_base.py   # stub docs
  retrieve.py          # retrieve(query, docs, k)
  memory.py            # per-user fact store, relevance-filtered injection
  agent.py             # ties retrieval + memory + citation-required prompting together
  evals/
    test_cases.py       # the 5+ cases above
    run_evals.py         # outcome-level pass/fail + rubric scoring, reports pass_rate
```

## Self-Check

Before considering this done, verify:

- [ ] A question outside the KB's coverage returns the explicit "no information" response — not a plausible-sounding guess.
- [ ] Every answer that does use KB content includes a citation, and the citation resolves to a real doc id.
- [ ] The same question asked for two different simulated users (different stored plan tiers) produces different, correctly plan-specific answers.
- [ ] A fact irrelevant to the current question (e.g. a past unrelated billing dispute, when the question is about password reset) is not injected into that prompt — check this by inspecting what actually got sent to the LLM, not just the final answer.
- [ ] `run_evals.py` produces a pass rate and a per-case breakdown, and re-running it on unmodified code gives the same result (no flaky test-case design).

This project's real point isn't the KB or the memory store — it's whether the retrieval, citation, and memory-injection *decisions* are correct and testable, which the eval suite is what actually proves.

## Interview Practice

**1.** Walk through what happens in your system when a user asks a question the KB doesn't cover — trace it from the retrieval call through to the final response, and identify exactly where "no relevant information" gets decided.

**2.** A reviewer asks why memory facts are filtered by relevance before injection instead of just always including the user's plan tier and recent issues. Defend the design, including a case where always-including would produce a worse answer.

**3.** Your eval suite has 5 cases and they all pass. A colleague says that's sufficient coverage. Push back — what kinds of failures could still exist that these 5 cases wouldn't catch?
