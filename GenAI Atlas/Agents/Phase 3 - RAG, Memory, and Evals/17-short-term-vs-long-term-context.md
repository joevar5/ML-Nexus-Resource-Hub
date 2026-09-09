# Day 17: Short-Term vs. Long-Term Context

## Concept

Short-term context is what's in the current prompt — the active task's history (Day 4). Long-term memory (Day 16) is what persists across sessions. The engineering problem this lesson addresses: **how much of long-term memory gets pulled into short-term context, and when**, because pulling in too little makes the agent forget things it should know, and pulling in too much crowds out the actual task with irrelevant background.

The wrong default is "always inject everything the memory store has about this user." That scales badly (token cost, latency) and actively hurts quality — a prompt padded with 40 loosely-relevant facts makes it harder for the model to weight the one fact that actually matters for the current question (context dilution, same underlying issue as Day 14's oversized chunks).

A better default: **retrieve long-term memory the same way you retrieve RAG context — relevance-scored against the current query, with a budget.**

```python
def build_context(current_task, memory_store, max_memory_tokens=500):
    relevant_facts = memory_store.semantic.search(current_task, top_k=10)
    selected = []
    used_tokens = 0
    for fact in relevant_facts:  # already sorted by relevance
        if used_tokens + fact.token_count > max_memory_tokens:
            break
        selected.append(fact)
        used_tokens += fact.token_count
    return selected
```

This treats long-term memory injection as its own small retrieval problem with its own token budget, separate from (and typically much smaller than) the budget for the actual task's working context. A useful mental model: short-term context is the *workspace*, long-term memory retrieval is a *tool call that happens to run automatically at the start of a task* — it should be scoped and bounded the same way any other retrieval is.

The other direction matters too: not everything belongs in long-term memory. A one-off detail relevant only to the current task (a file path being edited right now) shouldn't get promoted into semantic memory just because it appeared in an interaction — that pollutes future retrieval with task-specific noise that's irrelevant to any other session. Promotion to long-term memory should be selective (Day 16's `should_extract_facts` gate), not automatic.

## Coding Problem

Write `build_context(current_task: str, memory_facts: list[dict], max_tokens: int, relevance_fn) -> list[dict]` where `memory_facts` is `[{"text": str, "token_count": int}, ...]` and `relevance_fn(current_task, fact_text) -> float` scores relevance (mocked). Sort facts by relevance score descending, then greedily select facts (highest relevance first) until adding the next one would exceed `max_tokens`. Return the selected list, preserving relevance order.

## Quiz

### Question 1: Why Not Inject Everything

**Why is "always inject every fact the memory store has about this user" a bad default?**

A) Memory stores have a hard limit of 5 facts
B) It scales poorly in token cost/latency, and a prompt padded with many loosely-relevant facts makes it harder for the model to weight the one fact that actually matters — the same context-dilution problem as oversized RAG chunks
C) It's technically impossible to inject more than one fact per prompt
D) There is no downside — more context is always better

**Answer**: B

**Explanation**: Unbounded injection has both a cost problem (more tokens, more latency) and a quality problem — diluting the prompt with low-relevance facts makes the genuinely important one harder for the model to prioritize, mirroring the oversized-chunk dilution issue from Day 14.

### Question 2: Memory Injection as Retrieval

**What does treating long-term memory injection "the same way you retrieve RAG context" mean in practice?**

A) Memory and RAG must literally use the same vector database
B) Score memory facts for relevance to the current task and select within a token budget, rather than injecting the entire memory store unconditionally
C) Memory should never be retrieved — only injected in full
D) This comparison doesn't apply to memory systems

**Answer**: B

**Explanation**: Applying the RAG mindset to memory means: don't dump everything in, retrieve what's relevant to the current query and bound it by a budget — the same relevance-scoring-plus-budget approach used for retrieving document chunks.

### Question 3: What Shouldn't Become Long-Term Memory

**Why shouldn't a one-off task detail (e.g. "editing config.yaml right now") be automatically promoted to long-term semantic memory?**

A) File paths cannot be stored as text
B) It's specific to the current task and irrelevant to future sessions — promoting it pollutes future memory retrieval with task-specific noise that won't be useful again
C) Long-term memory has no capacity limits, so this doesn't matter
D) All details should always be promoted to memory automatically

**Answer**: B

**Explanation**: Not every detail that appears during a task is a durable fact about the user or system worth remembering forever. Promoting task-scoped, one-off details to long-term memory adds noise that degrades future retrieval relevance without providing lasting value — selective promotion (Day 16) exists specifically to filter this out.

## Interview Practice

**1.** Your agent's prompts are getting expensive because every request injects a large chunk of long-term memory "just in case it's relevant." Walk through how you'd redesign this to cut cost without losing genuinely useful context.

**2.** A colleague argues "just put everything in one context window, short-term and long-term both — simpler than managing a separate retrieval step for memory." Explain the concrete downside of this approach as the memory store grows.

**3.** Describe a real scenario where injecting an irrelevant-but-technically-related memory fact actively made an agent's answer worse, not just longer. What made it harmful rather than just unnecessary?
