# Day 16: Memory in Agents

## Concept

"Memory" in an agent context means information that persists *across* interactions, not just within a single loop (that's just Day 4's state within one task). Three distinct kinds get conflated under the word "memory" — treating them as one thing is where most memory-system designs go wrong:

- **Episodic memory** — a record of specific past interactions ("on March 3rd, the user asked about refunds and I told them X"). Useful for continuity across sessions. Typically stored as timestamped events, retrieved by recency or relevance to the current query.
- **Semantic memory** — facts distilled from past interactions, not tied to a specific event ("the user prefers concise answers", "the user's company uses Kubernetes"). Usually built by summarizing episodic memory periodically, not stored per-interaction.
- **Procedural memory** — learned patterns about *how* to do something ("when this user asks about billing, check their account tier first"). Rarest to implement explicitly; often approximated by few-shot examples pulled from past successful interactions.

The engineering question for all three is the same: **what gets written, when, and how is it retrieved later.** Naive approaches store every message ever exchanged and retrieve by embedding similarity at query time — this works for episodic recall but doesn't produce semantic memory (raw messages aren't facts) and doesn't scale storage/retrieval cost indefinitely.

A working pattern for a long-lived agent:

```python
def update_memory(interaction, memory_store):
    memory_store.episodic.append(interaction)  # always log the raw event
    if should_extract_facts(interaction):        # e.g. every N interactions, or on session end
        facts = llm.extract_facts(memory_store.episodic.recent(n=10))
        memory_store.semantic.upsert(facts)      # deduplicated, keyed facts — not raw text
```

The failure to avoid: treating "memory" as "just retrieve relevant past messages via RAG and stuff them in context." That's episodic recall only — it doesn't consolidate into durable facts, meaning the same insight has to be "re-discovered" by embedding search every time rather than being known outright, and it scales badly since more history means more (and noisier) retrieval candidates over time.

## Coding Problem

Write `extract_and_upsert_facts(recent_messages: list[str], existing_facts: dict[str, str], llm) -> dict[str, str]` where `llm.extract_facts(messages) -> dict[str, str]` (mocked) returns new candidate facts as `{key: value}` pairs (e.g. `{"preferred_response_length": "concise"}`). Merge them into `existing_facts`, where a new fact **overwrites** an existing key only if the new value differs — return the updated dict, and separately return a list of `(key, old_value, new_value)` tuples for any keys that were actually changed, so a caller can log what memory changed and why.

## Quiz

### Question 1: The Three Kinds of Memory

**A user mentions in passing that they prefer short answers, and this preference should apply to all future interactions, not just be recalled as "on this date they said this." What kind of memory is this?**

A) Episodic memory — a record of the specific event
B) Semantic memory — a distilled, durable fact independent of any specific event
C) Procedural memory — a pattern about how to perform a task
D) This isn't a form of memory at all

**Answer**: B

**Explanation**: The specific moment the preference was stated is episodic; the preference itself, generalized and applied going forward regardless of when it was stated, is semantic memory — a fact distilled from an interaction, not the interaction itself.

### Question 2: Why RAG-Over-History Isn't Enough

**Why does "just retrieve relevant past messages via embedding search" fall short as a full memory system?**

A) Embedding search cannot be applied to conversation history
B) It only provides episodic recall — it doesn't consolidate repeated insights into durable, directly-known facts, so the same conclusion must be re-derived by search every time, and retrieval quality degrades as history grows
C) It works perfectly and needs no additional layer
D) It's technically impossible with current vector databases

**Answer**: B

**Explanation**: Retrieving raw past messages surfaces episodic content but doesn't produce semantic facts — nothing is "known" outright, everything must be re-found and re-inferred each time, and as history accumulates, both retrieval cost and noise increase.

### Question 3: Fact Merging

**In the coding problem, why overwrite an existing fact only when the new value actually differs, rather than always overwriting?**

A) To reduce API costs by skipping the extraction call entirely
B) So the system can produce an accurate change log (what changed and why) instead of recording a "change" every time an unchanged fact happens to be re-extracted
C) Overwriting is never correct behavior for facts
D) There's no practical reason — it's an arbitrary choice

**Answer**: B

**Explanation**: If facts are re-extracted periodically, the same true fact will often be re-derived unchanged. Only recording an update when the value genuinely changes keeps the change log meaningful — useful for auditing why the agent's understanding of the user shifted, and avoiding log noise from redundant re-confirmations.

## Interview Practice

**1.** Design the memory system for an assistant that's used daily over months. What gets written to episodic memory, what gets promoted to semantic memory, and what triggers that promotion?

**2.** A user says "the assistant used to remember I prefer metric units, now it doesn't." Walk through the possible causes across the write path, the storage, and the retrieval/injection path, and how you'd narrow it down.

**3.** Argue for or against giving users visibility into what an agent has stored about them as long-term memory. What are the product and trust tradeoffs either way?
