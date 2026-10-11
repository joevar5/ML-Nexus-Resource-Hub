# Day 4: State and Context

> **The model only knows what's in the prompt right now. Nothing else.**

Almost everything in this lesson follows from that one sentence. An agent that runs for many steps has to carry its own memory around, decide what's worth keeping, and fetch back what it has forgotten. This lesson walks through how to do that well.

## What you'll be able to do

- Explain why state management is essential for multi-step agent tasks.
- Implement context accumulation that preserves relevant information across loop iterations.
- Recognise the limits of fixed-context-window approaches.
- Compare raw context accumulation with structured state representations.
- Choose between short-term, long-term and retrieved memory, and keep long-term memory clean.

## Concept

### 1. The model remembers nothing

Picture an agent halfway through a task. You ask it to do step 5, and it has no idea what happened in steps 1 to 4. That's not a bug; it's how language models work.

An agent's **state** is everything it knows at a given step: the original goal, the tool-call history, results so far, and any outside state it's tracking (a file it's editing, a ticket it's updating). The model itself is **stateless**: each call starts from a blank slate.

So the agent has to re-write all of its state into the prompt, every single time. If it isn't in the prompt, the model can't know it. The downside is obvious: every step adds more, so the prompt only ever grows.

### 2. The window has a ceiling

Left alone, the prompt grows until it no longer fits, and it gets slow and expensive long before that. State size is capped by the context window, and cost and speed scale with how much you send. "Just use a bigger window" doesn't solve it. A larger window means:

- **Attention dilution** — more noise for the model to sift through.
- **Higher latency** — more tokens to process.
- **Higher cost** — you pay per token.

A smart 4K context usually beats a noisy 128K one. The goal is the **minimal sufficient context**: the least text that still lets the agent decide correctly.

The practical answer is to treat the window as a scarce resource and decide deliberately what stays, what gets compressed, and what gets dropped. Dropping is where it gets risky, because you might drop the one thing that mattered. That's what the next section is about.

### 3. Choose how to represent state

There are four common ways to decide what the prompt carries. None of them is best everywhere; each is a trade-off.

- **Full transcript** — append everything, word for word.
  - *Good for:* short tasks that finish well inside the window; nothing is ever lost.
  - *Breaks when:* the task runs long. Cost and latency grow every step, and you eventually hit the context limit.
- **Sliding window** — the simplest bounded version: pin the system prompt so it's never evicted, then keep only as many of the most recent messages as fit in the token budget, dropping the oldest ones once it's full. No compression, no memory of what got dropped — just a hard, predictable cutoff.
  - *Good for:* chat-style tasks where only the last few turns matter; cheap, no extra model calls.
  - *Breaks when:* something important is old. The original goal or an early constraint scrolls off and the agent forgets it. Fix by pinning the goal alongside the system prompt.
- **Structured state object** — keep a separate data structure (`{files_edited: [...], tests_passing: bool, remaining_todos: [...]}`) and only render a short summary of it into the prompt, not the raw history. More work to build, but scales to long tasks.
  - *Good for:* tasks with a small, known set of fields that matter, like a coding agent tracking files and tests.
  - *Breaks when:* something important doesn't fit the schema, because anything you didn't define a field for is lost.
- **Rolling summary + recent window** — summarize everything older than the last K steps into one short paragraph, keep the last K steps word for word. Unlike a sliding window, nothing is silently lost — it's compressed instead of dropped.
  - *Good for:* open-ended, long-running tasks (research, support threads) where you can't predict what will matter.
  - *Breaks when:* the summary drops or distorts a detail the agent needs later. It also costs an extra model call each time you re-summarize.

In practice these combine: a structured object for the facts you know matter, plus a rolling summary and recent window for everything else.

The sliding window and the rolling summary are the two you'll build most often, and both are easy to get subtly wrong, so the next two sections walk through working code.

### 4. Code: sliding window

The simplest working version: system messages are always kept, everything else is scanned newest-first and kept only while there's still budget for it.

```python
class SlidingMemory:
    def __init__(self, max_tokens: int, count_tokens):
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens
        self.messages = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def render(self) -> list[dict]:
        system = [m for m in self.messages if m["role"] == "system"]
        others = [m for m in self.messages if m["role"] != "system"]

        budget = self.max_tokens
        for m in system:
            budget -= self.count_tokens(m["content"])

        kept = []
        for msg in reversed(others):  # newest first
            tokens = self.count_tokens(msg["content"])
            if budget - tokens >= 0:
                budget -= tokens
                kept.append(msg)
            # else: silently dropped — this is the tradeoff vs. rolling summary

        return system + list(reversed(kept))
```

This is bounded and predictable — output is always `<= max_tokens` — but anything that falls off the back is just gone, no trace. That's the trade-off against the rolling summary in section 5, which keeps a compressed trace instead of dropping silently.

### 5. Code: rolling summary

A step up from a plain sliding window: instead of just dropping the oldest messages, collapse the oldest half of history into one short summary line and keep the newest half word for word. Repeat until it fits.

```python
def render_context(state: dict, max_tokens: int, token_counter) -> str:
    """Render agent state into a prompt string, trimming with a rolling summary
    if it doesn't fit under max_tokens."""
    history = list(state["history"])

    while True:
        rendered = _render(state, history)
        if token_counter(rendered) <= max_tokens or len(history) <= 1:
            return rendered

        # Collapse the oldest half into a one-line summary, keep the rest verbatim.
        cutoff = len(history) // 2
        summary = f"[{cutoff} earlier steps omitted]"
        history = [summary] + history[cutoff:]


def _render(state: dict, history: list) -> str:
    lines = [f"Goal: {state['goal']}"]
    for entry in history:
        lines.append(entry if isinstance(entry, str) else str(entry))
    return "\n".join(lines)
```

Every loop checks the token count fresh, so it never over-trims — it stops the moment the render fits, and always keeps at least the most recent step verbatim.

Both strategies only manage the conversation itself. Neither one notices when the *outside world* changes, which brings us to stale context.

### 6. Don't trust your own copy (stale context)

Here's a quieter failure: the agent acts on data that was true ten steps ago but isn't anymore. If something outside changes — a file gets edited by someone else, an API's data updates — but the agent's in-context copy isn't refreshed, it keeps reasoning over data that's no longer true. This is **stale context**.

The rule is simple: re-fetch ground truth for anything you're about to act on, and don't trust your own cached copy of it. That protects one action at a time. It doesn't help with a bigger problem: facts the agent will need later that have already fallen out of the window.

### 7. Move durable facts outside the window (short-term vs. long-term memory)

Say the user mentioned their preferred region on turn 3. By turn 80 it's gone from the window, and tomorrow's new conversation never had it at all.

Think about how you work through a busy day. You hold a few things in your head right now: what you're doing, what someone just said, the number you need to type in. That's your working memory, and it clears out fast. Then there's what you *know*: your friend's birthday, where you parked, how you like your coffee. That sticks around for years. Agents need the same split, for the same reason: you can't carry everything in your head all the time.

**Short-term memory** is everything inside the context window: the sliding window, the rolling summary, the scratchpad from [Day 2](02-the-agent-loop.md#5-designing-the-scratchpad). It's what the agent is "thinking about" right now. It's fast and always visible to the model, but it's small, and it vanishes the moment the conversation ends. It exists to help finish *this* task.

> *Example:* the agent is booking a flight. "The user wants Tuesday morning, and I've already checked two airlines" is short-term. Once the booking is done, nobody needs it again.

**Long-term memory** is stored *outside* the context window, in an external index or database. It holds durable facts that should survive across sessions: a user's preferences, credentials, project config. The model can't see it by default. It only shows up in the prompt when the agent goes and fetches it (section 8). It's slower to reach and needs upkeep (section 10), but it never falls off the end of a window.

> *Example:* "this user always wants an aisle seat" is long-term. Tomorrow, in a brand-new conversation, the agent should still know it.

**Episodic memory** is a *subtype* of long-term memory, not a separate storage system, and interviewers like asking about it. Instead of stable facts, it stores *things that happened*: "last Tuesday the deploy failed because the config file was missing." Each entry is tied to what happened and when, so the agent can learn from past experience: "this exact error showed up before, and here's what fixed it." What makes it "episodic" is the *kind of content*, not where it's kept: an episode can sit in a relational database, a document store, or a vector database, the same as any other long-term fact.

| | Short-term | Long-term: facts | Long-term: episodic |
|---|---|---|---|
| Holds | Current task state | Stable facts | Past events and outcomes |
| Lives in | The context window | Any external store (relational DB, document store, vector DB) | Same options as facts |
| Survives the session? | No | Yes | Yes |
| Typical example | Last few steps, scratchpad | "Prefers aisle seats" | "Deploy failed on Tuesday, fixed by X" |

**How to decide where something goes.** Ask one question: *would I be annoyed if the agent forgot this tomorrow?* If yes, it's long-term (or episodic, if it's an event). If it only matters until the task is done, it's short-term. It's about **lifespan, not importance**. A critical detail for the current task can still be short-term, and a minor preference can still be worth keeping forever.

Of course, a fact stored outside the window is only useful if the agent can get the *right* one back in when it needs it.

### 8. Get the right fact back (query-based retrieval)

Every strategy so far, the sliding window and the rolling summary, decides what to keep by **recency**: newest in, oldest out. That works for "what just happened," but it breaks the moment the most relevant fact isn't the most recent one. Go back to the region the user mentioned on turn 3. By turn 80 a recency-based window dropped it long ago, even though it's exactly what the agent needs to pick the right server.

It's like a desk that only holds the last ten papers you touched. Anything older goes in the bin, however important. **Query-based retrieval** swaps the desk for a filing cabinet with a search function: nothing is thrown away, and you pull out the right folder when you need it.

**How it works, step by step** (this is the semantic-search version, the most common one; see the four approaches below for the others):

1. **Store.** As facts come up ("deploys to eu-west-1", "prefers Python"), save each one in an external store, usually as an embedding (a vector that captures its meaning) plus the original text.
2. **Query.** When the agent hits a step that needs context, it turns the current question into an embedding too: "which region should I deploy to?"
3. **Match.** The store returns the top few facts closest in *meaning* to the query, so "deploys to eu-west-1" matches even though the words differ. This is semantic similarity.
4. **Inject.** Only those few facts go into the prompt, in the "retrieved" tier from section 9. Everything else stays in the store, costing no tokens.

**Why this beats recency:** a fact's relevance has nothing to do with when it was said. Retrieval lets turn 3 compete with turn 79 on merit, and only what's relevant spends your token budget.

**Retrieval isn't limited to vector databases.** Query-based retrieval is the overall strategy; semantic search is just one way to implement it. There are four common approaches:

1. **Keyword retrieval.** Search for exact words or phrases. *Example:* search for `eu-west-1` in incident records. Best when exact names, error codes, IDs or configuration keys matter.
2. **Semantic retrieval.** Search by meaning using embeddings and vector similarity. *Example:* "live application location" retrieves "production deployment region." Best when the query and the stored text use different wording.
3. **Metadata filtering.** Filter by structured attributes such as user ID, project, environment or timestamp. *Example:* search only incidents where `project=payments` and `environment=production`. Best for narrowing the search to the right entity or scope.
4. **Hybrid retrieval.** Combine keyword search, semantic similarity and metadata constraints. *Example:* find records similar to `ImagePullBackOff`, but only for the payments service in production. Best for many real-world agent systems where both exact matches and semantic meaning matter.

For production systems, hybrid retrieval is often a strong starting point when queries mix natural language with technical identifiers. The best choice depends on your data, query patterns and evaluation results.

**What can go wrong:**

- **Wrong matches.** Similarity isn't truth. A fact that sounds related but isn't can get pulled in and mislead the model, so keep the top-k small.
- **Missed matches.** If the query is vague or worded very differently from how the fact was stored, the right fact never surfaces.
- **Polluted store.** Retrieval is only as good as what's stored. Stale or duplicate facts get retrieved just as readily as good ones, which is why section 10 matters.

Retrieval doesn't replace the window. The window still holds the last few turns, and retrieval handles "what do we know that's relevant to this step, from anywhere in the past."

With retrieval in place, the prompt now has several sources competing for space: the system prompt, pinned facts, history, a summary and retrieved facts. If they all share one budget, whichever is biggest wins.

### 9. Give each source its own budget (tiered token budgeting)

A single shared budget — the approach in sections 4 and 5 — trims oldest-first (or compresses oldest-first) across everything at once. A burst of retrieved facts, or a long history, can crowd out something that should never be dropped.

The fix is to split the budget into **tiers**, each with a fixed share, so nothing important gets crowded out.

- **System** — instructions, always kept, never evicted.
- **Pinned** — facts that must survive no matter what (a user preference, a credential), reserved a small fixed slice.
- **Recent** — the last few turns, verbatim. Gets the biggest share, since it's what the model needs most.
- **Summary** — the compressed version of everything older, typically produced by a faster, cheaper model call.
- **Retrieved** — facts pulled on demand via section 8's query-based retrieval, only when actually relevant to the current step.

```python
def render_tiered(ctx: dict, max_tokens: int, count_tokens) -> list[dict]:
    """ctx: {"system": str, "pinned": list[dict], "recent": list[dict],
    "summary": str | None, "retrieved": list[str]}"""
    alloc = {"system": 0.20, "pinned": 0.05, "summary": 0.15, "retrieved": 0.10, "recent": 0.50}
    out = []

    if ctx["system"]:
        out.append({"role": "system", "content": ctx["system"]})

    for fact in ctx["pinned"]:
        if max_tokens * alloc["pinned"] - count_tokens(fact["content"]) >= 0:
            out.append(fact)

    if ctx["summary"]:
        out.append({"role": "system", "content": f"[Summary] {ctx['summary']}"})

    for fact in ctx["retrieved"]:
        if max_tokens * alloc["retrieved"] - count_tokens(fact) >= 0:
            out.append({"role": "system", "content": f"[Retrieved] {fact}"})

    for msg in reversed(ctx["recent"]):
        if max_tokens * alloc["recent"] - count_tokens(msg["content"]) >= 0:
            out.append(msg)

    return out
```

Each tier fills independently within its own slice, so a burst of retrieved facts can never push out the pinned facts or the system prompt — the failure mode a single shared budget is prone to.

**What stays, what shrinks, what goes:**

- **Stays in context** — the system prompt (always), recent turns verbatim, pinned facts, and active tool results until the agent has consumed them.
- **Gets summarized** — old turns, compressed into a single summary message.
- **Gets retrieved** — facts fetched from the external store on demand. Summary and retrieved facts compete for the same token budget, so keep both concise.
- **Eviction order** — when you must cut, drop completed tool results and irrelevant turns first; never evict pinned items or the system prompt.

One weak spot remains: the retrieved tier is only as good as the long-term store behind it, and so far nothing checks what goes into that store.

### 10. Keep long-term memory clean (memory pollution)

Long-term storage (section 7) brings its own failure mode: **memory pollution** — the store fills with facts that are wrong, stale, or duplicated. A one-off remark gets saved as if it were a confirmed preference; an old fact never gets updated when reality changes; the same fact gets written twice under slightly different wording. Each of these quietly degrades every future retrieval.

Two guards keep a long-term store trustworthy:

- **A write policy** — only accept writes from trusted sources (an explicit user confirmation, a verified tool result), and check for an existing near-duplicate before writing, refreshing it instead of creating a second copy. Reject unverified assumptions and anything that's only relevant to the current task — that belongs in short-term state, not permanent storage.
- **TTL (time-to-live) eviction** — give every fact an expiry. A periodic sweep removes anything past it, so the store can't silently accumulate facts nobody's checked in months.

```python
import time

class MemoryStore:
    TRUSTED_SOURCES = {"user_confirmed", "tool_verified"}

    def __init__(self, default_ttl_seconds: int = 3600):
        self.facts: dict[str, dict] = {}
        self.default_ttl = default_ttl_seconds

    def add_fact(self, key: str, value: str, source: str, ttl: int = None) -> str:
        if source not in self.TRUSTED_SOURCES:
            return "rejected: untrusted source"

        key = key.strip().lower()
        expires_at = time.time() + (ttl or self.default_ttl)

        existing = self.facts.get(key)
        if existing and existing["value"].strip().lower() == value.strip().lower():
            existing["expires_at"] = expires_at  # same fact seen again: refresh, don't duplicate
            return "refreshed"

        self.facts[key] = {"value": value, "expires_at": expires_at}
        return "stored"

    def get_fact(self, key: str) -> str | None:
        entry = self.facts.get(key.strip().lower())
        if not entry or time.time() > entry["expires_at"]:
            self.facts.pop(key.strip().lower(), None)
            return None
        return entry["value"]

    def clean_expired(self) -> int:
        now = time.time()
        expired = [k for k, v in self.facts.items() if now > v["expires_at"]]
        for k in expired:
            del self.facts[k]
        return len(expired)
```

Same principle as section 6's stale-context rule, just pointed at storage instead of a single call: don't trust a fact forever just because it was true once. Gate what goes in, and let what's overdue expire on its own.

### 11. Putting it together

**Decision guide.** Start from your task and add layers only when the previous layer fails.

| If your task... | Use | Because |
|---|---|---|
| Is short and fits easily | Full transcript | Nothing to lose, nothing to build |
| Is chat-like, recent turns matter most | Sliding window, with the goal pinned | Cheap and predictable |
| Has a few well-defined fields that matter | Structured state object | Exact, compact, scales to long tasks |
| Is long and open-ended | Rolling summary + recent window | Compresses instead of dropping |
| Needs facts from long ago or from past sessions | Long-term store + query-based retrieval | Relevance beats recency |
| Mixes several of the above | Tiered token budget | Each source gets a guaranteed slice |
| Writes to a long-term store | Write policy + TTL eviction | Keeps the store trustworthy |
| Acts on data that can change | Re-fetch before acting | Avoids stale context |

**Key ideas:** sliding window · system pinning · summarization layer · tiered budgeting · query-based retrieval · TTL eviction · memory write policy (classify ephemeral vs. persistent *before* storing; persistent facts need a source of truth and an expiry timestamp).

**Practice:** *Sliding Window Conversation Memory* (Medium) — keep the last N turns plus a pinned system prompt within a token budget; it's section 4's code. Follow-ups in the same Memory track: summarize dropped context, merge conflicting memory facts, retrieve memories by query, preserve mission-critical memory under pruning, and expire stale memories without losing active goals.

*Further reading: [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) (Anthropic) — the note-taking/scratchpad pattern from [Day 2](02-the-agent-loop.md#5-designing-the-scratchpad), and managing context as a finite resource.*

## Interview Practice

**1.** Why does an agent need to re-send its state on every single LLM call, instead of the model just "remembering" the last call?

<details>
<summary>Answer</summary>

LLM calls are stateless by default — the model has no memory of a previous call unless that information is physically present in the current prompt. So the goal, the history, and any intermediate results all have to be re-serialized into every single call, or the model has no way to know about them.
</details>

**2.** A junior engineer says "just make the context window bigger and we never have to worry about state management." What problems does that not solve?

<details>
<summary>Answer</summary>

Three problems remain even with a huge context window: cost and latency still scale with how much you send, since every token costs money and time regardless of the limit. Stale context is still possible — a bigger window doesn't refresh data that changed outside the conversation. And model attention still degrades over a very long, noisy context, even if it technically fits — more tokens isn't the same as more useful signal.
</details>

**3.** Why might tiered token budgeting beat a single shared budget (sliding window or rolling summary) for an agent that also uses retrieval?

<details>
<summary>Answer</summary>

With one shared budget, a burst of retrieved facts can crowd out whatever's left — including things that should never be dropped, like a pinned user preference or the system prompt. Splitting the budget into fixed tiers (system, pinned, recent, summary, retrieved) means each type has its own guaranteed slice, so retrieval volume in one tier can never starve another tier of its space.
</details>

**4.** What is memory pollution, and what two mechanisms prevent it?

<details>
<summary>Answer</summary>

Memory pollution is when a long-term memory store fills up with facts that are wrong, stale, or duplicated — for example, treating a single offhand remark as a confirmed preference, or never updating a fact after it changes. Two guards fix it: a write policy that only accepts facts from trusted sources and checks for near-duplicates before storing (refresh instead of re-write), and TTL eviction, where every fact expires and gets swept out automatically instead of sitting there indefinitely.
</details>

**5.** Describe the FIFO sliding-window approach for agent context management. What are its strengths, its critical weakness, and what mitigation strategies address that weakness? Compare with at least one alternative strategy.

<details>
<summary>Answer</summary>

Keep the pinned system prompt plus the newest messages that fit the token budget; evict the oldest first. Strengths: simple, bounded, predictable cost and latency, no extra model calls. Critical weakness: eviction is blind to importance, so early content — like the original goal or a user constraint — silently disappears once the window fills. Mitigations: pin the goal and key facts so they're never evicted, restate the goal each step, or replace dropped turns with a summary. The alternative is a rolling summary (or structured state object), which compresses instead of dropping, at the cost of an extra model call and some detail loss.
</details>

**6.** An agent starts with the goal "Find and compare prices of three laptops." After 8 steps, it suddenly asks "Which laptop do you want me to find?" The trace shows early steps mentioning the goal are gone from the context. Diagnose the issue and propose two solutions.

<details>
<summary>Answer</summary>

Diagnosis: a plain FIFO sliding window evicted the early messages, including the one that stated the goal, so the model no longer knows its task. Solutions: (1) pin the goal alongside the system prompt so it's never evicted, or re-render it at the top of every prompt; (2) keep a structured state object (`goal`, `laptops_found`, `remaining`) or a rolling summary that always carries the goal forward.
</details>

**7.** Compare sliding-window truncation, summarisation-based compression, and structured state objects for context management. For each, describe a best-case scenario and a failure scenario. Design an adaptive context manager that switches strategies.

<details>
<summary>Answer</summary>

Sliding window — best: short chat-style tasks where only recent turns matter; fails when the goal or an early constraint scrolls off. Summarisation — best: open-ended, long tasks with no fixed schema (research); fails when the summary drops or distorts a detail the agent later needs. Structured state — best: tasks with well-defined fields (coding agent tracking files and tests); fails when something important doesn't fit the schema.

Adaptive manager: start with the full transcript while it fits comfortably; once usage passes a threshold (say 60–70% of the budget), switch to rolling summary + recent window; if the task has a known schema, track it in a structured object and render that instead. Always pin the system prompt and goal, and fall back to a more aggressive strategy only when token usage requires it.
</details>

**8.** What is the difference between short-term, long-term, and episodic memory?

<details>
<summary>Answer</summary>

Short-term memory is the current task's context — the window, scratchpad, intermediate state — and resets between conversations. Long-term memory holds durable facts (preferences, credentials, project config) that survive across sessions. Episodic memory stores records of specific past events and outcomes, so the agent can recall what happened and when, not just stable facts.
</details>
