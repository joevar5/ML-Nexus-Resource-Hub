# Day 4: State and Context

## Concept

**The model only knows what's in the prompt right now. Nothing else.**

### 1. What "State" Means Here

An agent's state is everything it knows at a given step: the original goal, the tool-call history, results so far, and any outside state it's tracking (a file it's editing, a ticket it's updating). The model itself remembers nothing between calls — each call is stateless. So all of that state has to be re-written into the prompt, every single time.

### 2. Why State Has a Ceiling

State size is capped by the context window, and cost/speed scale with how much state you send. Send everything, forever, and you'll eventually hit the limit — or just get slow and expensive long before you hit it.

### 3. Four Ways to Represent State

- **Full transcript** — append everything, word for word. Simple, but grows without limit.
- **Sliding window** — the simplest bounded version: pin the system prompt so it's never evicted, then keep only as many of the most recent messages as fit in the token budget, dropping the oldest ones once it's full. No compression, no memory of what got dropped — just a hard, predictable cutoff.
- **Structured state object** — keep a separate data structure (`{files_edited: [...], tests_passing: bool, remaining_todos: [...]}`) and only render a short summary of it into the prompt, not the raw history. More work to build, but scales to long tasks.
- **Rolling summary + recent window** — summarize everything older than the last K steps into one short paragraph, keep the last K steps word for word. Unlike a sliding window, nothing is silently lost — it's compressed instead of dropped.

### 4. Stale Context

A common bug: **stale context**. If something outside changes — a file gets edited by someone else, an API's data updates — but the agent's in-context copy of it isn't refreshed, the agent keeps reasoning over data that's no longer true. Rule: re-fetch ground truth for anything you're about to act on. Don't trust your own cached memory of it.

### 5. Code: Sliding Window

The simplest working version of section 3's sliding window: system messages are always kept, everything else is scanned newest-first and kept only while there's still budget for it.

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

This is bounded and predictable — output is always `<= max_tokens` — but anything that falls off the back is just gone, no trace. That's the tradeoff against section 6's rolling summary, which keeps a compressed trace instead of dropping silently.

### 6. Code: Rolling Summary

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

### 7. Short-Term vs. Long-Term Memory

Everything covered so far — sliding window, rolling summary, the scratchpad from [Day 2](02-the-agent-loop.md#5-designing-the-scratchpad) — is **short-term memory**: it lives entirely inside the context window and resets the moment the conversation ends. It's built for *this* task, right now.

**Long-term memory** is a different tier: durable facts meant to survive across sessions — a user's stated preferences, credentials, project config. It's stored outside the context window, in an external index, and pulled back in only when needed (see section 9).

The rule for choosing which tier something belongs in is about **lifespan**, not importance: state that only matters for finishing the current task belongs in short-term memory. A fact the user would expect the agent to still know tomorrow belongs in long-term storage.

### 8. Beyond Recency: Query-Based Retrieval

Every strategy so far — sliding window, rolling summary — decides what to keep based on **recency**: newest in, oldest out. That works for "what just happened," but it breaks down the moment the most relevant fact isn't the most recent one. If a user mentioned their deployment region on turn 3 and it's now turn 80, a pure recency-based window has already dropped it.

**Query-based retrieval** fixes this by turning memory into something you can search instead of a FIFO queue. Facts get stored as they accumulate; when the agent needs something specific, it queries the store by meaning (semantic similarity), not by "how recently was this said." This is what powers the "retrieved" tier in section 9 — pulling in exactly the relevant fact on demand, regardless of how long ago it entered memory.

### 9. A Sharper Tool: Tiered Token Budgeting

A single shared budget — the approach in sections 5 and 6 — trims oldest-first (or compresses oldest-first) across everything at once. A step up: split the budget into **tiers**, each with a fixed share, so nothing important can get crowded out just because history happened to grow long or retrieval happened to return a lot.

- **System** — instructions, always kept, never evicted.
- **Pinned** — facts that must survive no matter what (a user preference, a credential), reserved a small fixed slice.
- **Recent** — the last few turns, verbatim. Gets the biggest share, since it's what the model needs most.
- **Summary** — the compressed version of everything older.
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

### 10. Long-Term Memory: What to Keep, and for How Long

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

Same principle as section 4's stale-context rule, just pointed at storage instead of a single call: don't trust a fact forever just because it was true once. Gate what goes in, and let what's overdue expire on its own.

*Further reading: [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) (Anthropic) — the note-taking/scratchpad pattern from [Day 2](02-the-agent-loop.md#5-designing-the-scratchpad), and managing context as a finite resource.*

## Interview Practice

**1.** Why does an agent need to re-send its state on every single LLM call, instead of the model just "remembering" the last call?

<details>
<summary>Answer</summary>

LLM calls are stateless by default — the model has no memory of a previous call unless that information is physically present in the current prompt. So the goal, the history, and any intermediate results all have to be re-serialized into every single call, or the model has no way to know about them.
</details>

**2.** You're building an agent for a task that could run for hours. Which state representation would you use, and how would you decide what gets summarized vs. kept word-for-word?

<details>
<summary>Answer</summary>

Rolling summary + recent window. Keep the last K steps verbatim — they're what the model is most likely to need full detail on right now — and collapse everything older into a short one-line summary instead of dropping it. As the task grows, K can shrink or the summary can get denser, but something should always remain from early steps (like the original goal) so nothing important disappears entirely.
</details>

**3.** A junior engineer says "just make the context window bigger and we never have to worry about state management." What problems does that not solve?

<details>
<summary>Answer</summary>

Three problems remain even with a huge context window: cost and latency still scale with how much you send, since every token costs money and time regardless of the limit. Stale context is still possible — a bigger window doesn't refresh data that changed outside the conversation. And model attention still degrades over a very long, noisy context, even if it technically fits — more tokens isn't the same as more useful signal.
</details>

**4.** Compare "structured state object" and "rolling summary + recent window." Give a task where each would clearly be the better choice.

<details>
<summary>Answer</summary>

Structured state object fits a task with a small number of well-defined fields that matter — like a coding agent tracking `files_edited`, `tests_passing`, `remaining_todos`. You always know exactly what state looks like, and rendering it is just formatting those fields.

Rolling summary fits open-ended tasks where you can't predict the shape of what matters in advance — like a research agent following leads wherever they go. There's no fixed schema to hang the state on, so compressing the raw transcript is the more practical option.
</details>

**5.** What is the difference between short-term, long-term, and query-based (retrieved) memory?

<details>
<summary>Answer</summary>

Short-term memory lives inside the context window for the current task only and resets between conversations — the sliding window, the scratchpad, intermediate state. Long-term memory holds durable facts meant to survive across sessions, stored outside the context window in an external index. Query-based retrieval is how long-term memory actually gets used: instead of pulling facts back in by recency, the agent searches the store by meaning and pulls in only what's relevant to the current step.
</details>

**6.** Why might tiered token budgeting beat a single shared budget (sliding window or rolling summary) for an agent that also uses retrieval?

<details>
<summary>Answer</summary>

With one shared budget, a burst of retrieved facts can crowd out whatever's left — including things that should never be dropped, like a pinned user preference or the system prompt. Splitting the budget into fixed tiers (system, pinned, recent, summary, retrieved) means each type has its own guaranteed slice, so retrieval volume in one tier can never starve another tier of its space.
</details>

**7.** What is memory pollution, and what two mechanisms prevent it?

<details>
<summary>Answer</summary>

Memory pollution is when a long-term memory store fills up with facts that are wrong, stale, or duplicated — for example, treating a single offhand remark as a confirmed preference, or never updating a fact after it changes. Two guards fix it: a write policy that only accepts facts from trusted sources and checks for near-duplicates before storing (refresh instead of re-write), and TTL eviction, where every fact expires and gets swept out automatically instead of sitting there indefinitely.
</details>
