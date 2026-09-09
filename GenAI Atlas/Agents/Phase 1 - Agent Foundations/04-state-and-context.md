# Day 4: State and Context

## Concept

An agent's "state" is everything it knows at a given step: the original goal, the tool-call history, intermediate results, and any external state it's tracking (a file it's editing, a ticket it's updating). Almost all of that state has to be re-serialized into the context window on every single LLM call, because the model itself has no memory between calls — each call is stateless.

This creates a hard constraint: **state size is bounded by context window size**, and cost/latency scale with how much of that state you send. Three common state representations, in increasing sophistication:

- **Full transcript.** Append everything verbatim. Simple, but grows unbounded and eventually exceeds the context window or gets prohibitively expensive.
- **Structured state object.** Keep a separate data structure (e.g. `{files_edited: [...], tests_passing: bool, remaining_todos: [...]}`) and render only a compact summary of it into the prompt, rather than the raw history. Requires more engineering but scales to long tasks.
- **Rolling summary + recent window.** Summarize everything older than the last K steps into a short paragraph, keep the last K steps verbatim. Balances fidelity (recent detail preserved) against size.

A subtle bug that shows up constantly in production agents: **stale context**. If external state changes (a file is edited by something else, an API's data updates) but the agent's in-context representation of that state isn't refreshed, the agent reasons over data that's no longer true. Any agent that holds a long-lived session should re-fetch ground truth for anything it's about to act on, not trust its own cached memory of it.

## Coding Problem

Write `render_context(state: dict, max_tokens: int, token_counter) -> str` that renders an agent's state into a prompt string, using the rolling-summary strategy: if the full rendered state exceeds `max_tokens` (measured via `token_counter`, a function you can assume is given), collapse the oldest half of `state["history"]` into a one-line summary string (`f"[{n} earlier steps omitted]"`) and keep the newest half verbatim, repeating the collapse until it fits or only 1 step remains.

## Quiz

### Question 1: Why State Must Be Re-Sent

**Why does an agent need to re-serialize its full relevant state into the prompt on every LLM call?**

A) Because LLMs cache previous calls internally and forget only sometimes
B) Because each LLM call is stateless — the model has no memory of prior calls unless it's in the current prompt
C) Because it's required by the OpenAI API terms of service
D) It doesn't — state persists automatically between calls

**Answer**: B

**Explanation**: LLM inference calls are stateless by default. Anything the model needs to "remember" — goal, history, intermediate results — must be included in that call's prompt, which is exactly why context management is a core agent design problem.

### Question 2: State Representation Tradeoff

**What's the main downside of using "full transcript" as your state representation?**

A) It's too complex to implement
B) It grows unbounded, eventually exceeding the context window or becoming too expensive/slow
C) It never includes enough detail
D) It requires a database

**Answer**: B

**Explanation**: Appending everything verbatim is simple to implement but doesn't scale — token count grows linearly with steps, hitting context limits or cost/latency problems on longer-running tasks.

### Question 3: Stale Context

**An agent edited a file, then another process modified that same file, then the agent's next step still refers to its old in-memory copy of the file's contents. What's this failure called, and what's the fix?**

A) Goal drift; fix by re-stating the goal
B) Stale context; fix by re-fetching ground truth for anything about to be acted on, rather than trusting cached state
C) Tool schema mismatch; fix by validating the tool's JSON schema
D) This isn't a real failure mode

**Answer**: B

**Explanation**: This is stale context — the agent's internal representation of external state has diverged from reality. The fix is to re-fetch current ground truth before acting, not to assume previously-observed state is still accurate.

## Interview Practice

**1.** You're building an agent that runs for hours on a long-form task. Describe the state representation you'd use and how you'd decide what gets summarized versus kept verbatim as the task progresses.

**2.** A junior engineer says "just increase the context window so we never have to worry about state management." Identify three specific problems this doesn't solve, even with an effectively unlimited context window.

**3.** Compare the "structured state object" and "rolling summary + recent window" strategies. Describe a task where each would clearly be the better choice, and explain what property of the task drives that choice.
