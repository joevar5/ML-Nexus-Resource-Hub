# Day 29: Capstone Project

## Brief

Build one agent that ties together every phase of this path: tool use (Phase 2), RAG + memory + evals (Phase 3), and production concerns (Phase 4). This is the largest project in the path — budget more than one sitting.

**Goal:** an agent that helps a user manage a personal reading list — add articles by URL, answer questions about saved articles using their content, and recommend what to read next based on stated preferences remembered across sessions.

## Requirements

**Tools (Phase 2)**
- `add_article(url) -> {id, title, status}` — stub fetching; store canned content keyed by a few test URLs.
- `list_articles(filter) -> list[{id, title, status}]`
- `search_saved_articles(query) -> list[{id, snippet}]` — retrieval over saved content (Day 13-14).
- All tool calls validated against schema before execution (Day 7); all wrapped with structured error handling, not raised exceptions (Day 8).

**RAG + Memory (Phase 3)**
- Questions about saved articles ("what did that piece on RAG chunking say about overlap?") must be answered with citations back to the specific saved article (Day 15), and must say "I don't have that saved" rather than guess if nothing matches.
- Maintain a small preference memory (e.g. `{"prefers_topics": ["agents", "infra"], "avoid": ["crypto"]}`), updated when the user states a preference, and used — relevance-filtered, not dumped whole (Day 17) — when producing "what to read next" recommendations.
- An eval suite (Day 18-19) with at least 6 cases covering: correct retrieval + citation, correct "not found" behavior, a memory-dependent recommendation, an irrelevant-memory-not-injected case, and at least one rubric-scored case for recommendation quality.

**Production (Phase 4)**
- A guardrail (Day 21) preventing `add_article` from being called with a non-URL string, and any destructive action (e.g. a hypothetical `delete_article`) requires confirmation (Day 22) even though this is a low-stakes domain — implement it anyway, as practice.
- Full tracing (Day 23) of every LLM and tool call, with a `to_summary()` report (calls, latency, errors) per run.
- A short written debugging exercise: deliberately break one thing (e.g. make `search_saved_articles` return stale/wrong results for one test case) and use only your trace log to localize and diagnose it (Day 24), writing up what you found the way Day 28's analysis was written.

## Self-Check

- [ ] All eval suite cases pass, and you can explain what each one is actually testing for.
- [ ] A citation-verification check (Day 15) catches a manually-constructed bad citation in a test case.
- [ ] Preference memory correctly changes the "what to read next" answer between two simulated users with different stated preferences.
- [ ] The guardrail blocks a malformed `add_article` call before it reaches the tool, not after.
- [ ] The deliberate-bug debugging writeup identifies the actual injected fault using only the trace, without looking at the source code first.

This capstone doesn't require any real external APIs — every requirement above is testable with stub data. The point is whether the agent's control flow, grounding, memory scoping, and safety checks are all correct and demonstrably tested, not whether it's connected to the real internet.

## Interview Practice

**1.** Present your capstone as if walking an interviewer through a system you built at work. Cover scope, architecture, safety, and evals in under 5 minutes — practice the compressed version, not the full build writeup.

**2.** An interviewer asks "what would you do differently if you had another week?" Answer specifically for this project, not generically — name the actual weakest part of what you built.

**3.** Your deliberate-bug debugging exercise (from the requirements) found the injected fault. Explain how you found it using only the trace, and what that process tells you about a gap (if any) in your original tracing design.
