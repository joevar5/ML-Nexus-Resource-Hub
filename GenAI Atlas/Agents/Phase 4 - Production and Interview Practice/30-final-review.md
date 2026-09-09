# Day 30: Final Review

## Concept

No new material today — this is a deliberate consolidation pass. The 30 days covered four layers, and the point of this review is to be able to state, from memory, the one core idea of each layer and how they connect:

- **Foundations (Days 1-5):** an agent is a loop, not a call. The loop's state is everything it knows, re-serialized every step because inference is stateless. Every failure mode traces back to a wrong intermediate state being treated as ground truth with nothing catching it.
- **Tools and execution (Days 6-12):** the model requests, your code executes — that's a trust boundary, not a formality. Schema quality determines call accuracy. Errors are data to feed back into the loop, not exceptions to crash on. Retries need idempotency awareness, not blind repetition.
- **RAG, memory, evals (Days 13-20):** retrieval quality bottlenecks the whole RAG pipeline — the generation step trusts whatever context it's given. Grounding requires citation *and* verification, not just instruction. Memory splits into episodic/semantic/procedural, and injecting it is itself a retrieval problem with its own budget. Evals exist because agent behavior is non-deterministic — without a fixed suite you can't tell improvement from noise.
- **Production (Days 21-29):** guardrails are deterministic code, not a polite request to the model. Human-in-the-loop is scoped to where risk justifies the delay, not applied uniformly. Observability means structured, queryable traces, not print statements. Cost grows faster than step count because context accumulates. Architecture splits are justified by role incoherence, not tool count.

The thread connecting all four layers: **agents fail when something goes unchecked** — an unchecked assumption, an unvalidated tool call, an unverified citation, an unbounded loop, an unlogged step. Every mitigation taught in this path is some form of an explicit check inserted at a point the model itself has no built-in reason to look back and verify.

## Coding Problem

Write `audit_agent_design(design: dict) -> list[str]` that takes a spec like `{"has_termination_condition": bool, "validates_tool_calls": bool, "has_guardrails": bool, "has_tracing": bool, "has_evals": bool}` and returns a list of missing-safeguard warnings (one string per `False` value, naming which safeguard is missing and, briefly, why it matters — e.g. `"Missing termination condition: agent can loop indefinitely without a step/cost ceiling"`). This is a compressed version of the review checklist you should be able to run against any agent design, including your own capstone from Day 29.

## Quiz

### Question 1: The Common Thread

**What single idea connects the failure modes and mitigations across all four phases of this path?**

A) Bigger models fix every failure mode
B) Agents fail when some intermediate state or assumption goes unchecked; nearly every mitigation taught is an explicit validation step inserted where the model has no built-in reason to look back
C) All failures are caused by insufficient tool count
D) There is no unifying pattern — each failure mode is unrelated to the others

**Answer**: B

**Explanation**: From hallucinated tool calls to stale context to unverified citations to unbounded loops, the pattern is consistent: something wasn't checked before being trusted. The mitigations — validation, guardrails, tracing, evals — are all forms of inserting that missing check.

### Question 2: Guardrails vs. Prompting

**Across Days 21-22, why are guardrails implemented as deterministic code rather than as prompt instructions to the model?**

A) Prompt instructions are always ignored entirely
B) A prompt instruction is a soft constraint the model can still violate under unusual input; deterministic code makes non-compliance impossible rather than just less likely
C) Guardrails and prompts are functionally identical
D) Deterministic code cannot be applied to LLM-based systems

**Answer**: B

**Explanation**: This is the same principle from Day 21: a hard-coded check that blocks execution outright doesn't depend on the model choosing to comply. It's a real constraint, not a request.

### Question 3: Evals as the Feedback Loop

**Why do evals (Day 18-19) matter for every other phase of this path, not just RAG?**

A) Evals only apply to retrieval-augmented systems
B) Evals are the only way to tell whether a change to any part of the system (a prompt, a guardrail, a tool schema, an architecture split) actually improved behavior versus just changing it, given that agent output is non-deterministic
C) Evals are optional once an agent passes manual testing once
D) Evals replace the need for tracing and observability

**Answer**: B

**Explanation**: Every mitigation taught across this path is a claim that a change makes the agent better. Evals are what turn that claim into something measurable and repeatable — without them, "better" is just an impression, exactly the gap Day 18 opened with.

## Interview Practice

**1.** You're asked to design an agent from scratch in a 45-minute interview. Walk through, in order, the first five decisions you'd make before writing any code, and justify why that order — not just what the decisions are.

**2.** A hiring manager says "we don't need guardrails or evals for our MVP, we'll add them once we have real users." Argue both sides: when is this a reasonable call, and when is it a mistake that will be expensive to walk back later?

**3.** Pick any two lessons from this path that felt most connected to each other (e.g. tool schemas and hallucinated tool calls, or memory injection and RAG chunking) and explain the mechanism that links them — not just that they're related, but specifically how a weakness in one causes a failure in the other.
