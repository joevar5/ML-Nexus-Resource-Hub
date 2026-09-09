# Day 26: Agent Architecture

## Concept

"Should this be one agent or multiple?" is the central architecture question once a system grows past a simple loop. Two dominant patterns:

**Single agent, many tools.** One LLM, one loop, a large registry of tools covering the whole task domain. Simpler to reason about and debug (one trace, Day 23) — but as the tool count grows, tool selection accuracy degrades (more overlapping/confusable options, Day 7), and the system prompt needed to describe all of them grows large enough to dilute attention on any single instruction.

**Multi-agent, specialized roles.** Multiple agents, each with a narrow tool set and focused system prompt, coordinated by either a fixed pipeline or a "orchestrator" agent that routes subtasks to the right specialist. Examples: a research agent that only searches/reads, handing findings to a writing agent that only drafts, coordinated by a top-level agent that decides the sequence. This scales tool-selection accuracy (each agent only picks from a small, coherent set) at the cost of coordination complexity — now you have to get hand-offs between agents right, which is its own failure surface (Day 5's "compounding errors," but between agents instead of within one).

```
Single agent:            Multi-agent (orchestrated):
                              orchestrator
   [agent + 20 tools]      /       |        \
                     research   writer    reviewer
                    (3 tools) (2 tools)  (2 tools)
```

The decision rule that holds up in practice: **split when tool sets genuinely don't overlap in purpose and the task naturally decomposes into distinct roles a human team would also separate** (research vs. writing vs. review). Don't split just because there are "a lot of tools" — a single agent with 15 tools that are all variations on "query different data sources" is still one coherent role and splitting it adds coordination overhead without a corresponding accuracy gain. Splitting is a response to role incoherence, not tool count alone.

A related, narrower pattern worth knowing: **sub-agents as tools.** An orchestrator agent calls a specialist agent the same way it calls any other tool — the specialist runs its own internal loop and returns a final result, invisible to the orchestrator's own context. This bounds each specialist's context to its own subtask rather than accumulating the entire multi-agent conversation in one place, which is often the real motivation for the architecture, not just "division of labor."

## Coding Problem

Write `should_split_agent(tool_descriptions: list[str], similarity_fn) -> bool` where `similarity_fn(desc_a, desc_b) -> float` scores how related two tool purposes are (mocked). Return `True` (recommend splitting into specialists) if the tools cluster into 2+ groups with low average cross-group similarity but high average within-group similarity — implement a simple version: compute all pairwise similarities, and return `True` if the minimum pairwise similarity is below a threshold `0.3` while at least one pair scores above `0.7` (evidence of genuinely disjoint sub-groups, not just noisy uniform similarity).

## Quiz

### Question 1: When to Split

**A single agent has 15 tools, all of which query different internal data sources but serve the same underlying purpose (answering data questions). Should this be split into multiple specialized agents?**

A) Yes, always split once tool count exceeds 10
B) Not necessarily — splitting is justified by role incoherence (genuinely distinct purposes), not tool count alone; 15 tools serving one coherent role don't automatically benefit from splitting and the split would add coordination overhead without a clear accuracy gain
C) No system should ever have more than one agent
D) Splitting is required whenever more than 5 tools exist

**Answer**: B

**Explanation**: The decision rule is about whether tool sets serve genuinely distinct purposes that a human team would also separate, not raw tool count. 15 tools all doing "query a data source" are still one coherent role — splitting adds hand-off complexity without addressing an actual tool-selection confusion problem.

### Question 2: The Cost of Multi-Agent Systems

**What new failure surface does a multi-agent architecture introduce that a single-agent system doesn't have?**

A) Multi-agent systems eliminate all failure modes present in single agents
B) Coordination/hand-off between agents becomes its own source of error — information can be lost, misrepresented, or misinterpreted at the boundary between two agents, similar to compounding errors but occurring between agents rather than within one
C) Multi-agent systems cannot use tools at all
D) There is no additional failure surface — multi-agent systems are strictly safer

**Answer**: B

**Explanation**: Splitting into specialists trades one risk (tool-selection confusion in a large single agent) for another (hand-off errors between agents) — the orchestrator or downstream agent can misinterpret or lose information passed from an upstream specialist, a new failure mode specific to the multi-agent structure.

### Question 3: Sub-Agents as Tools

**What's the main practical benefit of treating a specialist agent as a "tool" called by an orchestrator, versus having all agents share one flat conversation?**

A) It makes the system use fewer LLM calls overall
B) It bounds each specialist's context to its own subtask — the orchestrator only sees the specialist's final result, not its full internal working history, preventing unbounded context accumulation across the whole multi-agent system
C) It removes the need for any tool schemas
D) It has no real benefit over a flat shared conversation

**Answer**: B

**Explanation**: If every agent's full internal reasoning were shared in one conversation, context would grow even faster than in a single agent (Day 25's cost concern, multiplied). Treating specialists as callable tools that return only a final result keeps each agent's context scoped to its own subtask.

## Interview Practice

**1.** You're asked to design a multi-agent system for automated code review (style, security, and test-coverage checks). Walk through whether this should be one agent with three tool categories or three specialist agents, and defend your choice using the decision rule from this lesson.

**2.** A team splits a single-agent system into 5 specialist agents expecting better accuracy, but overall task success rate drops. Walk through the likely causes, focused on the coordination layer specifically.

**3.** Describe a concrete failure you'd expect at the hand-off boundary between a research agent and a writing agent, and the specific check you'd add to catch it before it reaches the user.
