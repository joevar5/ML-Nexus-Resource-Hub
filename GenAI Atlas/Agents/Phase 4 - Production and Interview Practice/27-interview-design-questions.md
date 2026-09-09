# Day 27: Interview Design Questions

## Concept

Agent system-design interview questions are usually open-ended prompts like "design an AI agent that handles customer support tickets" — there's no single correct answer, but there is a checklist of concerns a strong answer covers and a weak answer skips. Interviewers are grading whether you reach for the right concerns unprompted, not whether you recite a specific architecture.

A structure that reliably covers what's expected, in order:

1. **Clarify scope first.** What actions can the agent actually take (read-only vs. can it modify state)? What's the failure cost if it's wrong (a wrong FAQ answer vs. a wrong refund)? This changes the entire design — don't skip straight to architecture.
2. **State the loop and tools.** What's the agent loop shape (Day 2), and what's the minimal tool set (Day 6-7)? Keep it to what the task actually needs — listing 20 speculative tools reads as not having thought about scope.
3. **Address grounding.** If the task involves answering questions, where does the agent's knowledge come from — RAG (Day 13), and how is it kept accurate (citations, Day 15)?
4. **Address safety explicitly.** Guardrails (Day 21) and human-in-the-loop (Day 22) for the specific actions this agent can take — not generically, tied to the actual risk identified in step 1.
5. **Address failure and observability.** How do you know it's working (evals, Day 18) and how do you debug it when it's not (tracing, Day 23-24)?
6. **Address cost/scale only if asked, or briefly if time allows.** This is usually a secondary concern relative to correctness and safety unless the interviewer specifically probes it.

The single most common weak answer: jumping straight to "I'd use LangGraph with a ReAct loop and a vector database" without first establishing what the agent is actually allowed to do and what happens when it's wrong. Tool/framework choice is the least interesting part of the answer — the risk model and the failure-handling plan are what distinguish a strong answer.

## Practice Prompt

Design an agent that triages incoming support tickets: reads the ticket, decides which team it belongs to (billing, technical, account), and either routes it automatically or drafts a response for human review.

Work through the 6 steps above out loud (or in writing) before looking at anything else in this course. Specifically answer: what's the failure cost of a wrong routing decision vs. a wrong drafted response, and does that change whether each action needs human-in-the-loop?

## Quiz

### Question 1: Where to Start

**Why does a strong answer to "design a customer support agent" start with clarifying scope (what actions it can take, cost of being wrong) before discussing architecture?**

A) Scope is irrelevant to the design and can be assumed
B) The failure cost and allowed actions directly determine the right architecture — a read-only FAQ agent and an agent that can issue refunds need very different guardrail and HITL designs, so starting with architecture without this context risks designing for the wrong risk profile
C) Interviewers only care about framework names
D) Scope should only be discussed at the very end, after the full architecture is described

**Answer**: B

**Explanation**: The appropriate level of guardrails, human oversight, and error tolerance depends entirely on what the agent can actually do and what a mistake costs. Establishing this first ensures the rest of the design is calibrated to the actual risk, not a generic template.

### Question 2: The Weak Answer Pattern

**What does jumping straight to "I'd use LangGraph with a ReAct loop and a vector database" signal as a weak answer, per this lesson?**

A) It's always the correct first thing to say
B) It skips the risk model and failure-handling plan — the parts that actually differentiate designs — in favor of naming a framework, which is the least interesting part of the answer
C) LangGraph is never an appropriate tool to mention
D) Vector databases should never be mentioned in a design interview

**Answer**: B

**Explanation**: Framework and tool choice are implementation details that don't demonstrate design judgment. What interviewers are evaluating is whether you identify the actual risks (safety, correctness, failure handling) for the specific task — naming a framework first sidesteps that entirely.

### Question 3: Cost/Scale Priority

**Per this lesson's suggested structure, when should cost and scale be addressed in an agent design answer?**

A) First, before anything else
B) Only if specifically asked, or briefly if time allows — it's typically secondary to correctness and safety concerns unless the interviewer probes it directly
C) Cost and scale should never be mentioned in a design interview
D) Cost is the single most important factor in every agent design

**Answer**: B

**Explanation**: The structure prioritizes scope, loop/tools, grounding, safety, and observability as the core of a strong answer — cost and scale matter, but are usually secondary unless the interviewer specifically steers the conversation there, and leading with them can crowd out the more important risk-focused discussion.

## Interview Practice

**1.** Design an agent that manages calendar scheduling on a user's behalf, including declining or rescheduling meetings autonomously. Apply the 6-step structure from this lesson out loud, spending real time on step 1 (scope) before moving on.

**2.** An interviewer interrupts your design answer and asks "what's the one thing most likely to go wrong with this system in production?" Practice answering this cold, for the design you just built in Question 1.

**3.** Compare how your answer would change if the same scheduling agent were read-only (it can only suggest changes, never execute them) versus fully autonomous. Which parts of your design from Question 1 would you keep, and which would you drop?
