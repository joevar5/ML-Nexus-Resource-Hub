# Day 28: Mixed Hard Problem

## Problem

An agent you built handles internal IT support tickets — it can look up account info, reset passwords, and escalate to a human. Three days after launch, you get this bug report: *"The agent reset a password for the wrong employee twice this week."*

You have access to the full trace log (Day 23) for both incidents. This is a debugging + design problem combining Days 5, 8, 21, 22, and 24 — work through it before checking the analysis below.

**Part A — Diagnosis.** Given only the symptom ("wrong employee, twice"), list at least 3 distinct root causes that would produce this exact symptom, and for each, name what in the trace log would confirm or rule it out.

**Part B — Immediate fix.** Independent of root cause, what's the fastest change you'd make to stop this from happening again *today*, even before full diagnosis is complete?

**Part C — Structural fix.** Once you know the root cause, what changes to the system (not just a patch) prevent this class of bug, not just this instance?

## Analysis

**Part A — plausible root causes:**

1. **Tool argument confusion** — the agent correctly decided "reset a password" but passed the wrong `employee_id` to the tool. Trace check: compare the LLM's stated intent (e.g. "resetting password for employee referenced in the ticket") against the actual `arguments` in the logged tool call — a mismatch here confirms this cause.
2. **Ambiguous identification upstream** — the ticket itself referenced an employee ambiguously (e.g. by first name only, and two employees share it), and retrieval/lookup picked the wrong match. Trace check: look at the lookup tool's call and result — if it returned one match, was there a legitimate ambiguity in the input the agent should have flagged instead of silently picking one?
3. **Stale context (Day 4)** — the correct employee ID was established earlier in a longer ticket thread, but by the reset step, that detail had fallen out of context (truncated history) and the model regenerated a plausible-but-wrong ID instead. Trace check: is the correct ID present verbatim in the exact prompt sent for the reset call, or only earlier in the full conversation?

**Part B — immediate fix:** This is exactly the case Day 21-22 exist for. Password reset is an irreversible-enough, high-cost action (locks out or exposes account access) that it should require explicit human confirmation before executing — regardless of root cause, adding a guardrail that blocks `reset_password` pending human approval stops the symptom today without needing full diagnosis first.

**Part C — structural fix:** Depends on which Part A cause is confirmed — argument validation against a verified employee-ID source (not model-derived) if cause 1, a disambiguation step that surfaces multiple matches to a human rather than auto-selecting if cause 2, or re-injecting the confirmed target ID into every subsequent prompt in the ticket thread if cause 3. The point of Part A's trace analysis is that the structural fix should target the confirmed cause, not all three defensively — a targeted fix is verifiable against the eval suite (Day 18); a shotgun fix isn't.

## Quiz

### Question 1: Diagnosis Before Structural Fix

**Why does Part C's structural fix depend on first confirming the specific root cause from Part A, rather than implementing all three possible fixes defensively?**

A) Implementing all fixes at once is always the safest approach
B) A targeted fix addressing the confirmed cause can be verified against the eval suite and is far more likely to actually solve the observed problem; a shotgun approach adds untested complexity without confirming any of it addresses the real cause
C) Root cause analysis is unnecessary once a symptom is observed
D) There's no meaningful difference between guessing and confirming a cause

**Answer**: B

**Explanation**: Without confirming which cause actually occurred, you risk adding complexity (three separate fixes) where only one was needed, none of which is validated against the actual failure. Diagnosis-first lets you apply and verify a fix that's known to address what actually happened.

### Question 2: Immediate vs. Structural Fix

**Why is adding a human-approval guardrail for password resets the right immediate fix, independent of which root cause (Part A) turns out to be correct?**

A) It only works if the root cause is confirmed first
B) All three plausible root causes result in the same dangerous action (wrong-target password reset) executing autonomously — blocking that specific action pending approval stops the symptom regardless of which upstream cause produced it
C) Guardrails cannot be added until a root cause is known
D) Human approval doesn't actually prevent this kind of error

**Answer**: B

**Explanation**: The immediate fix targets the point where all three root causes converge — the action itself executing without a check. A guardrail there stops the harmful outcome regardless of which specific upstream cause is eventually confirmed, buying time for proper diagnosis.

### Question 3: Trace-Based Diagnosis

**Why is comparing the model's stated intent against the actual logged tool call arguments (Part A, cause 1) a useful diagnostic check?**

A) It's not useful — intent and arguments are always identical
B) It distinguishes a reasoning failure (the model didn't understand what to do) from an execution/argument-passing failure (the model understood correctly but the wrong value ended up in the actual call) — these need different fixes
C) This check only applies to tool calls involving passwords
D) Stated intent is never logged and can't be compared

**Answer**: B

**Explanation**: If the model's reasoning correctly identifies the right employee but the actual tool call carries the wrong ID, the bug is in argument construction, not reasoning — a completely different fix than if the model's reasoning itself was wrong. This is the same localize-the-divergence discipline from Day 24, applied to this specific incident.

## Interview Practice

**1.** Present your Part A/B/C analysis from this lesson to an interviewer as if this were a live incident review. Practice stating the immediate fix before the root cause is even confirmed — explain why that ordering is defensible.

**2.** A senior engineer disagrees with your Part C structural fix, arguing a different root cause is more likely. Defend your conclusion using only what would be visible in the trace log, not speculation.

**3.** This incident involved password resets. Pick a different high-stakes action (e.g. issuing a refund, deleting a record) and redo Parts A-C for it — do the same three root causes apply, or does the new action surface a different one?
