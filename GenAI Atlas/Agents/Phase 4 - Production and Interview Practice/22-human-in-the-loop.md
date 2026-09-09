# Day 22: Human-in-the-Loop

## Concept

Human-in-the-loop (HITL) means an agent pauses and waits for explicit human approval before taking a specific action, rather than acting fully autonomously. It's not a fallback for a broken agent — it's a deliberate design choice for actions where the cost of an autonomous mistake outweighs the cost of the delay a human check introduces.

Three common HITL patterns, matched to different risk levels:

- **Approve-before-execute.** The agent proposes an action, execution is blocked until a human clicks approve. Used for irreversible or high-cost actions (Day 21). The agent should present the proposed action clearly enough that a human can actually evaluate it — "approve this tool call: `send_email(...)`" with the full argument payload shown, not just "approve?".
- **Review-after-execute.** The action executes immediately, but is logged for human review, with the ability to reverse it if flagged. Used when the action is reversible and speed matters more than pre-approval — e.g. draft responses that get sent but can be recalled.
- **Escalate-on-uncertainty.** The agent proceeds autonomously for cases it's confident about, and routes to a human specifically when it's uncertain (low retrieval confidence, an out-of-policy request, conflicting signals). This is the pattern that scales — most cases never need a human, only the genuinely ambiguous ones do.

```python
def maybe_require_approval(action: dict, risk_policy: dict) -> str:
    risk = risk_policy.get(action["name"], "low")
    if risk == "high":
        return "blocked_pending_approval"
    elif risk == "medium":
        return "execute_and_log_for_review"
    return "execute"
```

The design tension: too much HITL and the agent provides no efficiency gain over a human doing the task directly (every action waits on a person). Too little and you've removed the safety mechanism guardrails (Day 21) were meant to complement. The escalate-on-uncertainty pattern is usually the right target for a mature system — it requires the agent to have a real confidence signal (not just "always ask"), which is a harder engineering problem than the other two patterns but scales far better.

## Coding Problem

Write `maybe_require_approval(action, risk_policy, confidence_score, confidence_threshold)` extending the pattern above with escalate-on-uncertainty: even if `risk_policy` marks an action as `"low"` risk, return `"blocked_pending_approval"` if `confidence_score < confidence_threshold` (the agent isn't sure enough about this action, regardless of its baseline risk level). Otherwise fall through to the risk-based logic above.

## Quiz

### Question 1: Approve-Before-Execute

**Why should an approve-before-execute prompt show the full action payload (e.g. the actual email content and recipient) rather than just "approve this action?"**

A) It's a legal requirement in all jurisdictions
B) A human can't meaningfully evaluate a decision they can't actually see the details of — an uninformative approval prompt defeats the purpose of having a human check at all
C) Showing details always slows down approval time with no benefit
D) The full payload is only needed for debugging, not approval

**Answer**: B

**Explanation**: The entire value of human-in-the-loop is a genuine check on the proposed action. If the human can't see what they're approving, the "approval" is a rubber stamp that provides none of the safety benefit HITL is meant to add.

### Question 2: Escalate-on-Uncertainty

**Why is "escalate-on-uncertainty" described as the pattern that scales best, compared to always requiring approval?**

A) It requires no engineering effort to implement
B) Most cases proceed autonomously and only genuinely uncertain cases route to a human, so human review time is spent where it's actually needed rather than on every single action
C) It removes the need for any guardrails
D) Escalation is never actually triggered in practice

**Answer**: B

**Explanation**: Requiring approval for every action doesn't scale — it caps throughput at human review speed. Escalating only on genuine uncertainty concentrates limited human attention on the cases that actually need it, letting the confident majority proceed without a bottleneck.

### Question 3: The HITL Design Tension

**What's the risk of applying human-in-the-loop too broadly (requiring approval for nearly everything)?**

A) There is no downside to maximal human oversight
B) The agent provides little to no efficiency gain over a human just doing the task directly, since every action still waits on a person
C) Excessive HITL makes the agent less safe
D) HITL cannot be applied to more than one action type

**Answer**: B

**Explanation**: The entire value proposition of an agent is reducing human effort per task. If every single action requires a human to review and approve, the system has reintroduced the same bottleneck it was meant to remove — the design goal is scoping HITL to where the risk actually justifies the cost.

## Interview Practice

**1.** Design the HITL policy for an agent that processes expense reports. Which actions get approve-before-execute, which get review-after-execute, and which need no human involvement at all? Justify each tier.

**2.** A product lead wants to remove all human approval steps after 3 months of "no incidents" to improve speed. Argue both sides of this decision, including what evidence would actually justify it.

**3.** Describe how you'd build a genuine confidence signal for escalate-on-uncertainty, for an agent that doesn't have an obvious numeric confidence score (e.g. a tool-calling agent, not a classifier). What proxy would you use instead?
