# Day 21: Guardrails

## Concept

Guardrails are checks that constrain what an agent is allowed to do or say, enforced outside the model's own judgment — because the model's judgment is exactly what's unreliable in the cases guardrails exist for. Three layers, applied at different points in the pipeline:

- **Input guardrails.** Check the incoming request before it reaches the model: prompt injection detection (is the user trying to override system instructions), PII detection (should this even be processed, or redacted first), obviously out-of-scope requests (a support agent being asked to write malware).
- **Action guardrails.** Check a proposed tool call before executing it: is this action within the agent's allowed scope (a read-only agent should never be able to call a `delete_*` tool even if it tries), does this action exceed a safety threshold (a refund agent capped at approving refunds under $50 without escalation), is this a destructive/irreversible action that needs confirmation first.
- **Output guardrails.** Check the generated response before it's shown to the user: does it leak something it shouldn't (another user's data pulled in via a tool call), does it violate a content policy, is it consistent with what the guardrail-checked actions actually did (the answer claims a refund was processed, but the tool call for it never actually ran).

```python
def check_action_guardrail(tool_call: dict, allowed_tools: set[str], limits: dict) -> tuple[bool, str]:
    if tool_call["name"] not in allowed_tools:
        return False, f"Tool '{tool_call['name']}' is not in the allowed set for this agent"
    if tool_call["name"] == "issue_refund" and tool_call["arguments"].get("amount", 0) > limits.get("max_refund", 0):
        return False, "Refund amount exceeds auto-approval limit — requires human escalation"
    return True, ""
```

The design principle underneath all three layers: **guardrails are deterministic code, not another LLM call asked to "please be careful."** Prompting a model to "never approve refunds over $50" is a soft constraint the model can still violate under adversarial or unusual input. A hard-coded check that literally cannot execute the tool call if the amount exceeds the limit is a real constraint — it doesn't rely on the model complying, it makes non-compliance impossible at the execution layer.

## Coding Problem

Write `check_action_guardrail(tool_call, allowed_tools, limits)` matching the pattern above, extended with a check for **irreversible actions**: if `tool_call["name"]` is in a fixed set `IRREVERSIBLE_ACTIONS = {"delete_account", "send_payment", "cancel_subscription"}`, return `(False, "requires human confirmation")` unless `tool_call["arguments"].get("human_confirmed") is True` — i.e. irreversible actions are blocked by default and only proceed with an explicit confirmation flag already set.

## Quiz

### Question 1: Where Guardrails Apply

**A refund agent is asked to approve a $500 refund, but its policy caps auto-approval at $50. Which guardrail layer should catch this?**

A) Input guardrail, before the request reaches the model
B) Action guardrail, checking the proposed tool call before executing it
C) Output guardrail, after the response is already generated
D) No guardrail is needed — the model will refuse on its own

**Answer**: B

**Explanation**: This is a constraint on what action is allowed to execute, not on the input request's content or the final text output — it belongs at the action-guardrail layer, checked against the proposed tool call's arguments before that call runs.

### Question 2: Why Guardrails Must Be Deterministic

**Why is "prompt the model to never approve refunds over $50" not sufficient as the only safeguard?**

A) Prompts cannot contain numbers
B) A prompt instruction is a soft constraint the model can still violate under unusual, adversarial, or edge-case input — it relies on the model complying rather than making non-compliance impossible
C) Refund logic cannot be expressed in a prompt at all
D) There's no difference between a prompt instruction and a hard-coded check

**Answer**: B

**Explanation**: A prompted rule shapes behavior probabilistically, the same limitation as prompted structured output (Day 10). A deterministic code check that blocks execution outright doesn't depend on the model choosing to comply — it removes the possibility of violating the rule at the execution layer.

### Question 3: Irreversible Actions

**Why should irreversible actions (delete account, send payment) be blocked by default rather than allowed like any other tool call?**

A) Irreversible actions should never be automatable under any circumstances
B) Because a mistake in an irreversible action can't be undone, requiring an explicit confirmation step (human-in-the-loop) before executing reduces the blast radius of an agent error to something recoverable
C) This distinction doesn't matter — all tool calls carry equal risk
D) Irreversible actions are technically incapable of being called by an LLM agent

**Answer**: B

**Explanation**: The asymmetry between a reversible mistake (can be corrected) and an irreversible one (can't) justifies treating irreversible actions differently — requiring explicit confirmation adds a deliberate checkpoint precisely where an agent error would otherwise be unrecoverable.

## Interview Practice

**1.** Design the guardrail layer for an agent that can modify a production database. Walk through input, action, and output guardrails specifically, and identify which layer would catch a prompt-injection attempt versus a destructive query.

**2.** A teammate argues "we told the model in the system prompt never to do X, that should be enough." Explain concretely why this isn't sufficient, and what you'd add.

**3.** Describe a guardrail that's too strict — one that would block legitimate actions often enough to make the agent frustrating to use. How would you tell the difference between "appropriately cautious" and "overly restrictive" in practice?
