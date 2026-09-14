# Day 3: Goals and Task Breakdown

## Concept

**A goal is not a plan. Something has to turn one into the other.**

### 1. Why Break a Goal Down

Give an agent a vague goal like "improve our onboarding flow" and it will either freeze up (no clear first step) or guess a plan that looks fine but ignores real constraints. Neither looks like an error — it just quietly gives a wrong result.

Fix: split the goal into small, checkable subtasks before doing anything. Small steps are easy to check. One big vague step isn't.

### 2. Two Ways to Plan

Same idea as [Day 2](02-the-agent-loop.md#3-picking-a-loop-pattern)'s ReAct vs. Plan-and-Execute, just applied to planning instead of the loop.

**Single-pass** — write the whole plan first, then run it.

```
Goal: "Summarize this repo's open issues by severity"
Plan:
  1. List all open issues
  2. Classify each by severity
  3. Group and count by severity
  4. Write summary
```

Cheap and easy to review. Weak point: if step 3 finds something unexpected, steps 1-2 don't know that and nothing adjusts.

**Interleaved (plan-act-replan)** — plan just one or two steps, run them, look at what actually happened, then plan the next bit based on that. No long plan gets locked in early, so nothing goes stale.

This costs more — you're paying for a planning call at almost every step instead of one call for the whole thing. But it pays off exactly when things don't go as expected: a tool returns something odd, an API errors out, or one step turns out to secretly be two. Single-pass would just plow ahead with its original plan; interleaved notices and re-plans right there.

### 3. Which One to Use

Ask: **how predictable are the steps?**

- Predictable, fixed checklist → **single-pass**. Cheaper, easy to review upfront.
- Depends on what you find as you go → **interleaved**. Costs more, but adapts.

Common middle ground: single-pass for the big phases, interleaved inside each phase.

### 4. Goal Drift

As an agent runs longer, tool results pile up and the original goal — said once, way back — gets buried. The agent starts following "what recent history suggests" instead of the real goal. This is called **goal drift**.

Fix: re-send the exact goal text with every single prompt, not just the first one. Don't rely on it being "somewhere up there."

### 5. Check the Plan Before Trusting It

A plan comes from an LLM call, so it can come back broken — too many steps, zero steps, a step that isn't even text. Same rule as Day 2's tool calls: validate it, retry a couple of times, and if it still fails, fall back to something safe instead of looping forever.

### 6. Code: Single-Pass with Validation

```python
import logging

logger = logging.getLogger("goal_decomposition")
logging.basicConfig(level=logging.INFO, format="%(message)s")

MAX_RETRIES = 2


def decompose_goal(goal: str, max_subtasks: int, llm_call=None) -> list[str]:
    """Turn a goal into a validated list of subtasks."""
    llm_call = llm_call or _mock_llm_call

    for attempt in range(MAX_RETRIES + 1):
        raw = llm_call(goal, max_subtasks)
        error = _validate(raw, max_subtasks)

        if error is None:
            logger.info("Decomposed on attempt %d: %s", attempt + 1, raw)
            return raw

        logger.warning("Attempt %d rejected: %s", attempt + 1, error)

    # Retries used up — fall back instead of hanging.
    logger.warning("All attempts failed; falling back to [goal]")
    return [goal]


def _validate(subtasks, max_subtasks: int) -> str | None:
    if not isinstance(subtasks, list) or len(subtasks) == 0:
        return "0 items or not a list"
    if len(subtasks) > max_subtasks:
        return f"{len(subtasks)} items, over max_subtasks={max_subtasks}"
    if not all(isinstance(s, str) for s in subtasks):
        return "an item is not a string"
    return None


def _mock_llm_call(goal: str, max_subtasks: int) -> list[str]:
    return [f"Step {i + 1} toward: {goal}" for i in range(min(3, max_subtasks))]
```

Cap the retries, log every rejection, and always keep a fallback so the agent never just hangs.

### 7. Plans as a Graph, Not a List

A flat list assumes every step waits for the one before it. But often two steps don't depend on each other — "fetch revenue" and "fetch expenses" can run at the same time; only "calculate profit" needs both done. Model this as a **DAG**: each task has an id, a `depends_on` list, and a `tool_hint` (which tool handles it).

A graph plan needs two extra checks a list didn't need:

- **Dangling reference** — a task depends on an id that doesn't exist.
- **Cycle** — task A depends on B, B depends on A. Nothing can ever run.

```python
def validate_plan(tasks: list[dict]) -> str | None:
    """tasks: [{"id": str, "depends_on": list[str], "tool_hint": str}, ...]"""
    ids = {t["id"] for t in tasks}

    for t in tasks:
        missing = [d for d in t["depends_on"] if d not in ids]
        if missing:
            return f"task {t['id']!r} depends on unknown task(s): {missing}"

    # Kahn's algorithm: keep removing tasks with no unresolved deps.
    # Anything left over at the end means a cycle.
    remaining = {t["id"]: set(t["depends_on"]) for t in tasks}
    resolved: set[str] = set()

    while remaining:
        ready = [tid for tid, deps in remaining.items() if deps <= resolved]
        if not ready:
            return f"cycle detected among: {sorted(remaining)}"
        for tid in ready:
            resolved.add(tid)
            del remaining[tid]

    return None
```

Once a plan passes both checks: run every task with no pending dependencies (in parallel if you want), and each time one finishes, check if it unblocked anything new. Same validate-then-run rule as section 5 — just applied to a graph instead of a list.

*Further reading: [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic).*

## Interview Practice

**1.** Goal: "clean up our AWS spend." How would you decompose it, and when would you switch from single-pass to interleaved planning?

<details>
<summary>Answer</summary>

Start single-pass: list resources, pull cost data, flag idle/oversized ones. That part is predictable — no need to re-plan after every step.

Switch to interleaved once you're about to *act* (delete a volume, resize an instance). Now each decision depends on what the last check found, and a wrong guess is costly. Read-only investigation → single-pass. Risky actions → interleaved.
</details>

**2.** Describe a case where goal drift made an agent "succeed" at the wrong thing. What one change fixes it?

<details>
<summary>Answer</summary>

Goal: "fix the bug causing checkout failures." Twenty steps in, the agent notices an unrelated but real issue and "fixes" that instead, since it's what's freshest in its context — then reports success. Nothing in the recent history was wrong; the original goal just got buried.

Fix: re-send the exact goal text in every prompt, not just the first one.
</details>

**3.** Compare single-pass vs. plan-act-replan on cost, speed, and handling surprises. Which would you default to?

<details>
<summary>Answer</summary>

Single-pass: one planning call, cheap, fast — but doesn't adapt if something unexpected happens.

Plan-act-replan: a call before nearly every step, so it costs and takes more time — but handles surprises a fixed plan can't.

Default to single-pass. It's cheaper, and you'll find out quickly if it's not enough. Only switch to interleaved once single-pass actually breaks.
</details>

**4.** Design a plan + execution engine for a 50-step pipeline with branching and error recovery.

<details>
<summary>Answer</summary>

**Plan:** a DAG. Each task has an id, `depends_on`, a `tool_hint`, and a `status` (pending / running / done / failed / skipped).

**Execution:** run every task whose dependencies are all `done` (in parallel if possible). Each time one finishes, check what it unblocked, then run those.

**On failure:** mark that task `failed`, mark everything downstream `skipped` (they can never run now), but let every other branch keep going. Save status after each task finishes, so a crash means resuming from where it stopped, not starting over. Report done/failed/skipped clearly at the end instead of hiding it.
</details>
