# Day 3: Goals and Task Breakdown

## Concept

**A goal is not a plan. Something has to turn one into the other.**

### 1. Why Break a Goal Down

Give an agent a vague goal like "improve our onboarding flow" and it will either freeze (no clear first step) or invent a plan that ignores real constraints. Neither looks like an error. You just get a quietly wrong result.

Fix: split the goal into small subtasks you can check, *before* doing anything. A small step is easy to verify. One big vague step is not.

*Read more: the "Task Decomposition" section of [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (Lilian Weng).*

### 2. Two Ways to Plan

This is [Day 2](02-the-agent-loop.md#3-picking-a-loop-pattern)'s ReAct vs. Plan-and-Execute choice, applied to planning.

**Single-pass:** write the whole plan first, then run it.

```
Goal: "Summarize this repo's open issues by severity"
Plan:
  1. List all open issues
  2. Classify each by severity
  3. Group and count by severity
  4. Write summary
```

Cheap and easy to review. Weak spot: if step 3 finds something unexpected, the plan doesn't change.

**Interleaved (plan, act, re-plan):** plan one or two steps, run them, look at the result, then plan the next bit. Nothing goes stale. It costs more (a planning call at almost every step), but it pays off when a tool returns something odd or one step turns out to be two.

*Read more: [Plan-and-Execute Agents](https://www.langchain.com/blog/planning-agents) (LangChain).*

### 3. Which One to Use

Ask: **how predictable are the steps?**

- Known in advance (a fixed checklist) → **single-pass**.
- Depends on what you find → **interleaved**.

Common middle ground: single-pass for the big phases, interleaved inside each phase.

### 4. Goal Drift

The longer an agent runs, the more tool results pile up, and the original goal (said once, long ago) gets buried. The agent starts following whatever the recent history suggests instead of the real goal. This is **goal drift**.

Fix: put the exact goal text in every prompt, not just the first one.

### 5. Check the Plan Before Trusting It

A plan comes from an LLM call, so it can come back broken: zero steps, 40 steps, or a "step" that isn't text. Same rule as Day 2's tool calls: **validate, retry a couple of times, then fall back** to something safe instead of looping forever.

```python
def decompose_goal(goal, max_subtasks, llm_call, max_retries=2):
    for _ in range(max_retries + 1):
        plan = llm_call(goal, max_subtasks)
        if (isinstance(plan, list)
                and 0 < len(plan) <= max_subtasks
                and all(isinstance(s, str) for s in plan)):
            return plan
    return [goal]  # retries used up: treat the whole goal as one step
```

Cap the retries and always keep a fallback so the agent never hangs. In real code, also log each rejected plan so you can see why.

### 6. Plans as a Graph, Not a List

A list assumes every step waits for the one before it. But "fetch revenue" and "fetch expenses" don't depend on each other, so they can run at the same time. Only "calculate profit" needs both. Model this as a **DAG** (a graph with arrows and no loops): each task has an `id` and a `depends_on` list.

A graph plan needs two checks a list didn't:

- **Dangling reference:** a task depends on an id that doesn't exist.
- **Cycle:** A depends on B and B depends on A, so nothing can ever run.

```python
def validate_plan(tasks: list[dict]) -> str | None:
    """tasks: [{"id": str, "depends_on": list[str]}, ...]"""
    ids = {t["id"] for t in tasks}
    for t in tasks:
        missing = [d for d in t["depends_on"] if d not in ids]
        if missing:
            return f"task {t['id']!r} depends on unknown task(s): {missing}"

    # Keep running whatever has all its dependencies done.
    # If we get stuck with tasks left over, there's a cycle.
    remaining = {t["id"]: set(t["depends_on"]) for t in tasks}
    done: set[str] = set()
    while remaining:
        ready = [tid for tid, deps in remaining.items() if deps <= done]
        if not ready:
            return f"cycle detected among: {sorted(remaining)}"
        for tid in ready:
            done.add(tid)
            del remaining[tid]
    return None
```

Once the plan passes both checks: run everything with no pending dependencies (in parallel if you can), and each time a task finishes, see what it unblocked.

### 7. LangGraph: A Ready-Made Graph Runner

You don't have to write that "run what's ready" loop yourself. **LangGraph** is a library built on this idea: your agent is a graph of **nodes** (one function per step), joined by **edges** (what runs next), sharing one **state** that every node reads and updates.

The revenue/expenses/profit example:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict, total=False):
    revenue: float
    expenses: float
    profit: float

def fetch_revenue(state):  return {"revenue": 120.0}
def fetch_expenses(state): return {"expenses": 80.0}
def calc_profit(state):    return {"profit": state["revenue"] - state["expenses"]}

g = StateGraph(State)
g.add_node("fetch_revenue", fetch_revenue)
g.add_node("fetch_expenses", fetch_expenses)
g.add_node("calc_profit", calc_profit)

g.add_edge(START, "fetch_revenue")    # both start together, so they run in parallel
g.add_edge(START, "fetch_expenses")
g.add_edge(["fetch_revenue", "fetch_expenses"], "calc_profit")  # waits for BOTH
g.add_edge("calc_profit", END)

g.compile().invoke({})  # {'revenue': 120.0, 'expenses': 80.0, 'profit': 40.0}
```

Compare with section 6: `depends_on` became edges, and "run what's ready" is handled for you.

Beyond running a DAG, it gives you:

- **Conditional edges:** a function reads the state and picks the next node. This is how you build loops (Day 2's ReAct cycle is "model → tool → back to model") and branches ("plan check failed → go back to the planner"). Loops are allowed on purpose, so you need a step limit instead of a cycle check.
- **Checkpointing:** state is saved after every node, so a crashed or paused run resumes where it stopped. This is what makes human approval steps (Day 22) practical.

**In industry,** teams like Klarna, Uber, LinkedIn and Replit have publicly described using it for support bots, code-migration agents, SQL assistants and coding agents. Those are LangChain's own case studies, so treat the numbers as vendor claims. The common thread is workflows that need retries, approval gates, resuming after failure, or several agents handing off. For a simple single-tool agent, a plain loop is enough.

*Read more:*
- *[LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) (official docs)*
- *[Is LangGraph used in production?](https://www.langchain.com/blog/is-langgraph-used-in-production) (LangChain)*
- *[Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic)*

## Interview Practice

**1.** Describe how you would break down a high-level user request into a structured plan for an agent to execute. Walk through the validation checks you would run on the plan before execution begins. Use "Write a quarterly business report for Q4 2025 using company data" as your example.

<details>
<summary>Answer</summary>

**Step 1: Pin down the goal.** "Quarterly report" is vague. Before planning, the agent needs the basics: which company data (finance DB, CRM?), which metrics (revenue, costs, churn?), who reads it, and what format. If the request doesn't say, ask once or write the assumptions into the plan so they can be reviewed.

**Step 2: Decompose into small, checkable tasks.** Each one has an `id`, a `depends_on` list, and the tool it uses:

| id | Task | Depends on |
|---|---|---|
| t1 | Pull Q4 2025 revenue | none |
| t2 | Pull Q4 2025 expenses | none |
| t3 | Pull Q3 2025 numbers for comparison | none |
| t4 | Calculate profit and quarter-over-quarter change | t1, t2, t3 |
| t5 | Write the narrative summary | t4 |
| t6 | Build charts and assemble the report | t4, t5 |

t1–t3 are independent, so they can run in parallel.

**Step 3: Validate before running anything.**

- **Shape:** it's a list of tasks, there's at least one, there's a sane maximum (say 15), and every task has the required fields.
- **No dangling references:** every id in `depends_on` exists.
- **No cycles:** the graph can actually be finished.
- **Tools exist:** every `tool_hint` is a real tool the agent has, and it has permission to use it (read-only on the finance DB, not write).
- **Covers the goal:** the last task produces the thing the user asked for (a report), and the date range in every data task says Q4 2025, not "last quarter". This is the check that catches goal drift.
- **Cost/time cap:** estimated calls and tokens fit the budget.

**Step 4: If validation fails,** send the specific error back to the planner and retry (cap it at 2 or 3 tries). After that, fall back to asking the user instead of running a broken plan.

The point of all of this: a bad plan is cheap to fix before step 1 and expensive to fix after step 4.
</details>

**2.** Compare static planning (all steps upfront) with dynamic planning (re-plan after each step) for a multi-city business trip agent. What are the trade-offs in correctness, latency, and cost? When do you prefer each?

<details>
<summary>Answer</summary>

Example trip: London → Paris → Berlin, with flights, hotels, and meetings in each city.

**Static plan.** Decide everything up front: book the three flights, three hotels, then schedule meetings.
- *Cost:* lowest. One planning call.
- *Latency:* lowest, and independent bookings (hotels, say) can run in parallel.
- *Correctness:* weakest when the world pushes back. If the Paris flight is sold out, the plan still tries to book the Paris hotel and the Berlin leg that depends on being in Paris.

**Dynamic plan.** Plan the next step or two, act, look at the result, plan again.
- *Cost:* higher. A planning call after nearly every step, with a growing history.
- *Latency:* higher. The steps are sequential because each one waits on the last.
- *Correctness:* best. When the Paris flight fails, the agent sees that and moves to a train or shifts the city order. It notices before it books non-refundable hotels.

**When I'd pick each:**
- Static for stable, low-risk work where the steps really are known, such as pulling the same weekly report.
- Dynamic when things can fail or change midway, and especially when steps have side effects that are hard to undo (bookings that cost money).

**What I'd build for the trip agent:** a hybrid. Plan the whole trip once (cities, dates, rough budget), then run the bookings step by step, re-planning only when a step fails or a price moves past a threshold. Search and read-only lookups can run in parallel up front, but the actual purchases happen in order. This keeps the cost near the static plan most of the time and pays for re-planning only when something actually goes wrong.
</details>

**3.** Design a task-planning system for a 50-step data-processing pipeline with branching and error recovery. Describe the plan data structures, execution engine, and failure handling.

<details>
<summary>Answer</summary>

**Plan data structure: a DAG.** Fifty steps with branches can't be a flat list. Each task is a record:

```
{
  id, name, tool,
  depends_on: [ids],
  condition: optional "run only if X",   # for branching
  retries_left, timeout,
  status: pending | running | done | failed | skipped,
  output_ref                              # pointer to stored result, not the data itself
}
```

Large intermediate data (tables, files) goes to storage and tasks pass a reference. Otherwise the state gets huge.

**Execution engine.** A scheduler loop:
1. Find every `pending` task whose dependencies are all `done`.
2. Evaluate its `condition`. If false, mark it `skipped` (a branch not taken).
3. Run the ready tasks in parallel, up to a concurrency limit.
4. When one finishes, save its status, then look for newly unblocked tasks. Repeat until nothing is pending or running.

**Branching.** A task like "check data quality" writes a result, and later tasks use `condition` on it ("if bad rows > 5%, go to clean; else go straight to load"). A branch that isn't taken is `skipped`, which counts as satisfied for anything downstream that merges the branches back together.

**Failure handling, in layers:**
- *Transient errors* (timeouts, rate limits): retry with backoff, capped at a few tries.
- *Still failing:* try a fallback if one exists (a backup source, a simpler method).
- *Permanent failure:* mark the task `failed` and everything that depends on it `skipped`, since it can never run. Independent branches keep going. Don't kill the whole pipeline for one bad branch.
- *Unsafe to continue* (a bad write may have happened): stop and ask a human.

**Recovery and idempotency.** Persist the plan and every status after each task (a database, or LangGraph checkpointing). If the process crashes, restart, skip everything already `done`, and resume. That only works if tasks are *idempotent* (safe to run twice), so use "upsert" instead of "append".

**Plan validation.** Before running: check for cycles, dangling dependencies, and missing tools, same as in the lesson.

**Visibility.** At the end, report exactly what's done, failed, and skipped, with the reason. Never hide a partial result behind "success".
</details>

**4.** Goal: "clean up our AWS spend." How would you decompose it, and when would you switch from single-pass to interleaved planning?

<details>
<summary>Answer</summary>

Start single-pass: list resources, pull cost data, flag idle/oversized ones. That part is predictable, so there's no need to re-plan after every step.

Switch to interleaved once you're about to *act* (delete a volume, resize an instance). Now each decision depends on what the last check found, and a wrong guess is costly. Read-only investigation → single-pass. Risky actions → interleaved.
</details>

**5.** Describe a case where goal drift made an agent "succeed" at the wrong thing. What one change fixes it?

<details>
<summary>Answer</summary>

Goal: "fix the bug causing checkout failures." Twenty steps in, the agent notices an unrelated but real issue and "fixes" that instead, since it's what's freshest in its context, then reports success. Nothing in the recent history was wrong; the original goal just got buried.

Fix: re-send the exact goal text in every prompt, not just the first one.
</details>
