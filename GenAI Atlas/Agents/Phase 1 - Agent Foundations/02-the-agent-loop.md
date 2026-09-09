# Day 2: The Agent Loop

## Concept

**Reason → Act → Observe → Repeat.**

### 1. Why You Need a Loop at All

You ask an LLM a question, it answers — that's one turn, done in a single call. But the second the task needs more than one step — look something up, run a calculation, check a database, *then* combine all of that into a real answer — a single call won't cut it. You need a loop.

Without a loop, an LLM can only answer from what it already memorized during training. It can't look anything up, run code, or touch a live API — it's stuck guessing. A loop is what turns a static "knowledge base with a chat window" into something that can actually go do things and adjust based on what it finds.

Every agent framework you've heard of — LangGraph, function-calling loops, custom while-loops — is really just a variation on the same underlying pattern. Learn it once and you'll recognize it everywhere.

### 2. The Cycle Itself

At its core, the loop has three repeating phases: **Thought → Action → Observation.**

1. **Thought** — the model reasons about the current state: what it knows, what it still needs, what to do next.
2. **Action** — it produces a structured command: a tool name plus arguments.
3. **Observation** — the tool runs, its result gets appended back onto the conversation.

Round and round it goes until the model decides it has enough information and emits a final answer instead of another action.

```
1. Build a prompt from: system instructions + goal + conversation/tool history
2. Call the LLM
3. Parse the response: is it a final answer, or a tool call?
4. If tool call: execute it, append the result to history, go to 1
5. If final answer: stop
```

*Further reading: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the original paper this whole cycle is named after.*

### 3. Picking a Loop Pattern

The cycle above is called **ReAct** (Reason + Act), and it's the one you'll hit most often: think one step, act one step, look at what happened, repeat. It's popular because it's simple to implement, easy to debug (you can read the trace turn by turn), and it's the default most tool-calling APIs are built around.

It isn't the only pattern, though — and knowing the alternatives is what lets you pick the right one instead of defaulting to ReAct out of habit:

- **Plan-and-Execute** — the model writes out the *whole* multi-step plan up front, then executes each step in order, only replanning if something goes wrong. Fewer LLM calls than ReAct's think-after-every-step approach, but it commits to a plan before seeing any real-world feedback, so it's worse when the environment is unpredictable.
- **Reflexion / self-critique loops** — after acting, the model explicitly critiques its own output before deciding what to do next, rather than just reacting to the raw tool result. Useful when correctness matters more than speed, since it catches mistakes the plain ReAct loop would just plow through.

The deciding factor is usually **how predictable the task is**:

- **Unpredictable, step depends on last result** (browsing the web, debugging code, exploring an unfamiliar database) → use **ReAct**.
- **Predictable, steps known upfront** (a fixed checklist: fetch data → clean it → generate report) → use **Plan-and-Execute**. Fewer LLM calls.
- **Wrong answer is expensive** (high-stakes or irreversible actions) → use **Reflexion**. Not worth it for cheap, low-risk lookups.

These three sit inside a bigger set of seven recurring patterns that show up across production agents — worth knowing even outside the loop-design question. [*The 7 Design Patterns Every AI Agent Developer Should Know*](https://pub.towardsai.net/the-7-design-patterns-every-ai-agent-developer-should-know-in-2026-c77f28b51565) covers all of them; here's the "which one, when" summary:

| Pattern | What it does | Use it when |
|---|---|---|
| **Reflection** | Agent critiques its own output and revises before returning a final result | Output quality matters more than speed — self-critique measurably improves correctness |
| **ReAct** | Interleaves reasoning with tool calls in a continuous loop | The task is exploratory/open-ended and each step depends on what the last one revealed |
| **Plan-and-Execute** | Separates planning (write the whole strategy) from execution (run each step) | The task is long and structured — you don't want reasoning to drift mid-stream |
| **Tool Use** | Lets the agent call external functions via defined schemas | Always — it's the foundational capability every other pattern builds on |
| **Multi-Agent Collaboration** | Specialized agents work in parallel or in sequence, each scoped to one domain | The task genuinely splits into separate concerns — start single-agent, graduate up only as complexity demands |
| **Memory Management** | Carries agent state across turns (in-context, external, episodic, or procedural) | Context is filling up or the task spans many turns — layer memory in progressively |
| **Human-in-the-Loop** | Inserts a human checkpoint at a decision point, or lets a human review async | The action is high-stakes or irreversible — a risk-management mechanism, not a UX afterthought |

Whichever pattern you pick, it inherits the same problem: nothing so far stops the loop from running forever. That's the next piece.

### 4. Stop Conditions

Consider a common scenario: a team ships a research agent, then steps away for a break. When they return, the agent has been running for 40 minutes, calling the same search tool over 200 times, with the API costs climbing the entire time. Nothing was wrong with the model — the loop simply had no instruction telling it when to stop.

This is one of the most common ways agents fail in production, and also one of the most preventable. **A loop with no exit condition is just an infinite loop with extra steps.** It will keep consuming tokens and credits indefinitely, because to the model, "keep going" and "keep going forever" look exactly the same unless something outside it draws the line.

That "something outside it" is a stop condition. A reliable agent needs all four of these, not just one:

- **Final-answer detection** — the model emits a clearly reserved format (such as `Final Answer: <result>`) that you check for after every turn. The format needs to be strict enough that a normal sentence can never be mistaken for it.
- **A step budget** (`max_iterations`) — a hard ceiling on how many tool-calling turns the agent is allowed, with no exceptions once it's reached. There is no situation where an unbounded agent is a safe default. 10–15 steps is a reasonable starting point.
- **Repeated-action detection** — if the agent calls the exact same tool with the exact same arguments more than 2–3 times, that is not persistence, it is a stall. Models have no sense of frustration or wasted effort, so they will repeat a failing call as calmly on attempt 50 as on attempt 1. This has to be caught externally.
- **A graceful fallback** — when every other condition has been triggered and there's still no answer, the agent should return something honest, like *"I could not find an answer after N steps,"* rather than a guessed answer or a crash.

Three of these four conditions depend on the agent being able to look back at what already happened — you cannot detect a repeated action without remembering the last one. So stop conditions are only as reliable as the memory they're reading from, which is the next design question.

*Further reading: [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic) — the workflow vs. agent distinction, and the antipatterns (monolithic agents, over-engineered planning) that lead to this kind of runaway loop.*

### 5. Designing the Scratchpad

There's a quieter way the same agent could fail: it stops after 3 steps instead of 200, but returns the wrong answer, or forgets a constraint it was given at the start. No runaway loop, no unusual cost — just a confidently incorrect result. The underlying cause is usually the same: too little attention was paid to what the model could actually see when it made its decision.

That "what it can see" is called the **scratchpad**, and how it's designed is one of the highest-leverage decisions in the whole system. It functions as the agent's entire working memory — everything it knows about the task lives there, and only there. There is no side channel and no separate memory it can consult; if something isn't in the scratchpad, the model has no way of knowing it. It holds the system instructions once, at the start, followed by a repeating block for each turn: Thought, Action, Observation.

Two practices make the difference between a scratchpad that supports the model and one that confuses it:

- **Use a consistent delimiter, and label every section the same way.** Separate turns with something unambiguous (such as `---`), and label Thought, Action, and Observation identically every time. A language model is, fundamentally, a pattern-matching system reading its own transcript back to itself — a format that shifts between turns makes it noticeably harder for the model to locate what happened last. A clean, repeatable format is often the difference between an agent that finishes in 5 turns and one that takes 20, on the same task.
- **Manage its size proactively, not reactively.** Appending every tool result indefinitely grows the context linearly with each step — a 20-step agent can approach the context window, or simply become slow and expensive, well before that limit is hit. Check token usage every turn, and once it crosses roughly 70–80% of the model's limit, trim the oldest turns and replace them with a short summary instead of dropping them silently. A silent drop is often how an agent ends up forgetting the one constraint that mattered.

A poorly managed scratchpad and a missing stop condition tend to produce the same symptom from the outside — an agent that keeps going, or one that stops having lost track of the task. That overlap is useful: when a loop gets stuck, the cause is almost always in one of these two places.

*Further reading: [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) (Anthropic) — a deeper look at the note-taking/scratchpad pattern and managing context as a finite resource.*

### 6. Debugging a Stuck Loop

Once an agent is stuck, the next question is how to actually find the cause. A useful way to think about it: **an agent that loops isn't broken, it's communicating.** Unlike a program that fails with a clear stack trace, a looping agent produces page after page of plausible-looking actions that lead nowhere — which can feel unreadable until you know what to look for. The signal is almost always in the overall pattern, not in any single line.

Four patterns account for most of what shows up in practice:

- **No Final Answer is ever emitted.** The agent keeps acting because, from its perspective, the task is never finished — usually because the prompt never defined a clear way to signal completion. Fix: state the exact output format required, in the system prompt, with an example.
- **Hallucinated tool names.** The model calls a tool that was never registered, because it seemed like something that should exist. Fix: validate every tool call against a known registry before executing it, and reject unrecognized names explicitly.
- **Format errors.** The action text is close to what the parser expects but not exact, so parsing fails silently and the loop stalls with no visible error. Fix: tighten the parser or relax the format — either is fine, as long as it's applied consistently.
- **Ignored tool errors.** A tool returns an error, and the agent treats it as ordinary data rather than a failure, then repeats the same action. Fix: mark errors distinctly (for example, an explicit `ERROR:` prefix) so the model can tell a failure apart from a valid result.

A practical way to avoid reading every line of a long trace: check four numbers first — total steps, unique tools called, repeated (tool, args) pairs, and tool error count. These usually reveal the shape of the problem immediately. A trace with 40 steps and only 2 unique tools points to a stall; one with 40 steps and 35 unique tools points to the agent simply losing direction — two different problems with two different fixes.

*Further reading: [AI Agent Observability: Tracing, Testing, and Improving Agents](https://www.langchain.com/resources/agent-observability) (LangChain) — how production teams instrument agents so this kind of diagnosis doesn't rely on guesswork.*

### 7. Putting It Together

The version below combines everything covered so far into one implementation: the ReAct cycle (section 2), a real scratchpad with consistent delimiters (section 5), all four stop conditions (section 4), and structured logging so the trace can actually be debugged afterward (section 6) instead of just guessed at.

```python
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("agent_loop")
logging.basicConfig(level=logging.INFO, format="%(message)s")

MAX_STEPS = 15
DELIMITER = "---"
FINAL_RE = re.compile(r"^Final Answer:\s*(.+)", re.IGNORECASE)


# --- Section 5: Scratchpad -------------------------------------------------
# Holds the system prompt once, then one labeled Thought/Action/Observation
# block per turn. This is the *only* thing the model sees each iteration.
@dataclass
class Turn:
    thought: str
    action: str
    action_input: str
    observation: str

@dataclass
class Scratchpad:
    system_prompt: str
    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        logger.info(
            "%s Turn %d\nThought: %s\nAction: %s(%s)\nObservation: %s",
            DELIMITER, len(self.turns), turn.thought,
            turn.action, turn.action_input, turn.observation,
        )

    def render(self) -> str:
        parts = [self.system_prompt]
        for i, t in enumerate(self.turns, 1):
            parts.append(
                f"{DELIMITER}\nTurn {i}:\nThought: {t.thought}\n"
                f"Action: {t.action}\nAction Input: {t.action_input}\n"
                f"Observation: {t.observation}"
            )
        return "\n".join(parts)


def normalize_args(args: str) -> str:
    return " ".join(sorted(args.strip().lower().split()))


def run_agent(goal: str, tools: dict, max_steps: int = MAX_STEPS) -> str:
    pad = Scratchpad(system_prompt=f"Goal: {goal}")
    action_counts: dict[tuple[str, str], int] = {}

    for step in range(max_steps):
        response = llm.call(pad.render())

        # --- Section 4: stop condition #1 — final-answer detection --------
        match = FINAL_RE.match(response.text.strip())
        if match:
            logger.info("%s Stopped: final answer on step %d", DELIMITER, step)
            return match.group(1)

        tool_name, args = response.tool_call.name, response.tool_call.args

        # --- Section 4: stop condition #2 — repeated-action detection -----
        key = (tool_name, normalize_args(args))
        action_counts[key] = action_counts.get(key, 0) + 1
        if action_counts[key] > 2:
            logger.warning("%s Stopped: repeated action %s", DELIMITER, key)
            return "I could not find an answer: the agent repeated the same action."

        # --- Section 6: surface tool errors distinctly, don't hide them ---
        try:
            result = execute_tool(tool_name, args, tools)
        except Exception as exc:
            result = f"ERROR: {exc}"
            logger.warning("%s Tool error on step %d: %s", DELIMITER, step, exc)

        pad.add_turn(Turn(
            thought=response.thought,
            action=tool_name,
            action_input=args,
            observation=result,
        ))

    # --- Section 4: stop condition #3 — step budget exhausted / fallback --
    logger.warning("%s Stopped: step budget of %d exhausted", DELIMITER, max_steps)
    return f"I could not find an answer after {max_steps} steps."
```

Every stop condition from section 4 is checked on every turn, the scratchpad is rendered with the same delimiter and labels on every call (section 5), and each turn — plus every stop trigger and tool error — gets logged, so a stuck run can be diagnosed from the log alone using the four-number check from section 6 (total steps, unique tools, repeated pairs, error count).


## Practice

Apply what this lesson covered — the ReAct cycle, stop conditions, and scratchpad design — to a hands-on problem:

- [ReAct Loop + Scratchpad Summary](https://www.agenticprep.io/problems/react-loop-scratchpad-summary)