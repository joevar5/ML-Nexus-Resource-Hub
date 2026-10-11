# Exercise 08: Cost-Aware LLM Routing

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** [Exercise 07](../exercise-07-multi-tenant-llm-platform/README.md), [Lab 08](../../labs/lab-08-multi-model-serving.md), and [Lesson 09: Production LLM Best Practices](../../09-production-llm-best-practices.md)
**Hardware:** One small local model (or a cheap hosted one). Hosted API keys are optional.

## Goal

Build a router that sends each request to the cheapest model that can handle it well. Compare three ways of choosing, and measure how much money you save and how much quality you give up.

## Why it matters

Most traffic is easy. A short factual question does not need your biggest model, and sending everything to it is the most common way to overspend. A good router can cut the bill sharply, but a bad one quietly gives users worse answers. So the goal is never "cheapest". It is "cheapest at the quality we promised".

## Before you start

Pick three tiers. Model names change every few months, so decide by role and fill in whatever is current:

| Tier | Role | Example |
|---|---|---|
| 0 | Small and cheap, often self-hosted | A 7B–8B open model on your vLLM server |
| 1 | Mid-size, good all-rounder | A current small or "mini" hosted model |
| 2 | Largest, for hard cases | A current frontier model |

Write the price per million input and output tokens for each tier in one config file. Everything else in this exercise reads from it.

Build the test set before you build the router (Step 1). If you tune the router while looking at the test set, your results will be too good to be true.

## Steps

### Step 1: Build a test set (30 min)

Collect 100–200 realistic queries that cover easy, medium, and hard cases. For each one, write a reference answer or a scoring rule. Split it in two: a **tuning set** you may look at, and a **held-out set** you only use for the final numbers.

Decide how to score answers. Exact match works for extraction and classification. For open-ended answers use an LLM judge, as in [Exercise 12](../exercise-12-llm-evaluation-harness/README.md), and spot-check some of its verdicts by hand.

### Step 2: Run the two baselines (15 min)

Send every query to tier 0, then every query to tier 2. Record cost, quality, and latency. These are your bounds: the cheapest you could be and the best you could be.

### Step 3: Rule-based router (30 min)

Start with simple rules based on length, keywords, and whether the request needs code or math.

```python
def route_by_rules(query: str) -> int:
    q = query.lower()
    if any(k in q for k in ("prove", "derive", "step by step", "debug", "refactor")):
        return 2
    if len(query) < 300:
        return 0
    return 1
```

Rules are fast and free. They will also misroute some queries. Count how many.

### Step 4: Learned router (60 min)

Replace the rules with a small classifier. Two good options:

1. **A prompt to a small model** that returns the difficulty. Force a strict answer format and parse it carefully, with a fallback to tier 1 when parsing fails.
2. **A trained classifier.** Embed each query, then train logistic regression on "did the small model answer this correctly?" using your tuning set. This is cheaper and faster at runtime than an LLM call. Open-source projects such as RouteLLM take this approach, so read their approach before you write your own.

```python
def route_by_llm(query: str) -> int:
    r = small.chat.completions.create(
        model=TIER0, temperature=0, max_tokens=2,
        messages=[{"role": "system",
                   "content": "Rate the difficulty of the user's question: 0 easy, 1 medium, 2 hard. Reply with one digit."},
                  {"role": "user", "content": query}],
    )
    text = r.choices[0].message.content.strip()
    return int(text) if text in {"0", "1", "2"} else 1     # fall back to the middle tier
```

### Step 5: Escalation (45 min)

Start at the chosen tier. If the answer looks unreliable, retry on the next tier up. The weak point is how you decide that an answer is "unreliable".

Do not use phrases like "I don't know" as your only signal, because wrong answers are usually confident. Better signals:
- The output fails a format check (invalid JSON, missing field).
- The model's token log-probabilities are low (if the server gives them to you).
- A cheap verifier or the small model itself rates its own answer.
- The tier-0 and tier-1 answers disagree.

Track how often you escalate. If it is nearly always, the first tier is wasted money and added latency.

### Step 6: Compare everything on the held-out set (15 min)

| Strategy | Total cost | Quality | p50 latency | p95 latency | % escalated |
|---|---|---|---|---|---|
| Always tier 0 | | | | | |
| Always tier 2 | | | | | |
| Rules | | | | | |
| Learned router | | | | | |
| Learned + escalation | | | | | |

Fill it with your own numbers. Include the router's own cost and latency, since the classifier is not free.

### Step 7: Make it safe to run (30 min)

- **Log every routing decision** with the tier chosen, the reason, and the outcome, so you can find mistakes later.
- **Cache routing decisions** for repeated queries.
- **Add a kill switch** that sends everything to one tier if the router misbehaves.
- **Roll changes out gradually**: send 5% of traffic to a new router, compare quality, then widen.
- **Watch for drift**: if the mix of queries changes, a router trained on old data will slowly get worse.

## Deliverables

1. The router with rule, learned, and escalation modes.
2. The test set, the scoring code, and the results table.
3. `RESULTS.md`: the table plus a paragraph on what surprised you.
4. `ROUTING_PLAYBOOK.md`: how to add a new tier, how to re-train the router, and how to roll back.

## Check yourself

- [ ] You report numbers from the held-out set, not the tuning set.
- [ ] Cost is lower than "always tier 2" by a meaningful amount (aim for 50% or more), with quality within your promised range.
- [ ] You can name queries that your router sent to the wrong tier and explain why.
- [ ] The router's own cost and latency are in your totals.

## Common pitfalls

- **Judging with a weak scorer.** A noisy judge hides real quality loss. Validate it against some human-labelled answers.
- **Unparsed classifier output.** A model that replies "The query is moderately hard" instead of a digit will crash a naive `int()`. Parse strictly and fall back.
- **Latency stacking.** A classifier call plus a failed first attempt plus an escalation can be slower than going straight to the big model. Measure p95, not just the average.
- **Prices that are out of date.** Keep them in config and check them against the providers' pricing pages.
- **Tuning on the test set.** Once you have looked at the held-out set to fix the router, it is no longer held out.
