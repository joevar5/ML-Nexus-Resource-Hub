# Day 19: Scoring Agent Behavior

## Concept

Day 18 introduced LLM-as-judge as a necessary but weak eval signal. This lesson is about making that signal trustworthy enough to actually rely on.

The core risk: an LLM judge scoring another LLM's output can be miscalibrated in ways that aren't obvious from looking at a handful of scores — it might systematically favor longer answers, be lenient on a specific failure type, or simply disagree with what a human would say is correct. Trusting an unvalidated judge means you might be optimizing against the judge's biases, not the actual task.

Three practices make LLM-as-judge trustworthy rather than just convenient:

- **Rubric-based scoring, not vague scoring.** "Rate this answer 1-10" produces noisy, hard-to-interpret scores. "Does the answer cite a source? (yes/no). Does the answer directly address the question asked? (yes/no). Is the answer under 200 words? (yes/no)" produces specific, checkable, more consistent judgments. Break "quality" into concrete criteria before asking the judge to score anything.
- **Calibration against human labels.** Take a sample of cases (50-100 is a reasonable start), have a human score them against the same rubric, and compute agreement between the judge and the human labels. If agreement is low, the rubric is ambiguous or the judge model isn't suited to this task — fix the rubric or the judge before trusting it on the full eval set.
- **Position bias awareness.** When a judge compares two outputs (A vs. B), some judge models show a measurable bias toward whichever one is presented first, independent of actual quality. Mitigate by running the comparison both orders (A-then-B and B-then-A) and only counting a preference as real if it's consistent both ways.

```python
def judge_with_rubric(output: str, rubric_criteria: list[str], judge_llm) -> dict:
    results = {}
    for criterion in rubric_criteria:
        prompt = f"Output:\n{output}\n\nCriterion: {criterion}\nAnswer strictly yes or no."
        results[criterion] = judge_llm.call(prompt).strip().lower().startswith("yes")
    return results
```

The takeaway isn't "don't use LLM judges" — it's that an LLM judge is itself a component that needs its own validation, the same way you'd validate any other measurement tool before trusting readings from it.

## Coding Problem

Write `check_judge_agreement(human_labels: list[bool], judge_labels: list[bool]) -> dict` that computes agreement rate (fraction where both match), and separately computes the **false positive rate** (judge said `True`/pass, human said `False`/fail) and **false negative rate** (judge said `False`, human said `True`) as fractions of total cases. Return `{"agreement": float, "false_positive_rate": float, "false_negative_rate": float}`. Raise `ValueError` if the two lists have different lengths.

## Quiz

### Question 1: Rubric vs. Vague Scoring

**Why does "rate this 1-10" tend to produce noisier judge output than a rubric of specific yes/no criteria?**

A) LLMs cannot produce numbers
B) A vague, holistic score conflates many different quality dimensions into one number with no clear standard, while specific yes/no criteria give the judge a concrete, checkable question to answer each time
C) 1-10 scales are always more accurate than yes/no
D) There's no meaningful difference between the two approaches

**Answer**: B

**Explanation**: A single holistic score forces the judge to implicitly weigh multiple factors (accuracy, tone, length, citation) without a defined standard, producing inconsistent results. Breaking quality into specific criteria makes each judgment narrower and more consistent to answer.

### Question 2: Calibration

**Why is comparing judge labels against human labels on a sample of cases a necessary step before trusting an LLM judge at scale?**

A) It's not necessary — LLM judges are always accurate out of the box
B) An unvalidated judge might be systematically biased (e.g. favoring longer answers) in ways not obvious from a few examples; measuring agreement with human judgment reveals whether the judge and rubric are actually trustworthy
C) Calibration is only needed for numerical scores, not yes/no judgments
D) Human labeling is always more expensive and should be skipped entirely

**Answer**: B

**Explanation**: Without checking agreement against ground-truth human judgment, systematic judge biases stay invisible — you'd be optimizing against whatever the judge happens to reward, which may not match what actually constitutes a good answer.

### Question 3: Position Bias

**When an LLM judge compares two outputs (A vs. B) and shows a preference, why should you re-run the comparison with the order swapped before trusting that preference?**

A) Swapping order is required by the LLM provider's terms of service
B) Some judge models exhibit measurable bias toward whichever output is presented first, independent of actual quality — checking both orders and requiring a consistent result filters this out
C) Order never affects an LLM judge's output
D) This is only relevant when comparing more than 2 outputs

**Answer**: B

**Explanation**: Position bias is a documented failure mode where presentation order alone shifts a judge's preference. Running both orderings and only trusting a preference that holds in both directions distinguishes a genuine quality signal from an artifact of ordering.

## Interview Practice

**1.** Your LLM judge agrees with human labels 60% of the time on a calibration sample. Walk through what you'd investigate first — is it the rubric, the judge model, or the human labels themselves that's unreliable?

**2.** A stakeholder wants to skip human calibration entirely and trust the LLM judge from day one "since it's faster." Make the case for why this is risky, with a concrete failure scenario.

**3.** Design a rubric (3-5 criteria) for judging whether an agent's customer support response is good. Explain why you chose those specific criteria over other plausible ones.
