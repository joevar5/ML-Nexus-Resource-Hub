# Day 10: Structured Output

## Concept

An agent's downstream code (the loop, tool dispatch, UI) needs to parse the model's response programmatically — but LLMs generate free-form text by default. Structured output closes that gap by constraining generation to match a schema, so `json.loads(response)` reliably succeeds instead of occasionally getting prose, markdown fences, or a JSON object with an extra trailing sentence.

Two mechanisms, different reliability guarantees:

- **Prompted JSON.** Ask the model to "respond only with valid JSON matching this schema." Works most of the time, but the model can still wrap output in ```json fences, add a preamble ("Sure, here's the JSON:"), or produce almost-valid JSON (trailing commas, unescaped quotes). Requires defensive parsing on your end regardless.
- **Constrained decoding / native structured output** (OpenAI's `response_format: json_schema`, Anthropic's tool-use-as-output pattern, Gemini's `response_schema`). The provider enforces the schema at the token-sampling level — the model literally cannot emit a token that would violate the schema. This is strictly more reliable than prompting alone and should be preferred whenever available.

Even with native structured output, two things still need handling: (1) schema *validity* doesn't guarantee semantic *correctness* — a response can be well-formed JSON with a `"city": "Atlantis"` value that's syntactically fine but meaningless; and (2) you still need a fallback parse path for providers/models that don't support the feature.

```python
import json, re

def parse_structured_output(raw: str) -> dict:
    # Defensive parse: strip markdown fences if present, then parse
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    return json.loads(cleaned)  # raises if still invalid — let the caller decide the retry policy
```

## Coding Problem

Write `parse_structured_output(raw: str, schema_keys: set[str]) -> dict` that: strips markdown code fences if present, parses JSON, and validates that the parsed object's keys are exactly `schema_keys` (no extra, none missing). Raise a `ValueError` with a message naming which keys are missing and which are unexpected, rather than a generic parse error — the caller needs to know *why* it failed to decide whether to retry with a corrective prompt.

## Quiz

### Question 1: Prompted JSON's Weakness

**Why is asking the model to "respond only with valid JSON" in the prompt not fully reliable on its own?**

A) Models cannot generate JSON syntax at all
B) The model can still add preambles, markdown fences, or near-valid JSON (trailing commas, etc.) despite the instruction — it's a request, not an enforced constraint
C) JSON is not supported by any LLM provider
D) Prompted JSON is always 100% reliable and needs no defensive parsing

**Answer**: B

**Explanation**: A prompt instruction shapes behavior probabilistically — it isn't enforced at the decoding level. The model usually complies but can still wrap output in explanatory text or fences, which is why defensive parsing (stripping fences, catching parse errors) is still necessary.

### Question 2: Constrained Decoding

**How does native structured output (e.g. `response_format: json_schema`) differ from prompting the model to produce JSON?**

A) It's exactly the same mechanism with a different name
B) The provider enforces the schema during token sampling, so the model literally cannot emit a token that violates it — a stronger guarantee than a prompt instruction
C) It only works for schemas with a single field
D) It removes the need for any error handling whatsoever

**Answer**: B

**Explanation**: Constrained decoding operates at the sampling level, restricting which tokens are even eligible for generation at each step. This is a hard guarantee on syntactic validity, unlike prompting, which only influences the model's behavior without enforcing it.

### Question 3: Validity vs. Correctness

**A structured output response is syntactically valid JSON matching the schema, but contains `"user_age": -5`. What does this demonstrate?**

A) The schema validation is broken
B) Schema validity (well-formed, right keys/types) does not guarantee semantic correctness (values that make sense) — both need checking separately
C) This can never happen with structured output
D) Negative numbers are invalid JSON

**Answer**: B

**Explanation**: A schema can enforce "this field is a number" but not "this number is a plausible age." Structured output solves the parsing problem, not the correctness problem — semantic validation (range checks, business logic) is still a separate, necessary layer.

## Interview Practice

**1.** A teammate says "once we switch to native structured output, we don't need any validation code anymore." Explain specifically what structured output does and doesn't guarantee, and what validation still needs to exist.

**2.** Design the parsing and validation layer for an agent whose final answer must be `{"decision": "approve"|"deny", "reason": str, "confidence": float between 0 and 1}`. Walk through what you check beyond basic JSON schema validity.

**3.** You're using a model/provider that doesn't support native structured output. Describe the defensive parsing strategy you'd use instead, and how you'd measure whether it's reliable enough to ship.
