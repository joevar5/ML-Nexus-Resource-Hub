# Day 7: Tool Schemas

## Concept

A tool schema is how you tell the model what a function does and how to call it — usually JSON Schema describing the function name, a natural-language description, and a `parameters` object. The model doesn't read your source code; it only sees the schema. Schema quality is directly responsible for tool-call accuracy.

```json
{
  "name": "search_flights",
  "description": "Search for flights between two airports on a given date. Returns a list of available flights with price and duration.",
  "parameters": {
    "type": "object",
    "properties": {
      "origin": {"type": "string", "description": "3-letter IATA airport code, e.g. 'JFK'"},
      "destination": {"type": "string", "description": "3-letter IATA airport code, e.g. 'LHR'"},
      "date": {"type": "string", "description": "ISO 8601 date, e.g. '2026-03-15'"},
      "max_price": {"type": "number", "description": "Optional maximum price in USD"}
    },
    "required": ["origin", "destination", "date"]
  }
}
```

What makes schemas good or bad in practice:

- **Ambiguous descriptions cause wrong calls.** "date" with no format specified will get called with "March 15th", "03/15", "next Friday" — all valid English, none of them parseable by your backend. Specify format explicitly.
- **Overlapping tools confuse tool selection.** Two tools named `search_flights` and `find_flights` with similar descriptions will get called inconsistently — the model can't reliably pick the "right" one because there isn't one. Keep the tool set orthogonal: each tool should do something no other tool does.
- **Required vs. optional matters.** Marking a field required when it's genuinely optional forces the model to invent a value rather than omit it — which produces hallucinated defaults. Only mark fields required if the call is meaningless without them.
- **Enums beat free text where possible.** If `sort_by` can only be `"price"` or `"duration"`, declare it as an enum, not a free-text string — it eliminates an entire class of invalid values at the schema level rather than needing runtime validation.

## Coding Problem

Write `validate_call_against_schema(call: dict, schema: dict) -> list[str]` that returns a list of error strings (empty if valid) for a tool call `{"name": ..., "arguments": {...}}` against a JSON-Schema-style `parameters` object. Check: (1) every key in `required` is present in `arguments`, (2) no argument key exists that isn't in `properties`, (3) if a property has an `"enum"` list, the argument's value is one of them. You don't need full JSON Schema — just these three checks.

## Quiz

### Question 1: Ambiguous Formats

**A tool's `date` parameter is described only as `{"type": "string"}` with no format guidance. What's the likely consequence?**

A) The model will always default to ISO 8601 automatically
B) The model may pass dates in inconsistent formats ("March 15th", "03/15") that your backend can't reliably parse
C) The API will reject the tool definition
D) No consequence — string types are always safe

**Answer**: B

**Explanation**: The model only knows what the schema tells it. An unconstrained string description gives no signal about expected format, so the model reasonably produces natural-language dates that a downstream parser wasn't built to handle.

### Question 2: Overlapping Tools

**Why do two tools with similar names and descriptions (e.g. `search_flights` and `find_flights`) hurt tool-call accuracy?**

A) APIs only allow one tool to be registered at a time
B) The model has no reliable way to distinguish which one to pick since their descriptions don't differentiate a real use case
C) It doubles the token cost with no other effect
D) It causes the API to return an error

**Answer**: B

**Explanation**: Tool selection is done by the model reading descriptions and matching intent. If two tools' descriptions don't correspond to genuinely different situations, tool choice becomes arbitrary and inconsistent — the fix is making the tool set orthogonal.

### Question 3: Required Fields

**Why should a field only be marked `required` if the call is genuinely meaningless without it?**

A) Required fields cost more tokens
B) Marking a genuinely-optional field as required forces the model to invent a plausible-sounding value rather than omit it
C) The JSON Schema spec forbids optional fields
D) It has no real effect on model behavior

**Answer**: B

**Explanation**: If the schema says a field must be present, the model will supply something rather than fail — and when it has no real value to give, that something is a hallucinated guess. Reserving `required` for truly mandatory fields avoids manufacturing false data.

## Interview Practice

**1.** You inherit an agent with 12 tools and notice it frequently calls the wrong one. Walk through how you'd diagnose whether the problem is the tool descriptions, overlapping tool purposes, or something else entirely.

**2.** A colleague argues "just make every parameter a free-text string, it's simpler and the model will figure out the format." Argue against this, using a specific example of where it breaks.

**3.** Design the schema for a tool that books a meeting room, including which fields are required, which are optional, and which should be enums rather than free text. Justify each choice.
