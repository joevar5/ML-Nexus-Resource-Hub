# Day 6: Tool Calling Basics

## Concept

Modern LLM APIs (OpenAI, Anthropic, Gemini) support native "tool calling" (also called function calling): you describe a set of functions the model can invoke, and instead of only returning text, the model can return a structured request to call one of them.

The flow, concretely:

1. You send the API a list of tool definitions (name, description, JSON schema for arguments) alongside the prompt.
2. The model decides — based on the prompt and the tool descriptions — whether to respond with text or with a tool call (`{"name": "get_weather", "arguments": {"city": "Boston"}}`).
3. Your code, not the model, actually executes the function. The model never runs code itself; it only requests that you run it.
4. You append the function's return value to the conversation and call the model again, so it can use the result to decide the next step or produce a final answer.

This last point is the one people miss: **the model doesn't call the tool — it asks you to.** The API returns a structured request; your application code is responsible for validating that request, executing it safely, and feeding the result back. This is exactly why tool execution is a security boundary: never execute a tool call with unchecked arguments (e.g. passing a model-generated string straight into a shell command or SQL query).

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
}]
response = llm.call(messages, tools=tools)
if response.tool_call:
    result = get_weather(**response.tool_call.arguments)  # you execute it
    messages.append({"role": "tool", "content": result})
    response = llm.call(messages, tools=tools)  # call again with the result
```

## Coding Problem

Write `execute_tool_call(tool_call: dict, registry: dict[str, callable]) -> dict` that looks up `tool_call["name"]` in `registry`, and if found, calls it with `tool_call["arguments"]` unpacked as keyword arguments, returning `{"ok": True, "result": ...}`. If the name isn't in the registry, or the call raises a `TypeError` (wrong/missing arguments), return `{"ok": False, "error": <message>}` instead of letting the exception propagate — the agent loop should be able to feed this failure back to the model rather than crashing.

## Quiz

### Question 1: Who Executes the Tool

**When an LLM returns a tool call, who actually runs the underlying function?**

A) The LLM provider's servers run it automatically
B) Your application code — the model only returns a structured request; you must execute it
C) It runs inside the model's own sandboxed environment
D) The user must manually copy-paste and run it

**Answer**: B

**Explanation**: Tool/function calling APIs return a structured request (name + arguments) — they do not execute anything server-side. Your code is responsible for validating and running the call, then feeding the result back.

### Question 2: Security Boundary

**Why is tool execution described as a security boundary?**

A) Because LLM providers charge more for tool calls
B) Because the model can request arbitrary arguments, and unchecked execution (e.g. passing them straight into a shell command) creates an injection risk
C) Because tools can only be called once per session
D) It isn't a security concern — the model is trusted by design

**Answer**: B

**Explanation**: Arguments in a tool call are model-generated text, not verified input. Executing them without validation — e.g. interpolating a "city" argument straight into a shell command — opens the same injection risks as any untrusted user input.

### Question 3: The Full Round Trip

**After a tool call executes, what needs to happen for the agent to use the result?**

A) Nothing — the model automatically sees function return values
B) The result must be appended to the conversation/messages and the model must be called again with it included
C) The result is only used for logging, never for the model's next decision
D) A new conversation must be started from scratch

**Answer**: B

**Explanation**: The model has no visibility into a function's return value unless it's explicitly included in the next call's context. The round trip is: model requests a call, your code executes it, you append the result to history, then call the model again.

## Interview Practice

**1.** Explain the full round trip of a tool call to someone who thinks the model "just runs the function." Be specific about which parts happen on the model provider's side versus your own infrastructure.

**2.** A teammate wants to let the agent execute raw SQL queries it generates, directly against production. Identify the specific risks and describe the minimum safeguards you'd require before agreeing to this design.

**3.** Design the tool-calling flow for an agent that needs to call two tools where the second depends on the first's result. Walk through exactly what's in each of the three LLM calls involved.
