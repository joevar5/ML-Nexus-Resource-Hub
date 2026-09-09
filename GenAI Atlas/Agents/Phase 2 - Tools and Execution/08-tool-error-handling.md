# Day 8: Tool Error Handling

## Concept

Tools fail: networks time out, APIs return 500s, arguments are invalid, rate limits get hit. How a tool failure is surfaced to the model determines whether the agent recovers gracefully or falls apart.

The key principle: **feed errors back into the conversation as data, don't let them crash the loop.** A tool call that raises an unhandled exception should be caught at the execution layer and converted into a structured error message the model can read and reason about — the same way a human debugging a failed API call reads the error message rather than the program terminating.

```python
def safe_execute(tool_call, registry):
    try:
        fn = registry[tool_call["name"]]
        return {"ok": True, "result": fn(**tool_call["arguments"])}
    except KeyError:
        return {"ok": False, "error": f"Unknown tool: {tool_call['name']}"}
    except TypeError as e:
        return {"ok": False, "error": f"Invalid arguments: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Tool execution failed: {e}"}
```

Not all errors should be handled the same way once they reach the model:

- **Retryable errors** (timeout, rate limit, transient 500) — the model or a retry wrapper should try again, often with backoff.
- **Non-retryable errors** (invalid arguments, permission denied, resource doesn't exist) — retrying identically will fail identically. The model needs to change its approach, not repeat the call.
- **Errors the model can't fix** (a third-party service is down) — the agent should recognize this and either fail gracefully with a clear message to the user, or fall back to an alternative path, rather than retrying forever.

The distinction matters because an agent that treats every error as retryable will loop on non-retryable failures (Day 5's "looping" failure mode), and an agent that treats every error as fatal will give up on transient network blips that a simple retry would have fixed.

## Coding Problem

Write `classify_error(error_message: str) -> str` that returns `"retryable"` if the message contains any of `"timeout"`, `"rate limit"`, `"503"`, `"connection reset"` (case-insensitive substring match), `"non_retryable"` if it contains `"invalid"`, `"permission denied"`, `"not found"`, and `"unknown"` otherwise. This classification should drive whether an agent's retry wrapper attempts the call again or surfaces the failure to the model for a different approach.

## Quiz

### Question 1: Handling Tool Exceptions

**Why should a tool execution failure be converted into a structured error message rather than allowed to raise an unhandled exception?**

A) Unhandled exceptions are always faster
B) An unhandled exception crashes the agent loop entirely; a structured error lets the model see the failure and decide how to respond
C) It's required by the OpenAI API
D) There's no difference in outcome

**Answer**: B

**Explanation**: If an exception propagates unhandled, the entire agent process typically terminates. Catching it and returning a structured `{"ok": False, "error": ...}` result keeps the loop alive and gives the model the information it needs to adapt.

### Question 2: Retryable vs. Non-Retryable

**An agent's tool call fails with "invalid argument: city must be a 3-letter code". What's the correct response?**

A) Retry the identical call — it will likely succeed the second time
B) This is a non-retryable error; the model needs to change the argument, not repeat the same call
C) Abandon the entire task immediately
D) Silently ignore the error and continue

**Answer**: B

**Explanation**: An invalid-argument error is deterministic — calling with the same bad argument again produces the same failure. The correct recovery is for the model to see the error and supply a corrected argument, not to retry blindly.

### Question 3: Why the Distinction Matters

**What goes wrong if an agent treats every tool error as retryable, with no distinction for non-retryable failures?**

A) Nothing — retrying is always safe
B) It can loop indefinitely on errors that will never succeed no matter how many times they're retried (e.g. invalid input), wasting steps/cost without progress
C) It makes the agent faster
D) It only affects logging output

**Answer**: B

**Explanation**: Retrying a non-retryable error (like malformed input) doesn't fix anything — it just repeats the same failure, which is exactly the "looping" failure mode from Day 5. Classifying errors lets the agent retry only what retrying can actually fix.

## Interview Practice

**1.** Describe how you'd design the boundary between "errors your infrastructure should catch and retry automatically" and "errors that should be surfaced to the model for a different approach." Give one example of each.

**2.** A production agent's error rate spiked overnight. Walk through the first three things you'd check, and how you'd tell whether it's a retryable transient issue versus a real regression.

**3.** A junior engineer wraps every single tool call in a generic `try/except: return None`. Identify what real failure information this design throws away, and what it costs the agent downstream.
