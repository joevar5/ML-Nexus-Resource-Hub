# Ex 07: Streaming Relay with Heartbeats and Cancellation (CPU, auto-graded)

**Goal:** question 2 — stream tokens over SSE, keep idle connections alive, and
stop the GPU work the moment a user leaves. Read Lecture 04 first.
**Time:** ~2 h. **Needs:** Python 3.10+ only (pure asyncio).

## Part 1 — the core logic (graded)

Implement `sse_relay` in `starter.py` to the spec in its docstring, then
`python check.py` until it prints `PASS (5/5)`. The grader's fake engine
records whether it was closed (its `finally:` block stands in for
`engine.abort(request_id)`) and how many tokens were pulled after the client left.

The hard case is a client that leaves while the engine is in the middle of a
slow step: a plain `async for` loop can't notice until the next token arrives.

## Part 2 — wire it into a real proxy (not graded)

1. Put `sse_relay` behind FastAPI as in Lecture 04: turn `httpx`'s
   `aiter_lines()` into the `upstream` iterator (strip the `data: ` prefix),
   and set `client_gone` from a small task that polls `request.is_disconnected()`.
2. Point it at `vllm serve <small model>`. Start a 2,000-token generation with
   `curl -N`, press Ctrl-C after a second, and show in the vLLM log (or the
   `vllm:num_requests_running` gauge) that the request was aborted.
3. Put nginx in front with default settings and watch tokens arrive in bursts;
   then add `proxy_buffering off;` and compare.

## Pass criteria

`check.py` prints PASS; for Part 2, the vLLM log line or metric showing the
abort, and a before/after note on nginx buffering.
