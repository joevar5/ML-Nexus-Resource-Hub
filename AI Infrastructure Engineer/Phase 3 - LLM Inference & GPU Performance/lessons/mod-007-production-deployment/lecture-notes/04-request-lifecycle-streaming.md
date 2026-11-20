# Lecture 04: The Request Lifecycle, Streaming and Cancellation

Answers questions **1** (what happens between a request arriving and the first
token leaving) and **2** (stream over SSE, handle disconnects, cancel in-flight
generation) of [the 25 questions](../../../25-questions.md).

## 1. From request to first token

```
client ──TLS──▶ gateway / LB ──▶ router ──▶ API server ──▶ engine scheduler ──▶ GPU
         auth, rate limit    picks replica   tokenize,      waiting queue →      prefill →
                             (load, prefix)  chat template  admit when KV fits   first token
```

| Stage | What happens | Typical cost | What makes it slow |
|---|---|---|---|
| Network + gateway | TLS, auth, rate limits, routing | 1–20 ms | cross-region hops, cold TLS |
| Router | pick a replica (Mod 6 Lecture 03) | < 1 ms | a hot replica chosen for its cache |
| API server | parse JSON, apply chat template, tokenize | 1–10 ms | huge prompts, Python under load |
| **Queue** | waits until the scheduler admits it: needs a batch slot **and** free KV blocks | 0 ms to seconds | load above capacity, long prompts ahead |
| **Prefill** | runs the whole prompt (minus prefix-cache hits), in chunks if chunked prefill is on | ≈ `2 · params · prompt_tokens / (peak FLOPs · MFU)` | long prompts, low MFU |
| Sample + detokenize | logits → token → text | < 1 ms | structured-output masks (Lecture 01, XGrammar) |
| Stream out | first SSE chunk written and flushed | network RTT | proxy buffering (below) |

**TTFT = network + routing + tokenize + queue + prefill + sample + network.**
Under load, *queue* usually dominates; at low load, *prefill* does.

Worked numbers, 8B model in BF16 on one H100 (989 TFLOP/s peak, assume 50% MFU),
2,000-token prompt, no cache hit:
prefill ≈ `2 · 8e9 · 2000 / (989e12 · 0.5)` ≈ **65 ms**. Each later token needs one
decode step ≥ `16 GB / 3.35 TB/s` ≈ **4.8 ms** plus KV reads (Mod 1 Lecture 1.5).
So a lone request sees TTFT ≈ 70–90 ms and ~200 tokens/s; a loaded server adds queue
time to TTFT and stretches each decode step as the batch grows.

## 2. Streaming with Server-Sent Events

LLM APIs stream with SSE over a normal HTTP response:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

data: {"choices":[{"delta":{"content":"Hel"}}]}

data: {"choices":[{"delta":{"content":"lo"}}]}

: keep-alive

data: [DONE]

```

- One event = `data: <payload>` followed by a **blank line** (`\n\n`).
- Lines starting with `:` are comments; send one every ~15 s as a **heartbeat** so idle
  timeouts on load balancers (often 60 s) don't cut a slow generation.
- `data: [DONE]` ends the stream (OpenAI convention).
- **Turn off buffering** in every proxy on the path, or tokens arrive in bursts:
  nginx `proxy_buffering off;` (or the response header `X-Accel-Buffering: no`),
  no gzip on `text/event-stream`, and long read timeouts.

## 3. Disconnects and cancellation

If a user closes the tab, the GPU will otherwise keep decoding up to `max_tokens`
for nobody. That burns batch slots and KV blocks other users are queuing for.

What must happen, end to end:

1. **Detect** the disconnect: the write to the socket fails, or the framework
   reports it (`await request.is_disconnected()` in Starlette/FastAPI).
2. **Propagate** it: every hop must close *its* upstream connection. A proxy that
   keeps reading from vLLM after the client left defeats the engine's own check.
3. **Abort** in the engine: vLLM's OpenAI server aborts a request when its client
   disconnects, which frees the sequence's KV blocks at the next scheduler step.
4. **Count** it: export aborted requests and wasted tokens as metrics.

A proxy that does this correctly (FastAPI + httpx):

```python
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
client = httpx.AsyncClient(base_url="http://vllm:8000", timeout=None)

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    body["stream"] = True

    async def relay():
        async with client.stream("POST", "/v1/chat/completions", json=body) as up:
            async for line in up.aiter_lines():
                if await req.is_disconnected():
                    break                      # leaving the `async with` closes the upstream
                if line:                       # aiter_lines strips the blank separators
                    yield (line + "\n\n").encode()

    return StreamingResponse(relay(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

Starlette also cancels the generator when the client goes away, which unwinds the
`async with` and closes the upstream stream; the explicit check covers servers and
middleware that don't. Exercise 07 makes you build and test this logic without a GPU.

### Other ways a stream ends early

| Case | Do this |
|---|---|
| User presses "stop" (client keeps the connection) | expose a cancel call (`DELETE /requests/{id}`) that aborts by request id |
| Server-side deadline (e.g. 120 s total) | abort and send a final event with `finish_reason: "timeout"` |
| Slow client (reads slower than you generate) | bounded buffer; past the bound, abort rather than hold KV forever |
| Upstream replica dies mid-stream | send an error event and close; **don't** silently retry on another replica, or the user gets a duplicated or inconsistent answer |
| Retry by the client | make it a new request; idempotency keys only make sense before the first token |

## Checklist

- [ ] TTFT broken into queue + prefill + network in your metrics (Lecture 05).
- [ ] Every proxy hop flushes per event (no buffering) and has a read timeout above your longest gap.
- [ ] Heartbeat comment every ~15 s.
- [ ] Client disconnect → upstream closed → engine abort, verified with a test.
- [ ] Aborted-request and wasted-token counters exported.
