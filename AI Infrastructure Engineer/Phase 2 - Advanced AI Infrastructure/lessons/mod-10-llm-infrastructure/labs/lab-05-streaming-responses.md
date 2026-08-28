# Lab 05: Streaming Responses to Clients

**Duration:** 60 min  **Prerequisites:** vLLM running (lab 01)

## Objective
Stream tokens from vLLM to a client using Server-Sent Events (SSE), demonstrating reduced time-to-first-token vs blocking responses.

## Steps

### 1. Streaming via OpenAI SDK
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

stream = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role":"user","content":"Tell me a story about a debugging session."}],
    max_tokens=400,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
```

### 2. Wrap with a FastAPI proxy that streams to browsers
```python
# proxy.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx, json

app = FastAPI()

@app.post("/stream")
async def stream(req: Request):
    body = await req.json()
    async def gen():
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", "http://localhost:8000/v1/chat/completions",
                                json={"model": body["model"],
                                      "messages": body["messages"],
                                      "stream": True}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

### 3. Run and test with curl
```bash
uvicorn proxy:app --port 9000 &
curl -N -X POST http://localhost:9000/stream \
  -H 'content-type: application/json' \
  -d '{"model":"mistralai/Mistral-7B-Instruct-v0.3",
       "messages":[{"role":"user","content":"Stream me 200 tokens of poetry."}]}'
```

### 4. Browser client (HTML)
```html
<!doctype html><html><body><pre id="out"></pre><script>
const resp = await fetch("/stream", { method:"POST", body: JSON.stringify({
  model: "mistralai/Mistral-7B-Instruct-v0.3",
  messages: [{role:"user", content: "Tell me a long story."}]
})});
const r = resp.body.getReader();
const dec = new TextDecoder();
while (true) {
  const { done, value } = await r.read();
  if (done) break;
  for (const line of dec.decode(value).split("\n")) {
    if (line.startsWith("data: ") && line.slice(6) !== "[DONE]") {
      const j = JSON.parse(line.slice(6));
      document.getElementById("out").textContent += j.choices[0].delta.content || "";
    }
  }
}
</script></body></html>
```

### 5. Measure TTFT
```python
import time, requests
t0 = time.perf_counter()
r = requests.post("http://localhost:9000/stream", stream=True,
                  json={"model":"mistralai/Mistral-7B-Instruct-v0.3",
                        "messages":[{"role":"user","content":"Write a haiku."}]})
for line in r.iter_lines():
    if line and line.startswith(b"data: ") and b"delta" in line:
        print(f"TTFT: {(time.perf_counter()-t0)*1000:.0f}ms")
        break
```
Expect TTFT 100-500ms vs ~3-5s for the full response.

## Validation
- [ ] Tokens visibly stream to the terminal/browser, not arrive in one block.
- [ ] TTFT < 1s on a warm model.
- [ ] Cancelling the request mid-stream stops generation (vLLM detects client disconnect).

## Cleanup
```bash
pkill -f uvicorn
```

## Troubleshooting
- **Whole response arrives at once** — Reverse proxy is buffering. Disable for the SSE path (nginx: `proxy_buffering off;`).
- **`event: error` mid-stream** — Look at vLLM logs; usually OOM or rate-limit.
- **Connection drops after 60s** — Default proxy idle timeout. Bump to several minutes for long generations.
