"""Ex 07 — an SSE relay that streams tokens, sends heartbeats and cancels on disconnect.
Pure asyncio (no web framework) so it can be tested without a GPU. Then `python check.py`.

Spec for `sse_relay`:
- For each chunk `c` from `upstream`, yield b"data: " + c.encode() + b"\\n\\n".
- If no chunk arrives for `heartbeat_s` seconds, yield the comment b": keep-alive\\n\\n" (and keep waiting).
- When `upstream` ends normally, yield b"data: [DONE]\\n\\n" and stop.
- If `upstream` raises, yield b"event: error\\ndata: upstream failed\\n\\n" and stop (no [DONE]).
- As soon as `client_gone` is set — even while waiting for a slow upstream chunk — stop yielding
  and make sure the upstream is closed: no more tokens may be pulled from it, and its cleanup
  (`finally:` blocks, i.e. the engine abort) must run. Do not send [DONE] to a client that left.

Hint: keep the pending `upstream.__anext__()` in a task and `asyncio.wait` on it together with
`client_gone.wait()` and a timeout. You cannot `aclose()` an async generator while one of its
`__anext__()` calls is still running — cancel and await that task first.
"""
from __future__ import annotations
import asyncio
from typing import AsyncIterator


async def sse_relay(upstream: AsyncIterator[str], client_gone: asyncio.Event,
                    heartbeat_s: float = 15.0) -> AsyncIterator[bytes]:
    raise NotImplementedError
    yield b""  # makes this an async generator; remove when you implement it
