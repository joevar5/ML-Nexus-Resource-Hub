"""Grades starter.sse_relay. Prints PASS when every scenario behaves as specified."""
import asyncio
import sys
import time
import starter as s


class FakeEngine:
    """An upstream token stream that records how many tokens were pulled and whether it was closed."""
    def __init__(self, tokens=None, delay=0.0, first_delay=0.0, fail_after=None):
        self.tokens, self.delay, self.first_delay, self.fail_after = tokens, delay, first_delay, fail_after
        self.pulled = 0
        self.closed = False

    async def stream(self):
        try:
            i = 0
            while self.tokens is None or i < len(self.tokens):
                await asyncio.sleep(self.first_delay if i == 0 else self.delay)
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("GPU fell off the bus")
                self.pulled += 1
                yield self.tokens[i] if self.tokens else f"t{i}"
                i += 1
        finally:
            self.closed = True       # stands in for "engine.abort(request_id)"


async def collect(gen, gone=None, stop_after=None, limit_s=2.0):
    out, t0 = [], time.monotonic()
    async for chunk in gen:
        out.append(chunk)
        if stop_after is not None and len([c for c in out if c.startswith(b"data: ")]) == stop_after:
            gone.set()
        if time.monotonic() - t0 > limit_s:
            raise AssertionError("relay did not finish in time")
    return out, time.monotonic() - t0


async def normal():
    e, gone = FakeEngine(["Hel", "lo", "!"]), asyncio.Event()
    out, _ = await collect(s.sse_relay(e.stream(), gone, heartbeat_s=1.0))
    assert out == [b"data: Hel\n\n", b"data: lo\n\n", b"data: !\n\n", b"data: [DONE]\n\n"], out


async def heartbeat():
    e, gone = FakeEngine(["x"], first_delay=0.35), asyncio.Event()
    out, _ = await collect(s.sse_relay(e.stream(), gone, heartbeat_s=0.1))
    hb = [c for c in out if c == b": keep-alive\n\n"]
    assert len(hb) >= 2, f"expected >= 2 heartbeats during a 0.35 s wait, got {len(hb)}: {out}"
    assert out[-2:] == [b"data: x\n\n", b"data: [DONE]\n\n"], out


async def disconnect_fast_stream():
    e, gone = FakeEngine(None, delay=0.01), asyncio.Event()
    out, dt = await collect(s.sse_relay(e.stream(), gone, heartbeat_s=1.0), gone, stop_after=5)
    await asyncio.sleep(0.05)
    assert e.closed, "upstream was not closed after the client left (engine keeps generating)"
    assert e.pulled <= 7, f"pulled {e.pulled} tokens; should stop right after the client left"
    assert b"data: [DONE]\n\n" not in out, "sent [DONE] to a client that disconnected"


async def disconnect_while_upstream_blocked():
    e, gone = FakeEngine(None, delay=10.0), asyncio.Event()
    e.first_delay = 0.0
    out, dt = await collect(s.sse_relay(e.stream(), gone, heartbeat_s=5.0), gone, stop_after=1)
    assert dt < 0.5, f"relay took {dt:.2f} s to stop; it must not wait for the next upstream token"
    await asyncio.sleep(0.05)
    assert e.closed, "upstream not closed when the client left during a slow generation step"


async def upstream_error():
    e, gone = FakeEngine(["a", "b", "c"], fail_after=2), asyncio.Event()
    out, _ = await collect(s.sse_relay(e.stream(), gone, heartbeat_s=1.0))
    assert out == [b"data: a\n\n", b"data: b\n\n", b"event: error\ndata: upstream failed\n\n"], out


fails = []
for name, fn in [("normal stream + [DONE]", normal), ("heartbeat while waiting", heartbeat),
                 ("client disconnects mid-stream", disconnect_fast_stream),
                 ("client disconnects while upstream is blocked", disconnect_while_upstream_blocked),
                 ("upstream error event", upstream_error)]:
    try:
        asyncio.run(asyncio.wait_for(fn(), 5))
    except NotImplementedError:
        fails.append(f"  {name}: not implemented")
    except AssertionError as e:
        fails.append(f"  {name}: {e}")
    except TimeoutError:
        fails.append(f"  {name}: did not finish within 5 s (stuck waiting on the upstream?)")
    except Exception as e:  # noqa: BLE001
        fails.append(f"  {name}: raised {type(e).__name__}: {e}")
if fails:
    print(f"FAIL ({len(fails)}/5)"); print("\n".join(fails)); sys.exit(1)
print("PASS (5/5)")
