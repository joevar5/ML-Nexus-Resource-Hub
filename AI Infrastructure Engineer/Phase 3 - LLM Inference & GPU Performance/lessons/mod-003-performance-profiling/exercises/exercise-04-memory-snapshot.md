# Ex 04: Predict, Then Measure, Inference Memory

**Goal:** account for every gigabyte during generation — weights + KV cache +
workspace — and check your formula against the allocator. Practice for
question 3 (KV-cache sizing).
**Time:** ~2 h. **Needs:** NVIDIA GPU.

## Steps

1. **Predict** for your Ex 01 model (read `config.json`): weights bytes, and KV
   bytes per token = `2 · layers · kv_heads · head_dim · 2 bytes`. Predict
   memory after prefill of 2,000 tokens at batch 4, and after 500 more decode steps.
2. **Measure** with the snapshot API:

```python
torch.cuda.memory._record_memory_history(max_entries=200_000)
out = model.generate(**batch, max_new_tokens=500)
torch.cuda.memory._dump_snapshot("gen.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
print(torch.cuda.max_memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30)
```

   Open `gen.pickle` at <https://pytorch.org/memory_viz>.
3. Repeat with `cache_implementation="static"` and compare the shape of the curve.

## Questions (`MEMORY.md`)

- Predicted vs measured: weights, KV at the two points, peak. Within 10%?
- What is the staircase you see with the dynamic cache? (Hint: `torch.cat`
  reallocating the cache every step.) What does the static cache change?
- `memory_reserved − memory_allocated` at the end: how much is allocator cache /
  fragmentation? How does vLLM's PagedAttention avoid this (Lecture 03 deep dive)?

## Pass criteria

KV prediction within 10% of the measured growth, and all three questions answered with numbers.
