# Ex 03: Find and Remove Launch Overhead in Decode

**Goal:** prove decode on a small model is CPU/launch-bound, then fix it with
CUDA graphs via `torch.compile`. Answers parts of questions 9 and 14.
**Time:** ~3 h. **Needs:** NVIDIA GPU, PyTorch ≥ 2.3, `transformers` with static cache.

## Steps

1. Profile 20 decode steps in eager mode:

```python
from torch.profiler import profile, ProfilerActivity, schedule
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=2, warmup=3, active=15)) as prof:
    for _ in range(20):
        step()          # one decode forward with past_key_values
        prof.step()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
prof.export_chrome_trace("eager.json")   # open in https://ui.perfetto.dev
```

2. Record: CPU time per step, CUDA time per step, number of kernel launches per step.
3. Switch to a static KV cache and compiled decode:
   `model.generation_config.cache_implementation = "static"` and
   `model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)`.
   Warm up (compilation takes a while), then profile again.
4. Measure tokens/s for 128 new tokens, batch 1, both modes.

## Deliverable: `DIAGNOSIS.md` (5 bullets + a table)

| mode | CPU ms/step | GPU ms/step | launches/step | tokens/s |
|---|---|---|---|---|

## Pass criteria

- Eager mode shows CPU time per step ≥ GPU time per step (launch-bound).
- Compiled mode shows a **≥ 1.5× tokens/s** gain at batch 1, and fewer
  graph-launched kernels visible as a single `cudaGraphLaunch`.
- One bullet explaining why the gain shrinks for a 7B model or batch 64.
