# Ex 05: Cold Start — Break It Down, Then Cut It

**Goal:** time every stage of starting a vLLM replica and remove the biggest
ones. Feeds the T_start you used in Ex 03. **Time:** ~3 h. **Needs:** one GPU
with ≥ 24 GB, Docker.

## Steps

1. Baseline, from nothing cached: `docker pull vllm/vllm-openai:<version>`,
   then `vllm serve Qwen/Qwen2.5-7B-Instruct` downloading weights from the Hub.
   Time: image pull, weight download, "Loading weights took …", CUDA graph
   capture / compile, first request served (poll `/health` then send one request).
2. Mitigations, one at a time, re-timing each:
   - Weights pre-downloaded to local NVMe (`HF_HUB_OFFLINE=1`, mounted cache).
   - Image already present (what a pre-pull DaemonSet gives you).
   - `--enforce-eager` (skips CUDA graph capture: faster start, slower decode —
     measure both sides).
   - Faster loading: `--load-format runai_streamer` or `fastsafetensors` if your
     vLLM version supports it.
3. For each, also record steady-state TPOT so you see what you traded.

## Report

| setup | image | weights | load | warmup | total to first token | steady TPOT |
|---|---|---|---|---|---|---|

## Pass criteria

- Baseline and ≥ 3 mitigations measured.
- Total cold start cut by ≥ 5× from the baseline.
- One paragraph: which mitigations you'd use in production and their cost
  (disk, memory, image size, decode speed).
