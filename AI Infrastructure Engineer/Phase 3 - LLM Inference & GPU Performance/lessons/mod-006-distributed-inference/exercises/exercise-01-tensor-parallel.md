# Ex 01: Tensor Parallelism — What TP=2 Costs and Buys

**Goal:** measure TP on real hardware and connect the result to the numbers
from Ex 02. **Time:** ~3 h. **Needs:** a 2-GPU machine (2× L4 / A10G / A100;
cloud spot instances are fine). Do Ex 02 first.

## Steps

1. Model: `Qwen/Qwen2.5-7B-Instruct` (fits one 24 GB GPU in BF16 with a small KV cache).
2. Three deployments:
   - **A:** TP = 1 on one GPU
   - **B:** TP = 2 (`--tensor-parallel-size 2`)
   - **C:** two TP = 1 replicas, one per GPU, with a round-robin proxy (or two
     ports and a client that alternates)
3. Check the interconnect first: `nvidia-smi topo -m` (NVLink, or PCIe and
   which path?). Write it down.
4. Load test each with `vllm bench serve`: 512-token prompts, 256-token outputs,
   concurrency 1 and 64. Record TTFT p50, TPOT p50, output tok/s, and the KV
   cache blocks each engine reports at start-up.

## Report

| setup | link | KV blocks | TTFT p50 @1 | TPOT p50 @1 | tok/s @64 |
|---|---|---|---|---|---|

## Pass criteria

- All three rows filled for your hardware.
- You explain: why B has the best single-request TPOT (two GPUs' bandwidth on
  one request) but C has the best throughput per GPU when the model fits on one
  GPU (no all-reduce, Lecture 01 "When NOT to parallelize").
- Use `tp_bytes_per_step` from Ex 02 and your link bandwidth to estimate the
  all-reduce time per decode step in B; compare with the measured TPOT gap.
