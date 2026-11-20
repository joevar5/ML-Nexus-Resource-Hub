# Ex 04: MIG — Many Small Models on One Big GPU

**Goal:** decide when partitioning one A100/H100 beats giving a small model a
whole GPU. **Time:** ~3 h. **Needs:** an A100 or H100 with admin rights
(MIG is not available on L4/A10G/consumer GPUs).

## Steps

1. Baseline: one `Qwen2.5-1.5B-Instruct` vLLM instance on the full GPU; measure
   max goodput (TTFT p99 ≤ 500 ms, TPOT p99 ≤ 50 ms) with `vllm bench serve`.
2. Enable MIG and create e.g. three `2g.20gb` (A100 80GB) or equivalent
   instances: `nvidia-smi -i 0 -mig 1`, `nvidia-smi mig -cgi <profile ids> -C`,
   `nvidia-smi -L` to list the MIG device UUIDs.
3. Run one vLLM instance per slice (`CUDA_VISIBLE_DEVICES=<MIG-uuid>`), each
   with `--gpu-memory-utilization 0.9`. Measure goodput of each and the sum.
4. Isolation test: saturate slice 1 with a heavy load while slice 2 serves a
   light steady load; record slice 2's TTFT/TPOT before and during.
5. Compare with time-slicing (no MIG, three processes on the full GPU) under
   the same test.

## Report

| setup | instances | total goodput | p99 TPOT of the light tenant under a noisy neighbour |
|---|---|---|---|

## Pass criteria

- Table filled for full-GPU, MIG and time-slicing.
- Evidence that MIG isolates the light tenant and time-slicing does not.
- A recommendation: for which model sizes and tenancy needs MIG is worth the
  lost flexibility (no slice can use the whole GPU's memory or bandwidth).
