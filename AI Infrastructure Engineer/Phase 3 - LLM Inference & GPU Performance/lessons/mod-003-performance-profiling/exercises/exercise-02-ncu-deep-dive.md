# Ex 02: ncu on a Prefill GEMM vs a Decode GEMV

**Goal:** use Nsight Compute to show, with counters, that the same linear layer
is compute-bound in prefill and memory-bound in decode.
**Time:** ~3 h. **Needs:** NVIDIA GPU, `ncu` (needs admin/perf-counter access;
on cloud VMs use `sudo` or a provider that exposes counters).

## Steps

1. Isolate one MLP projection from your Ex 01 model (e.g. `up_proj`, shape
   `hidden → intermediate`) and run it on:
   - **prefill:** `x` of shape `[1, 512, hidden]`
   - **decode:** `x` of shape `[1, 1, hidden]` and `[32, 1, hidden]`
2. Profile one launch of each:
   `ncu --set full -k regex:gemm|gemv|cutlass -s 3 -c 1 -o proj python proj.py`
3. From the report record: Duration, SM throughput %, DRAM throughput %,
   achieved occupancy, roofline position, top 2 warp stall reasons.
4. Compute by hand: FLOPs = `2·T·d_in·d_out`, bytes ≈ weights `d_in·d_out·2` (+ activations),
   AI = FLOPs / bytes. Compare with your GPU's BF16 ridge point.

## Deliverable: `NCU_REPORT.md`

| case | tokens T | AI (hand) | duration | SM % | DRAM % | bound by | stall #1 |
|---|---|---|---|---|---|---|---|

## Pass criteria

- Hand AI and ncu's roofline agree on the side of the ridge for all three cases.
- Decode T = 1 shows DRAM % ≫ SM %; prefill shows SM (tensor) % high.
- One paragraph: why does T = 32 decode cost almost the same time as T = 1?
  (This is *why batching works*, question 6.)
