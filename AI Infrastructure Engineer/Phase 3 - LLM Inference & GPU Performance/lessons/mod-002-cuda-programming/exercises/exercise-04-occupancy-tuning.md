# Ex 04: Occupancy Tuning — Predict, Then Measure

**Goal:** test the Mod 1 occupancy model against the real hardware, and see
for yourself that higher occupancy is not always faster.
**Time:** ~3 h. **Needs:** NVIDIA GPU, your K3/K4 kernels from Ex 02,
your occupancy calculator from Mod 1 Ex 03.

## Steps

1. Compile with `-Xptxas -v` and record registers/thread and shared bytes/block
   for K3 (TS = 16, 32) and K4.
2. **Predict** theoretical occupancy for each with your Mod 1 calculator (use
   your GPU's limits: `torch.cuda.get_device_properties(0)` gives SM count;
   max warps/SM is 64 on A100/H100, 48 on L4/A10G/RTX 30–40, 32 on T4).
3. Ask the runtime: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (via a tiny
   C++ helper in your `load_inline` source) — does it agree with step 2?
4. **Measure** achieved occupancy and time with
   `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum`.
5. Sweep: for K4, force register limits with `__launch_bounds__(512, minBlocks)`
   for minBlocks ∈ {1, 2, 3}; record registers, spills (`-Xptxas -v`), occupancy, GFLOPS.

## Deliverable: `OCCUPANCY_REPORT.md`

| kernel | regs | smem | predicted occ. | runtime API occ. | achieved occ. | GFLOPS | spills |
|---|---|---|---|---|---|---|---|

## Pass criteria

- Predicted and runtime-API occupancy agree for every row (if not, explain:
  register allocation granularity, 1 KB reserved shared memory per block).
- At least one case where raising occupancy did **not** raise GFLOPS, with the reason.
- A recommendation: the configuration you would ship, and why.
