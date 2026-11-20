# Ex 03: Measure Your Fabric with nccl-tests

**Goal:** know your interconnect's real numbers before blaming the model, and
connect them to the TP/EP costs from Mod 6 Ex 02. **Time:** ~2 h.
**Needs:** any machine with ≥ 2 GPUs (multi-node part optional).

## Steps

1. Build: `git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make`
   (add `MPI=1` for multi-node).
2. Topology: `nvidia-smi topo -m` — NVLink (`NV#`), or PCIe through a switch
   (`PIX`/`PXB`) or through the CPU (`PHB`/`SYS`)? Write it down.
3. Single node:
   `./build/all_reduce_perf -b 8 -e 4G -f 2 -g <num_gpus>` and the same for
   `alltoall_perf`.
4. Read the output correctly: **algbw** = size / time; **busbw** = algbw ×
   `2(n−1)/n` for all-reduce, and is the number to compare with link speed.
5. Optional multi-node: the same with `mpirun` across 2 nodes; note the busbw
   drop when traffic crosses InfiniBand/Ethernet.
6. Force the slow path once to see it: `NCCL_P2P_DISABLE=1` (no direct GPU-to-GPU).

## Report

- Plot busbw vs message size for all-reduce and all-to-all (log x-axis). Mark
  where it stops being latency-bound and where it saturates.
- Peak busbw vs the link's spec; the ratio.
- Using `tp_bytes_per_step` from Mod 6 Ex 02, the all-reduce time per decode
  step of a 70B model at TP = your GPU count on *this* fabric, with the
  measured busbw **at that message size** (not the peak).

## Pass criteria

All of the above, plus one sentence: would you run TP across these GPUs, and
why? If anything looks anomalous (busbw far below spec), the `NCCL_DEBUG=INFO`
line that explains it (transport chosen: P2P/NVLink, SHM, or NET).
