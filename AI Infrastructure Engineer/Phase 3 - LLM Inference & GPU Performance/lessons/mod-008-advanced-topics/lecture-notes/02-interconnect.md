# Lecture 02: NVLink + InfiniBand + GPUDirect

## NVLink

NVIDIA proprietary high-speed link between GPUs (and to NVSwitch).
- H100 NVLink 4: 900 GB/s aggregate per GPU (~450 GB/s each way)
- 8 GPUs / NVSwitch on a node share full bandwidth
- vs PCIe Gen4 x16: ~32 GB/s each way; Gen5 x16: ~64 GB/s each way — 7–14× slower

NVLink is what makes TP fast within a node. Without it, TP is a bad idea.

## InfiniBand

Across-node fabric:
- HDR (200 Gb/s) or NDR (400 Gb/s) per port
- Multi-rail for higher bandwidth
- RDMA: zero-copy memory access between nodes

PP across nodes uses IB to avoid copying activations through CPU memory.

## GPUDirect RDMA

GPU → NIC → remote NIC → remote GPU, all without CPU involvement.
Critical for multi-node training; enabled via `nccl` + IB drivers.

## Diagnosing fabric issues

- `ib_send_bw`: measure raw IB bandwidth
- `nccl-tests`: measure all-reduce throughput across GPUs
- `nvidia-smi topo -m`: shows interconnect topology

## When fabric matters

- Single-node TP (training or inference): NVLink topology
- Multi-node training, PP/EP inference, and prefill→decode KV transfer: IB bandwidth + GPUDirect
- Inference with one model replica per GPU: barely matters

---

## Deep dive: how it works (core logic)

### The bandwidth ladder (approximate; check your system)
| Link | Rough bandwidth per GPU |
| :--- | :--- |
| HBM (on-package) | 3–8 TB/s |
| NVLink (H100) | ~900 GB/s total (~450 each way) |
| PCIe Gen5 x16 | ~64 GB/s each way |
| InfiniBand NDR port | ~50 GB/s (400 Gb/s) |
Each step down is roughly 10× slower. A design that fits the NVLink domain is far cheaper than one that spans nodes. Cost model for a message: `time ≈ α (latency) + bytes / β (bandwidth)`; small messages are latency-bound, big ones bandwidth-bound.

### Collectives
- **All-reduce (ring):** reduce-scatter then all-gather; each GPU sends/receives `2(n−1)/n × bytes`. Bandwidth-optimal, latency grows with n. **Tree** algorithms cut latency for small messages. [NCCL](https://github.com/NVIDIA/nccl) picks the algorithm by size and topology.
- **All-to-all:** every GPU sends a different chunk to every other: `(n−1)/n` of its data leaves the GPU. This is the MoE pattern; it stresses bisection bandwidth and is hard to overlap.

### Device-initiated communication ([NCCL device-initiated communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html))
Traditionally the CPU launches a communication kernel, so tiny messages pay launch latency. Device-side APIs let a *running* kernel put/get over NVLink (load/store, hardware multicast) or issue RDMA directly, so comms fuse with compute. This is the basis of low-latency MoE dispatch such as [DeepEP](https://github.com/deepseek-ai/DeepEP).

### Topology matters
[Multi-node NVLink Systems Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/) covers NVLink domains that span multiple nodes (GB200 NVL racks) alongside InfiniBand, which changes where tensor parallelism is affordable. For scale-out Ethernet, see the [Ultra Ethernet 1.0.3 Specification](https://ultraethernet.org/wp-content/uploads/sites/20/2026/08/UE-Specification-1.0.3.pdf). Moving KV state between prefill and decode pools needs a transport layer such as [NIXL](https://github.com/ai-dynamo/nixl), which abstracts memory and network backends.

### Read the originals

- [NCCL](https://github.com/NVIDIA/nccl)
- [NCCL device-initiated communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [Multi-node NVLink Systems Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/)
- [Ultra Ethernet 1.0.3 Specification](https://ultraethernet.org/wp-content/uploads/sites/20/2026/08/UE-Specification-1.0.3.pdf)
- [NIXL](https://github.com/ai-dynamo/nixl)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
