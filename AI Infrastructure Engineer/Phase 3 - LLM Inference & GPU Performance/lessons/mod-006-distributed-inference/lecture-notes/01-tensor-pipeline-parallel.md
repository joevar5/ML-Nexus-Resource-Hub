# Lecture 01: Tensor + Pipeline Parallelism

## Tensor parallel (TP)

Split each layer's weights across N GPUs. Each GPU computes its slice; an
all-reduce combines results.

- **Pros**: low latency (no pipeline bubbles)
- **Cons**: requires fast interconnect (NVLink); communication scales with batch × seq

Use for: 7B-70B models on a single node (8 GPUs).

## Pipeline parallel (PP)

Split model into N sequential stages, each on a different GPU. Process a
mini-batch as several "micro-batches" so all stages stay busy.

- **Pros**: communication is small (only activations between stages)
- **Cons**: bubbles at start + end; latency adds up

Use for: 100B+ models across multiple nodes.

## Combined

Frontier models use 3D parallelism: TP within a node, PP across nodes, plus
data parallel for replicas.

In vLLM, `--tensor-parallel-size 8 --pipeline-parallel-size 2` serves a dense
model across two 8-GPU nodes; MoE models add expert parallelism
(`--enable-expert-parallel`), see the deep dive and question 23.

## When NOT to parallelize

A model that fits on one GPU (or one quantized GPU) is faster + cheaper to
serve replicated. Don't tensor-parallel a 7B model across 2 GPUs to brag.

---

## Deep dive: how it works (core logic)

### Tensor parallelism: split inside the layer ([Megatron-LM](https://arxiv.org/abs/1909.08053))
Take an MLP: `Y = GeLU(X·A)`, `Z = Y·B`.
- Split `A` by **columns** across GPUs: each GPU computes its slice of `Y` independently. GeLU is elementwise, so no communication is needed yet.
- Split `B` by **rows**: each GPU multiplies its `Y` slice by its rows of `B`, producing a *partial sum* of `Z`.
- One **all-reduce** adds the partial sums.
Attention is the same idea: split the heads across GPUs, then all-reduce after the output projection. Total: **2 all-reduces per layer** in the forward pass.

Cost: a ring all-reduce moves ≈ `2·(n−1)/n × tensor_bytes` per GPU. For a batch of T tokens with hidden size h in FP16, the tensor is `T·h·2` bytes, every layer, twice. That is only affordable over NVLink (hundreds of GB/s), not PCIe or Ethernet. So **TP stays inside a node** ([NCCL](https://github.com/NVIDIA/nccl) implements the collectives).

### Pipeline parallelism: split across layers
Give each GPU a contiguous group of layers; activations flow between stages (small point-to-point sends, cheap even across nodes). To keep stages busy, split the batch into *m* micro-batches. Idle "bubble" fraction ≈ `(p−1)/(m+p−1)` for p stages; more micro-batches shrink it. In serving, PP adds per-token latency (a token passes through all stages), so use it only when the model doesn't fit otherwise.

### Expert parallelism for MoE
An MoE layer has E experts; a router picks top-k per token. Put different experts on different GPUs ([GShard](https://arxiv.org/abs/2006.16668)):
1. Router assigns each token to its experts, subject to a per-expert **capacity** (overflow tokens are dropped or rerouted).
2. **All-to-all dispatch:** every GPU sends tokens to the GPUs holding their experts.
3. Experts compute.
4. **All-to-all combine:** results return and are weighted-summed.
Two all-to-alls per MoE layer; cost grows with tokens and with cross-node distance. [DeepSeek-V3](https://arxiv.org/html/2412.19437) shows a full model-system design (routed + shared experts, balanced routing); [MegaScale-Infer](https://arxiv.org/abs/2504.02263) covers serving MoE at scale, with communication overlapped with compute.

### Choosing
Fits in one GPU → don't parallelize. Doesn't fit → TP across the NVLink domain; if still too big or MoE → EP across nodes; PP only if needed. Replicate (data parallel) the whole arrangement for throughput.

### Read the originals

- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [NCCL](https://github.com/NVIDIA/nccl)
- [GShard](https://arxiv.org/abs/2006.16668)
- [DeepSeek-V3](https://arxiv.org/html/2412.19437)
- [MegaScale-Infer](https://arxiv.org/abs/2504.02263)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
