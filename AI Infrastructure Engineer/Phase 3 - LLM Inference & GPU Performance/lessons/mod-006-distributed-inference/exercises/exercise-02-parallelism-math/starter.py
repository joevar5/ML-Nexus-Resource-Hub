"""Ex 02 — parallelism math (CPU-only). Fill in every function, then `python check.py`.
Units: bytes and seconds; link bandwidth in GB/s with GB = 1e9 bytes."""
from __future__ import annotations


def ring_allreduce_bytes_per_gpu(tensor_bytes: float, n: int) -> float:
    """Bytes each GPU sends in a ring all-reduce of a tensor of `tensor_bytes` over n GPUs (0 if n == 1)."""
    raise NotImplementedError


def tp_bytes_per_step(tokens: int, hidden: int, layers: int, tp: int, bytes_per_elem: float = 2) -> float:
    """Per-GPU all-reduce traffic for one forward step under Megatron-style tensor parallelism:
    2 all-reduces per layer (attention + MLP), each on a [tokens, hidden] activation."""
    raise NotImplementedError


def comm_time_s(num_bytes: float, link_gbs: float) -> float:
    """Bandwidth-only transfer time (ignore latency)."""
    raise NotImplementedError


def pp_bubble_fraction(stages: int, microbatches: int) -> float:
    """Idle fraction of a GPipe-style pipeline: (p - 1) / (m + p - 1)."""
    raise NotImplementedError


def moe_alltoall_bytes_per_gpu(tokens_per_gpu: int, hidden: int, top_k: int, ep: int,
                               bytes_per_elem: float = 2) -> float:
    """Bytes one GPU sends for one MoE layer: dispatch (each token's activation to each of its top_k
    experts) plus combine (the results coming back). With experts spread evenly over `ep` GPUs, a
    fraction (ep - 1) / ep of the copies leave the GPU."""
    raise NotImplementedError


def weight_bytes_per_gpu(params: float, bytes_per_param: float, tp: int, pp: int) -> float:
    """Weights held by one GPU when the model is split tp-ways inside layers and pp-ways across layers."""
    raise NotImplementedError
