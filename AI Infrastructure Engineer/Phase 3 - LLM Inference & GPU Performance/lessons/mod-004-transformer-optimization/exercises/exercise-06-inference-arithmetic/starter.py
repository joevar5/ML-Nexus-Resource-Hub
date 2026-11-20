"""Exercise 06 — inference arithmetic. Fill in every function, then run `python check.py`.

Units: bytes, seconds, GB = 1e9 bytes (vendor convention for bandwidth/capacity).
"""
from __future__ import annotations


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per_elem: float) -> float:
    """Bytes of K and V cached for ONE token across all layers (one sequence)."""
    raise NotImplementedError


def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, bytes_per_elem: float,
                   tokens_per_seq: int, num_seqs: int) -> float:
    """Total KV cache for `num_seqs` sequences of `tokens_per_seq` tokens each."""
    raise NotImplementedError


def decode_step_floor_s(weight_bytes: float, kv_bytes_read: float, bandwidth_gbs: float) -> float:
    """Lower bound on one decode step: every weight and every cached KV byte read once from HBM."""
    raise NotImplementedError


def max_concurrent_seqs(gpu_mem_gb: float, weight_bytes: float, kv_bytes_per_seq: float,
                        reserve_frac: float = 0.1) -> int:
    """How many full-length sequences fit after weights, keeping `reserve_frac` of memory free
    for activations/workspace. Return 0 if the weights alone do not fit."""
    raise NotImplementedError


def spec_expected_tokens(alpha: float, gamma: int) -> float:
    """Expected tokens per target forward pass with acceptance rate alpha and draft length gamma.
    Handle alpha == 1 (every draft accepted)."""
    raise NotImplementedError


def spec_speedup(alpha: float, gamma: int, c: float) -> float:
    """Walltime speedup vs plain decoding; c = draft step cost / target step cost."""
    raise NotImplementedError


def best_gamma(alpha: float, c: float, max_gamma: int = 10) -> int:
    """gamma in 1..max_gamma with the highest speedup (smallest gamma on ties)."""
    raise NotImplementedError
