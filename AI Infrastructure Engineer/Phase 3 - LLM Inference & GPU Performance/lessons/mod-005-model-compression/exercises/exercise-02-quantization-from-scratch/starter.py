"""Exercise 02 — quantization from scratch (numpy, CPU). Fill in the TODOs, then `python check.py`.

Conventions
- Weights W have shape [out_features, in_features]; activations X have shape [tokens, in_features];
  the layer computes Y = X @ W.T.
- Symmetric quantization: scale = absmax / qmax, qmax = 2**(bits-1) - 1, q = clip(round(x / scale), -qmax, qmax).
- "per_row=True" means one scale per row: per output channel for W, per token for X.
- If an absmax is 0, use scale = 1 for that row (avoid dividing by zero).
"""
from __future__ import annotations
import numpy as np


def quantize(x: np.ndarray, bits: int, per_row: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (q, scale). q is an int array; scale is a scalar array (per_row=False) or shape [rows, 1]."""
    raise NotImplementedError


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def fake_quant(x: np.ndarray, bits: int, per_row: bool = False) -> np.ndarray:
    """quantize then dequantize — the value the matmul actually sees."""
    raise NotImplementedError


def fake_quant_groupwise(w: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    """Per-row AND per-group: each run of `group_size` consecutive input columns in a row gets its own
    scale (this is how AWQ/GPTQ int4 checkpoints store weights). in_features is a multiple of group_size."""
    raise NotImplementedError


def smoothquant_scales(x_absmax: np.ndarray, w_absmax: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Per input channel j: s_j = x_absmax_j**alpha / w_absmax_j**(1 - alpha).
    x_absmax: max |X[:, j]| over calibration tokens; w_absmax: max |W[:, j]| over output rows."""
    raise NotImplementedError


def relative_error(ref: np.ndarray, approx: np.ndarray) -> float:
    """Frobenius-norm relative error ||ref - approx|| / ||ref||."""
    raise NotImplementedError
