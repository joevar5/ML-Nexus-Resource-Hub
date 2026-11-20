# Lecture 02: Kernel Fusion

## Why fusion matters

A transformer block has ~30 ops. Each unfused op = a kernel launch + a global
memory round-trip. Fusion combines multiple ops into one kernel — fewer
launches, intermediate values stay in registers / shared memory.

## How to get fusion

| Approach | When |
|---|---|
| `torch.compile` | first try; fuses what it can automatically |
| `torch.jit.script` (older) | legacy systems |
| Triton | write fused kernel in Python |
| Hand-written CUDA | last resort; for hottest kernels |

## torch.compile

```python
model = torch.compile(model, mode="reduce-overhead")
```

Every mode fuses ops through the Inductor backend (which emits Triton kernels).
`mode="reduce-overhead"` additionally captures **CUDA graphs** to remove launch
overhead (big win for small-batch decode); `mode="max-autotune"` also
benchmarks kernel configurations. Often 1.3–2× speedup with one line.

## Triton example

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

Triton's strength: write GPU-fast kernels in Python; let the compiler handle
launch config + memory layout. Used by `torch.compile` (Inductor), vLLM and SGLang.

## Common fusion opportunities

- Elementwise chains: `bias + GELU + dropout` → one kernel
- LayerNorm + linear projection
- Attention forward (FlashAttention does this)
- Softmax + dropout

---

## Deep dive: how it works (core logic)

### Why fusion wins: count the bytes
RMSNorm: `y = x · rsqrt(mean(x²) + ε) · w`. Unfused PyTorch eager runs separate kernels (square, mean, add-eps/rsqrt, multiply, multiply by weight). Each one **reads and writes the whole tensor in HBM**. For an N-byte tensor that is several N of reads and writes. The math is trivial (about 3 FLOPs per element) so the op is purely bandwidth-bound. A fused kernel reads `x` once, does everything in registers, writes `y` once = **2N bytes**. Speed-up ≈ the ratio of bytes moved.

### A Triton kernel, line by line ([Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html), [Triton programming guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html))
```python
@triton.jit
def rmsnorm(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row  = tl.program_id(0)                       # one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row*stride + cols, mask=mask, other=0.).to(tl.float32)
    var = tl.sum(x*x, axis=0) / N                 # reduction stays on-chip
    y = x * tl.rsqrt(var + eps) * tl.load(w_ptr + cols, mask=mask)
    tl.store(y_ptr + row*stride + cols, y, mask=mask)
```
Triton's model: you write the program for one **block of data**; the compiler handles thread mapping, coalescing and shared memory. Accumulate in FP32 even with FP16/BF16 inputs. Details of the design are in the [Triton paper](https://eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf); the code and tests are in the [Triton repository](https://github.com/triton-lang/triton).

### What to fuse in a transformer
Elementwise + normalization + residual adds, bias+activation (GeLU/SwiGLU) after a matmul (fuse in the epilogue), RoPE + QKV split, and attention itself (FlashAttention is the ultimate fusion: QKᵀ, softmax and ·V in one kernel). `torch.compile` automates many of the simple ones.

### The hidden cost: numerics ([Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/))
A fused or split reduction changes addition order, hence the last bits of the result. If the reduction strategy depends on batch size, the same request produces different logits depending on its batch-mates. The remedy is batch-invariant RMSNorm, matmul and attention: fixed reduction order regardless of batch, at a modest speed cost. Decide whether you need bit-exact reproducibility (evals, debugging, RL) before you pick fused kernels.

### Read the originals

- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton programming guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- [Triton paper](https://eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [Triton repository](https://github.com/triton-lang/triton)
- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
