# Ex 03: Tree Reduction

**Goal:** implement a sum of N floats three ways and explain each gap with the
memory hierarchy. This is the Harris reduction deck in miniature
([Mod 1 digest §2](../../mod-001-gpu-fundamentals/source-digest.md)).
**Time:** ~3 h. **Needs:** NVIDIA GPU.

## Variants

1. **Atomic baseline:** every thread does `atomicAdd(out, x[i])`. (Shows why
   contended atomics serialize.)
2. **Block reduction in shared memory, sequential addressing:** each block
   loads 2 elements per thread (first add during load), reduces with
   `if (tid < s) sdata[tid] += sdata[tid + s]`, writes one partial sum; a
   second launch (or one `atomicAdd` per block) combines partials.
3. **Warp shuffle:** grid-stride loop accumulates many elements per thread,
   then `warp_sum` with `__shfl_down_sync`, then one shared slot per warp,
   then the first warp reduces those. One `atomicAdd` per block.

```cuda
__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
```

## Measure

N ∈ {1M, 64M}. Bandwidth = `4·N / time`. Compare to `torch.sum` and to the
copy ceiling from Ex 01.

## Pass criteria

- Results match `torch.sum(x.double())` to `rtol=1e-4` (FP32 sums differ in
  the last bits by reduction order; explain why in one line).
- At N = 64M, variant 3 reaches **≥ 80% of `torch.sum`** bandwidth.
- A table of the three variants with GB/s and a one-line reason for each gap.
