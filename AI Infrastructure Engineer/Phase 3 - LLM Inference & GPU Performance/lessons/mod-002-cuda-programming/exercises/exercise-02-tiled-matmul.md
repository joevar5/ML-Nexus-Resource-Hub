# Ex 02: The SGEMM Ladder

**Goal:** climb the first rungs of the matmul ladder from the lecture deep dive
and *measure* each step's gain, as in Boehm's worklog.
**Time:** ~6 h. **Needs:** NVIDIA GPU. Read Lecture 02 and the
[Mod 2 digest §1](../source-digest.md) first.

## Steps

Square FP32 matrices, N = 4096 (also test N = 1000 for bounds handling).
Compare against `torch.matmul` with TF32 **off**, so cuBLAS runs plain SGEMM:
`torch.backends.cuda.matmul.allow_tf32 = False`.

1. **K1 naive:** one thread per C element, `threadIdx.x` → row (deliberately uncoalesced).
2. **K2 coalesced:** swap the mapping so `threadIdx.x` walks along a row of C/B.
3. **K3 shared-memory tiles:** the `matmul_tiled` kernel from Lecture 02, tile
   size TS ∈ {8, 16, 32} (TS×TS threads, so TS = 64 is impossible: 4,096 > 1,024
   threads per block). Add bounds checks so N = 1000 works.
4. **K4 register tiling (1D):** each thread computes TM = 8 outputs in a column,
   keeping the 8 accumulators in registers. Block tile 64×64, BK = 8, 512 threads.
5. **Stretch, K5 2D register tiling:** each thread computes an 8×8 patch (BM = BN = 128, BK = 8, 256 threads).

Starter for K3 (fill the TODOs):

```cuda
template <int TS>
__global__ void sgemm_smem(int N, const float* A, const float* B, float* C) {
    __shared__ float As[TS][TS], Bs[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y, col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (N + TS - 1) / TS; ++t) {
        // TODO: load As/Bs with bounds checks (write 0 when out of range)
        __syncthreads();
        // TODO: inner product over k < TS
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}
```

## Measure

- GFLOPS = `2·N³ / time`. Report % of cuBLAS.
- For K3, run `ncu --section MemoryWorkloadAnalysis --section Occupancy` and record
  achieved occupancy and `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` counts.

## Pass criteria

- Every kernel matches `torch.matmul` with `atol=1e-3, rtol=1e-3` at N = 1000 and 4096.
- A table K1 → K4 (→ K5) with GFLOPS and % of cuBLAS. Expect roughly
  K1 ~1–2%, K2 ~8%, K3 ~10–15%, K4 ~30–40%, K5 ~60–70% (GPU-dependent).
- **K4 ≥ 30% of cuBLAS; K5 (if done) ≥ 50%.**
- One sentence per step naming the bottleneck it removed.
