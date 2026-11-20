# Quiz Answers — Module 02

| Q | Answer | Why |
|:-:|:-:|---|
| 1 | c | Whole warps, with room for several blocks per SM. 256 is the usual start; tune from there. |
| 2 | b | A warp's 32 × 4-byte reads fall in one 128-byte span → 4 sectors, one request. |
| 3 | c | ~20–30 vs ~400–800 cycles. "1000× faster" is a myth; see Mod 1 Lecture 1.3. |
| 4 | a | Same bank, *different* address, same warp → serialized. Same address is a broadcast, not a conflict (so d is wrong). |
| 5 | d | Lanes are masked off per path; cost ≈ sum of both paths. |
| 6 | a, b, c, d | All four are occupancy caps; the smallest one binds (Mod 1 Lecture 1.4). |
| 7 | a | 12.8% in Boehm's SGEMM worklog (kernel 3). Each thread still does one FMA per two shared loads. |
| 8 | b | 2D blocktiling (8×8 per thread) reached 68.7%. More outputs per thread → more FLOPs per byte loaded. |
| 9 | c | `CUDAExtension` wraps nvcc + the host compiler; `load_inline` JIT-builds for quick experiments. |
| 10 | a | The host returns right after queuing the launch. Time with CUDA events or sync before stopping the clock. |
| 11 | c | Grid first, block second. Easy to flip when new. |
| 12 | a | Register-to-register exchange inside a warp; the building block of warp reductions. |
| 13 | b | It maximizes *theoretical occupancy*, which is not the same as speed (Volkov). Benchmark. |
| 14 | c | Row stride 33 words → lane *i* reading column 0 hits bank *i*. |
| 15 | b | Warm up (several iterations) before timing. CUDA graphs are not involved unless you use them. |
