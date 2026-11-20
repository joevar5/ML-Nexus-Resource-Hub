# Source Digest — Transformer Optimization

An **advanced layer** on top of the lecture notes. What the papers actually say, as steps and numbers. I read each paper's text (abstract and the sections quoted); where something wasn't extractable I say so.

---

## 1. Online softmax: the trick behind FlashAttention
**Source:** [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, NVIDIA)

1. **Problem:** the safe softmax needs **three passes** over the row (max, sum, normalize), so three memory reads.
2. **Idea:** compute the running max **and** the running sum in **one pass**; when the max grows, rescale the old sum by `exp(old_max − new_max)`.
3. **Result:** fewer memory accesses → **up to 1.3× faster softmax**, and **up to 5× for Softmax+TopK fused**.
4. **Why it matters:** this is the identity that lets attention be computed block by block without ever holding a full row.

---

## 2. FlashAttention (v1): be IO-aware
**Source:** [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al.)

1. **Thesis:** approximate attention rarely gave real speedups because it ignored **reads/writes between GPU memory levels**. Make the algorithm **IO-aware**.
2. **Method:** **tile** Q, K, V into blocks that fit in on-chip SRAM, using the online-softmax trick; never write the N×N matrix to HBM. Backward pass **recomputes** attention from saved statistics instead of storing it.
3. **Guarantees:** exact attention, fewer HBM accesses than standard, provably optimal over a range of SRAM sizes; memory becomes linear in sequence length.
4. **Results reported:** **15%** faster end-to-end BERT-large (seq 512) vs the MLPerf 1.1 record; **3×** on GPT-2 (seq 1K); **2.4×** on Long-Range Arena (1K–4K); better perplexity from longer context (0.7 on GPT-2); first Transformers above chance on Path-X (16K tokens, 61.4%) and Path-256 (64K, 63.1%).
5. **Extension:** block-sparse FlashAttention, faster than other approximate methods.

---

## 3. FlashAttention-2: fix the work partitioning
**Source:** [FlashAttention-2](https://arxiv.org/abs/2307.08691)

1. **Starting point:** v1 reached only **25–40%** of theoretical FLOPs/s, well below GEMM efficiency.
2. **Cause:** poor work partitioning across blocks and warps → low occupancy or extra shared-memory traffic.
3. **Three fixes:**
   - **Fewer non-matmul FLOPs.** On A100: **312 TFLOPs/s** for FP16 matmul vs only **19.5 TFLOPs/s** for FP32 non-matmul, so *each non-matmul FLOP is ~16× more expensive*. The algorithm is tweaked (e.g. keep only the row-wise logsumexp, delay rescaling) to spend time on matmuls.
   - **Parallelize over the sequence length**, even for one head: the outer loop over row blocks is embarrassingly parallel (loop order swapped, an idea first used in Triton), raising occupancy.
   - **Split work between warps inside a block** to reduce shared-memory communication.
4. **Result:** ~**2×** over v1, **50–73%** of peak on A100, up to **225 TFLOPs/s** per A100 in end-to-end GPT training (72% MFU).

---

## 4. FlashAttention-3: use Hopper's asynchrony
**Source:** [FlashAttention-3](https://arxiv.org/abs/2407.08608)

1. **Starting point:** FA-2 achieved only **35%** utilization on H100 (vs 80–90% for GEMM).
2. **Three techniques:**
   - **Warp specialization:** producer warps move data with TMA; consumer warps run tensor-core MMAs; overlapped.
   - **Interleave matmul and softmax** (ping-pong scheduling between warpgroups) so the slow softmax hides behind the next matmul.
   - **FP8:** block quantization + **incoherent processing** (a random orthogonal transform that spreads outliers) to keep accuracy.
3. **Hardware catch:** FP8 WGMMA needs *k-major* layouts, which clash with the FP32 accumulator layout when you chain two GEMMs in one kernel, so the algorithm is reshaped.
4. **Results:** FP16 **1.5–2.0×** faster than FA-2, up to **740 TFLOPs/s (75% utilization)**; FP8 close to **1.2 PFLOPs/s**, with **2.6× lower numerical error** than a baseline FP8 attention.

*(FlashAttention-4 and FlashInfer are not read; see "Not read" below.)*

---

## 5. Shrinking the KV cache by model design
**Sources:** [GQA](https://arxiv.org/abs/2305.13245), [DeepSeek-V2](https://arxiv.org/abs/2405.04434)

**GQA (Ainslie et al.):**
1. Decoder inference is bottlenecked by loading weights **and all keys/values every step**.
2. **MQA** (one KV head) cuts that sharply but can hurt quality and training stability.
3. **GQA** uses an intermediate number of KV heads (more than 1, fewer than query heads). GQA with groups = number of heads is regular multi-head.
4. **Recipe:** convert an existing multi-head checkpoint by mean-pooling heads, then **uptrain with ~5% of the original pre-training compute**.
5. Result: quality close to multi-head attention with speed comparable to MQA.

**DeepSeek-V2 (MLA):** *(only the abstract's figures were extractable)*
1. **Multi-head Latent Attention** compresses the KV cache into a **latent vector**.
2. Reported: **KV cache −93.3%**, **max generation throughput 5.76×**, **42.5% lower training cost** vs DeepSeek 67B.
3. Model: **236B total, 21B active** per token, **128K context** (MoE with DeepSeekMoE).

---

## 6. Speculative decoding: the original math
**Sources:** [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html) (arXiv 2211.17192), [Chen et al.](https://arxiv.org/abs/2302.01318) (DeepMind)

1. **Observation:** decode is memory-bound, so *scoring several tokens in parallel costs about the same as scoring one*.
2. **Procedure:** a cheap draft model proposes γ tokens → the target scores all in one pass → accept left to right with a **modified rejection sampling** rule → on first rejection, sample from the **adjusted distribution `p′ = norm(max(0, p − q))`**. Output distribution is **identical** to the target's.
3. **Acceptance:** `E[min(1, p/q)] = Σ min(p, q)`. Call this overlap α.
4. **Expected tokens per target call:** `(1 − α^(γ+1)) / (1 − α)` (a capped geometric variable).
5. **Walltime improvement (Theorem 3.8):** `(1 − α^(γ+1)) / ((1 − α)(γc + 1))`, where *c* = draft cost ÷ target cost.
6. **Results:** Leviathan: **2–3×** on T5-XXL with identical outputs, no retraining. Chen: **2–2.5×** on Chinchilla 70B in a distributed setup.

---

## 7. Drafts without a separate model
**Sources:** [Medusa](https://arxiv.org/abs/2401.10774), [EAGLE](https://arxiv.org/abs/2401.15077), [EAGLE-3](https://arxiv.org/abs/2503.01840)

**Medusa:**
1. Extra decoding **heads** on the target predict several following tokens in parallel; no separate draft model to maintain.
2. A **tree-based attention** mask verifies many candidate continuations in one step.
3. **Medusa-1:** heads fine-tuned on a **frozen** backbone → lossless. **Medusa-2:** fine-tuned **jointly** with the backbone → better predictions and speed, but needs a special training recipe to preserve capabilities. Extensions include self-distillation when no training data is available.

**EAGLE:**
1. Two observations: autoregression at the **feature level** (second-to-top-layer) is easier than at the token level, but feature uncertainty limits it.
2. Fix: feed the draft head a **token sequence advanced by one time step**, which resolves the uncertainty.
3. Result on LLaMA2-Chat 70B: **2.7×–3.5× latency speedup, ~2× throughput**, same output distribution.

**EAGLE-3:**
1. Finding: more training data helped EAGLE little, because of the feature-prediction constraint.
2. Changes: **direct token prediction** instead of features, and **fusion of low/mid/high-level features** from several layers, trained with **"training-time test"** (the draft is trained on its own multi-step outputs).
3. Results: speedup up to **6.5×**, ~**1.4×** better than EAGLE-2; in **SGLang** at batch size 64, **1.38× throughput**.

**Caveat visible in the numbers:** the throughput gain at batch 64 (1.38×) is far smaller than the single-stream speedup (up to 6.5×). Speculation helps most at low batch.

---

## Not read
[FlashAttention-4](https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf), [FlashInfer](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html), the [Triton paper](https://eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf), [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html), the exact MLA equations in DeepSeek-V2, and Medusa's speedup tables. The lecture-note deep dive describes these from general knowledge, so treat its details as unverified until you open the source.
