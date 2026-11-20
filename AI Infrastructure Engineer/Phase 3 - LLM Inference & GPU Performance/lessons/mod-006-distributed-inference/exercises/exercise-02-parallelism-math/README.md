# Ex 02: Parallelism Math — TP, PP and EP Costs (CPU-only, auto-graded)

**Goal:** be able to say, with numbers, why TP stays inside a node, when PP's
bubble hurts, and what an MoE all-to-all costs. Questions 22 and 23.
**Time:** ~2 h. **Needs:** Python 3 only. Read Lecture 01's deep dive first.

## What to do

1. Implement the six functions in `starter.py`.
2. `python check.py` until it prints `PASS (10/10)`.

## Then answer (in `ANSWERS.md`)

1. Llama-3-70B decode, batch 64, TP = 8, BF16. Per-step TP traffic per GPU,
   and its time over NVLink 4 (~450 GB/s per direction), PCIe Gen5 (~50 GB/s)
   and 200 Gb/s Ethernet (25 GB/s). Compare with the decode-step floor from
   Mod 4 Ex 06 (140 GB of BF16 weights ÷ 8 GPUs ÷ 3,350 GB/s per GPU ≈ 5.2 ms). Which links
   make TP=8 pointless?
2. The same for an 8K-token prefill. Why is prefill more tolerant of slow links
   than you might expect? (Compare comm time with prefill *compute* time:
   `2 · 70e9 · 8192 FLOPs ÷ (8 × 989e12 × 0.5 MFU)`.)
3. A 4-stage pipeline serving decode with only 4 sequences in flight: what
   fraction of the GPUs idles? What changes with 32 sequences?
4. DeepSeek-V3-style MoE layer (hidden 7168, top-8, EP = 32, 4,096 tokens per
   GPU): all-to-all bytes per GPU per layer, and the time over 400 Gb/s
   InfiniBand (50 GB/s). Why does DeepSeek overlap two micro-batches
   (Mod 6 digest §2)?

## Pass criteria

`check.py` prints PASS and every answer shows formula + number + a one-line conclusion.
