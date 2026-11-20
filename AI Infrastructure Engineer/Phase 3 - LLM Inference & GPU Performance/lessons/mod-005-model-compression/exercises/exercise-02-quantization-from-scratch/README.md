# Ex 02: Quantization from Scratch (CPU-only, auto-graded)

**Goal:** understand *why* the quantization methods in Lecture 01 exist by
building them in numpy and watching the error move. Do this before Ex 01.
**Time:** ~2 h. **Needs:** Python 3 + numpy.

## What to do

1. Implement the six functions in `starter.py` (symmetric quantize/dequantize,
   per-row scales, group-wise scales, SmoothQuant scales, relative error).
2. Run `python check.py` until it prints `PASS (6/6)`. The checks build a
   synthetic layer whose activations have four outlier channels 60× larger
   than the rest — the situation real LLMs are in.

## Then answer (in `ANSWERS.md`), with numbers from your code

1. Relative output error of a W4 (per-row) layer vs W4 group-128. Why do
   AWQ/GPTQ checkpoints use groups, and what do the extra scales cost in bits
   per weight (FP16 scale per 128 weights)?
2. W8A8 with per-tensor activation scales: what is the error before and after
   SmoothQuant? Try α ∈ {0.25, 0.5, 0.75, 0.9}; which is best and why?
3. Replace per-tensor activation scales with **per-token** (per-row) scales
   without smoothing. How much of the SmoothQuant gain does that recover, and
   why doesn't it fix outlier *channels*?
4. Question 11: given these results, which scheme would you pick for (a) a 70B
   model on one H100 at batch 1, (b) the same model at batch 128? Use the
   table in Lecture 01.

## Pass criteria

`check.py` prints PASS and all four answers cite your measured errors.
