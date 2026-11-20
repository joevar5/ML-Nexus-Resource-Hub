# Ex 06: Inference Arithmetic (CPU-only, auto-graded)

**Goal:** be able to size an LLM deployment on a whiteboard: KV cache, how many
users fit, the decode speed ceiling, and whether speculative decoding pays.
These are the numbers behind questions 3, 10, 12 and 25 of
[the 25 questions](../../../../25-questions.md).
**Time:** ~2 h. **Needs:** Python 3 only.

## What to do

1. Read Lecture 01 (KV cache), Lecture 03 (speculation formula) and the
   Mod 1 Lecture 1.5 deep dive (why decode is memory-bound).
2. Implement the seven functions in `starter.py`.
3. Run `python check.py` until it prints `PASS (14/14)`.

## Then answer, using your functions (write the answers in `ANSWERS.md`)

1. **Question 3.** Llama-3-70B (80 layers, 8 KV heads, head_dim 128), 32K
   context, 100 concurrent users. KV cache in FP16 and in FP8? How many
   H100 80 GB GPUs does the KV cache alone need?
2. Same model, but **without** GQA (64 KV heads). How much bigger? This is
   why every modern model uses GQA or MLA.
3. An 8B BF16 model on one H100 (3,350 GB/s). What is the best possible
   single-stream tokens/s? With 64 users at 4K context each, how much does
   reading the KV cache add to each decode step?
4. Draft acceptance falls from 0.8 to 0.5 when users switch from code to
   creative writing. With c = 0.05, what γ should the server use for each,
   and what speedup does each get?

## Pass criteria

`check.py` prints PASS, and all four answers show the formula used and a number.
