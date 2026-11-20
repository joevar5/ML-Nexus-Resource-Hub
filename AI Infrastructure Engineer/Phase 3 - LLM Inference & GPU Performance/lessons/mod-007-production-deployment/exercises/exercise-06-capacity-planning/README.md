# Ex 06: Capacity and Cost Model (CPU, auto-graded)

**Goal:** questions 24 and 25 — size a ChatGPT-style service from first
principles and prove a 70% cost cut, in code you can re-run when an assumption
changes. Read Lecture 06 first.
**Time:** ~2 h. **Needs:** Python 3 only.

## Part 1 (graded)

Implement the eight functions in `starter.py`; `python check.py` until
`PASS (11/11)`. The cases reproduce Lecture 06's numbers.

## Part 2 — use it (`DESIGN.md`)

1. Re-run the design for **your** assumptions: pick a different model (e.g. 8B
   on L4 at $0.80/GPU-hour, or a 70B on 8×H100 with TP=8) and a different
   traffic shape. Report peak rps, replicas, GPUs at peak, $/day, $/1M tokens.
2. Sensitivity: which single assumption moves $/day the most if it is 30% wrong
   (DAU, prompt length, output length, peak ratio, MFU, decode step time)?
3. Question 24: starting from a static-peak BF16 baseline, list the levers you
   would apply, each with its cost factor and its **quality risk**, and show the
   combined factor is ≤ 0.30.
4. If you ran Mod 3/4/6 benchmarks, replace the first-principles replica
   capacity with your measured goodput and report how far the estimate was off.

## Pass criteria

PASS from `check.py`, and every answer in `DESIGN.md` shows its numbers.
