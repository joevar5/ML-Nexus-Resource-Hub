# Ex 01: Quantize a Real LLM Three Ways and Measure the Loss

**Goal:** answer question 11 with your own numbers: BF16 vs FP8 vs INT4
weight-only, measured on quality, memory and speed at two batch sizes.
**Time:** ~4 h. **Needs:** one GPU with ≥ 24 GB (FP8 needs Ada/Hopper: L4,
L40S, H100; on A100 replace FP8 with INT8 W8A8). Do Ex 02 first.

## Steps

1. Model: `Qwen/Qwen2.5-7B-Instruct` (or any 7–8B instruct model).
2. Produce two quantized checkpoints with `llmcompressor` (Lecture 01, Tooling):
   `FP8_DYNAMIC`, and `W4A16` (GPTQ-style, 256 calibration samples from
   `HuggingFaceH4/ultrachat_200k`). Or use published AWQ/FP8 checkpoints of the
   same model and say so.
3. **Quality:** WikiText-2 perplexity (`lm_eval --tasks wikitext`) and one task
   eval (`gsm8k` 5-shot or 500 MMLU questions) for all three.
4. **Speed:** `vllm bench throughput` or `vllm bench serve` with 512-token
   inputs and 256-token outputs, at concurrency 1 and 64.
5. **Memory:** checkpoint size on disk and the KV cache blocks vLLM reports at
   startup (`# GPU blocks`) — quantized weights leave room for more KV.

## Report (`QUANT_REPORT.md`)

| variant | disk GB | KV blocks | ppl | task score | tok/s @1 | tok/s @64 |
|---|---|---|---|---|---|---|

## Pass criteria

- All 3 rows filled; quality deltas stated vs BF16 (e.g. "+0.08 ppl, −0.4 pp gsm8k").
- You can explain why INT4 wins at concurrency 1 but may lose to FP8 at 64
  (dequantization cost once decode stops being memory-bound).
- A one-paragraph recommendation for a chat service on H100 and on A100.
