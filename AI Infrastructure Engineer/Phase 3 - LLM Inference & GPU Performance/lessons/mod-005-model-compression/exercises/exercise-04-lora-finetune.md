# Ex 04: Serving Many LoRA Adapters on One Base Model

**Goal:** measure the serving side of LoRA (question 21's multi-tenant angle).
Training LoRA adapters was covered in Phase 2
[project 102](../../../../Phase%202%20-%20Advanced%20AI%20Infrastructure/projects/project-102-text-to-sql-finetuning/);
reuse that adapter or public adapters for the same base model.
**Time:** ~3 h. **Needs:** one GPU with ≥ 24 GB, vLLM.

## Steps

1. Pick a base model with ≥ 4 compatible adapters (your project 102 adapter
   plus public ones from the Hub, or create extra ones by fine-tuning 50 steps
   on different tiny datasets — they only need to load).
2. Serve them unmerged:

```bash
vllm serve <base> --enable-lora --max-loras 4 --max-lora-rank 64 \
  --lora-modules a1=<path1> a2=<path2> a3=<path3> a4=<path4>
```

3. Load test with 64 concurrent requests, each picking an adapter at random,
   then with all requests on one adapter, then with the base model only.
4. Separately, merge one adapter into the base (`peft` `merge_and_unload`) and
   serve it as a plain model under the same load.

## Report

| setup | tok/s | TTFT p50 | TTFT p99 |
|---|---|---|---|
| base only | | | |
| 1 adapter (unmerged) | | | |
| 4 adapters mixed (unmerged) | | | |
| 1 adapter merged | | | |

## Pass criteria

- All 4 rows; the cost of unmerged LoRA (vs merged) stated as a percentage.
- Adapter size on disk vs base model size.
- Recommendation: when to merge, when to serve unmerged, and what `--max-loras`
  trades against KV cache memory.
