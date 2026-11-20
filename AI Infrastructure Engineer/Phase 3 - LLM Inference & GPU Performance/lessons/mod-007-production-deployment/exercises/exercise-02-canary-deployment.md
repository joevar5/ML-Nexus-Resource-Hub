# Ex 02: Roll Out a Quantized Model Safely, with Automatic Rollback

**Goal:** question 19 — ship an FP8 (or INT4) version of a model behind a
canary judged on **goodput and quality**, and prove the rollback works.
Phase 1/2 covered Kubernetes canaries in general; this is the LLM-specific part.
**Time:** ~4 h. **Needs:** a Kubernetes cluster with one GPU node (your Phase 2
project 105 cluster), Argo Rollouts, Prometheus.

## Steps

1. Two images/configs of the same model: **stable** = BF16, **canary** = the FP8
   checkpoint from Mod 5 Ex 01. Plus a deliberately **broken canary** (e.g.
   the INT4 checkpoint with a wrong chat template, or `--max-model-len 512`).
2. An Argo `Rollout` with steps 5% → 25% → 50% → 100%, a pause between steps,
   and an `AnalysisTemplate` that fails the step when any of these trips:
   - TTFT p99 or TPOT p99 above SLO (from vLLM histograms, Lecture 05)
   - error rate > 1%
   - **quality**: a small job that sends 50 fixed eval prompts to the canary
     service and fails if exact-match/score drops more than X vs stable
     (Argo `Job` metric provider).
3. Keep stable pods running at full size until promotion completes, so rollback
   is instant (no cold start, Mod 6 Ex 05).
4. Run it twice: the good FP8 canary (should promote) and the broken one
   (should roll back automatically).

## Pass criteria

- Both runs recorded (Argo dashboard screenshots or `kubectl argo rollouts get` output).
- The broken canary is rolled back by the analysis, not by hand, and the trip
  reason is the metric you expected.
- A short runbook: what you check before promoting manually, and how long a
  rollback takes end to end.
