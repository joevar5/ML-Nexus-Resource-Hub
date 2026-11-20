# Ex 05: End-to-End — an LLM Service You Would Put on Call

**Goal:** combine this module into one service. Generic deployment hygiene
(containers, Helm, CI/CD) was built in Phase 1 project 05 and Phase 2 project
105 — reuse them; spend your time on the LLM-specific parts.
**Time:** ~6 h. **Needs:** your Phase 2 cluster with one GPU node.

## Must have

1. vLLM serving one model, behind your Ex 07 streaming proxy (heartbeats,
   cancellation) and a router (Mod 6 Ex 04 policy if you have ≥ 2 replicas).
2. The Lecture 05 metrics in Prometheus and one Grafana dashboard: TTFT/TPOT
   p50/p99, goodput, queue, KV utilization, preemptions, prefix hit rate,
   $/1M tokens.
3. SLOs (TTFT p99, TPOT p99, availability) with **burn-rate alerts**.
4. Autoscaling on waiting requests with a warm minimum sized by Mod 6 Ex 03.
5. Rollouts through the Ex 02 canary with automatic rollback.
6. A `RUNBOOK.md` covering: p99 TTFT alert (use Lecture 05's diagnosis table),
   KV cache full / preemptions rising, GPU node lost, bad model rollout.

## Pass criteria

- A 30-minute load test replaying a bursty trace (BurstGPT or your own) with
  the dashboard screenshot attached.
- One injected incident (kill a replica, or deploy the broken canary) handled
  by following your runbook, written up as a short postmortem.
