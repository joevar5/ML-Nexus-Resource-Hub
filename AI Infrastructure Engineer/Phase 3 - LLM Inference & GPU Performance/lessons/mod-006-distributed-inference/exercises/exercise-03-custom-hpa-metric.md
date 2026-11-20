# Ex 03: Autoscaling with Slow Cold Starts — Size the Headroom

**Goal:** question 20. Phase 2 already built queue-depth autoscaling
([mod-10 lesson 11](../../../../Phase%202%20-%20Advanced%20AI%20Infrastructure/lessons/mod-10-llm-infrastructure/11-autoscaling-on-queue-depth-keda.md),
[project 105](../../../../Phase%202%20-%20Advanced%20AI%20Infrastructure/projects/project-105-gpu-cluster-autoscaling/)).
This exercise adds the part those skip: how cold-start time decides how much
idle capacity you must pay for.
**Time:** ~3 h. **Needs:** your project 105 cluster, or just Python for parts 1–2.

## Part 1 — measure (or estimate) your cold start

Break a scale-up into stages and time each one (from project 105 logs, or
estimate from Lecture 02's list): node provisioning, image pull, weight
download, weight load, warmup. Total = **T_start**.

## Part 2 — headroom arithmetic

Traffic can grow at most **g** requests/s per minute during a ramp (take
g = 10% of peak per minute). One replica serves **c** requests/s at your
latency SLO (measure it with `vllm bench serve`).

1. Spare capacity needed so the queue doesn't explode while a replica starts:
   `headroom_rps ≥ g · T_start` → replicas of headroom = `ceil(g · T_start / c)`.
2. Compute it for T_start = 8 min (cold node), 3 min (warm node, cached
   image), 45 s (local weight cache + pre-pulled image) and 15 s
   (ServerlessLLM-style loading, Mod 6 digest §4).
3. Cost per day of that headroom at $2.50 per GPU-hour.

## Part 3 (cluster) — prove it

With KEDA on `vllm:num_requests_waiting`, run the same step-load test with
and without a warm minimum of headroom replicas, and with and without a
pre-pull DaemonSet. Plot queue depth and p99 TTFT over time.

## Pass criteria

- The four-row table of headroom replicas and $/day vs T_start.
- One paragraph: which cold-start stage is worth engineering away first for
  your setup, and how much money it saves per day.
- (Part 3) the plots, with the SLO line drawn.
