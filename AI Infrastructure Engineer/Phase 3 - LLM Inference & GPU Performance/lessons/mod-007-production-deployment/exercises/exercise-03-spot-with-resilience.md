# Ex 03: Offline Batch Inference on Spot GPUs

**Goal:** cut the cost of a large offline job (evals, synthetic data,
embeddings, nightly summaries) with spot GPUs, without losing work when nodes
are reclaimed. **Time:** ~3 h. **Needs:** cloud account with spot GPUs (or
simulate preemption by killing the process).

## Steps

1. Job: 50,000 prompts → completions with vLLM's offline `LLM.generate` (much
   higher throughput than the server for batch work; measure both once).
2. Make it **resumable**: process in shards of e.g. 500 prompts; write each
   shard's output atomically (write to `tmp`, then rename) to object storage or
   a mounted volume; on start, skip shards that already exist.
3. Handle the interruption notice (AWS: poll the instance metadata
   `spot/instance-action` endpoint; GCP: `preempted` metadata; Kubernetes: the
   pod's `SIGTERM` + `terminationGracePeriodSeconds`): stop taking new shards
   and finish or abandon the current one cleanly.
4. Run it on spot, force at least two interruptions, and let it resume.

## Report

Total runtime, shards redone, $/job on spot vs on-demand (use the actual
hourly prices you paid), and output tokens per dollar.

## Pass criteria

- Final output has exactly 50,000 results, no duplicates, after ≥ 2 interruptions.
- The cost comparison, and the work lost per interruption (≤ one shard).
