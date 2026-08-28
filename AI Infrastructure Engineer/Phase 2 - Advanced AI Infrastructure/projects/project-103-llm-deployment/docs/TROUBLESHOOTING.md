# Troubleshooting — Project 103 (LLM Deployment)

Top failure modes for the LLM deployment project. Each entry: symptom → diagnosis → remediation → prevention.

> **Tip:** Before opening any of these runbooks, capture the current state with `kubectl describe pod <llm-serving-pod>`, `kubectl logs <pod> --tail=200`, and `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv`. Most issues become obvious once you have that snapshot.

---

## 1. Out-of-Memory (OOM) at model load

**Symptom**
- Pod crashes during startup with `torch.cuda.OutOfMemoryError`, `CUDA out of memory`, or container exits with code 137.
- `nvidia-smi` shows GPU memory pegged near 100% before the crash.

**Diagnosis**
1. Compute required memory: `params × bytes_per_param + activations + KV cache + overhead`.
   - Llama-3.1-8B in fp16 ≈ 16 GB just for weights.
   - KV cache per sequence: `2 × n_layers × n_heads × head_dim × seq_len × bytes`. For Llama-3.1-8B at 4k context: ~0.5 GB/seq.
2. Compare against GPU memory (`nvidia-smi --query-gpu=memory.total`).
3. Check `max_num_seqs`, `max_model_len`, `gpu_memory_utilization` in vLLM config.

**Remediation**
- Lower `gpu_memory_utilization` from default 0.9 → 0.85 to leave headroom.
- Reduce `max_model_len` to actual usage (don't reserve KV for context you never use).
- Reduce `max_num_seqs` (parallel sequences).
- Quantize weights (AWQ, GPTQ, or FP8 with Transformer Engine on H100/L40S).
- Use a larger GPU (A10G 24 GB → A100 40/80 GB → H100 80 GB).

**Prevention**
- Always compute the memory budget before deploy. Document it in the Helm values.
- Add an alert on `nvidia_gpu_memory_used_bytes / total > 0.95`.

---

## 2. Slow time-to-first-token (TTFT)

**Symptom**
- First token after a `/generate` request takes 2-10+ seconds. P95 TTFT > 1 s under load.

**Diagnosis**
1. Distinguish prefill vs decode latency. vLLM exposes `vllm:time_to_first_token_seconds_bucket`.
2. Check queue depth (`vllm:num_requests_waiting`) — high values mean the scheduler isn't admitting your request.
3. Check prompt length distribution — a single 32k-token prompt can dominate prefill.

**Remediation**
- Enable prefix caching if many requests share a common prefix (system prompt, few-shot examples).
- Increase `max_num_batched_tokens` if GPU is under-utilized during prefill.
- For very long prompts, evaluate chunked prefill (vLLM 0.6+).
- Add speculative decoding (vLLM `--speculative-model`) for short outputs.

**Prevention**
- Set an SLO on TTFT P95 (e.g., < 500 ms for prompts ≤ 2k tokens).
- Alert when `vllm:num_requests_waiting` > 0 for > 60 s.

---

## 3. Throughput plateau (tokens/sec stops scaling)

**Symptom**
- Adding GPUs or replicas doesn't increase aggregate tokens/sec.

**Diagnosis**
1. Check per-GPU utilization: `nvidia-smi dmon -s u`. If GPU sits at 50%, the bottleneck is elsewhere.
2. Check CPU utilization — tokenization, sampling, JSON serialization can be CPU-bound.
3. Check network: in tensor-parallel mode, NCCL all-reduce latency grows with GPU count; nccl-tests can quantify.

**Remediation**
- Pin tokenizer to a worker pool; avoid running it on the request-handling thread.
- Switch to `--enforce-eager false` so vLLM uses CUDA graphs for fixed batch sizes.
- For multi-GPU, prefer NVLink-connected GPUs over PCIe (NVLink is ~10x faster).
- If quantizing, ensure your framework can fuse quantization into the kernel (avoid dequant→matmul→requant).

**Prevention**
- Add a roofline analysis to the project deliverables. Knowing your compute-bound vs memory-bound regime guides every subsequent decision.

---

## 4. NCCL hangs in multi-GPU tensor parallelism

**Symptom**
- Pod logs show `NCCL WARN Cuda failure` or stays silent for minutes.
- `nvidia-smi` shows GPUs at 0% utilization.

**Diagnosis**
1. Set `NCCL_DEBUG=INFO` and inspect the all-reduce ring formation.
2. Check that all GPUs see each other: `nvidia-smi topo -m` should show NV# (NVLink) or PIX (PCIe) between TP-grouped GPUs.
3. Verify `CUDA_VISIBLE_DEVICES` matches the topology you expect.

**Remediation**
- Force a specific NIC: `NCCL_SOCKET_IFNAME=eth0` (or whatever your pod's NIC is).
- If the cluster has IB but you're not using it, set `NCCL_IB_DISABLE=1` to fall back cleanly to TCP.
- Restart pods with `kubectl rollout restart`. Stale NCCL state can survive process restarts when init containers leave shared memory dirty.

**Prevention**
- Run `nccl-tests` as a pre-flight job in every new cluster to catch interconnect issues before your model load attempts.
- Pin NCCL versions in the container image. NCCL bugs are real and version-specific.

---

## 5. KV cache eviction churn (high tail latency under load)

**Symptom**
- P99 latency 10x higher than P50. Throughput drops as concurrent requests increase.

**Diagnosis**
- Check `vllm:gpu_cache_usage_perc` — if it sits at > 95% during load, the scheduler is constantly evicting and re-prefilling.
- Check `vllm:num_preempted_requests_total` — > 0 means preemption is happening.

**Remediation**
- Reduce `max_num_seqs` so fewer sequences compete for KV blocks.
- Increase GPU (more total cache).
- For chat applications, enable prefix caching to dedupe system prompts across sessions.
- Consider PagedAttention's `swap_space` to bleed off pressure to CPU memory (slower but bounded).

**Prevention**
- Treat KV cache as a first-class capacity dimension in your load model. The standard "requests per second" framing under-specifies LLM serving.

---

## 6. Streaming SSE connections drop mid-generation

**Symptom**
- Clients receive partial output then `connection closed`.
- Server logs show no errors.

**Diagnosis**
1. Check ingress/load-balancer idle timeout (most defaults are 30-60 s; long generations exceed that).
2. Check `kubectl describe ingress` for `proxy-read-timeout` and `proxy-send-timeout` annotations.
3. Confirm the client is actually consuming the stream (some clients buffer SSE).

**Remediation**
- Set ingress annotations: `nginx.ingress.kubernetes.io/proxy-read-timeout: "600"` and `proxy-send-timeout: "600"`.
- Send periodic `: keep-alive` comments every 15-20 s in your SSE stream so intermediaries don't time out.
- For very long generations, consider splitting into chunks with separate requests.

**Prevention**
- Include long-generation tests in your CI suite (generate ≥ 2,000 output tokens).

---

## 7. Model loads slowly (5+ minutes)

**Symptom**
- Pod stays in `Running` but not `Ready` for several minutes after start.

**Diagnosis**
1. Check whether the model is being downloaded from the Hub on every pod start (look for HuggingFace download log lines).
2. Check the `readinessProbe` — if it polls a `/ready` endpoint that only flips when the model is fully loaded, the wait time is the actual load time.

**Remediation**
- Pre-bake the model into the container image (or use a model PVC that's pre-populated).
- Use a higher-bandwidth storage class (e.g., `gp3` 1 GiB/s instead of `gp2`).
- For sharded models, prefetch shards in parallel.
- Increase `readinessProbe.initialDelaySeconds` so K8s doesn't restart the pod while it's still loading.

**Prevention**
- Pre-pull container images on nodes via DaemonSet.
- Document expected cold-start time per (model, hardware) combo in the project README.

---

## 8. FP8 numerical drift (quantized output diverges from baseline)

**Symptom**
- FP8/AWQ/GPTQ quantized model produces qualitatively worse output than the fp16 baseline (e.g., longer outputs, more hallucinations, broken JSON).

**Diagnosis**
1. Compare evaluation metrics on a held-out set (perplexity, accuracy, instruction-following score).
2. Check if your calibration data matches inference distribution. Calibrating on Wikitext but serving chat will hurt.
3. Check whether activation scaling is per-tensor (FP8) or per-channel (AWQ); the wrong scale strategy degrades quality.

**Remediation**
- Re-calibrate AWQ/GPTQ with a sample that matches your serving distribution.
- For FP8 with Transformer Engine, ensure `fp8_recipe` is set correctly (DelayedScaling vs Current).
- Try a different quantization scheme; AWQ generally preserves quality better than GPTQ for chat.

**Prevention**
- Always evaluate quantized variants on your actual evaluation set before promoting to production. The "1.5pp" threshold from the project spec is conservative for a reason.

---

## 9. HPA never scales up (or never scales down)

**Symptom**
- Replica count stuck at min (or max) despite changing load.

**Diagnosis**
1. `kubectl describe hpa <name>` — look at the conditions and target/current values.
2. Confirm metrics are reaching Prometheus and the HPA's metric source (Prometheus Adapter, KEDA).
3. For GPU workloads, CPU metrics don't reflect actual load — make sure your HPA is using a request-based metric (queue depth, RPS, GPU utilization).

**Remediation**
- Switch HPA to a custom metric: `vllm:num_requests_waiting` is an excellent signal.
- Use KEDA's `prometheus` scaler if Prometheus Adapter is flaky.
- Set `behavior.scaleDown.stabilizationWindowSeconds: 300` to avoid flapping.

**Prevention**
- Load-test the HPA explicitly. Run a step-load test and verify the scale-up and scale-down both happen as expected.

---

## 10. Queue saturation under burst load

**Symptom**
- Burst of requests causes a backlog; new requests sit in the queue for seconds-to-minutes.

**Diagnosis**
- `vllm:num_requests_waiting` spikes.
- Per-tenant tracking shows one tenant consuming all capacity.

**Remediation**
- Add per-tenant rate limiting at the API layer (e.g., a `RateLimitMiddleware` in FastAPI keyed by API key).
- Shed load by returning HTTP 429 when `num_requests_waiting > N`.
- Provision more replicas (HPA on queue depth, see #9).
- For mixed workloads, consider tier-aware routing — small models for cheap/fast, big models for premium.

**Prevention**
- Always model for burst, not just steady-state. P99 throughput matters more than mean.

---

## 11. Cost surprise after deploy

**Symptom**
- Monthly cloud bill includes a much higher GPU spend than projected.

**Diagnosis**
1. Check actual GPU-hours: `(replicas × hours_running × on_demand_rate)`.
2. Compare against your project's COST.md analysis.
3. Look for: zombie pods (a deployment scaled to 10 that never scaled down), spot interruptions causing re-launches, inefficient batching (low GPU utilization but full provisioning).

**Remediation**
- Right-size: if GPU utilization is < 40% steady-state, you have too many replicas.
- Use spot instances for batch / non-critical paths.
- Add a Prometheus query for `nvidia_gpu_utilization` and alert when low utilization persists.

**Prevention**
- Include a weekly cost review in your runbook. Costs drift; what was right last quarter may be wrong this quarter.

---

## 12. Tokenization mismatch causes garbage output

**Symptom**
- Output is incoherent, character-for-character. May contain BPE artifacts (`Ġthe` instead of ` the`).

**Diagnosis**
1. Check that the tokenizer version matches the model version. Llama-3 vs Llama-3.1 use different tokenizers.
2. Compare `tokenizer.encode("hello world")` between training-time and serving-time.

**Remediation**
- Pin the tokenizer version (and model version) in the container image.
- Add a startup self-check: encode + decode a canary string, compare to expected output.

**Prevention**
- Bake the tokenizer into the model artifact. Never download "the latest" tokenizer at runtime.

---

## 13. Image build fails with CUDA mismatch

**Symptom**
- `docker build` succeeds but the container crashes with `CUDA driver version is insufficient` on a GPU node.

**Diagnosis**
- Container CUDA version > host driver's max-supported CUDA version.
- `nvidia-smi` on host shows the driver version (e.g., `535.x` supports CUDA up to 12.2).

**Remediation**
- Match the container's CUDA version to the host driver: use `nvcr.io/nvidia/cuda:12.2.2-runtime-ubuntu22.04` (or whatever matches).
- Upgrade the host driver if you genuinely need a newer CUDA.

**Prevention**
- Standardize on a (driver, CUDA) pair across your cluster. Document in cluster setup notes.

---

## 14. Prometheus scraping causes high CPU on the server pod

**Symptom**
- Serving pod CPU spikes every 15 s coinciding with Prometheus scrapes.

**Diagnosis**
- Check `vllm` metrics cardinality. Per-request labels can explode cardinality.

**Remediation**
- Drop high-cardinality labels via Prometheus relabel config.
- Reduce metric collection interval if 15 s overhead is meaningful.

**Prevention**
- Audit metrics cardinality before deploy. Use `prometheus_tsdb_head_series` to verify.

---

## 15. Pod evicted mid-generation by spot interruption

**Symptom**
- In-flight requests fail with connection reset; pod restarts on a new node 30-60 s later.

**Diagnosis**
- Check node events: `kubectl describe node <node>` — look for `SpotInterruption` or `Preempted`.

**Remediation**
- Add a `PreStop` hook with `terminationGracePeriodSeconds: 120` so the pod can drain in-flight requests.
- Use a service like AWS's Spot Interrupt Notice handler (2-minute warning) to gracefully shed load.
- For latency-critical paths, prefer on-demand or reserved instances.

**Prevention**
- Plan for interruption. Spot is great for cost, but only with the right architecture.

---

## See also

- `docs/ARCHITECTURE.md` — system design and component overview
- `docs/DEPLOYMENT.md` — production deployment guide
- `docs/OPTIMIZATION.md` — performance tuning playbook
- `docs/COST.md` — cost analysis and optimization
- `docs/RAG.md` — retrieval-augmented generation patterns (if applicable to your variant)
