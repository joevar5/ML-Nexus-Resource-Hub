# Lesson 11: Autoscaling LLM Serving on Queue Depth (KEDA)

[Lesson 06](./06-llm-serving-optimization.md) made one GPU serve as many requests as it can. This lesson covers what to do when one GPU is not enough, or when most of the day it is too much: adding and removing model replicas automatically as load changes, without paying for idle GPUs or letting users queue.

It is the working guide for [Project 105](../../projects/project-105-gpu-cluster-autoscaling/README.md).

**Prerequisites:** [Lesson 02](./02-vllm-deployment.md), [Lesson 06](./06-llm-serving-optimization.md), the Kubernetes GPU scheduling lesson in [mod-04](../mod-04-kubernetes/02-gpu-scheduling.md), and the Prometheus lesson in [mod-08](../mod-08-monitoring-observability/01-prometheus-metrics.md).

### Contents

1. [What LLM Autoscaling Is All About](#what-llm-autoscaling-is-all-about)
2. [Core Logic: Scale on the Queue, Not the CPU](#core-logic-scale-on-the-queue-not-the-cpu)
3. [Step-by-Step: Autoscale vLLM with KEDA](#step-by-step-autoscale-vllm-with-keda)
4. [Practical Exercise](#practical-exercise)
5. [Key Takeaways](#key-takeaways)
6. [Additional Resources](#additional-resources)

---

## What LLM Autoscaling Is All About

A GPU costs money every hour it exists, busy or not. Traffic is rarely flat: it peaks at some hours and nearly vanishes at others. A fixed number of replicas is either too many (wasted money) or too few (users wait). Autoscaling adjusts the replica count to demand.

LLM serving makes this harder than a normal web service in three ways:

| Difference | Why it hurts |
|---|---|
| **Slow to start** | A new replica must get a GPU node and load many gigabytes of weights. This takes minutes, not seconds. |
| **Wrong signal** | CPU usage says almost nothing about a GPU-bound server |
| **Expensive** | Each extra replica is a whole GPU, so over-scaling costs real money |

---

## Core Logic: Scale on the Queue, Not the CPU

### Why CPU Autoscaling Fails

The Kubernetes Horizontal Pod Autoscaler (HPA) scales on CPU or memory by default. A vLLM server can be fully saturated while its CPU sits low, because the work happens on the GPU. The default signal misses the overload.

### The Signal to Use

vLLM exposes Prometheus metrics at `/metrics`. Two are useful for scaling:

| Metric | Meaning | Use |
|---|---|---|
| `vllm:num_requests_waiting` | Requests queued, not yet running | Best scale-up signal: it is non-zero only when the server is full |
| `vllm:num_requests_running` | Requests being processed | Shows how busy a replica is |

Metric names can change between vLLM versions, so open `/metrics` on your own server and confirm them before you write a query.

A queue means users are waiting *now*. That is the direct measure of "not enough capacity", and it ties to your latency target ([Lesson 06](./06-llm-serving-optimization.md) calls it TTFT).

### The Arithmetic

Choose a target of waiting requests per replica, then derive the replica count:

```
desired replicas = ceil(total waiting requests ÷ target waiting per replica)
```

The autoscaler does this calculation for you. Your job is choosing a good target by measuring where latency starts to break (Lesson 06, Step 1).

### Why KEDA

KEDA (Kubernetes Event-driven Autoscaling) extends Kubernetes autoscaling with external signals. It reads a Prometheus query, feeds the result to an HPA it manages, and can also scale a deployment down to zero. You describe the rule in a `ScaledObject`.

### The Cold-Start Problem

Because a new replica takes minutes to be ready, reacting only after a queue forms is too late. Three tools help:

| Tool | What it does |
|---|---|
| **Minimum replicas** | Never scale below one if users can't wait for a cold start |
| **A lower target** | Scale up while the queue is still short, to leave headroom |
| **Fast model loading** | Cache weights on a local volume or node disk so a new replica doesn't download them again |

Scale **down** slowly. A long cooldown avoids killing a replica that you will need again in five minutes.

---

## Step-by-Step: Autoscale vLLM with KEDA

### Step 1: Deploy vLLM With GPU Requests

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
spec:
  replicas: 1
  selector:
    matchLabels: {app: vllm}
  template:
    metadata:
      labels: {app: vllm}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest   # pin an exact version in real use
          args: ["--model", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "--max-model-len", "4096"]
          ports: [{containerPort: 8000}]
          resources:
            limits: {nvidia.com/gpu: 1}
          readinessProbe:
            httpGet: {path: /health, port: 8000}
            initialDelaySeconds: 60
            periodSeconds: 10
```

The readiness probe matters: Kubernetes must not send traffic to a replica that is still loading weights. The scraping annotations depend on how your Prometheus is set up, so match your cluster's configuration.

### Step 2: Install KEDA

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda --namespace keda --create-namespace
```

### Step 3: Describe the Scaling Rule

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm
  minReplicaCount: 1        # 0 saves money but every first request pays a cold start
  maxReplicaCount: 4        # a hard cap on cost
  pollingInterval: 15       # seconds between checks
  cooldownPeriod: 300       # seconds before scaling in
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        query: sum(vllm:num_requests_waiting)
        threshold: "5"      # target waiting requests per replica
```

Field names follow KEDA's current API; check the docs for your installed version. `maxReplicaCount` is your budget guard, so set it deliberately.

### Step 4: Generate Load and Watch It Scale

Use the benchmark client from Lesson 06 with rising concurrency, and watch the replica count and queue together:

```bash
kubectl get hpa -w
kubectl get pods -l app=vllm -w
```

Record a timeline: when load rose, when the queue formed, when the new replica became ready, and when the queue cleared. The gap between "queue formed" and "replica ready" is your cold-start cost.

### Step 5: Scale the Nodes Too

New pods need GPU nodes. A pod waiting for a node shows as `Pending`. A node autoscaler (the Kubernetes Cluster Autoscaler or Karpenter) adds GPU nodes when pods can't be scheduled. Node provisioning adds more minutes to the cold start, so this is the slowest part of the chain.

### Step 6: Consider Spot Instances

Spot (preemptible) GPU nodes can cost much less, but the cloud can take them back with short notice. Keep at least one on-demand replica, put the rest on spot, and make sure a lost replica's in-flight requests are retried by the client or gateway.

### Step 7: Price It

Report cost at idle and at peak:

```
idle cost/hour = min replicas × GPU hourly price
peak cost/hour = max replicas × GPU hourly price
```

Compare against a fixed fleet sized for peak. The saving is the point of the whole exercise.

---

## Practical Exercise

Autoscale a small vLLM deployment:

1. Deploy vLLM with one GPU and a readiness probe
2. Confirm `vllm:num_requests_waiting` appears in Prometheus
3. Add a KEDA `ScaledObject` with `threshold: "5"` and `maxReplicaCount: 3`
4. Load-test with rising concurrency and record the timeline
5. Report the cold-start gap and the idle versus peak cost

*No cluster? Read the manifests, and write the timeline you would expect. On a rented GPU, use a single-node cluster such as k3s, with a small model. It costs a few dollars for an hour.*

---

## Key Takeaways

1. A GPU costs money idle or busy, so autoscaling is mostly a cost tool
2. CPU is the wrong signal for GPU-bound LLM servers: scale on queue depth (`vllm:num_requests_waiting`)
3. KEDA reads a Prometheus query and manages an HPA, and can scale to zero
4. Cold starts take minutes (node plus weight loading), so keep a minimum replica, scale up early, and cache weights
5. Scale down slowly and cap the maximum replicas to cap the bill
6. Spot nodes save money but need an on-demand floor and client retries
7. Always report idle and peak cost, plus the cold-start timeline

---

## Additional Resources

- [KEDA documentation](https://keda.sh/docs/)
- [KEDA Prometheus scaler](https://keda.sh/docs/latest/scalers/prometheus/)
- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics.html)
- [Karpenter](https://karpenter.sh/)
- [Lesson 06: LLM Serving Optimization](./06-llm-serving-optimization.md)

---

**Next Lesson:** [12-preference-tuning-dpo.md](./12-preference-tuning-dpo.md): Preference Tuning With DPO
