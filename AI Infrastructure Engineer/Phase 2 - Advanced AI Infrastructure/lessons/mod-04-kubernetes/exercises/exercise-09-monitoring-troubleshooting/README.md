# Exercise 09: Kubernetes Monitoring and Troubleshooting Mastery

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Lab 05 monitoring stack; debug methodology from lab 08

## Objective

Install kube-prometheus-stack + Pixie (or similar) and use them to diagnose 5 categorized incidents: networking, resource, scheduling, configuration, and state.

## Requirements

1. Full monitoring stack installed and healthy.
2. 5 deliberate incidents injected; diagnose each in < 10 minutes.
3. Document the diagnosis path per incident.
4. Build a personal `K8S_DEBUG_PLAYBOOK.md`.

## Step-by-step

### Step 1 — Install (30 min)
Per mod-101 lab 05. Verify Prometheus, Grafana, kube-state-metrics all running.

### Step 2 — Add Pixie for live cluster inspection (30 min)
```bash
bash -c "$(curl -fsSL https://withpixie.ai/install.sh)"
px deploy
```
Pixie auto-instruments without code changes; gives real-time per-pod metrics + network flows.

### Step 3 — Incident 1: Pod can't reach external API (20 min)
```bash
kubectl apply -f incident-1-egress-blocked.yaml
# (NetworkPolicy denies all egress)
```
Diagnose using: `kubectl describe netpol`, `kubectl exec pod -- curl -v https://example.com`, Pixie network flow.

### Step 4 — Incident 2: Pod OOMKilled (20 min)
```bash
kubectl apply -f incident-2-low-memory.yaml
# (memory limit 64Mi but app needs 200Mi)
```
Diagnose using: `kubectl describe pod` (OOMKilled in lastState), `kubectl top pod`, Grafana cgroup memory chart.

### Step 5 — Incident 3: Pod stuck Pending (20 min)
```bash
kubectl apply -f incident-3-unschedulable.yaml
# (requests > any node capacity)
```
Diagnose: `kubectl describe pod` (Events at bottom), `kubectl get nodes -o wide`, `kubectl top nodes`.

### Step 6 — Incident 4: Config drift (20 min)
```bash
kubectl apply -f incident-4-bad-config.yaml
# (ConfigMap has invalid JSON → app crashes on read)
```
Diagnose: `kubectl logs pod`, `kubectl exec -- cat /etc/config/...`, then `kubectl describe cm`.

### Step 7 — Incident 5: Service has no endpoints (20 min)
```bash
kubectl apply -f incident-5-selector-mismatch.yaml
# (Service selector wrong)
```
Diagnose: `kubectl get endpoints svc`, `kubectl get pods --show-labels`, compare to selector.

### Step 8 — Compile playbook (30 min)
For each incident type, document the 3-step diagnostic flow:
- Symptom → Command to confirm → Fix

## Deliverables

1. Working monitoring + Pixie.
2. All 5 incidents reproduced + diagnosed + fixed.
3. `K8S_DEBUG_PLAYBOOK.md`.

## Validation

- [ ] Each diagnosis took < 10 minutes.
- [ ] Fix applied; cluster healthy after each.
- [ ] Playbook covers all 5 categories with concrete commands.

## Common pitfalls

- **Not reading Events** — Always start with `kubectl describe`.
- **Trusting `kubectl logs` alone** — Sometimes the issue is the kubelet, not the app.
- **Forgetting `--previous`** — For CrashLoopBackOff, you need logs from the prior instance.
