# Exercise 02: Control Plane Tour

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 01

## Objective

Explore Kubernetes' control plane components (kube-apiserver, etcd, kube-scheduler, kube-controller-manager) by reading their logs, exec'ing into their pods, and triggering observable effects. By the end you'll understand the request flow from `kubectl apply` to a running pod.

## Why this matters

Kubernetes' magic isn't magic — it's 5 components doing specific jobs. Engineers who understand the components diagnose 10× faster than engineers who treat K8s as a black box.

## Steps

### Step 1 — Inventory the control plane (15 min)
```bash
kubectl get pods -n kube-system
# Expect: kube-apiserver, etcd, kube-scheduler, kube-controller-manager, kube-proxy, coredns
```

### Step 2 — Read kube-apiserver logs during apply (30 min)
```bash
kubectl logs -n kube-system kube-apiserver-kind-control-plane -f &

# In another terminal:
kubectl apply -f deployment.yaml

# Observe the apiserver logs showing admission, validation, persistence
```

Identify in the logs:
- The mutating admission webhooks.
- The validation step.
- The final write to etcd.

### Step 3 — Inspect etcd directly (30 min)
```bash
kubectl -n kube-system exec etcd-kind-control-plane -- sh
# Inside:
ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  get /registry/deployments/default/iris-api --print-value-only | head -30
```
You see the deployment object as etcd stores it.

### Step 4 — Watch the scheduler (30 min)
```bash
kubectl logs -n kube-system kube-scheduler-kind-control-plane -f &

# Create a pod with unschedulable requirements:
kubectl run busy --image=nginx --requests=cpu=100,memory=1Ti

# Scheduler logs report 0/N nodes available
```
Remove the absurd request → pod schedules → log shows it.

### Step 5 — Controller manager in action (30 min)
```bash
kubectl logs -n kube-system kube-controller-manager-kind-control-plane -f &

# Scale a deployment up:
kubectl scale deploy/iris-api --replicas=10
# Controller logs the ReplicaSet updates and pod creates

# Kill a pod:
kubectl delete pod -l app=iris --random
# Controller immediately creates a replacement (logs show it)
```

### Step 6 — Trace a single request end-to-end (30 min)
Run `kubectl --v=8 get pods` to see the API call.
Then `kubectl get --raw /api/v1/namespaces/default/pods` to make a raw call.
Then in etcd: `etcdctl get /registry/pods/default/ --prefix --keys-only`.

You've now seen the full chain: client → apiserver → etcd → response.

## Deliverables

1. `OBSERVATIONS.md` documenting what each component does, with log excerpts.
2. Diagram of the request flow you traced.

## Validation

- [ ] You can name what each control-plane component does in one sentence.
- [ ] You watched the apiserver process a real apply.
- [ ] You read a Kubernetes object directly out of etcd.
- [ ] You observed scheduler + controller-manager react to your actions in real-time.

## Stretch goals

- Disable the scheduler temporarily (`kubectl -n kube-system scale --replicas=0 deploy/kube-scheduler` on managed clusters; on kind: rename the manifest in /etc/kubernetes/manifests). Try to schedule a pod. Re-enable.
- Walk through the source code of one controller (e.g., the deployment controller in kubernetes/kubernetes).
- Trace a Pod creation through the audit log (enable audit logging first).

## Common pitfalls

- **etcd corruption from manual writes** — Never write to etcd directly in production.
- **Disabling controllers** — Cluster appears to work until you change anything.
- **`kubectl --v=8` is verbose** — Use to debug, never as the default.
