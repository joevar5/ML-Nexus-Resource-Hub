# Lab 07: NetworkPolicies for East-West Isolation

**Duration:** 60 min  **Prerequisites:** Cluster with a NetworkPolicy-aware CNI (Calico, Cilium); kind users: install Calico

## Objective
Default-deny all inter-pod traffic, then explicitly allow only the connections your app needs. Verify allowed connections work and denied connections fail.

## Steps

### 1. Install Calico on kind
```bash
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.0/manifests/calico.yaml
kubectl rollout status -n kube-system ds/calico-node
```

### 2. Three test workloads
```bash
kubectl create deploy front --image=nginx
kubectl create deploy api   --image=hashicorp/http-echo --port=5678 -- -text="api"
kubectl create deploy db    --image=postgres -- POSTGRES_PASSWORD=app
kubectl expose deploy front --port=80
kubectl expose deploy api   --port=80 --target-port=5678
kubectl expose deploy db    --port=5432
```

### 3. Baseline — everything reaches everything
```bash
kubectl run probe --rm -it --image=curlimages/curl --restart=Never -- sh
# Inside probe:
curl -s --max-time 3 front  # ok
curl -s --max-time 3 api    # ok
nc -zv db 5432              # ok
exit
```

### 4. Default deny ingress + egress
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: default-deny-all }
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
```
Re-run probe — everything times out.

### 5. Allow DNS (required for almost everything)
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: allow-dns }
spec:
  podSelector: {}
  policyTypes: [Egress]
  egress:
    - to:
        - namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: kube-system } }
          podSelector: { matchLabels: { k8s-app: kube-dns } }
      ports:
        - { protocol: UDP, port: 53 }
        - { protocol: TCP, port: 53 }
```

### 6. Allow front → api
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: allow-front-to-api }
spec:
  podSelector: { matchLabels: { app: api } }
  policyTypes: [Ingress]
  ingress:
    - from:
        - podSelector: { matchLabels: { app: front } }
      ports: [{ protocol: TCP, port: 5678 }]
---
# Also allow front's egress to api
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: allow-front-egress }
spec:
  podSelector: { matchLabels: { app: front } }
  policyTypes: [Egress]
  egress:
    - to: [{ podSelector: { matchLabels: { app: api } } }]
      ports: [{ protocol: TCP, port: 5678 }]
```

### 7. Re-test
From the front pod: `curl api` works. From any other pod: it doesn't.

## Validation
- [ ] Before policies: all probes succeed.
- [ ] After default-deny + allow-dns: only DNS works.
- [ ] After allow-front-to-api: only the front pod reaches api.
- [ ] `db` is still isolated.

## Cleanup
```bash
kubectl delete netpol --all
kubectl delete deploy,svc front api db
```

## Troubleshooting
- **Policies have no effect** — CNI doesn't enforce NetworkPolicies (e.g., flannel). Use Calico or Cilium.
- **DNS breaks everything after default-deny** — Forgot `allow-dns` policy. Always add it first.
- **`namespaceSelector` doesn't match** — kube-system needs label `kubernetes.io/metadata.name: kube-system` (default in K8s 1.21+).
