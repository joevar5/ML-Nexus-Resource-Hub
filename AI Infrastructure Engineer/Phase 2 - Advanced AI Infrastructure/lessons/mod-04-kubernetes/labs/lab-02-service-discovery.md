# Lab 02: Service Discovery — ClusterIP, NodePort, LoadBalancer

**Duration:** 45 min  **Prerequisites:** Lab 01 complete

## Objective
Understand how Services route traffic to Pods, the difference between ClusterIP/NodePort/LoadBalancer, and how DNS works inside the cluster.

## Steps

### 1. ClusterIP (default)
```yaml
apiVersion: v1
kind: Service
metadata: { name: web }
spec:
  selector: { app: web }
  ports: [{ port: 80, targetPort: 80 }]
```
```bash
kubectl apply -f svc.yaml
kubectl get svc web
```

### 2. Hit it from inside the cluster
```bash
kubectl run probe --rm -it --image=curlimages/curl --restart=Never -- sh
# Inside the probe container:
curl http://web                       # by short name (same namespace)
curl http://web.default.svc.cluster.local
nslookup web                          # resolves to ClusterIP
exit
```

### 3. NodePort — exposes a port on every node
```yaml
spec:
  type: NodePort
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30080
```
```bash
kubectl apply -f svc.yaml
curl http://localhost:8080            # via kind port-mapping
```

### 4. LoadBalancer (cloud) or via metallb (local)
On a cloud cluster, `type: LoadBalancer` allocates an external IP. On kind, install metallb for the same UX:
```bash
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.14.5/config/manifests/metallb-native.yaml
# Then configure an address pool (see metallb docs).
```

### 5. Inspect endpoints
```bash
kubectl get endpoints web              # one IP per Ready pod
kubectl describe svc web | head -20
```

### 6. Service vs label selector mismatches
Change the Deployment's pod label to `app: web-v2`. Service selector still says `app: web`. Watch endpoints empty out:
```bash
kubectl get endpoints web -w
```

## Validation
- [ ] DNS resolves `web` to the ClusterIP from inside the cluster.
- [ ] `kubectl get endpoints web` lists one IP per Ready pod.
- [ ] Breaking the label selector → endpoints empty → external traffic fails.

## Cleanup
```bash
kubectl delete svc web
```

## Troubleshooting
- **`Service has no endpoints`** — Label selector doesn't match any Ready pod. Check `kubectl get pods --show-labels`.
- **NodePort unreachable from host** — kind's port mapping must include the nodePort (see kind-config.yaml from mod-101 lab 03).
- **DNS resolution fails inside cluster** — CoreDNS pod unhealthy. `kubectl get pods -n kube-system | grep dns`.
