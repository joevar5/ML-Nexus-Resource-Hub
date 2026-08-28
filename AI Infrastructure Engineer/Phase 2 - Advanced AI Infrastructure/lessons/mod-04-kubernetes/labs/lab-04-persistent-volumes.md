# Lab 04: Persistent Volumes for Stateful Workloads

**Duration:** 75 min  **Prerequisites:** Lab 03 complete

## Objective
Provision storage via PersistentVolumeClaims, deploy a StatefulSet (Postgres), and verify data survives pod restarts.

## Steps

### 1. Check StorageClasses
```bash
kubectl get storageclass
```
kind ships with `standard` (local-path provisioner).

### 2. Static PVC + Pod
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: data }
spec:
  accessModes: [ReadWriteOnce]
  resources: { requests: { storage: 1Gi } }
  storageClassName: standard
---
apiVersion: v1
kind: Pod
metadata: { name: writer }
spec:
  containers:
    - name: write
      image: busybox
      command: ["sh", "-c", "echo hello > /data/hello.txt; sleep 3600"]
      volumeMounts: [{ name: data, mountPath: /data }]
  volumes:
    - name: data
      persistentVolumeClaim: { claimName: data }
```
```bash
kubectl apply -f pvc.yaml
kubectl get pvc                 # Bound
kubectl exec writer -- cat /data/hello.txt   # hello
kubectl delete pod writer
kubectl apply -f pvc.yaml       # re-creates writer (different pod, same PVC)
kubectl exec writer -- cat /data/hello.txt   # hello — persisted
```

### 3. StatefulSet for Postgres
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata: { name: pg }
spec:
  serviceName: pg
  replicas: 1
  selector: { matchLabels: { app: pg } }
  template:
    metadata: { labels: { app: pg } }
    spec:
      containers:
        - name: pg
          image: postgres:15
          env:
            - { name: POSTGRES_PASSWORD, value: app }
          ports: [{ containerPort: 5432 }]
          volumeMounts: [{ name: data, mountPath: /var/lib/postgresql/data, subPath: pgdata }]
  volumeClaimTemplates:
    - metadata: { name: data }
      spec:
        accessModes: [ReadWriteOnce]
        resources: { requests: { storage: 5Gi } }
---
apiVersion: v1
kind: Service
metadata: { name: pg }
spec:
  clusterIP: None
  selector: { app: pg }
  ports: [{ port: 5432 }]
```
```bash
kubectl apply -f pg.yaml
kubectl get pvc                 # data-pg-0 Bound
kubectl exec -it pg-0 -- psql -U postgres -c "CREATE TABLE t (x int); INSERT INTO t VALUES (42);"
kubectl delete pod pg-0
sleep 5
kubectl exec -it pg-0 -- psql -U postgres -c "SELECT * FROM t;"  # 42 — survived
```

## Validation
- [ ] PVC binds within a few seconds.
- [ ] Data persists across pod deletion (both the writer pod and the StatefulSet pod).
- [ ] StatefulSet pod is named `pg-0` (predictable, ordinal).

## Cleanup
```bash
kubectl delete sts pg svc pg
kubectl delete pvc data-pg-0 data
```

## Troubleshooting
- **PVC stuck Pending** — No StorageClass with available capacity. On kind, ensure local-path provisioner is running: `kubectl get pods -n local-path-storage`.
- **Data gone after pod restart** — You used an `emptyDir` volume by mistake. Recheck the manifest.
- **StatefulSet pods don't get assigned stable hostnames** — Headless Service is missing. Service must have `clusterIP: None`.
