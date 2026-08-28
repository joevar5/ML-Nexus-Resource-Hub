# Exercise 12: Stateful Postgres on Kubernetes

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Lab 04 (PVs)

## Objective

Deploy production-shape Postgres on K8s using the CloudNative-PG operator: 3-replica cluster with streaming replication, automated backups to S3, point-in-time recovery, monitoring.

## Requirements

1. CloudNative-PG operator installed.
2. Cluster with 1 primary + 2 replicas.
3. Continuous WAL archive to S3 (Minio for lab).
4. Automated daily backup.
5. Test PITR: restore to a specific point in time.
6. Prometheus metrics from Postgres exporter.

## Step-by-step

### Step 1 — Install operator (15 min)
```bash
kubectl apply --server-side -f https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/release-1.23/releases/cnpg-1.23.1.yaml
kubectl rollout status -n cnpg-system deployment/cnpg-controller-manager
```

### Step 2 — Run Minio for S3-compatible storage (15 min)
```bash
helm install minio bitnami/minio --set auth.rootUser=minio --set auth.rootPassword=minio123
```

### Step 3 — Cluster definition (45 min)
```yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata: { name: pg }
spec:
  instances: 3
  primaryUpdateStrategy: unsupervised

  bootstrap:
    initdb:
      database: app
      owner: app

  storage:
    size: 10Gi
    storageClass: standard

  monitoring:
    enablePodMonitor: true

  backup:
    barmanObjectStore:
      destinationPath: "s3://backups/pg"
      endpointURL: "http://minio:9000"
      s3Credentials:
        accessKeyId: { name: minio-creds, key: ACCESS_KEY_ID }
        secretAccessKey: { name: minio-creds, key: SECRET_ACCESS_KEY }
      wal:
        compression: gzip
      data:
        compression: gzip
    retentionPolicy: "30d"

  resources:
    requests: { memory: 1Gi, cpu: 500m }
    limits:   { memory: 2Gi, cpu: 1 }
```

### Step 4 — Scheduled backup (15 min)
```yaml
apiVersion: postgresql.cnpg.io/v1
kind: ScheduledBackup
metadata: { name: pg-daily }
spec:
  schedule: "0 2 * * *"
  cluster: { name: pg }
  backupOwnerReference: cluster
```

### Step 5 — Verify (30 min)
```bash
kubectl get cluster pg          # status Ready
kubectl get pods -l postgresql=pg

# Primary + 2 replicas
kubectl exec -it pg-1 -- psql -U app -d app -c "INSERT INTO test VALUES (1,'hello');"

# Verify replication
kubectl exec -it pg-2 -- psql -U app -d app -c "SELECT * FROM test;"
```

### Step 6 — Manual backup + restore (30 min)
```bash
kubectl create -f - <<EOF
apiVersion: postgresql.cnpg.io/v1
kind: Backup
metadata: { name: manual-1 }
spec: { cluster: { name: pg } }
EOF

# Wait until status: completed
# Now simulate disaster:
kubectl delete cluster pg

# Restore to a new cluster from backup
kubectl apply -f - <<EOF
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata: { name: pg-restored }
spec:
  instances: 1
  bootstrap:
    recovery:
      backup: { name: manual-1 }
  ...rest as original
EOF
```

### Step 7 — Point-in-time recovery (30 min)
```yaml
bootstrap:
  recovery:
    backup: { name: manual-1 }
    recoveryTarget:
      targetTime: "2026-05-22 15:30:00.000000+00"
```

### Step 8 — Monitoring (15 min)
PodMonitor created automatically. Add Grafana dashboard:
```bash
helm install grafana grafana/grafana --set adminPassword=admin
# Import dashboard 9628 (PostgreSQL Database)
```

## Deliverables

1. Cluster + ScheduledBackup manifests.
2. Demonstration of automated + manual backup.
3. Successful PITR.
4. Grafana dashboard showing Postgres metrics.

## Validation

- [ ] 3 pods Ready; replication lag near zero.
- [ ] Backup files in Minio bucket.
- [ ] Restored cluster has the same data + can be PITR'd.
- [ ] Metrics visible in Grafana.

## Common pitfalls

- **CNPG-PG version mismatch** — Operator version must support the Postgres major version.
- **WAL archive lag** — If S3 endpoint is slow, WAL builds up and Postgres goes read-only.
- **Restore to same cluster name** — Conflicts; restore to a new name and rename only after verification.
- **Forgetting `recoveryTarget` precision** — Without it, recovery goes to end of WAL (not point-in-time).
