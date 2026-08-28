# Exercise 07: Secret Management Across IaC + Runtime

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Vault basics (lab 06)

## Objective

Implement an end-to-end secret management workflow: HashiCorp Vault as source of truth, Terraform fetches secrets at provisioning time, Kubernetes apps fetch via External Secrets Operator (ESO). No plaintext secrets in Git anywhere.

## Why this matters

Secrets in Git → credential rotation requires repo changes → friction → secrets don't rotate. With Vault + ESO, rotation is automated and audit-logged. Engineers can deploy infrastructure without ever seeing a credential.

## Requirements

1. Vault running in production-shape mode (not dev).
2. Terraform reads from Vault for provider credentials.
3. K8s Secrets generated from Vault via ESO.
4. Automated 30-day rotation on at least one secret.
5. Audit log queryable.

## Step-by-step

### Step 1 — Vault production-shape setup (45 min)
Don't use dev mode. Production-shape:
```yaml
# vault-helm-values.yaml
server:
  ha:
    enabled: true
    replicas: 3
    raft:
      enabled: true
  dataStorage:
    enabled: true
    size: 10Gi
  auditStorage:
    enabled: true
    size: 5Gi
```
```bash
helm install vault hashicorp/vault -n vault --create-namespace -f vault-helm-values.yaml
# Initialize + unseal (5 keys, 3 needed)
kubectl exec -n vault vault-0 -- vault operator init
# Distribute keys to 5 trusted humans in different time zones.
```

### Step 2 — Enable secrets engine (15 min)
```bash
vault secrets enable -path=ml -version=2 kv
vault kv put ml/iris/db username=app password=initial-pass
vault kv put ml/iris/api-token value=abc123
```

### Step 3 — Terraform reads from Vault (45 min)
```hcl
data "vault_kv_secret_v2" "db" {
  mount = "ml"
  name  = "iris/db"
}

resource "aws_db_instance" "iris" {
  username = data.vault_kv_secret_v2.db.data["username"]
  password = data.vault_kv_secret_v2.db.data["password"]
  ...
}
```
Never write the password to Terraform variables or state-readable inputs (use `sensitive = true`).

### Step 4 — External Secrets Operator (30 min)
```bash
helm install external-secrets external-secrets/external-secrets -n external-secrets --create-namespace
```

### Step 5 — SecretStore + ExternalSecret (30 min)
```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata: { name: vault, namespace: iris }
spec:
  provider:
    vault:
      server: "http://vault.vault:8200"
      path: "ml"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "iris"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata: { name: iris-db, namespace: iris }
spec:
  refreshInterval: "1h"
  secretStoreRef: { name: vault, kind: SecretStore }
  target: { name: iris-db, creationPolicy: Owner }
  data:
    - secretKey: username
      remoteRef: { key: iris/db, property: username }
    - secretKey: password
      remoteRef: { key: iris/db, property: password }
```

K8s Secret `iris-db` materializes from Vault. Pod consumes via `secretKeyRef` like any other secret.

### Step 6 — Rotate (30 min)
Manual:
```bash
NEW=$(openssl rand -base64 32)
vault kv put ml/iris/db username=app password="$NEW"
# ESO refreshes within 1h (or trigger manually: kubectl annotate externalsecret iris-db force-sync=$(date +%s))
# Pod env vars use the new password on next restart (or use fsync for live reload)
```

Automated rotation: Vault DB Secrets Engine generates short-lived (1h TTL) DB credentials, rotates without app changes.

### Step 7 — Audit log (15 min)
```bash
vault audit enable file file_path=/vault/audit/audit.log
# Every Vault operation logged with actor + secret path
```
Query: `tail -f /vault/audit/audit.log | jq 'select(.request.path | startswith("ml/iris"))'`

## Deliverables

1. Vault HA installed.
2. Terraform pulling secrets from Vault.
3. ESO syncing at least 2 secrets into K8s.
4. Rotation demonstrated end-to-end (Vault update → K8s Secret updates within 1h).
5. `SECRET_LIFECYCLE.md`: how to add a secret, rotate, retire.

## Validation

- [ ] No plaintext secrets in any Git repo.
- [ ] Terraform applies without engineer having seen the password.
- [ ] ExternalSecret reflects Vault changes within refresh interval.
- [ ] Audit log captures every access.

## Stretch goals

- Use **Vault Database Secrets Engine** for dynamic short-lived DB credentials.
- Add **AWS IAM credential generation** via Vault (no static AWS keys).
- Implement **PKI** (cert issuance via Vault for mTLS).

## Common pitfalls

- **Terraform state contains secrets** — Mark them `sensitive = true`; encrypt the state at rest.
- **ESO refresh interval too long** — Rotation has long lag. Trade refresh frequency vs Vault load.
- **Plaintext fallback in Helm chart** — `valuesFile` with default secret values. Always require secret via env at runtime.
- **Audit log on local disk** — Lost if Vault pod recreated. Ship to S3/SIEM.
