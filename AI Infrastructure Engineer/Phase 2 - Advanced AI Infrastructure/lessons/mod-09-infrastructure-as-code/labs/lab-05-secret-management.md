# Lab 06: Secret Management with Vault + Terraform

**Duration:** 75 min  **Prerequisites:** Docker, Terraform 1.6+

## Objective
Run HashiCorp Vault locally, configure it via Terraform, store secrets, retrieve them programmatically, and rotate them.

## Steps

### 1. Start Vault dev mode
```bash
docker run -d --name vault -p 8200:8200 --cap-add IPC_LOCK \
  -e VAULT_DEV_ROOT_TOKEN_ID=root \
  -e VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200 \
  hashicorp/vault:1.16
export VAULT_ADDR=http://localhost:8200
export VAULT_TOKEN=root
vault status
```

### 2. Terraform configures Vault
```hcl
terraform {
  required_providers {
    vault = { source = "hashicorp/vault", version = "~> 4.0" }
  }
}
provider "vault" { address = "http://localhost:8200"; token = "root" }

# KV v2 engine
resource "vault_mount" "kvv2" {
  path = "secret"; type = "kv"; options = { version = "2" }
}

# Policy for the app
resource "vault_policy" "app" {
  name   = "iris-app"
  policy = <<EOT
path "secret/data/iris/*" { capabilities = ["read"] }
EOT
}

# AppRole for the app to authenticate
resource "vault_auth_backend" "approle" { type = "approle" }
resource "vault_approle_auth_backend_role" "app" {
  backend         = vault_auth_backend.approle.path
  role_name       = "iris-app"
  token_policies  = [vault_policy.app.name]
}

# Seed initial secret
resource "vault_kv_secret_v2" "db" {
  mount = vault_mount.kvv2.path
  name  = "iris/db"
  data_json = jsonencode({ username = "iris_app"; password = "initial-pass-rotate-me" })
}
```
```bash
terraform init && terraform apply
```

### 3. App fetches secret via AppRole
```python
import hvac
client = hvac.Client(url="http://localhost:8200")
role_id   = "<from-vault-cli>"
secret_id = "<from-vault-cli>"
client.auth.approle.login(role_id=role_id, secret_id=secret_id)
secret = client.secrets.kv.v2.read_secret_version(mount_point="secret", path="iris/db")
print(secret["data"]["data"])
```

Get the IDs:
```bash
vault read auth/approle/role/iris-app/role-id
vault write -f auth/approle/role/iris-app/secret-id
```

### 4. Rotate
```hcl
data_json = jsonencode({ username = "iris_app"; password = "new-rotated-pass" })
```
`terraform apply` → reads now return the new value. Old version retained (KV v2 keeps history).

### 5. Read previous version
```python
secret = client.secrets.kv.v2.read_secret_version(
    mount_point="secret", path="iris/db", version=1)
```

### 6. Audit log
```bash
vault audit enable file file_path=/tmp/vault-audit.log
cat /tmp/vault-audit.log | head
```

## Validation
- [ ] AppRole login succeeds.
- [ ] App reads secret without using the root token.
- [ ] Rotation produces a new version; old version still readable.
- [ ] Audit log records reads.

## Cleanup
```bash
terraform destroy -auto-approve
docker stop vault && docker rm vault
```

## Troubleshooting
- **`Vault is sealed`** — Dev mode auto-unseals, but if you restart the container without `-e VAULT_DEV_ROOT_TOKEN_ID`, it'll generate a new token.
- **`permission denied`** — Policy doesn't grant the path. Test with `vault policy read iris-app`.
- **AppRole secret-id expires** — Default TTL applies; set `secret_id_ttl=0` on the role for non-expiring (test only).
