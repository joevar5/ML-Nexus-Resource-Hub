# Lab 03: ConfigMaps and Secrets

**Duration:** 60 min  **Prerequisites:** Lab 02 complete

## Objective
Externalize app configuration with ConfigMaps and Secrets, mount as env vars and as files, and rotate without restarting unnecessarily.

## Steps

### 1. Create a ConfigMap from literals and a file
```bash
echo "log_level: info" > app.conf
kubectl create configmap app-config --from-literal=APP_ENV=staging --from-file=app.conf=app.conf
kubectl get cm app-config -o yaml
```

### 2. Mount as env vars
```yaml
spec:
  containers:
    - name: api
      envFrom:
        - configMapRef: { name: app-config }
```

### 3. Mount the file at /etc/app/
```yaml
      volumeMounts:
        - name: cfg
          mountPath: /etc/app
  volumes:
    - name: cfg
      configMap: { name: app-config }
```

### 4. Create a Secret
```bash
kubectl create secret generic db-creds \
  --from-literal=username=app \
  --from-literal=password='s3cr3t!'
kubectl get secret db-creds -o yaml      # base64-encoded
```

### 5. Project Secret keys as env vars
```yaml
      env:
        - name: DB_USER
          valueFrom:
            secretKeyRef: { name: db-creds, key: username }
        - name: DB_PASS
          valueFrom:
            secretKeyRef: { name: db-creds, key: password }
```

### 6. Rotate Secret without restart (file mounts only)
```bash
kubectl create secret generic db-creds --from-literal=password='n3wp4ss' --dry-run=client -o yaml \
  | kubectl apply -f -
```
File mounts update within ~60s (kubelet sync). Env-var injections require pod restart.

### 7. Encrypted secrets — SealedSecrets or External Secrets
Sketch:
```bash
helm install sealed-secrets sealed-secrets/sealed-secrets -n kube-system
echo -n 'mypass' | kubectl create secret generic test --dry-run=client --from-file=password=/dev/stdin -o yaml \
  | kubeseal --format yaml > sealed-secret.yaml      # safe to commit
```

## Validation
- [ ] `kubectl exec pod -- env | grep APP_ENV` shows `staging`.
- [ ] `kubectl exec pod -- cat /etc/app/app.conf` shows your file.
- [ ] After rotating Secret values, file mounts reflect new values within 60s without pod restart.
- [ ] Trying to view the Secret with `kubectl get secret db-creds -o yaml` shows base64 (not plaintext).

## Cleanup
```bash
kubectl delete cm app-config secret db-creds
```

## Troubleshooting
- **Pod doesn't restart on env change** — Env vars are injected at start. Restart manually: `kubectl rollout restart deployment/...`.
- **Secret looks "encrypted" but isn't** — Base64 is encoding, not encryption. Use SealedSecrets or External Secrets for at-rest encryption.
- **File mount didn't update** — Check kubelet sync interval; on kind it's 60s by default.
