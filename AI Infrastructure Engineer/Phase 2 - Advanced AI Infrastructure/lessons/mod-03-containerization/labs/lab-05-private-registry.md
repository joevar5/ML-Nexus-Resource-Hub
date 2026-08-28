# Lab 05: Stand Up a Private Container Registry

**Duration:** 45 min  **Prerequisites:** Docker installed

## Objective
Run a local Docker Registry v2, push and pull images to/from it, then add basic auth and TLS.

## Steps

### 1. Run an insecure local registry
```bash
docker run -d -p 5000:5000 --name registry registry:2
docker tag iris-api:0.2 localhost:5000/iris-api:0.2
docker push localhost:5000/iris-api:0.2
docker pull localhost:5000/iris-api:0.2
```

### 2. Add basic auth
```bash
mkdir auth
docker run --rm --entrypoint htpasswd httpd:2 -Bbn admin admin > auth/htpasswd

docker rm -f registry
docker run -d -p 5000:5000 \
  -v "$PWD/auth:/auth" \
  -e REGISTRY_AUTH=htpasswd \
  -e REGISTRY_AUTH_HTPASSWD_REALM='Registry Realm' \
  -e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
  --name registry registry:2

docker logout localhost:5000
docker pull localhost:5000/iris-api:0.2     # 401
docker login localhost:5000                 # admin/admin
docker pull localhost:5000/iris-api:0.2     # 200
```

### 3. Add TLS
Generate a self-signed cert:
```bash
mkdir certs
openssl req -newkey rsa:4096 -nodes -sha256 -x509 -days 365 \
  -subj '/CN=registry.local' \
  -addext 'subjectAltName=DNS:registry.local,DNS:localhost,IP:127.0.0.1' \
  -keyout certs/domain.key -out certs/domain.crt
```
Trust the cert on the Docker host (macOS: `security add-trusted-cert`; Linux: copy to `/etc/docker/certs.d/registry.local:5000/ca.crt`).

Restart registry with TLS:
```bash
docker rm -f registry
docker run -d -p 5000:5000 \
  -v "$PWD/auth:/auth" -v "$PWD/certs:/certs" \
  -e REGISTRY_AUTH=htpasswd -e REGISTRY_AUTH_HTPASSWD_REALM='Realm' \
  -e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
  -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
  -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
  --name registry registry:2
```

### 4. List and inspect
```bash
curl -u admin:admin -k https://localhost:5000/v2/_catalog
curl -u admin:admin -k https://localhost:5000/v2/iris-api/tags/list
```

## Validation
- [ ] Push without auth fails after step 2.
- [ ] Push without TLS trust fails after step 3.
- [ ] `curl /v2/_catalog` lists `iris-api`.

## Cleanup
```bash
docker rm -f registry
rm -rf auth certs
```

## Troubleshooting
- **`http: server gave HTTP response to HTTPS client`** — Docker daemon expects HTTPS; you're talking to plain HTTP. Add `localhost:5000` to `insecure-registries` in daemon.json for insecure-mode tests.
- **Self-signed cert rejected** — Trust the CA on the host; restart Docker daemon after installing.
