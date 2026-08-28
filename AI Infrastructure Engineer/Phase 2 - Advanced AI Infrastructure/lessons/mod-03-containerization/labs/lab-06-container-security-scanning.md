# Lab 06: Security-Scan Images with Trivy

**Duration:** 45 min  **Prerequisites:** Docker; Trivy installed

## Objective
Scan a real Docker image for OS, language, and secret vulnerabilities; produce a report; and configure a CI gate that fails on HIGH/CRITICAL findings.

## Steps

### 1. Install Trivy
```bash
brew install trivy                # macOS
# Linux: see https://aquasecurity.github.io/trivy/latest/getting-started/installation/

trivy --version
```

### 2. Scan an image
```bash
docker pull python:3.11             # known to have some old vulns
trivy image --severity HIGH,CRITICAL --format table python:3.11
```

### 3. Generate JSON for tooling
```bash
trivy image --format json --output trivy-report.json python:3.11
jq '.Results[].Vulnerabilities | length' trivy-report.json
```

### 4. Scan filesystem for secrets
```bash
trivy fs --scanners secret --severity HIGH,CRITICAL .
```

### 5. Scan IaC and Dockerfile
```bash
trivy config Dockerfile
trivy config terraform/
```

### 6. CI gate
GitHub Actions step:
```yaml
- name: Trivy scan
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: ghcr.io/${{ github.repository }}/iris-api:${{ github.sha }}
    severity: 'HIGH,CRITICAL'
    exit-code: '1'           # fail the build on any HIGH/CRITICAL
    ignore-unfixed: true
    format: 'sarif'
    output: 'trivy.sarif'

- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v3
  with: { sarif_file: 'trivy.sarif' }
```

### 7. Suppress accepted risk
Create `.trivyignore`:
```
CVE-2023-12345  # accepted: only affects build tools not present at runtime
```

## Validation
- [ ] `trivy image python:3.11` reports at least one HIGH or CRITICAL CVE.
- [ ] Same scan on a minimal `python:3.11-slim-bookworm` reports fewer.
- [ ] CI workflow fails on a deliberately vulnerable image.

## Cleanup
```bash
rm -f trivy-report.json trivy.sarif .trivyignore
docker rmi python:3.11
```

## Troubleshooting
- **Trivy slow first run** — It's downloading the vulnerability DB. Subsequent scans use the cache.
- **False positives flagged** — Use `.trivyignore` or run with `--ignore-unfixed` to skip CVEs with no upstream fix yet.
- **Different severity counts on different runs** — DB updates daily; pin DB version in CI for reproducibility (`trivy image --db-repository ...` or vendor the DB).
