# Exercise 07: Author a Production-Grade Helm Chart

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Lab 05 (helm basics)

## Objective

Author a substantial, production-quality Helm chart for the iris-api service with templates, values overrides, dependencies, tests, NOTES.txt, and a CI workflow validating it.

## Requirements

1. Chart with at least 10 templates.
2. `values.yaml` covering image, replicas, resources, ingress, autoscaling, monitoring, security context.
3. Multiple `values-<env>.yaml` for dev/staging/prod.
4. Sub-chart dependency (e.g., postgresql for metadata).
5. `helm test` hooks validating the deployment works.
6. `NOTES.txt` with helpful post-install instructions.
7. CI runs `helm lint`, `helm template`, and `kubeval`/`kubeconform`.

## Step-by-step

### Step 1 — Scaffold (10 min)
```bash
helm create iris-api
cd iris-api
```

### Step 2 — Strip + rewrite (45 min)
Delete the boilerplate. Write minimal but complete templates:
- `templates/_helpers.tpl` — name and label helpers
- `templates/deployment.yaml`
- `templates/service.yaml`
- `templates/ingress.yaml` (conditional on `.Values.ingress.enabled`)
- `templates/hpa.yaml` (conditional)
- `templates/configmap.yaml`
- `templates/secret.yaml`
- `templates/serviceaccount.yaml`
- `templates/servicemonitor.yaml` (conditional on `.Values.monitoring.enabled`)
- `templates/networkpolicy.yaml` (conditional)
- `templates/tests/test-connection.yaml` — `helm test` hook

### Step 3 — values.yaml (30 min)
```yaml
replicaCount: 2
image:
  repository: iris-api
  tag: ""        # default to chart appVersion
  pullPolicy: IfNotPresent

ingress:
  enabled: false
  className: nginx
  hosts:
    - host: iris.local
      paths: [{ path: /, pathType: Prefix }]

autoscaling:
  enabled: false
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

resources:
  requests: { cpu: 100m, memory: 256Mi }
  limits:   { cpu: 500m, memory: 512Mi }

monitoring:
  enabled: false
  serviceMonitor:
    interval: 30s

postgresql:
  enabled: false       # use sub-chart for metadata storage
  auth: { username: iris, database: iris }
```

### Step 4 — Sub-chart dependency (30 min)
```yaml
# Chart.yaml
dependencies:
  - name: postgresql
    version: 15.5.0
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
```
```bash
helm dependency update
```

### Step 5 — Test hook (15 min)
```yaml
# templates/tests/test-connection.yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ include "iris-api.fullname" . }}-test-connection"
  annotations: { "helm.sh/hook": test }
spec:
  restartPolicy: Never
  containers:
    - name: wget
      image: busybox
      command: ["wget"]
      args: ["{{ include "iris-api.fullname" . }}:{{ .Values.service.port }}/health"]
```

### Step 6 — NOTES.txt (15 min)
```
Thank you for installing iris-api!

To get started:
{{- if .Values.ingress.enabled }}
  Visit: http://{{ (index .Values.ingress.hosts 0).host }}
{{- else }}
  kubectl port-forward svc/{{ include "iris-api.fullname" . }} 8080:80
  Then: curl http://localhost:8080/health
{{- end }}

Run tests:
  helm test {{ .Release.Name }}
```

### Step 7 — Per-env values (15 min)
- `values-dev.yaml`: 1 replica, no ingress, minimal resources.
- `values-staging.yaml`: 2 replicas, ingress with staging cert.
- `values-prod.yaml`: HPA on, monitoring on, ingress with prod cert, NetworkPolicy on.

### Step 8 — Install + verify (15 min)
```bash
helm install iris ./iris-api -f values-dev.yaml
helm test iris
helm upgrade iris ./iris-api -f values-staging.yaml --atomic
helm history iris
helm rollback iris 1
```

### Step 9 — CI workflow (15 min)
```yaml
on: pull_request
jobs:
  lint-and-template:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: azure/setup-helm@v4
      - run: helm lint iris-api/
      - run: helm dependency update iris-api/
      - run: helm template test iris-api/ -f iris-api/values-prod.yaml | kubeconform -strict -
```

## Deliverables

1. Complete chart with 10+ templates.
2. 3 per-env values files.
3. `helm test` hook.
4. CI workflow.
5. `CHART_README.md` documenting all values.

## Validation

- [ ] `helm lint` passes.
- [ ] `helm template` produces valid YAML for all three envs.
- [ ] `helm install` + `helm test` succeeds.
- [ ] `helm upgrade` swaps the values atomically.
- [ ] `helm rollback` reverts.
- [ ] CI validates on every PR.

## Stretch goals

- Add a **library chart** for shared template helpers used across multiple service charts.
- Add **unit tests** via `helm-unittest`.
- Publish the chart to a **helm repository** (chartmuseum, OCI registry).
