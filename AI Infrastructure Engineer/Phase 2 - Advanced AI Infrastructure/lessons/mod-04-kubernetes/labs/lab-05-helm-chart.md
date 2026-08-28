# Lab 05: Package an App with Helm

**Duration:** 75 min  **Prerequisites:** Helm 3 installed; kind cluster

## Objective
Convert the Lab 04 model-serving manifests into a parameterized Helm chart, install it, override values per environment, and roll back.

## Steps

### 1. Scaffold
```bash
helm create iris-chart
ls iris-chart/
# Chart.yaml  values.yaml  templates/  charts/  .helmignore
```

### 2. Replace template defaults
Strip out the generated boilerplate and write minimal templates:
```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "iris-chart.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector: { matchLabels: { app: {{ include "iris-chart.name" . }} } }
  template:
    metadata: { labels: { app: {{ include "iris-chart.name" . }} } }
    spec:
      containers:
        - name: api
          image: "{{ .Values.image.repo }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports: [{ containerPort: 8000 }]
```

### 3. values.yaml
```yaml
replicaCount: 2
image:
  repo: iris-api
  tag: "0.2"
  pullPolicy: Never
service:
  port: 80
  nodePort: 30080
```

### 4. Render and lint
```bash
helm lint iris-chart
helm template demo iris-chart           # render to stdout
helm template demo iris-chart -f values-prod.yaml
```

### 5. Install
```bash
helm install iris ./iris-chart
helm list
kubectl get all -l app=iris-chart
```

### 6. Upgrade
```bash
helm upgrade iris ./iris-chart --set replicaCount=4
helm history iris
```

### 7. Rollback
```bash
helm rollback iris 1
helm history iris
```

### 8. Uninstall
```bash
helm uninstall iris
```

## Validation
- [ ] `helm lint` returns 0 errors.
- [ ] `helm template` renders all manifests with values substituted.
- [ ] After `--set replicaCount=4`, the cluster shows 4 pods.
- [ ] `helm rollback` reverts replicas to 2.

## Cleanup
```bash
helm uninstall iris 2>/dev/null
rm -rf iris-chart
```

## Troubleshooting
- **Template render error** — `helm template --debug` shows the rendered output even when invalid. Look for missing `.Values.x.y` references.
- **Install hangs at "waiting"** — Use `--timeout 1m --atomic` so failures roll back automatically.
- **`Error: release iris failed: cannot re-use a name`** — Previous failed install left a record. `helm uninstall iris` first.
