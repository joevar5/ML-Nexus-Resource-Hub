# Lesson 07: Helm Package Manager

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand what Helm is and why it's valuable for ML infrastructure
- Install and configure Helm 3
- Deploy applications using Helm charts
- Create custom Helm charts for ML applications
- Use Helm values to configure deployments
- Manage chart repositories
- Implement Helm best practices for ML platforms
- Debug and troubleshoot Helm deployments

## Prerequisites
- Completed lessons 01-06 (Kubernetes fundamentals)
- Understanding of YAML and templating concepts
- Familiarity with Kubernetes resources (Deployments, Services, etc.)
- Basic command-line skills

## Introduction

**Helm** is the package manager for Kubernetes (like apt for Ubuntu, brew for macOS).

**Why Helm matters for ML infrastructure:**
- **Reusability:** Package complex ML stacks (MLflow, Kubeflow, etc.) for easy deployment
- **Configuration management:** Single values file to configure entire ML platform
- **Versioning:** Roll back to previous versions of ML infrastructure
- **Templating:** DRY principle - don't repeat Kubernetes YAML
- **Ecosystem:** 1000+ pre-built charts (Prometheus, Grafana, Airflow, etc.)

**Real-world examples:**
- **Uber:** Uses Helm to deploy ML platform components across 1000+ K8s clusters
- **Spotify:** Deploys personalization ML services via custom Helm charts
- **Airbnb:** Manages experimentation infrastructure with Helm
- **Netflix:** Deploys recommendation system components using Helm

**Without Helm:**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
# ... 50 more files
# How do you manage versions? How do you configure for dev vs prod?
```

**With Helm:**
```bash
helm install ml-platform ./ml-platform-chart --values prod-values.yaml
# Done! All resources deployed, configured, versioned.
```

## 1. Helm Concepts

### 1.1 Core Components

**Chart:**
- Package containing Kubernetes manifests
- Like a .deb or .rpm package
- Directory structure with templates and metadata

**Release:**
- Instance of a chart running in a cluster
- You can install the same chart multiple times (different releases)
- Example: `ml-platform-dev`, `ml-platform-staging`, `ml-platform-prod`

**Repository:**
- Collection of charts
- Can be public (ArtifactHub) or private (ChartMuseum, AWS ECR)

**Values:**
- Configuration parameters for a chart
- Override default settings
- Different values for dev/staging/prod

### 1.2 Helm Architecture (Helm 3)

```
┌─────────────────────────────────────────────┐
│           User's Machine                    │
│                                             │
│  ┌────────────┐                             │
│  │ Helm CLI   │                             │
│  └─────┬──────┘                             │
│        │                                    │
│        │ 1. helm install                    │
│        │    --values prod.yaml              │
│        │                                    │
└────────┼────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Kubernetes Cluster                  │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │    Kubernetes API Server           │    │
│  │  (Helm stores release info in      │    │
│  │   Secrets/ConfigMaps)              │    │
│  └────────────┬───────────────────────┘    │
│               │                             │
│               │ 2. Create resources         │
│               ▼                             │
│  ┌────────────────────────────────────┐    │
│  │  Deployment, Service, Ingress, ... │    │
│  └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘

Note: Helm 3 removed Tiller (server-side component)
All operations are client-side + K8s API
```

## 2. Installing and Using Helm

### 2.1 Install Helm 3

```bash
# Install Helm on Linux
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Or via package manager
# macOS
brew install helm

# Verify installation
helm version
# version.BuildInfo{Version:"v3.12.0", ...}

# Add common repositories
helm repo add stable https://charts.helm.sh/stable
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

### 2.2 Installing a Chart

**Example: Install Prometheus for monitoring**

```bash
# Search for Prometheus charts
helm search repo prometheus

# Install Prometheus
helm install my-prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace

# List installed releases
helm list -n monitoring
# NAME           NAMESPACE  REVISION  STATUS    CHART
# my-prometheus  monitoring 1         deployed  prometheus-15.10.0

# Check deployed resources
kubectl get all -n monitoring
```

### 2.3 Customizing with Values

**View default values:**

```bash
helm show values prometheus-community/prometheus > prometheus-default-values.yaml
```

**Create custom values file:**

```yaml
# prometheus-custom-values.yaml
server:
  persistentVolume:
    enabled: true
    size: 50Gi
    storageClass: gp3-balanced

  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 1000m
      memory: 2Gi

alertmanager:
  enabled: false  # Disable alertmanager

pushgateway:
  enabled: false  # Disable pushgateway
```

**Install with custom values:**

```bash
helm install my-prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace \
  --values prometheus-custom-values.yaml
```

### 2.4 Upgrading and Rolling Back

**Upgrade a release:**

```bash
# Modify values file
# ... edit prometheus-custom-values.yaml ...

# Upgrade with new values
helm upgrade my-prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --values prometheus-custom-values.yaml

# View release history
helm history my-prometheus -n monitoring
# REVISION  STATUS     CHART                 DESCRIPTION
# 1         superseded prometheus-15.10.0    Install complete
# 2         deployed   prometheus-15.10.0    Upgrade complete
```

**Roll back to previous version:**

```bash
# Rollback to revision 1
helm rollback my-prometheus 1 -n monitoring

# Verify rollback
helm history my-prometheus -n monitoring
# REVISION  STATUS     CHART                 DESCRIPTION
# 1         superseded prometheus-15.10.0    Install complete
# 2         superseded prometheus-15.10.0    Upgrade complete
# 3         deployed   prometheus-15.10.0    Rollback to 1
```

### 2.5 Uninstalling a Release

```bash
# Uninstall release (deletes all resources)
helm uninstall my-prometheus -n monitoring

# Keep history for potential rollback
helm uninstall my-prometheus -n monitoring --keep-history
```

## 3. Creating Custom Helm Charts

### 3.1 Chart Structure

```
ml-model-serving/
├── Chart.yaml          # Chart metadata
├── values.yaml         # Default configuration values
├── templates/          # Kubernetes manifest templates
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── _helpers.tpl   # Template helpers
│   └── NOTES.txt      # Post-install notes
├── charts/             # Dependent charts (subcharts)
└── README.md           # Chart documentation
```

### 3.2 Create a Chart for ML Model Serving

**Step 1: Generate chart scaffold**

```bash
helm create ml-model-serving
cd ml-model-serving
```

**Step 2: Define Chart.yaml**

```yaml
# Chart.yaml
apiVersion: v2
name: ml-model-serving
description: Helm chart for deploying ML model inference services
type: application
version: 1.0.0  # Chart version
appVersion: "1.0"  # Application version

keywords:
  - ml
  - machine-learning
  - inference
  - model-serving

maintainers:
  - name: ML Platform Team
    email: ml-platform@company.com

dependencies: []  # No subcharts for now
```

**Step 3: Define default values.yaml**

```yaml
# values.yaml
replicaCount: 3

image:
  repository: myregistry.io/bert-inference
  tag: "v1.0.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: nginx
  host: ml-api.example.com
  path: /bert
  tls:
    enabled: true
    secretName: ml-api-tls

resources:
  limits:
    cpu: 4000m
    memory: 8Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 2000m
    memory: 4Gi
    nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

model:
  name: "bert-base-uncased"
  path: "/models/bert"
  batchSize: 32

env:
  - name: LOG_LEVEL
    value: "INFO"
  - name: WORKERS
    value: "4"
```

**Step 4: Create templates/deployment.yaml**

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ml-model-serving.fullname" . }}
  labels:
    {{- include "ml-model-serving.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "ml-model-serving.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "ml-model-serving.selectorLabels" . | nindent 8 }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - name: http
          containerPort: {{ .Values.service.targetPort }}
          protocol: TCP

        env:
        - name: MODEL_NAME
          value: {{ .Values.model.name | quote }}
        - name: MODEL_PATH
          value: {{ .Values.model.path | quote }}
        - name: BATCH_SIZE
          value: {{ .Values.model.batchSize | quote }}
        {{- range .Values.env }}
        - name: {{ .name }}
          value: {{ .value | quote }}
        {{- end }}

        resources:
          {{- toYaml .Values.resources | nindent 12 }}

        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 60
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 30
          periodSeconds: 5
```

**Step 5: Create templates/service.yaml**

```yaml
# templates/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "ml-model-serving.fullname" . }}
  labels:
    {{- include "ml-model-serving.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
  - port: {{ .Values.service.port }}
    targetPort: http
    protocol: TCP
    name: http
  selector:
    {{- include "ml-model-serving.selectorLabels" . | nindent 4 }}
```

**Step 6: Create templates/ingress.yaml**

```yaml
# templates/ingress.yaml
{{- if .Values.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "ml-model-serving.fullname" . }}
  labels:
    {{- include "ml-model-serving.labels" . | nindent 4 }}
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: {{ .Values.ingress.className }}
  {{- if .Values.ingress.tls.enabled }}
  tls:
  - hosts:
    - {{ .Values.ingress.host }}
    secretName: {{ .Values.ingress.tls.secretName }}
  {{- end }}
  rules:
  - host: {{ .Values.ingress.host }}
    http:
      paths:
      - path: {{ .Values.ingress.path }}
        pathType: Prefix
        backend:
          service:
            name: {{ include "ml-model-serving.fullname" . }}
            port:
              number: {{ .Values.service.port }}
{{- end }}
```

**Step 7: Create templates/_helpers.tpl**

```yaml
# templates/_helpers.tpl
{{/*
Expand the name of the chart.
*/}}
{{- define "ml-model-serving.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ml-model-serving.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ml-model-serving.labels" -}}
helm.sh/chart: {{ include "ml-model-serving.chart" . }}
{{ include "ml-model-serving.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ml-model-serving.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml-model-serving.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ml-model-serving.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}
```

### 3.3 Testing the Chart

**Lint the chart:**

```bash
helm lint ./ml-model-serving
# ==> Linting ./ml-model-serving
# [INFO] Chart.yaml: icon is recommended
# 1 chart(s) linted, 0 chart(s) failed
```

**Dry-run installation:**

```bash
helm install bert-model ./ml-model-serving \
  --namespace ml-serving \
  --dry-run --debug
# This will print all rendered YAML without actually installing
```

**Template rendering:**

```bash
# Render templates locally
helm template bert-model ./ml-model-serving \
  --values prod-values.yaml > rendered-manifests.yaml

# Inspect rendered YAML
cat rendered-manifests.yaml
```

**Install for real:**

```bash
helm install bert-model ./ml-model-serving \
  --namespace ml-serving \
  --create-namespace

# Verify installation
helm list -n ml-serving
kubectl get all -n ml-serving
```

## 4. Environment-Specific Values

### 4.1 Development Environment

```yaml
# values-dev.yaml
replicaCount: 1

image:
  tag: "latest"  # Use latest for dev

service:
  type: NodePort  # Use NodePort for easy testing

ingress:
  enabled: false  # No Ingress in dev

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
    nvidia.com/gpu: 0  # No GPU in dev
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: false  # No autoscaling in dev

env:
  - name: LOG_LEVEL
    value: "DEBUG"  # Verbose logging in dev
```

**Deploy to dev:**

```bash
helm install bert-dev ./ml-model-serving \
  --namespace ml-dev \
  --values values-dev.yaml
```

### 4.2 Production Environment

```yaml
# values-prod.yaml
replicaCount: 5

image:
  repository: myregistry.io/bert-inference
  tag: "v1.2.3"  # Specific version in prod

service:
  type: ClusterIP

ingress:
  enabled: true
  className: nginx
  host: ml-api.company.com
  path: /bert
  tls:
    enabled: true
    secretName: ml-api-prod-tls

resources:
  limits:
    cpu: 4000m
    memory: 8Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 2000m
    memory: 4Gi
    nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 5
  maxReplicas: 20
  targetCPUUtilizationPercentage: 60

env:
  - name: LOG_LEVEL
    value: "WARNING"  # Less verbose in prod
  - name: WORKERS
    value: "8"
```

**Deploy to prod:**

```bash
helm install bert-prod ./ml-model-serving \
  --namespace ml-prod \
  --values values-prod.yaml
```

## 5. Advanced Helm Features

### 5.1 Chart Dependencies (Subcharts)

**Example: ML platform with MLflow + PostgreSQL**

```yaml
# ml-platform/Chart.yaml
apiVersion: v2
name: ml-platform
version: 1.0.0
dependencies:
  - name: postgresql
    version: 12.x.x
    repository: https://charts.bitnami.com/bitnami
  - name: minio
    version: 12.x.x
    repository: https://charts.bitnami.com/bitnami
```

**Install dependencies:**

```bash
cd ml-platform
helm dependency update
# Downloaded postgresql-12.8.0.tgz
# Downloaded minio-12.6.0.tgz
```

**Override subchart values:**

```yaml
# ml-platform/values.yaml
postgresql:
  auth:
    username: mlflow
    password: changeme
    database: mlflow
  primary:
    persistence:
      size: 50Gi

minio:
  auth:
    rootUser: admin
    rootPassword: changeme
  persistence:
    size: 500Gi
```

### 5.2 Conditional Resources

**Example: Enable GPU support conditionally**

```yaml
# values.yaml
gpu:
  enabled: true
  count: 1
  model: "Tesla-V100"
```

```yaml
# templates/deployment.yaml
spec:
  template:
    spec:
      {{- if .Values.gpu.enabled }}
      nodeSelector:
        nvidia.com/gpu.product: {{ .Values.gpu.model }}
      tolerations:
      - key: nvidia.com/gpu
        operator: Equal
        value: present
        effect: NoSchedule
      {{- end }}

      containers:
      - name: inference
        resources:
          {{- if .Values.gpu.enabled }}
          limits:
            nvidia.com/gpu: {{ .Values.gpu.count }}
          requests:
            nvidia.com/gpu: {{ .Values.gpu.count }}
          {{- end }}
```

### 5.3 Helm Hooks

**Use hooks to run jobs before/after install/upgrade:**

```yaml
# templates/pre-install-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "ml-model-serving.fullname" . }}-pre-install
  annotations:
    "helm.sh/hook": pre-install
    "helm.sh/hook-weight": "0"
    "helm.sh/hook-delete-policy": hook-succeeded
spec:
  template:
    spec:
      containers:
      - name: model-downloader
        image: amazon/aws-cli
        command: ["sh", "-c"]
        args:
          - |
            aws s3 cp s3://my-models/bert-v1.0.pth /models/
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage-pvc
      restartPolicy: Never
```

**Hook types:**
- `pre-install`: Run before install
- `post-install`: Run after install
- `pre-upgrade`: Run before upgrade
- `post-upgrade`: Run after upgrade
- `pre-delete`: Run before uninstall
- `post-delete`: Run after uninstall

## 6. Helm Best Practices for ML

### 6.1 Versioning Strategy

```yaml
# Use semantic versioning
Chart.yaml:
  version: 1.2.3  # Chart version (increment when chart changes)
  appVersion: "2.0.0"  # Application version (model version)

values.yaml:
  image:
    tag: "v2.0.0"  # Match appVersion
```

**Version management:**

```bash
# Install specific chart version
helm install bert ./ml-model-serving --version 1.2.3

# Upgrade to specific version
helm upgrade bert ./ml-model-serving --version 1.3.0

# List available versions
helm search repo ml-model-serving --versions
```

### 6.2 Secrets Management

**Don't commit secrets to values.yaml!**

**Option 1: Use separate secrets file (encrypted)**

```yaml
# secrets.yaml (DO NOT COMMIT)
postgresql:
  auth:
    password: "super-secret-password"
```

```bash
# Encrypt with git-crypt or SOPS
# Then install
helm install ml-platform ./ml-platform \
  --values values.yaml \
  --values secrets.yaml  # Encrypted secrets
```

**Option 2: Use external secrets management**

```yaml
# Use external-secrets operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ml-platform-secrets
spec:
  secretStoreRef:
    name: aws-secrets-manager
  target:
    name: ml-platform-db-password
  data:
  - secretKey: password
    remoteRef:
      key: prod/ml-platform/db-password
```

**Option 3: Pass values via CLI**

```bash
helm install ml-platform ./ml-platform \
  --set postgresql.auth.password="$(kubectl get secret db-password -o jsonpath='{.data.password}' | base64 -d)"
```

### 6.3 Resource Naming Conventions

```yaml
# _helpers.tpl
{{- define "ml-model.fullname" -}}
{{ .Release.Name }}-{{ .Chart.Name }}
{{- end }}

# Results in predictable names:
# bert-prod-ml-model-serving-deployment
# bert-prod-ml-model-serving-service
# bert-prod-ml-model-serving-ingress
```

### 6.4 Validation with JSON Schema

```yaml
# values.schema.json
{
  "$schema": "https://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["replicaCount", "image"],
  "properties": {
    "replicaCount": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100
    },
    "image": {
      "type": "object",
      "required": ["repository", "tag"],
      "properties": {
        "repository": {"type": "string"},
        "tag": {"type": "string"}
      }
    },
    "resources": {
      "type": "object",
      "properties": {
        "limits": {
          "type": "object",
          "properties": {
            "nvidia.com/gpu": {"type": "integer", "minimum": 0, "maximum": 8}
          }
        }
      }
    }
  }
}
```

**Helm will validate values against schema automatically!**

## 7. Publishing and Sharing Charts

### 7.1 Package a Chart

```bash
# Package chart into .tgz archive
helm package ml-model-serving/
# Successfully packaged chart and saved it to: ml-model-serving-1.0.0.tgz
```

### 7.2 Host Charts in Chart Repository

**Option 1: GitHub Pages (free)**

```bash
# Create gh-pages branch
git checkout --orphan gh-pages
git rm -rf .

# Package charts
helm package ../ml-model-serving
helm package ../ml-platform

# Create index
helm repo index . --url https://mycompany.github.io/helm-charts

# Commit and push
git add .
git commit -m "Initial charts"
git push origin gh-pages

# Users can now add your repo
helm repo add mycompany https://mycompany.github.io/helm-charts
helm repo update
helm search repo mycompany
```

**Option 2: ChartMuseum (self-hosted)**

```bash
# Install ChartMuseum
helm install chartmuseum stable/chartmuseum \
  --set env.open.DISABLE_API=false \
  --set persistence.enabled=true

# Upload chart
curl --data-binary "@ml-model-serving-1.0.0.tgz" \
  http://chartmuseum.example.com/api/charts
```

**Option 3: AWS ECR / GCP Artifact Registry / Azure ACR**

```bash
# AWS ECR example
aws ecr create-repository --repository-name helm-charts
helm push ml-model-serving-1.0.0.tgz oci://123456789.dkr.ecr.us-west-2.amazonaws.com/helm-charts
```

## 8. Debugging Helm Deployments

### 8.1 Common Issues

**Issue 1: Template rendering error**

```bash
helm install bert ./ml-model-serving
# Error: YAML parse error on ml-model-serving/templates/deployment.yaml:
# error converting YAML to JSON: yaml: line 25: mapping values are not allowed in this context

# Debug: Render templates
helm template bert ./ml-model-serving --debug
```

**Issue 2: Release in failed state**

```bash
helm list -n ml-serving
# NAME  STATUS  REVISION
# bert  failed  1

# Check release notes
helm get notes bert -n ml-serving

# Get all manifests of release
helm get manifest bert -n ml-serving

# Check values used
helm get values bert -n ml-serving

# Uninstall and reinstall
helm uninstall bert -n ml-serving
helm install bert ./ml-model-serving -n ml-serving
```

**Issue 3: Upgrade stuck/failed**

```bash
# Rollback to previous revision
helm rollback bert -n ml-serving

# Force upgrade (dangerous!)
helm upgrade bert ./ml-model-serving -n ml-serving --force

# Delete and reinstall (nuclear option)
helm uninstall bert -n ml-serving
helm install bert ./ml-model-serving -n ml-serving
```

## 9. Hands-On Exercise: Deploy ML Platform with Helm

**Objective:** Create a Helm chart that deploys:
- ML model inference service
- Prometheus for monitoring
- Grafana for dashboards

**Step 1: Create chart structure**

```bash
mkdir ml-platform-chart
cd ml-platform-chart
helm create .
```

**Step 2: Define dependencies in Chart.yaml**

```yaml
# Chart.yaml
apiVersion: v2
name: ml-platform
version: 1.0.0
dependencies:
  - name: prometheus
    version: 15.x.x
    repository: https://prometheus-community.github.io/helm-charts
  - name: grafana
    version: 6.x.x
    repository: https://grafana.github.io/helm-charts
```

**Step 3: Configure values**

```yaml
# values.yaml
mlModel:
  enabled: true
  replicaCount: 3
  image:
    repository: myregistry.io/bert-inference
    tag: v1.0.0

prometheus:
  server:
    persistentVolume:
      size: 50Gi

grafana:
  adminPassword: changeme
  persistence:
    enabled: true
    size: 10Gi
```

**Step 4: Install dependencies**

```bash
helm dependency update
```

**Step 5: Deploy**

```bash
helm install ml-platform . \
  --namespace ml-platform \
  --create-namespace \
  --values values.yaml

# Check deployment
helm list -n ml-platform
kubectl get all -n ml-platform
```

## 10. Summary

### Key Takeaways

✅ **Helm is the package manager for Kubernetes**
- Packages are called charts
- Instances are called releases
- Charts are versioned and reusable

✅ **Helm simplifies deployment:**
- One command to deploy complex applications
- Templating eliminates repetitive YAML
- Environment-specific configurations via values

✅ **Creating custom charts:**
- Use `helm create` to scaffold
- Define templates with Go templating
- Use values.yaml for configuration
- Test with `helm lint` and `helm template`

✅ **Best practices:**
- Use semantic versioning
- Separate dev/staging/prod values
- Never commit secrets to values.yaml
- Validate with JSON schema
- Use chart dependencies for complex stacks

✅ **Real-world ML use cases:**
- Deploy ML platforms (MLflow, Kubeflow)
- Package model serving applications
- Manage multi-environment deployments
- Version ML infrastructure

## Self-Check Questions

1. What's the difference between a Helm chart and a Helm release?
2. How do you override default values in a chart?
3. What command would you use to roll back a failed upgrade?
4. How do you create a chart with dependencies?
5. What's the purpose of templates/_helpers.tpl?
6. How would you deploy the same chart to dev, staging, and prod?
7. What are Helm hooks and when would you use them?
8. How do you troubleshoot a chart that won't install?

## Additional Resources

- [Helm Official Documentation](https://helm.sh/docs/)
- [Artifact Hub (chart repository)](https://artifacthub.io/)
- [Helm Best Practices](https://helm.sh/docs/chart_best_practices/)
- [Creating Charts Guide](https://helm.sh/docs/chart_template_guide/)
- [Helm Chart Testing](https://github.com/helm/chart-testing)

---

**Next lesson:** Monitoring and Troubleshooting Kubernetes
