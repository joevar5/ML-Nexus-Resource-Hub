# Exercise 06: Kubernetes Operator for ML Model Lifecycle

**Estimated Time**: 38-46 hours
**Difficulty**: Advanced
**Prerequisites**: Kubernetes, Python 3.9+, Kubernetes Operator Framework (Kopf), Docker

## Overview

Build a production-grade Kubernetes Operator that automates ML model lifecycle management - from deployment to auto-scaling, health monitoring, versioning, and automated rollback. Implement Custom Resource Definitions (CRDs) for ModelDeployment and ModelServer resources. This exercise teaches advanced Kubernetes patterns essential for building platform-level ML infrastructure automation.

In production ML platforms, operators are critical for:
- **Declarative Model Management**: Define models in YAML, operator handles deployment
- **Automated Operations**: Health checks, scaling, rollback without manual intervention
- **Version Management**: Deploy multiple model versions, A/B testing
- **Resource Optimization**: Auto-scale based on queue depth, GPU utilization
- **Self-Healing**: Restart failed pods, redeploy corrupted models

## Learning Objectives

By completing this exercise, you will:

1. **Design Custom Resource Definitions (CRDs)** for ML workloads
2. **Implement Kubernetes Operator** using Kopf framework
3. **Handle resource reconciliation loops** (desired state → actual state)
4. **Manage dependent resources** (Deployments, Services, HPAs, PVCs)
5. **Implement health checking** and auto-remediation
6. **Build status reporting** with conditions and events
7. **Handle upgrades and rollbacks** safely

## Business Context

**Real-World Scenario**: Your ML platform has 50 data scientists deploying 200+ models. Current problems:

- **Manual deployment complexity**: Each model requires Deployment, Service, HPA, PVC, ConfigMap YAML (100+ lines total)
- **Configuration drift**: Engineers manually edit resources, causing inconsistencies
- **No automated health checks**: Failed model loads go undetected for hours
- **Risky updates**: Model updates require manual kubectl commands, error-prone
- **No version history**: Can't easily rollback to previous model versions

Your task: Build an operator that:
- Deploys models from simple 20-line CRD: `kubectl apply -f model.yaml`
- Automatically creates all dependent resources (Deployment, Service, HPA, etc.)
- Monitors model health, auto-restarts on failures
- Manages model versions, supports A/B testing
- Implements safe rollback on errors

## Project Structure

```
exercise-06-k8s-operator-framework/
├── README.md
├── requirements.txt
├── Dockerfile
├── helm/
│   └── model-operator/              # Helm chart for operator
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml      # Operator deployment
│           ├── rbac.yaml            # ServiceAccount, Role, RoleBinding
│           └── crd.yaml             # ModelDeployment CRD
├── config/
│   ├── crd.yaml                     # ModelDeployment CRD definition
│   └── rbac.yaml                    # RBAC for operator
├── src/
│   └── model_operator/
│       ├── __init__.py
│       ├── operator.py              # Main operator logic
│       ├── reconciler.py            # Reconciliation loop
│       ├── resource_manager.py      # Manage K8s resources
│       ├── health_checker.py        # Model health monitoring
│       ├── version_manager.py       # Model versioning
│       ├── scaler.py                # Auto-scaling logic
│       └── utils.py                 # Helper functions
├── tests/
│   ├── test_operator.py
│   ├── test_reconciler.py
│   ├── test_resource_manager.py
│   └── fixtures/
│       └── sample_model_deployment.yaml
├── examples/
│   ├── simple-model.yaml            # Basic model deployment
│   ├── multi-version-model.yaml     # A/B testing example
│   └── gpu-model.yaml               # GPU model deployment
└── docs/
    ├── DESIGN.md                    # Architecture
    ├── CRD_REFERENCE.md             # CRD API spec
    └── OPERATIONS.md                # Operator guide
```

## Requirements

### Functional Requirements

1. **ModelDeployment CRD**:
   - Define model image, resources, replicas
   - Support multiple versions (A/B testing)
   - Configure health check endpoints
   - Set auto-scaling parameters

2. **Operator Capabilities**:
   - Create Deployment for model server
   - Create Service for model endpoint
   - Create HPA for auto-scaling
   - Create PVC for model storage (if needed)
   - Create ConfigMap for model config

3. **Reconciliation Logic**:
   - Detect CRD changes, update resources
   - Handle resource deletions (garbage collection)
   - Maintain desired state continuously
   - Handle conflicts gracefully

4. **Health Monitoring**:
   - Check /health endpoint every 30s
   - Restart pods on consecutive failures
   - Report status in CRD status field
   - Emit Kubernetes events

5. **Version Management**:
   - Deploy multiple model versions simultaneously
   - Traffic splitting between versions
   - Promote/rollback versions declaratively
   - Track version history

### Non-Functional Requirements

- **Reconciliation Speed**: <5 seconds from CRD change to resource update
- **Reliability**: Handle API server failures, retry with backoff
- **Observability**: Log all operations, expose Prometheus metrics
- **Safety**: Never delete resources without confirmation

## Implementation Tasks

### Task 1: CRD Design (6-7 hours)

Design Custom Resource Definition for ML models.

```yaml
# config/crd.yaml

apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: modeldeployments.ml.example.com
spec:
  group: ml.example.com
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              # Model configuration
              modelName:
                type: string
                description: "Name of the ML model"

              framework:
                type: string
                enum: ["tensorflow", "pytorch", "sklearn", "onnx"]
                description: "ML framework"

              # Container image
              image:
                type: string
                description: "Docker image for model server"

              # Model versions (for A/B testing)
              versions:
                type: array
                items:
                  type: object
                  properties:
                    name:
                      type: string  # "v1", "v2", etc.
                    weight:
                      type: integer  # Traffic percentage (0-100)
                      minimum: 0
                      maximum: 100
                    modelPath:
                      type: string  # S3/GCS path to model artifacts

              # Resource requirements
              resources:
                type: object
                properties:
                  requests:
                    type: object
                    properties:
                      cpu:
                        type: string
                      memory:
                        type: string
                      nvidia.com/gpu:
                        type: string
                  limits:
                    type: object
                    properties:
                      cpu:
                        type: string
                      memory:
                        type: string
                      nvidia.com/gpu:
                        type: string

              # Scaling configuration
              scaling:
                type: object
                properties:
                  minReplicas:
                    type: integer
                    minimum: 1
                  maxReplicas:
                    type: integer
                    minimum: 1
                  targetCPUUtilization:
                    type: integer
                    minimum: 1
                    maximum: 100

              # Health check
              healthCheck:
                type: object
                properties:
                  endpoint:
                    type: string  # "/health"
                  intervalSeconds:
                    type: integer
                    default: 30
                  timeoutSeconds:
                    type: integer
                    default: 5
                  failureThreshold:
                    type: integer
                    default: 3

          status:
            type: object
            properties:
              # Overall status
              phase:
                type: string
                enum: ["Pending", "Running", "Failed", "Degraded"]

              # Conditions (Ready, Healthy, Scaled)
              conditions:
                type: array
                items:
                  type: object
                  properties:
                    type:
                      type: string
                    status:
                      type: string
                      enum: ["True", "False", "Unknown"]
                    lastTransitionTime:
                      type: string
                      format: date-time
                    reason:
                      type: string
                    message:
                      type: string

              # Version status
              activeVersions:
                type: array
                items:
                  type: object
                  properties:
                    name:
                      type: string
                    replicas:
                      type: integer
                    healthyReplicas:
                      type: integer

              # Metrics
              metrics:
                type: object
                properties:
                  requestsPerSecond:
                    type: number
                  averageLatencyMs:
                    type: number
                  errorRate:
                    type: number

    subresources:
      status: {}  # Enable status subresource

    additionalPrinterColumns:
    - name: Framework
      type: string
      jsonPath: .spec.framework
    - name: Phase
      type: string
      jsonPath: .status.phase
    - name: Replicas
      type: integer
      jsonPath: .status.activeVersions[0].replicas
    - name: Age
      type: date
      jsonPath: .metadata.creationTimestamp

  scope: Namespaced
  names:
    plural: modeldeployments
    singular: modeldeployment
    kind: ModelDeployment
    shortNames:
    - md
```

**Example ModelDeployment**:

```yaml
# examples/simple-model.yaml

apiVersion: ml.example.com/v1
kind: ModelDeployment
metadata:
  name: sentiment-analysis
  namespace: ml-models
spec:
  modelName: sentiment-classifier
  framework: pytorch
  image: my-registry/sentiment-model:v1.0.0

  versions:
  - name: v1
    weight: 100
    modelPath: s3://models/sentiment/v1/model.pt

  resources:
    requests:
      cpu: "1"
      memory: "2Gi"
    limits:
      cpu: "2"
      memory: "4Gi"

  scaling:
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilization: 70

  healthCheck:
    endpoint: /health
    intervalSeconds: 30
    failureThreshold: 3
```

**Acceptance Criteria**:
- ✅ CRD validates all fields correctly
- ✅ Support for multiple model versions
- ✅ Status subresource enabled
- ✅ Additional printer columns display correctly
- ✅ Short name `md` works

---

### Task 2: Operator Framework Setup (7-9 hours)

Implement operator using Kopf framework.

```python
# src/model_operator/operator.py

import kopf
import kubernetes
from typing import Dict, Any
import logging
from .reconciler import ModelReconciler
from .health_checker import HealthChecker
from .version_manager import VersionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Kubernetes client
kubernetes.config.load_incluster_config()

# Initialize components
reconciler = ModelReconciler()
health_checker = HealthChecker()
version_manager = VersionManager()

@kopf.on.create('ml.example.com', 'v1', 'modeldeployments')
def create_model_deployment(spec, name, namespace, **kwargs):
    """
    Handle ModelDeployment creation

    Called when: kubectl apply -f model.yaml (first time)

    Tasks:
    1. Validate spec
    2. Create Deployment
    3. Create Service
    4. Create HPA
    5. Update status

    Args:
        spec: ModelDeployment spec
        name: Resource name
        namespace: Namespace
        **kwargs: Additional context (body, meta, status, etc.)
    """
    logger.info(f"Creating ModelDeployment: {namespace}/{name}")

    try:
        # TODO: Validate spec
        _validate_spec(spec)

        # TODO: Create resources
        reconciler.reconcile(name, namespace, spec)

        # TODO: Update status to "Pending"
        _update_status(name, namespace, phase="Pending", message="Resources created")

        logger.info(f"ModelDeployment {namespace}/{name} created successfully")

    except Exception as e:
        logger.error(f"Failed to create ModelDeployment {namespace}/{name}: {e}")
        _update_status(name, namespace, phase="Failed", message=str(e))
        raise

@kopf.on.update('ml.example.com', 'v1', 'modeldeployments')
def update_model_deployment(spec, name, namespace, old, new, diff, **kwargs):
    """
    Handle ModelDeployment updates

    Called when: kubectl apply -f model.yaml (changes detected)

    Tasks:
    1. Detect what changed (image, replicas, versions, etc.)
    2. Update corresponding resources
    3. Handle version changes (A/B testing)

    Args:
        old: Previous spec
        new: New spec
        diff: List of changes
    """
    logger.info(f"Updating ModelDeployment: {namespace}/{name}")
    logger.info(f"Changes: {diff}")

    try:
        # TODO: Reconcile to desired state
        reconciler.reconcile(name, namespace, spec)

        # TODO: Handle version changes
        if _versions_changed(old, new):
            version_manager.update_traffic_split(name, namespace, spec['versions'])

        _update_status(name, namespace, phase="Running", message="Updated successfully")

    except Exception as e:
        logger.error(f"Failed to update ModelDeployment {namespace}/{name}: {e}")
        _update_status(name, namespace, phase="Failed", message=str(e))
        raise

@kopf.on.delete('ml.example.com', 'v1', 'modeldeployments')
def delete_model_deployment(spec, name, namespace, **kwargs):
    """
    Handle ModelDeployment deletion

    Called when: kubectl delete modeldeployment <name>

    Tasks:
    1. Delete Deployment
    2. Delete Service
    3. Delete HPA
    4. Cleanup PVCs (if any)

    Note: Kubernetes handles garbage collection via ownerReferences
    """
    logger.info(f"Deleting ModelDeployment: {namespace}/{name}")

    try:
        # Resources will be auto-deleted via ownerReferences
        # Just log the deletion
        logger.info(f"ModelDeployment {namespace}/{name} deleted")

    except Exception as e:
        logger.error(f"Error during deletion: {e}")
        raise

@kopf.daemon('ml.example.com', 'v1', 'modeldeployments')
async def health_check_daemon(spec, name, namespace, stopped, **kwargs):
    """
    Background daemon for health checking

    Runs continuously while ModelDeployment exists.
    Checks model health every 30 seconds.

    Args:
        stopped: Event that signals daemon should stop
    """
    import asyncio

    interval = spec.get('healthCheck', {}).get('intervalSeconds', 30)

    logger.info(f"Starting health check daemon for {namespace}/{name}")

    while not stopped.is_set():
        try:
            # TODO: Check health
            is_healthy = await health_checker.check_health(name, namespace, spec)

            # TODO: Update status
            if not is_healthy:
                _update_status(
                    name,
                    namespace,
                    phase="Degraded",
                    message="Health check failing"
                )
                # TODO: Trigger remediation (restart pods)
                await health_checker.remediate(name, namespace)

        except Exception as e:
            logger.error(f"Health check error for {namespace}/{name}: {e}")

        # Wait for next check
        await asyncio.sleep(interval)

@kopf.timer('ml.example.com', 'v1', 'modeldeployments', interval=60.0)
def status_update_timer(spec, name, namespace, **kwargs):
    """
    Periodic status update timer

    Runs every 60 seconds to update metrics in status
    """
    try:
        # TODO: Fetch metrics from Prometheus
        metrics = _fetch_metrics(name, namespace)

        # TODO: Update status with metrics
        _update_status(
            name,
            namespace,
            metrics=metrics
        )

    except Exception as e:
        logger.error(f"Failed to update status for {namespace}/{name}: {e}")

def _validate_spec(spec: Dict[str, Any]) -> None:
    """Validate ModelDeployment spec"""
    required_fields = ['modelName', 'framework', 'image', 'versions']
    for field in required_fields:
        if field not in spec:
            raise ValueError(f"Missing required field: {field}")

    # Validate version weights sum to 100
    total_weight = sum(v['weight'] for v in spec['versions'])
    if total_weight != 100:
        raise ValueError(f"Version weights must sum to 100, got {total_weight}")

def _versions_changed(old: Dict, new: Dict) -> bool:
    """Check if versions changed"""
    return old.get('spec', {}).get('versions') != new.get('spec', {}).get('versions')

def _update_status(name: str, namespace: str, phase: str = None, message: str = None, metrics: Dict = None):
    """Update ModelDeployment status"""
    # TODO: Update status subresource via Kubernetes API
    raise NotImplementedError

def _fetch_metrics(name: str, namespace: str) -> Dict:
    """Fetch metrics from Prometheus"""
    # TODO: Query Prometheus for RPS, latency, error rate
    return {
        "requestsPerSecond": 0.0,
        "averageLatencyMs": 0.0,
        "errorRate": 0.0
    }

if __name__ == '__main__':
    kopf.run()
```

**Acceptance Criteria**:
- ✅ Operator handles create/update/delete events
- ✅ Background health checking works
- ✅ Periodic status updates
- ✅ Proper error handling and logging

---

### Task 3: Resource Manager (8-10 hours)

Manage dependent Kubernetes resources.

```python
# src/model_operator/resource_manager.py

from kubernetes import client
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manage Kubernetes resources for ModelDeployment"""

    def __init__(self):
        self.apps_api = client.AppsV1Api()
        self.core_api = client.CoreV1Api()
        self.autoscaling_api = client.AutoscalingV2Api()

    def create_or_update_deployment(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any],
        owner_reference: Dict
    ) -> client.V1Deployment:
        """
        Create or update Deployment for model server

        Deployment spec:
        - Replicas: spec.scaling.minReplicas
        - Container: spec.image
        - Resources: spec.resources
        - Env vars: MODEL_PATH, FRAMEWORK, etc.
        - Volume mounts: For model storage (if using PVC)

        Args:
            owner_reference: For garbage collection
        """
        deployment_name = f"{name}-deployment"

        # TODO: Build Deployment spec
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=deployment_name,
                namespace=namespace,
                owner_references=[owner_reference],
                labels={
                    "app": name,
                    "managed-by": "model-operator"
                }
            ),
            spec=client.V1DeploymentSpec(
                replicas=spec.get('scaling', {}).get('minReplicas', 1),
                selector=client.V1LabelSelector(
                    match_labels={"app": name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": name}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="model-server",
                                image=spec['image'],
                                ports=[client.V1ContainerPort(container_port=8080)],
                                env=[
                                    client.V1EnvVar(name="MODEL_NAME", value=spec['modelName']),
                                    client.V1EnvVar(name="MODEL_PATH", value=spec['versions'][0]['modelPath']),
                                    client.V1EnvVar(name="FRAMEWORK", value=spec['framework'])
                                ],
                                resources=self._build_resources(spec.get('resources', {})),
                                liveness_probe=self._build_probe(spec.get('healthCheck', {})),
                                readiness_probe=self._build_probe(spec.get('healthCheck', {}))
                            )
                        ]
                    )
                )
            )
        )

        # TODO: Create or update
        try:
            self.apps_api.read_namespaced_deployment(deployment_name, namespace)
            # Exists, update it
            self.apps_api.patch_namespaced_deployment(deployment_name, namespace, deployment)
            logger.info(f"Updated Deployment {namespace}/{deployment_name}")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Doesn't exist, create it
                self.apps_api.create_namespaced_deployment(namespace, deployment)
                logger.info(f"Created Deployment {namespace}/{deployment_name}")
            else:
                raise

        return deployment

    def create_or_update_service(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any],
        owner_reference: Dict
    ) -> client.V1Service:
        """
        Create or update Service for model endpoint

        Service spec:
        - Type: ClusterIP
        - Selector: app=<name>
        - Port: 8080
        """
        service_name = f"{name}-service"

        service = client.V1Service(
            metadata=client.V1ObjectMeta(
                name=service_name,
                namespace=namespace,
                owner_references=[owner_reference],
                labels={"app": name}
            ),
            spec=client.V1ServiceSpec(
                selector={"app": name},
                ports=[
                    client.V1ServicePort(
                        name="http",
                        port=8080,
                        target_port=8080,
                        protocol="TCP"
                    )
                ],
                type="ClusterIP"
            )
        )

        # TODO: Create or update
        try:
            self.core_api.read_namespaced_service(service_name, namespace)
            self.core_api.patch_namespaced_service(service_name, namespace, service)
            logger.info(f"Updated Service {namespace}/{service_name}")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                self.core_api.create_namespaced_service(namespace, service)
                logger.info(f"Created Service {namespace}/{service_name}")
            else:
                raise

        return service

    def create_or_update_hpa(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any],
        owner_reference: Dict
    ) -> client.V2HorizontalPodAutoscaler:
        """
        Create or update HorizontalPodAutoscaler

        HPA spec:
        - Min/max replicas from spec.scaling
        - Target CPU utilization
        """
        hpa_name = f"{name}-hpa"

        scaling_config = spec.get('scaling', {})

        hpa = client.V2HorizontalPodAutoscaler(
            metadata=client.V1ObjectMeta(
                name=hpa_name,
                namespace=namespace,
                owner_references=[owner_reference],
                labels={"app": name}
            ),
            spec=client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=f"{name}-deployment"
                ),
                min_replicas=scaling_config.get('minReplicas', 1),
                max_replicas=scaling_config.get('maxReplicas', 10),
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=scaling_config.get('targetCPUUtilization', 70)
                            )
                        )
                    )
                ]
            )
        )

        # TODO: Create or update
        try:
            self.autoscaling_api.read_namespaced_horizontal_pod_autoscaler(hpa_name, namespace)
            self.autoscaling_api.patch_namespaced_horizontal_pod_autoscaler(hpa_name, namespace, hpa)
            logger.info(f"Updated HPA {namespace}/{hpa_name}")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                self.autoscaling_api.create_namespaced_horizontal_pod_autoscaler(namespace, hpa)
                logger.info(f"Created HPA {namespace}/{hpa_name}")
            else:
                raise

        return hpa

    def _build_resources(self, resources: Dict) -> client.V1ResourceRequirements:
        """Build resource requirements"""
        return client.V1ResourceRequirements(
            requests=resources.get('requests', {}),
            limits=resources.get('limits', {})
        )

    def _build_probe(self, health_check: Dict) -> client.V1Probe:
        """Build liveness/readiness probe"""
        endpoint = health_check.get('endpoint', '/health')
        return client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path=endpoint,
                port=8080
            ),
            initial_delay_seconds=10,
            period_seconds=health_check.get('intervalSeconds', 30),
            timeout_seconds=health_check.get('timeoutSeconds', 5),
            failure_threshold=health_check.get('failureThreshold', 3)
        )
```

**Acceptance Criteria**:
- ✅ Create Deployment with correct spec
- ✅ Create Service exposing model endpoint
- ✅ Create HPA for auto-scaling
- ✅ Set ownerReferences for garbage collection
- ✅ Handle create vs update correctly

---

### Task 4: Reconciler (6-7 hours)

Implement reconciliation loop.

```python
# src/model_operator/reconciler.py

from .resource_manager import ResourceManager
from kubernetes import client
import logging

logger = logging.getLogger(__name__)

class ModelReconciler:
    """Reconcile desired state (CRD spec) with actual state (K8s resources)"""

    def __init__(self):
        self.resource_manager = ResourceManager()
        self.custom_api = client.CustomObjectsApi()

    def reconcile(self, name: str, namespace: str, spec: dict):
        """
        Reconcile ModelDeployment to desired state

        Steps:
        1. Get owner reference (for garbage collection)
        2. Create/update Deployment
        3. Create/update Service
        4. Create/update HPA
        5. Verify resources are healthy

        Args:
            name: ModelDeployment name
            namespace: Namespace
            spec: ModelDeployment spec
        """
        logger.info(f"Reconciling ModelDeployment {namespace}/{name}")

        # TODO: Get ModelDeployment for owner reference
        model_deployment = self.custom_api.get_namespaced_custom_object(
            group="ml.example.com",
            version="v1",
            namespace=namespace,
            plural="modeldeployments",
            name=name
        )

        owner_reference = {
            "apiVersion": "ml.example.com/v1",
            "kind": "ModelDeployment",
            "name": name,
            "uid": model_deployment['metadata']['uid'],
            "controller": True,
            "blockOwnerDeletion": True
        }

        # TODO: Create/update resources
        try:
            deployment = self.resource_manager.create_or_update_deployment(
                name, namespace, spec, owner_reference
            )

            service = self.resource_manager.create_or_update_service(
                name, namespace, spec, owner_reference
            )

            hpa = self.resource_manager.create_or_update_hpa(
                name, namespace, spec, owner_reference
            )

            logger.info(f"Reconciliation complete for {namespace}/{name}")

        except Exception as e:
            logger.error(f"Reconciliation failed for {namespace}/{name}: {e}")
            raise

    def get_status(self, name: str, namespace: str) -> dict:
        """
        Get current status of ModelDeployment

        Query Deployment, Service, HPA to determine:
        - Phase (Pending, Running, Failed, Degraded)
        - Active replicas
        - Health status
        """
        # TODO: Get Deployment status
        apps_api = client.AppsV1Api()
        deployment_name = f"{name}-deployment"

        try:
            deployment = apps_api.read_namespaced_deployment(deployment_name, namespace)
            replicas = deployment.status.replicas or 0
            ready_replicas = deployment.status.ready_replicas or 0

            if ready_replicas == 0:
                phase = "Pending"
            elif ready_replicas < replicas:
                phase = "Degraded"
            else:
                phase = "Running"

            return {
                "phase": phase,
                "replicas": replicas,
                "readyReplicas": ready_replicas
            }

        except client.exceptions.ApiException as e:
            if e.status == 404:
                return {"phase": "Pending", "replicas": 0, "readyReplicas": 0}
            raise
```

**Acceptance Criteria**:
- ✅ Reconcile creates all resources
- ✅ Reconcile updates resources on spec changes
- ✅ Owner references set correctly
- ✅ Status reflects actual state
- ✅ Handles errors gracefully

---

### Task 5: Health Checker (4-5 hours)

Implement health monitoring and remediation.

```python
# src/model_operator/health_checker.py

from kubernetes import client
import aiohttp
import logging

logger = logging.getLogger(__name__)

class HealthChecker:
    """Monitor model health and trigger remediation"""

    def __init__(self):
        self.core_api = client.CoreV1Api()
        self.apps_api = client.AppsV1Api()

    async def check_health(self, name: str, namespace: str, spec: dict) -> bool:
        """
        Check health of all model pods

        Calls /health endpoint on each pod
        Returns True if >80% of pods healthy
        """
        # TODO: Get all pods for deployment
        pod_list = self.core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app={name}"
        )

        if not pod_list.items:
            logger.warning(f"No pods found for {namespace}/{name}")
            return False

        # TODO: Check health of each pod
        healthy_count = 0
        total_count = len(pod_list.items)

        health_endpoint = spec.get('healthCheck', {}).get('endpoint', '/health')

        async with aiohttp.ClientSession() as session:
            for pod in pod_list.items:
                pod_ip = pod.status.pod_ip
                if not pod_ip:
                    continue

                try:
                    async with session.get(
                        f"http://{pod_ip}:8080{health_endpoint}",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            healthy_count += 1
                        else:
                            logger.warning(f"Pod {pod.metadata.name} health check failed: {response.status}")

                except Exception as e:
                    logger.error(f"Health check error for pod {pod.metadata.name}: {e}")

        health_percentage = healthy_count / total_count if total_count > 0 else 0
        is_healthy = health_percentage >= 0.8

        logger.info(f"Health check for {namespace}/{name}: {healthy_count}/{total_count} healthy")

        return is_healthy

    async def remediate(self, name: str, namespace: str):
        """
        Remediate unhealthy deployment

        Strategies:
        1. Delete unhealthy pods (let deployment recreate)
        2. Rollback to previous version (if applicable)
        3. Scale down to min replicas and back up
        """
        logger.info(f"Remediating {namespace}/{name}")

        # TODO: Delete unhealthy pods
        pod_list = self.core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app={name}"
        )

        deleted_count = 0
        for pod in pod_list.items:
            # TODO: Check if pod is unhealthy
            # For now, just delete pods with restartCount > 5
            if pod.status.container_statuses:
                restart_count = pod.status.container_statuses[0].restart_count
                if restart_count > 5:
                    logger.info(f"Deleting unhealthy pod: {pod.metadata.name}")
                    self.core_api.delete_namespaced_pod(pod.metadata.name, namespace)
                    deleted_count += 1

        logger.info(f"Deleted {deleted_count} unhealthy pods")
```

**Acceptance Criteria**:
- ✅ Check health endpoints
- ✅ Calculate health percentage
- ✅ Delete unhealthy pods
- ✅ Async implementation
- ✅ Proper error handling

---

### Task 6: Version Manager (4-5 hours)

Implement version management for A/B testing.

```python
# src/model_operator/version_manager.py

from kubernetes import client
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class VersionManager:
    """Manage multiple model versions for A/B testing"""

    def __init__(self):
        self.apps_api = client.AppsV1Api()
        self.custom_api = client.CustomObjectsApi()

    def update_traffic_split(self, name: str, namespace: str, versions: List[Dict]):
        """
        Update traffic split between versions using Istio VirtualService

        Example versions:
        [
            {"name": "v1", "weight": 80, "modelPath": "s3://..."},
            {"name": "v2", "weight": 20, "modelPath": "s3://..."}
        ]

        Creates/updates Istio VirtualService to route traffic
        """
        # TODO: Verify weights sum to 100
        total_weight = sum(v['weight'] for v in versions)
        if total_weight != 100:
            raise ValueError(f"Version weights must sum to 100, got {total_weight}")

        # TODO: Create Deployment for each version
        for version in versions:
            if version['weight'] > 0:
                self._create_version_deployment(name, namespace, version)

        # TODO: Update VirtualService (if using Istio)
        # For simplicity, assuming Istio is available
        try:
            self._update_virtual_service(name, namespace, versions)
        except Exception as e:
            logger.warning(f"Could not update VirtualService: {e}")

        logger.info(f"Updated traffic split for {namespace}/{name}")

    def _create_version_deployment(self, name: str, namespace: str, version: Dict):
        """Create Deployment for specific model version"""
        deployment_name = f"{name}-{version['name']}"

        # TODO: Create Deployment with version-specific config
        # Similar to ResourceManager.create_or_update_deployment
        # but with version-specific labels and env vars

        logger.info(f"Created/updated deployment for version {version['name']}")

    def _update_virtual_service(self, name: str, namespace: str, versions: List[Dict]):
        """Update Istio VirtualService for traffic splitting"""
        vs_name = f"{name}-vs"

        # Build route rules
        routes = []
        for version in versions:
            if version['weight'] > 0:
                routes.append({
                    "destination": {
                        "host": f"{name}-service",
                        "subset": version['name']
                    },
                    "weight": version['weight']
                })

        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": vs_name,
                "namespace": namespace
            },
            "spec": {
                "hosts": [f"{name}-service"],
                "http": [{"route": routes}]
            }
        }

        # TODO: Create or update VirtualService
        try:
            self.custom_api.patch_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=namespace,
                plural="virtualservices",
                name=vs_name,
                body=virtual_service
            )
        except client.exceptions.ApiException as e:
            if e.status == 404:
                self.custom_api.create_namespaced_custom_object(
                    group="networking.istio.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="virtualservices",
                    body=virtual_service
                )
```

**Acceptance Criteria**:
- ✅ Support multiple model versions
- ✅ Traffic splitting with weights
- ✅ Create separate deployments per version
- ✅ Integrate with Istio VirtualService
- ✅ Validate weight sums to 100

---

### Task 7: Testing and Documentation (4-5 hours)

```python
# tests/test_operator.py

import pytest
from unittest.mock import Mock, patch
from model_operator.operator import _validate_spec

def test_validate_spec_success():
    """Test valid spec passes validation"""
    spec = {
        "modelName": "test-model",
        "framework": "pytorch",
        "image": "test:v1",
        "versions": [{"name": "v1", "weight": 100, "modelPath": "s3://..."}]
    }
    _validate_spec(spec)  # Should not raise

def test_validate_spec_missing_field():
    """Test missing required field fails"""
    spec = {"modelName": "test-model"}
    with pytest.raises(ValueError, match="Missing required field"):
        _validate_spec(spec)

def test_validate_spec_weights():
    """Test version weights must sum to 100"""
    spec = {
        "modelName": "test-model",
        "framework": "pytorch",
        "image": "test:v1",
        "versions": [
            {"name": "v1", "weight": 60, "modelPath": "s3://..."},
            {"name": "v2", "weight": 30, "modelPath": "s3://..."}
        ]
    }
    with pytest.raises(ValueError, match="weights must sum to 100"):
        _validate_spec(spec)
```

## Expected Results

| Metric | Target | Measured |
|--------|--------|----------|
| **Reconciliation Time** | <5s | ________s |
| **Health Check Interval** | 30s | ________s |
| **CRD Validation** | 100% | ________% |

## Validation

Submit:
1. Complete operator implementation
2. CRD with validation
3. Test suite (>75% coverage)
4. Example ModelDeployments
5. Documentation (DESIGN.md, CRD_REFERENCE.md, OPERATIONS.md)

## Resources

- [Kubernetes Operators](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Kopf Framework](https://kopf.readthedocs.io/)
- [Kubebuilder](https://book.kubebuilder.io/)
- [Operator SDK](https://sdk.operatorframework.io/)

---

**Estimated Completion Time**: 38-46 hours

**Skills Practiced**:
- Kubernetes operators
- Custom Resource Definitions
- Reconciliation loops
- Resource management
- Health monitoring
- Version management
