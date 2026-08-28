# Exercise 11: Build a Custom Kubernetes Operator

**Duration:** 4 hours
**Difficulty:** Advanced
**Prerequisites:** Kubebuilder installed; Go familiarity

## Objective

Build a Kubernetes Operator using kubebuilder that manages a `ModelDeployment` CRD. The operator reconciles each `ModelDeployment` into a Deployment + Service + HPA + ServiceMonitor.

## Why this matters

Operators encode operational knowledge ("how do we deploy a model?") as code. Once written, every model deploys identically with no copy-paste. Teams that build operators have order-of-magnitude leverage over teams that hand-write manifests.

## Requirements

1. CRD `ModelDeployment` with spec.image, spec.replicas, spec.modelVersion, spec.resources.
2. Controller reconciling to: Deployment, Service, HPA, ServiceMonitor.
3. Status updates: replicas, condition (Healthy/Progressing/Failed), lastModelChange.
4. Finalizer cleaning up resources on delete.
5. Tests with envtest.

## Step-by-step

### Step 1 — Scaffold (30 min)
```bash
kubebuilder init --domain example.com --repo example.com/model-operator
kubebuilder create api --group ml --version v1alpha1 --kind ModelDeployment
```

### Step 2 — Define CRD types (30 min)
```go
// api/v1alpha1/modeldeployment_types.go
type ModelDeploymentSpec struct {
    Image        string                       `json:"image"`
    ModelVersion string                       `json:"modelVersion"`
    Replicas     *int32                       `json:"replicas,omitempty"`
    Resources    corev1.ResourceRequirements  `json:"resources,omitempty"`
}

type ModelDeploymentStatus struct {
    ReadyReplicas    int32              `json:"readyReplicas"`
    Conditions       []metav1.Condition `json:"conditions,omitempty"`
    LastModelChange  metav1.Time        `json:"lastModelChange,omitempty"`
}
```

### Step 3 — Reconcile loop (90 min)
```go
// internal/controller/modeldeployment_controller.go
func (r *ModelDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    var md mlv1alpha1.ModelDeployment
    if err := r.Get(ctx, req.NamespacedName, &md); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }
    
    // 1. Build desired Deployment from md.Spec
    desiredDep := buildDeployment(&md)
    if err := r.applyOwned(ctx, &md, desiredDep); err != nil { return ctrl.Result{}, err }
    
    // 2. Service
    desiredSvc := buildService(&md)
    if err := r.applyOwned(ctx, &md, desiredSvc); err != nil { return ctrl.Result{}, err }
    
    // 3. HPA
    desiredHPA := buildHPA(&md)
    if err := r.applyOwned(ctx, &md, desiredHPA); err != nil { return ctrl.Result{}, err }
    
    // 4. ServiceMonitor (if monitoring CRDs present)
    desiredSM := buildServiceMonitor(&md)
    if err := r.applyOwned(ctx, &md, desiredSM); err != nil { return ctrl.Result{}, err }
    
    // Update status
    return ctrl.Result{}, r.updateStatus(ctx, &md)
}
```

### Step 4 — Finalizer (15 min)
```go
const finalizerName = "modeldeployment.ml.example.com/finalizer"

// In Reconcile:
if md.DeletionTimestamp != nil {
    if controllerutil.ContainsFinalizer(&md, finalizerName) {
        // cleanup: delete owned PVCs, S3 objects, etc.
        controllerutil.RemoveFinalizer(&md, finalizerName)
        return ctrl.Result{}, r.Update(ctx, &md)
    }
    return ctrl.Result{}, nil
}
if !controllerutil.ContainsFinalizer(&md, finalizerName) {
    controllerutil.AddFinalizer(&md, finalizerName)
    return ctrl.Result{}, r.Update(ctx, &md)
}
```

### Step 5 — Tests with envtest (45 min)
```go
// internal/controller/modeldeployment_controller_test.go
var _ = Describe("ModelDeployment controller", func() {
    It("creates a Deployment", func() {
        md := &mlv1alpha1.ModelDeployment{...}
        Expect(k8sClient.Create(ctx, md)).Should(Succeed())
        
        depKey := types.NamespacedName{Name: md.Name, Namespace: md.Namespace}
        var dep appsv1.Deployment
        Eventually(func() error { return k8sClient.Get(ctx, depKey, &dep) },
                   time.Second*10, time.Millisecond*100).Should(Succeed())
    })
})
```

### Step 6 — Deploy + try it (30 min)
```bash
make manifests install
make run                    # local controller against cluster
# In another terminal:
kubectl apply -f - <<EOF
apiVersion: ml.example.com/v1alpha1
kind: ModelDeployment
metadata: { name: iris }
spec:
  image: iris-api:0.2
  modelVersion: v3.2.1
  replicas: 3
EOF

kubectl get all -l ml.example.com/model=iris
kubectl get modeldeployment iris -o yaml   # status populated
```

## Deliverables

1. Operator source (in `controllers/` and `api/`).
2. CRD installed.
3. At least 1 `ModelDeployment` resource creating all child resources.
4. envtest passing.
5. `OPERATOR_DESIGN.md` explaining the reconcile philosophy.

## Validation

- [ ] Creating a ModelDeployment creates Deployment + Service + HPA + (optional) ServiceMonitor.
- [ ] Editing the ModelDeployment's `image` triggers a rollout.
- [ ] Deleting the ModelDeployment removes all child resources.
- [ ] Tests pass.

## Stretch goals

- Add **webhook validation**: reject ModelDeployments with invalid specs at admission.
- Add **OpenAPI schema** validation on the CRD.
- Add **events** emitted via `r.Recorder.Event(...)`.

## Common pitfalls

- **Reconcile loop infinite update** — Re-applying owned resources triggers another reconcile. Use `apply` patches or check semantic equality first.
- **Forgetting OwnerReference** — Orphan resources after parent deletion. Always set via `controllerutil.SetControllerReference`.
- **Finalizer never removed** — Resource stuck Terminating forever. Test the delete path.
