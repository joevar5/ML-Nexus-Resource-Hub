# Module 04: Kubernetes Fundamentals

**Duration:** 60 hours · **Difficulty:** Intermediate to Advanced · **Prerequisites:** Modules 02–03

## Overview

Docker gets one container running reliably. Kubernetes is what happens when you need hundreds of them to survive node failures, traffic spikes, and 3am deploys without a human babysitting them. This module takes you from "what even is a pod" to running a real ML system on a cluster — deployments, GPU scheduling, autoscaling, and the debugging skills to fix it when it inevitably breaks.

## What you'll walk away with

- A working mental model of K8s architecture — control plane, nodes, and how scheduling decisions actually get made
- The ability to deploy, network, and scale an ML inference service with confidence
- GPU workloads running correctly on a cluster (this is where most people get stuck the first time)
- Helm charts you'd trust to package a real application
- A troubleshooting reflex: logs → describe → events, instead of guessing

## Lessons

| # | Lesson | Hours |
|---|--------|-------|
| 01 | [Kubernetes Introduction](./01-k8s-introduction.md) — containers vs. orchestration, local cluster, first deploy | 6 |
| 02 | [Kubernetes Architecture](./02-k8s-architecture.md) — control plane, nodes, how scheduling works | 8 |
| 03 | [Core Resources](./03-core-resources.md) — pods, multi-container patterns, namespaces, labels | 7 |
| 04 | [Deployments & Services](./04-deployments-services.md) — rolling updates, canary, service discovery | 8 |
| 05 | [Networking & Ingress](./05-networking-ingress.md) — CNI, network policies, TLS, ingress controllers | 7 |
| 06 | [Storage & Persistence](./06-storage-persistence.md) — PVs/PVCs, StorageClasses, StatefulSets | 7 |
| 07 | [Helm Package Manager](./07-helm-package-manager.md) — charts, values, templating | 6 |
| 08 | [GPU Scheduling](./08-gpu-scheduling.md) — device plugins, GPU sharing, multi-GPU pods | 5 |
| 09 | [Monitoring & Troubleshooting](./09-monitoring-troubleshooting.md) — metrics, logs, health probes, debugging | 6 |

## Hands-on

1. **First ML deployment** — 3 replicas of an inference service, live
2. **Full stack** — model server + Redis + PostgreSQL, wired with real networking
3. **Autoscaling** — HPA scaling pods on CPU/memory
4. **Ingress with TLS** — a real HTTPS endpoint for your ML API
5. **GPU deployment** — GPU-accelerated inference with resource limits set correctly

## Before you start

```bash
kubectl version --client
minikube status        # or: kubectl cluster-info (Docker Desktop K8s)
helm version            # optional, installed during the module
```

Local cluster (minikube, kind, or Docker Desktop) is enough for everything except the optional cloud-cluster exercises.

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 25 questions, 70% to pass
- **Practical** — deploy a complete ML system with 3+ replicas, load-balanced service, TLS ingress, ConfigMaps/Secrets, persistent storage, resource limits, health checks, and HPA

## The mistakes everyone makes here

- **CrashLoopBackOff / ImagePullBackOff** — `kubectl logs` and `kubectl describe pod` first, always
- **Service unreachable** — check that the service's selector actually matches your pod labels
- **Pods stuck Pending** — usually a resource request the cluster can't satisfy; check `kubectl top nodes`
- **GPU not detected inside a pod** — almost always the device plugin isn't installed or the node isn't labeled

## What's next

Pass the quiz, ship the practical, then choose your direction: **Module 05 (Data Pipelines)** or **Module 06 (MLOps)** — both build directly on what you deployed here.

More reading in [`resources.md`](./resources.md).
